"""Pace a Studio recording session: one scripted demonstration per episode.

At the session one person moved the leader arm and the other clicked **Start episode** and
**Accept** in Studio. Here the demonstrator is the leader arm (autopilot ``leader`` mode) and
this script tells you when to click, so each recorded episode is exactly one pick-and-place:

1. it turns the autopilot off and resets the scene (arm at home, cubes at new positions);
2. you click **Start episode** in Studio and press Enter here;
3. it turns the demonstrator on; the arm picks the episode's cube (picked at random) and places
   it in the PLACE AREA,
   recovering from a deliberate failure if this episode has one, and goes back home;
4. it turns the autopilot off again and you click **Accept** in Studio (or **Discard**).

Start the simulation with run_sim.py first, connect Studio's environment and turn on
**Teleoperate**. The episode judge's auto-reset is held off (120 s) while this runs, so the cubes
stay put until you accept.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

FAIL_KEYS = ("grasp_failed", "dropped", "timeouts", "missed")


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310
        return json.load(response)


def post(url: str, body: dict | None = None) -> None:
    data = json.dumps(body or {}).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")  # noqa: S310
    with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
        response.read()


def counters(url: str) -> dict:
    """The demonstrator's counters since the simulation started.

    Read from the episode judge's status: the plugin blanks the autopilot's own fields while it is off.
    """
    return get(f"{url}/health").get("episode", {}).get("demo") or {}


def run_episode(url: str, index: int, total: int, timeout_s: float) -> bool:
    """Run one demonstration; return whether it ended with the cube placed and the arm home."""
    post(f"{url}/autopilot", {"mode": "off"})
    post(f"{url}/reset")
    time.sleep(1.0)
    input(f"\n[{index}/{total}] Scene reset. Click 'Start episode' in Studio, then press Enter here... ")

    before = counters(url)
    post(f"{url}/autopilot", {"mode": "leader"})
    started, placed_seen = time.monotonic(), False
    while time.monotonic() - started < timeout_s:
        time.sleep(0.5)
        health = get(f"{url}/health")
        state, episode = health.get("autopilot", {}), health.get("episode", {})
        placed_seen = placed_seen or episode.get("phase") == "success_hold"
        if placed_seen and state.get("phase") == "idle" and not state.get("waiting_for_arm"):
            time.sleep(1.0)  # a moment of the arm resting at home ends the episode
            break
    else:
        post(f"{url}/autopilot", {"mode": "off"})
        print(f"  Not finished within {timeout_s:.0f} s. Click 'Discard' in Studio.")
        return False
    post(f"{url}/autopilot", {"mode": "off"})

    after = counters(url)
    if not after:
        print(f"  Done in {time.monotonic() - started:.1f} s. Click 'Accept' in Studio.")
        return True
    injected = [kind for kind in ("miss", "drop") if (after.get(f"injected_{kind}") or 0) > (before.get(f"injected_{kind}") or 0)]
    failures = sum((after.get(key) or 0) - (before.get(key) or 0) for key in FAIL_KEYS)
    note = f"deliberate {'/'.join(injected)} failure, {failures} failed tries, recovered" if injected else "clean"
    picked_before, picked_after = before.get("picked") or {}, after.get("picked") or {}
    cubes = [name.removeprefix("cube_") for name in picked_after if picked_after[name] > picked_before.get(name, 0)]
    cube = f"{'/'.join(cubes)} cube, " if cubes else ""
    print(f"  Done in {time.monotonic() - started:.1f} s ({cube}{note}). Click 'Accept' in Studio.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--timeout-s", type=float, default=60.0, help="Give up on an episode after this long")
    args = parser.parse_args()

    if not get(f"{args.url}/health").get("autopilot", {}).get("available"):
        raise SystemExit("This scene has no autopilot: start the simulation with run_sim.py")
    # Hold the episode judge's reset off, so the placed cube stays until you accept: the plugin
    # takes a dwell of at most 120 s, which leaves two minutes to click Accept.
    post(f"{args.url}/episode/auto-reset", {"enabled": True, "dwell_s": 120})
    done = 0
    try:
        for index in range(1, args.episodes + 1):
            done += run_episode(args.url, index, args.episodes, args.timeout_s)
            input("  Press Enter after clicking Accept (or Discard)... ")
    finally:
        post(f"{args.url}/autopilot", {"mode": "off"})
        post(f"{args.url}/episode/auto-reset", {"enabled": True, "dwell_s": 2.0})
    print(f"\n{done} of {args.episodes} episodes finished.")


if __name__ == "__main__":
    main()
