"""Run the pick-and-place demonstrator in drive mode and measure its success rate.

Start the simulation with run_sim.py first. A success is judged by the scene's episode
judge, independently of the demonstrator: in single_pick_place the plugin's auto-reset (the
cube rests on the disc for 5 s), in handson_pick_place handson_episode.py (the red cube rests
in the PLACE AREA for 2 s). Then the scene respawns the cube(s) and the demonstrator goes again.
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


def post(url: str, body: dict) -> None:
    request = urllib.request.Request(  # noqa: S310
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
        response.read()


def counters(health: dict) -> tuple[int, int, int, int, str | None]:
    autopilot, episode = health.get("autopilot", {}), health.get("episode", {})
    failed = sum(int(autopilot.get(key) or 0) for key in FAIL_KEYS)
    return (
        int(autopilot.get("attempts") or 0),
        int(autopilot.get("placed") or 0),
        failed,
        int(episode.get("episode_count") or 0),
        autopilot.get("phase"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=0, help="Stop after this many successful episodes instead")
    parser.add_argument("--stall-s", type=float, default=60.0, help="Stop when nothing changes for this long")
    args = parser.parse_args()

    health = get(f"{args.url}/health")
    if not health.get("autopilot", {}).get("available"):
        raise SystemExit("This scene has no autopilot: start the simulation with run_sim.py")
    if not health.get("episode", {}).get("enabled"):
        print("Warning: episode auto-reset is off, so successes are not counted and the cube is not respawned.")

    post(f"{args.url}/autopilot", {"mode": "drive"})
    time.sleep(0.5)
    start = get(f"{args.url}/health")
    a0, p0, f0, e0, _ = counters(start)
    last, last_change = None, time.monotonic()
    try:
        while True:
            a, p, f, e, phase = counters(get(f"{args.url}/health"))
            done = (p - p0) + (f - f0)
            if (a, p, f, e) != last:
                print(f"attempts {a - a0:3d}  placed {p - p0:3d}  failed {f - f0:3d}  successes {e - e0:3d}  ({phase})")
                last, last_change = (a, p, f, e), time.monotonic()
            if (args.episodes and e - e0 >= args.episodes) or (not args.episodes and done >= args.trials):
                break
            if time.monotonic() - last_change > args.stall_s:
                print(f"No progress for {args.stall_s:.0f} s; stopping.")
                break
            time.sleep(0.5)
        dwell = float(get(f"{args.url}/health").get("episode", {}).get("success_dwell_s") or 5.0)
        time.sleep(dwell + 1.5)  # let the last placement finish its success dwell
        end = get(f"{args.url}/health")
        _, p, f, e, _ = counters(end)
    finally:
        post(f"{args.url}/autopilot", {"mode": "off"})

    done, successes = (p - p0) + (f - f0), e - e0
    print(f"\nTrials {done}, successes {successes}, failures {f - f0}")
    injected = {
        kind: int(end["autopilot"].get(f"injected_{kind}") or 0) - int(start["autopilot"].get(f"injected_{kind}") or 0)
        for kind in ("miss", "drop")
    }
    if any(injected.values()):
        # With deliberate failures, the measure is whether every episode still ends in a success.
        print(f"Deliberate failures: miss {injected['miss']}, drop {injected['drop']}; episodes completed {successes}")
    elif done:
        print(f"Success rate: {100 * successes / done:.0f}%  (goal: 90%+)")


if __name__ == "__main__":
    main()
