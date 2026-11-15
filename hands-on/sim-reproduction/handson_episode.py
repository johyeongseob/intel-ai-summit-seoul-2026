"""Episode judge for the hands-on scene: success when a cube rests in the PLACE AREA.

One episode picks one cube, any of the three, and places it in the PLACE AREA. The judge watches
the simulation every control tick, independently of whoever moves the arm (the scripted
demonstrator, a teleoperator, or a trained policy):

- **success**: a cube's centre is inside the PLACE AREA square, the cube rests on the
  mat (not lifted) and is still, for ``success_dwell_s`` seconds of simulated time;
- then the scene resets: all three cubes go to new random positions, and the episode count
  goes up.

It takes the place of the plugin's cube-on-plate auto-reset (``episode_auto_reset.py``), with
the same methods, so the viewer's Episode panel and ``GET /health`` (``episode``) show it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from loguru import logger

from handson_scene import CUBE_HALF, CUBES, PLACE_AREA, PLACE_AREA_HALF

if TYPE_CHECKING:
    from collections.abc import Callable

Phase = Literal["idle", "success_hold"]

DEFAULT_DWELL_S = 2.0

RESTING_DZ = 0.01
"""The cube counts as resting when its centre is at most this far above its height on the mat."""
STILL_SPEED = 0.05
"""The cube counts as still below this linear speed (m/s)."""


class HandsOnEpisode:
    """Judge one pick-and-place episode at a time and reset the scene after each success."""

    def __init__(
        self,
        model: object,
        reset_scene: Callable[[], None],
        *,
        target_cubes: tuple[str, ...] = CUBES,
        success_dwell_s: float = DEFAULT_DWELL_S,
        active: bool = True,
    ) -> None:
        """Bind to a loaded hands-on scene.

        Args:
            model: The compiled scene.
            reset_scene: Resets the whole scene (all cubes) and tells the autopilot; the plugin's
                own scene reset.
            target_cubes: The cubes that count: all three, since each episode moves one picked at random.
            success_dwell_s: Seconds the cube must rest in the PLACE AREA.
            active: Whether success detection and resets run.
        """
        import mujoco  # noqa: PLC0415

        body = mujoco.mjtObj.mjOBJ_BODY
        self._targets: list[tuple[str, int, int]] = []
        for name in target_cubes:
            joint = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}:joint"))
            self._targets.append((name, int(mujoco.mj_name2id(model, body, name)), int(model.jnt_dofadr[joint])))
        self._target_label = target_cubes[0] if len(target_cubes) == 1 else "any cube"
        self._placed: str | None = None
        self._place = int(mujoco.mj_name2id(model, body, PLACE_AREA))
        self._reset_scene = reset_scene
        self._dwell_s = float(success_dwell_s)
        self._active = active
        self._phase: Phase = "idle"
        self._since: float | None = None
        self._countdown_s: float | None = None
        self._episode_count = 0

    # --- What the plugin calls (same as its EpisodeAutoReset) -------------------

    @property
    def active(self) -> bool:
        """Whether success detection and resets run."""
        return self._active

    def set_active(self, active: bool) -> None:  # noqa: FBT001
        """Pause or resume; pausing cancels a running countdown."""
        self._active = active
        if not active:
            self.notify_manual_reset()

    @property
    def dwell_s(self) -> float:
        """Seconds the cube must rest in the PLACE AREA."""
        return self._dwell_s

    def set_dwell(self, dwell_s: float) -> None:
        """Change the success dwell."""
        self._dwell_s = float(dwell_s)

    def notify_manual_reset(self) -> None:
        """Forget a running countdown after a reset from elsewhere (viewer, HTTP)."""
        self._phase, self._since, self._countdown_s, self._placed = "idle", None, None, None

    def status(self) -> dict[str, object]:
        """Return a JSON-friendly snapshot for ``/health`` and the viewer.

        ``demo`` carries the demonstrator's counters (attempts, failures, deliberate failures),
        which the plugin hides from the ``autopilot`` status while the autopilot is off.
        """
        from dataclasses import asdict  # noqa: PLC0415

        from pick_place_demo import ACTIVE_STATS  # noqa: PLC0415

        return {
            "demo": asdict(ACTIVE_STATS[0]) if ACTIVE_STATS else None,
            "enabled": True,
            "kind": "handson",
            "active": self._active,
            "phase": self._phase,
            "episode_count": self._episode_count,
            "success_dwell_s": self._dwell_s,
            "countdown_s": self._countdown_s,
            "cube": self._placed or self._target_label,
            "target": PLACE_AREA,
        }

    def update(self, model: object, data: object) -> None:  # noqa: ARG002
        """Advance success detection, the countdown and the reset by one control tick."""
        if not self._active:
            return
        now = float(data.time)
        placed = self.cube_in_place_area(data)
        if placed is None:
            if self._phase == "success_hold":
                logger.info("{} left the PLACE AREA — countdown cancelled", self._placed)
            self.notify_manual_reset()
            return
        if self._since is None:
            self._phase, self._since, self._placed = "success_hold", now, placed
            logger.info("{} in the PLACE AREA — next episode in {:.1f}s", placed, self._dwell_s)
        self._countdown_s = max(0.0, self._dwell_s - (now - self._since))
        if self._countdown_s <= 0.0:
            self._episode_count += 1
            logger.info("Episode {} done: {} placed; resetting the cubes", self._episode_count, placed)
            self.notify_manual_reset()
            self._reset_scene()

    # --- The success test -------------------------------------------------------

    def cube_in_place_area(self, data: object) -> str | None:
        """The target cube that rests, still, with its centre inside the PLACE AREA square, if any."""
        place = np.asarray(data.xpos[self._place])
        for name, body, dof in self._targets:
            cube = np.asarray(data.xpos[body])
            inside = bool(np.all(np.abs(cube[:2] - place[:2]) <= PLACE_AREA_HALF))
            resting = float(cube[2]) <= CUBE_HALF + RESTING_DZ
            speed = float(np.linalg.norm(data.qvel[dof : dof + 3]))
            if inside and resting and speed <= STILL_SPEED:
                return name
        return None

    def in_place_area(self, data: object) -> bool:
        """Whether a target cube rests, still, with its centre inside the PLACE AREA square."""
        return self.cube_in_place_area(data) is not None
