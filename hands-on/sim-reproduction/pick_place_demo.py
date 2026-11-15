"""Scripted pick-and-place demonstrator for the MuJoCo plugin's ``single_pick_place`` scene.

One cycle (one episode) picks the cube and puts it on the target disc:

idle -> approach -> descend -> close -> lift -> carry -> lower -> release -> rise -> idle

It reads privileged simulation state (cube and target poses) and reuses the plugin's
SO-101 ``ArmIK`` (grasp point position first, fingers down and jaw yaw second). It
implements the plugin's ``Demonstrator`` protocol, so the autopilot can run it in
``drive`` mode (it moves the arm) or ``leader`` mode (it only publishes targets).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from physicalai_mujoco_plugin.conveyor_demo import ARM_JOINTS, GRIPPER, ArmIK, _yaw_distance

Phase = Literal["idle", "approach", "descend", "close", "lift", "carry", "lower", "release", "rise"]


@dataclass
class PickPlaceConfig:
    """Motion parameters (metres, seconds, radians)."""

    cube: str = "block1"
    target: str = "target"
    cube_half: float = 0.02
    target_radius: float = 0.05
    grip_open: float = 0.8
    """Gripper angle while approaching; 0.5 gives a 5.2 cm gap, so this leaves room around a 4 cm cube."""
    grip_closed: float = -0.15
    home: tuple[float, float, float] = (0.20, 0.0, 0.12)
    """Grasp point position to wait at between cycles."""
    hover_z: float = 0.10
    carry_z: float = 0.10
    grasp_dz: float = 0.005
    """Grasp this far above the cube centre."""
    place_z: float = 0.045
    """Grasp point height when the jaws open over the target: the cube's bottom ~1.5 cm above the disc."""
    cart_speed: float = 0.25
    descend_speed: float = 0.10
    lift_speed: float = 0.15
    close_s: float = 0.4
    release_s: float = 0.4
    approach_timeout_s: float = 5.0
    lifted_dz: float = 0.03
    """The cube counts as lifted when its centre is this far above its resting height."""


@dataclass
class PickPlaceStats:
    """Counters since the simulation started (they survive mode switches)."""

    attempts: int = 0
    placed: int = 0
    grasp_failed: int = 0
    dropped: int = 0
    timeouts: int = 0


def is_pick_place_scene(model: object, config: PickPlaceConfig | None = None) -> bool:
    """Return whether *model* has the cube, the target and a single unprefixed SO-101."""
    import mujoco  # noqa: PLC0415

    cfg = config or PickPlaceConfig()
    bodies = (cfg.cube, cfg.target, "gripper")
    has_bodies = all(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0 for name in bodies)
    has_arm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINTS[0]) >= 0
    return has_bodies and has_arm


class PickPlaceDemonstrator:
    """Pick the cube and place it on the target disc, one control tick at a time."""

    def __init__(self, model: object, config: PickPlaceConfig | None = None) -> None:
        """Bind to a loaded model; IK runs on a private copy of the data."""
        import mujoco  # noqa: PLC0415

        self._mujoco = mujoco
        self.model = model
        self.config = config or PickPlaceConfig()
        self.ik = ArmIK(model)
        self.stats = PickPlaceStats()
        body = mujoco.mjtObj.mjOBJ_BODY
        self._cube = int(mujoco.mj_name2id(model, body, self.config.cube))
        self._target = int(mujoco.mj_name2id(model, body, self.config.target))
        joint = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{self.config.cube}:joint"))
        self._cube_dof = int(model.jnt_dofadr[joint])
        self._actuators = [
            int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)) for name in (*ARM_JOINTS, GRIPPER)
        ]
        self.reset()

    # --- Demonstrator protocol -------------------------------------------------

    @property
    def phase(self) -> Phase:
        """Current phase of the cycle."""
        return self._phase

    @property
    def target(self) -> str | None:
        """The cube while a cycle runs, else ``None``."""
        return None if self._phase == "idle" else self.config.cube

    def reset(self) -> None:
        """Forget the cycle; the next step starts from the arm's current pose."""
        self._phase: Phase = "idle"
        self._since = 0.0
        self._yaw = 0.0
        self._setpoint = np.zeros(3)
        self._q: np.ndarray | None = None
        self._grip = self.config.grip_open

    def status(self) -> dict[str, object]:
        """Return the counters for ``/health`` (same keys every call)."""
        return asdict(self.stats)

    def take_reset_request(self) -> bool:  # noqa: PLR6301
        """Never ask for a scene reset: a failed grasp is simply retried."""
        return False

    # --- Geometry --------------------------------------------------------------

    def _cube_pos(self, data: object) -> np.ndarray:
        return np.array(data.xpos[self._cube], dtype=np.float64)

    def _resting(self, data: object) -> bool:
        speed = float(np.linalg.norm(data.qvel[self._cube_dof : self._cube_dof + 3]))
        return self._cube_pos(data)[2] < self.config.cube_half + 0.01 and speed < 0.05  # noqa: PLR2004

    def _on_target(self, data: object) -> bool:
        cube, target = self._cube_pos(data), data.xpos[self._target]
        planar = float(np.hypot(cube[0] - target[0], cube[1] - target[1]))
        return planar <= self.config.target_radius - 0.005 and cube[2] < self.config.cube_half + 0.01  # noqa: PLR2004

    def _cube_yaw(self, data: object) -> float:
        """Yaw of the cube's most horizontal face normal (any face up works for a cube)."""
        rot = data.xmat[self._cube].reshape(3, 3)
        axis = rot[:, int(np.argmin(np.abs(rot[2, :])))]
        return float(np.arctan2(axis[1], axis[0]))

    def _choose_yaw(self, data: object, grasp: np.ndarray) -> float:
        """Pick the jaw yaw (4 face-aligned options) that is reachable, upright and needs the least wrist turn."""
        mujoco, ik = self._mujoco, self.ik
        q_now = self._q.copy()
        base = self._cube_yaw(data)
        best: tuple[tuple[bool, float, float], float] | None = None
        for k in range(4):
            yaw = base + k * np.pi / 2
            q, err = ik.solve(q_now, grasp, yaw, iters=40)
            ik.data.qpos[ik.qpos_adr] = q
            mujoco.mj_kinematics(self.model, ik.data)
            tilt = ik.tilt(ik.data)
            yaw_err = _yaw_distance(ik.gripper_yaw(ik.data), yaw)
            usable = err < 0.003 and tilt < np.radians(10) and yaw_err < np.radians(5)  # noqa: PLR2004
            score = (usable, -abs(float(q[4] - q_now[4])), -tilt)
            if best is None or score > best[0]:
                best = (score, yaw)
        return best[1]

    # --- Motion ----------------------------------------------------------------

    def _go(self, phase: Phase, now: float) -> None:
        self._phase, self._since = phase, now

    def _move_toward(self, goal: np.ndarray, dt: float, speed: float | None = None) -> None:
        step = goal - self._setpoint
        dist = float(np.linalg.norm(step))
        limit = (speed or self.config.cart_speed) * dt
        self._setpoint = goal.copy() if dist <= limit else self._setpoint + step * (limit / dist)

    def _raise_to(self, z: float, dt: float, speed: float | None = None) -> bool:
        goal = self._setpoint.copy()
        goal[2] = z
        self._move_toward(goal, dt, speed)
        return abs(self._setpoint[2] - z) < 0.003  # noqa: PLR2004

    def step(self, data: object, dt: float, *, apply: bool = True) -> np.ndarray:  # noqa: C901, PLR0912
        """Advance one control tick; write the targets to the actuators if *apply*.

        Returns:
            Joint targets in radians: the five arm joints, then the gripper.
        """
        cfg, now = self.config, float(data.time)
        if self._q is None:
            self._mujoco.mj_kinematics(self.model, data)
            self._q = data.qpos[self.ik.qpos_adr].copy()
            self._setpoint = self.ik.tcp(data).copy()
            self._yaw = self.ik.gripper_yaw(data)
        cube = self._cube_pos(data)
        grasp = cube + np.array([0.0, 0.0, cfg.grasp_dz])

        if self._phase == "idle":
            self._grip = cfg.grip_open
            self._move_toward(np.array(cfg.home), dt)
            if self._resting(data) and not self._on_target(data):
                self._yaw = self._choose_yaw(data, grasp)
                self.stats.attempts += 1
                self._go("approach", now)
        elif self._phase == "approach":
            goal = np.array([grasp[0], grasp[1], cfg.hover_z])
            self._move_toward(goal, dt)
            lined_up = _yaw_distance(self.ik.gripper_yaw(data), self._yaw) < np.radians(5)
            if np.linalg.norm(self._setpoint - goal) < 0.003 and lined_up:  # noqa: PLR2004
                self._go("descend", now)
            elif now - self._since > cfg.approach_timeout_s:
                self.stats.timeouts += 1
                self._go("rise", now)
        elif self._phase == "descend":
            self._move_toward(grasp, dt, cfg.descend_speed)
            if np.linalg.norm(self._setpoint - grasp) < 0.002:  # noqa: PLR2004
                self._go("close", now)
        elif self._phase == "close":
            self._grip = cfg.grip_closed
            if now - self._since > cfg.close_s:
                self._go("lift", now)
        elif self._phase == "lift":
            if self._raise_to(cfg.carry_z, dt, cfg.lift_speed):
                if cube[2] > cfg.cube_half + cfg.lifted_dz:
                    self._go("carry", now)
                else:
                    self.stats.grasp_failed += 1
                    self._grip = cfg.grip_open
                    self._go("rise", now)
        elif self._phase == "carry":
            if cube[2] < cfg.cube_half + 0.01:  # noqa: PLR2004 - fell out of the jaws
                self.stats.dropped += 1
                self._grip = cfg.grip_open
                self._go("rise", now)
            else:
                # Put the cube, not the grasp point, over the disc centre.
                offset = (self.ik.tcp(data) - cube)[:2]
                target = data.xpos[self._target]
                goal = np.array([target[0] + offset[0], target[1] + offset[1], cfg.carry_z])
                self._move_toward(goal, dt)
                if np.linalg.norm(self._setpoint - goal) < 0.003:  # noqa: PLR2004
                    self._go("lower", now)
        elif self._phase == "lower":
            if self._raise_to(cfg.place_z, dt, cfg.descend_speed):
                self._grip = cfg.grip_open
                self._go("release", now)
        elif self._phase == "release":
            self._grip = cfg.grip_open
            if now - self._since > cfg.release_s:
                self._go("rise", now)
        elif self._phase == "rise":
            self._grip = cfg.grip_open
            if self._raise_to(cfg.hover_z, dt):
                if self._on_target(data):
                    self.stats.placed += 1
                self._go("idle", now)

        self._q, _ = self.ik.solve(self._q, self._setpoint, self._yaw)
        targets = np.array([*self._q, self._grip], dtype=np.float64)
        if apply:
            for actuator, value in zip(self._actuators, targets, strict=True):
                data.ctrl[actuator] = value
        return targets
