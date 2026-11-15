"""Scripted pick-and-place demonstrator for the cube scenes.

It works in two scenes, with a config for each (:func:`config_for`):

- the plugin's ``single_pick_place``: one 4 cm cube and a round target disc (M3);
- the hands-on scene ``handson_pick_place``: 2 cm red, blue and yellow cubes and a square
  PLACE AREA; it moves only the target cube (red by default) and leaves the others (M4).

One cycle (one episode) picks the cube and puts it on the target:

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

ACTIVE_STATS: list[PickPlaceStats] = []
"""The counters of the demonstrator in this process, for the episode judge's status.

The plugin blanks a demonstrator's own status fields while the autopilot is off; the judge's
``episode`` status stays, so tools such as record_helper.py read the counters there.
"""


@dataclass
class PickPlaceConfig:
    """Motion parameters (metres, seconds, radians)."""

    cube: str = "block1"
    target: str = "target"
    cube_half: float = 0.02
    target_radius: float = 0.05
    """Radius of a round target (the disc)."""
    target_half: float | None = None
    """Half side of a square target (the PLACE AREA); ``None`` for a round one."""
    grip_open: float = 0.8
    """Gripper angle while approaching; 0.5 gives a 5.2 cm gap, so this leaves room around a 4 cm cube."""
    grip_closed: float = -0.15
    home: tuple[float, float, float] = (0.20, 0.0, 0.12)
    """Grasp point position to wait at between cycles."""
    hover_z: float = 0.10
    carry_z: float = 0.10
    grasp_dz: float = 0.005
    """Grasp this far above the cube centre."""
    jaw_offset: float = 0.0
    """Put the grasp point this far from the cube centre along the jaw's closing axis.

    The moving jaw pushes the cube against the fixed jaw. A small cube starts far from the fixed
    jaw, so it slides that far as the jaws close; starting the grasp point this much to the
    moving jaw's side puts the fixed jaw next to the cube instead.
    """
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
    max_failures: int = 3
    """Failed tries in a row on one cube before asking for a scene reset (a cube out of reach)."""
    follow_tol: float = 0.012
    """A phase only ends once the real grasp point is this close to the target one (metres).

    In ``leader`` mode the arm moves only after Studio sends the targets back, with some delay,
    and not at all before Studio teleoperates: the demonstrator waits for it instead of judging
    a grasp or a release before the arm got there.
    """
    idle_wait_s: float = 4.0
    """Seconds to wait for the cube to lie still on the mat (not resting on another cube, say)
    before asking for a scene reset."""

    # Deliberate failures, so recorded episodes also show how to recover (the presenter's advice).
    # At most one per episode, on its first try; the retry is always a normal pick.
    failure_rate: float = 0.0
    """Share of episodes that get one deliberate failure."""
    failure_kinds: tuple[str, ...] = ("miss", "drop")
    """``miss``: close beside the cube and lift nothing. ``drop``: a shallow grasp that slips during the carry."""
    seed: int | None = None
    """Seed for choosing which episodes fail and how; ``None`` draws a fresh one."""
    miss_offset: float = 0.02
    """``miss``: how far beside the cube (across the jaws) the jaws close."""
    drop_grasp_dz: float = 0.008
    """``drop``: grasp this far above the cube centre, so the pads hold only its top edge."""
    drop_grip: float = 0.03
    """``drop``: gripper angle for the shallow grasp; it still lifts the cube."""
    drop_slip_grip: float = 0.2
    """``drop``: the grip eases toward this angle during the carry, a few millimetres, until the cube slips."""
    drop_ease_s: tuple[float, float] = (0.4, 1.0)
    """``drop``: seconds the grip takes to ease, drawn per failure, so the cube falls at different points."""


@dataclass
class PickPlaceStats:
    """Counters since the simulation started (they survive mode switches)."""

    attempts: int = 0
    placed: int = 0
    grasp_failed: int = 0
    dropped: int = 0
    timeouts: int = 0
    missed: int = 0
    """Released, but the cube did not end up on the target."""
    injected_miss: int = 0
    """Deliberate ``miss`` failures started."""
    injected_drop: int = 0
    """Deliberate ``drop`` failures started."""


def handson_config(target_cube: str = "cube_red") -> PickPlaceConfig:
    """Config for the hands-on scene: 2 cm cubes, a 53 mm square PLACE AREA about 25 cm out."""
    return PickPlaceConfig(
        cube=target_cube,
        target="place_area",
        cube_half=0.01,
        target_half=0.0265,
        grip_open=0.5,  # a 5.2 cm gap: room around a 2 cm cube, and the jaws clear cubes 8 cm away
        home=(0.18, 0.0, 0.10),
        hover_z=0.08,
        carry_z=0.08,
        # Tuned offline (90 of 90 episodes placed on the first try over six seeds):
        grip_closed=0.0,  # a gentler squeeze: at -0.15 the 4.5 g cube sprang away ~2 cm on release
        jaw_offset=0.01,  # a centred 2 cm cube slid ~18 mm to the fixed jaw while closing
        place_z=0.017,  # the cube's bottom ~2 mm above the mat when the jaws open
    )


def config_for(model: object) -> PickPlaceConfig:
    """The config for the scene *model* was built from: hands-on if it has the PLACE AREA.

    Deliberate failures come from environment variables, which ``run_sim.py`` passes on to the
    simulation process: ``HANDSON_FAILURE_RATE`` (0 to 1, default 0), ``HANDSON_FAILURE_KINDS``
    (comma-separated, default ``miss,drop``) and ``HANDSON_SEED``.
    """
    import os  # noqa: PLC0415

    import mujoco  # noqa: PLC0415

    cfg = handson_config() if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "place_area") >= 0 else PickPlaceConfig()
    cfg.failure_rate = float(os.environ.get("HANDSON_FAILURE_RATE", cfg.failure_rate))
    kinds = os.environ.get("HANDSON_FAILURE_KINDS")
    if kinds:
        cfg.failure_kinds = tuple(kind.strip() for kind in kinds.split(",") if kind.strip())
    seed = os.environ.get("HANDSON_SEED")
    if seed:
        cfg.seed = int(seed)
    return cfg


def is_pick_place_scene(model: object, config: PickPlaceConfig | None = None) -> bool:
    """Return whether *model* has the cube, the target and a single unprefixed SO-101."""
    import mujoco  # noqa: PLC0415

    cfg = config or config_for(model)
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
        self.config = config or config_for(model)
        self.ik = ArmIK(model)
        self.stats = PickPlaceStats()
        ACTIVE_STATS[:] = [self.stats]
        body = mujoco.mjtObj.mjOBJ_BODY
        self._cube = int(mujoco.mj_name2id(model, body, self.config.cube))
        self._target = int(mujoco.mj_name2id(model, body, self.config.target))
        joint = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{self.config.cube}:joint"))
        self._cube_dof = int(model.jnt_dofadr[joint])
        self._actuators = [
            int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)) for name in (*ARM_JOINTS, GRIPPER)
        ]
        grip_joint = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER))
        self._grip_qadr = int(model.jnt_qposadr[grip_joint])
        self._grip_dadr = int(model.jnt_dofadr[grip_joint])
        self._rng = np.random.default_rng(self.config.seed)
        self._waiting = False
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
        """Forget the cycle; the next step starts from the arm's current pose.

        Called after every scene reset, so each episode draws its deliberate failure here.
        """
        cfg = self.config
        planned = None
        if cfg.failure_kinds and self._rng.random() < cfg.failure_rate:
            planned = str(self._rng.choice(cfg.failure_kinds))
        self._planned_failure: str | None = planned
        """The failure this episode's first try will make, if any."""
        self._failure: str | None = None
        """The failure the current try is making."""
        self._ease_s = 1.0
        self._carry_start = 0.0
        self._idle_since = float("inf")
        """When the idle arm started waiting for a pickable cube; set on the first idle tick."""
        self._phase: Phase = "idle"
        self._since = 0.0
        self._yaw = 0.0
        self._setpoint = np.zeros(3)
        self._q: np.ndarray | None = None
        self._grip = self.config.grip_open
        self._failures = 0
        self._reset_requested = False
        self._released = False

    def status(self) -> dict[str, object]:
        """Return the counters for ``/health`` (same keys every call).

        ``waiting_for_arm`` is true while the arm has not reached the published targets yet: in
        ``leader`` mode, until Studio sends them back as actions.
        """
        return {**asdict(self.stats), "waiting_for_arm": self._waiting}

    def take_reset_request(self) -> bool:
        """Return ``True`` once after ``max_failures`` failed tries in a row (say, a cube out of reach).

        A failed try is otherwise simply retried from where the cube lies.
        """
        requested, self._reset_requested = self._reset_requested, False
        return requested

    def _failed(self) -> None:
        self._failures += 1
        if self._failures >= self.config.max_failures:
            self._failures, self._reset_requested = 0, True

    # --- Geometry --------------------------------------------------------------

    def _cube_pos(self, data: object) -> np.ndarray:
        return np.array(data.xpos[self._cube], dtype=np.float64)

    def _resting(self, data: object) -> bool:
        speed = float(np.linalg.norm(data.qvel[self._cube_dof : self._cube_dof + 3]))
        return self._cube_pos(data)[2] < self.config.cube_half + 0.01 and speed < 0.05  # noqa: PLR2004

    def _on_target(self, data: object) -> bool:
        cfg = self.config
        cube, target = self._cube_pos(data), data.xpos[self._target]
        if cfg.target_half is not None:  # square PLACE AREA: the cube's centre inside it
            inside = bool(np.all(np.abs(cube[:2] - target[:2]) <= cfg.target_half))
        else:  # round disc
            inside = float(np.hypot(cube[0] - target[0], cube[1] - target[1])) <= cfg.target_radius - 0.005  # noqa: PLR2004
        return inside and cube[2] < cfg.cube_half + 0.01  # noqa: PLR2004

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
            self._released = False
            self._failure = None
            self._move_toward(np.array(cfg.home), dt)
            on_target = self._on_target(data)
            if on_target or self._resting(data) or self._idle_since > now:
                self._idle_since = now
            elif now - self._idle_since > cfg.idle_wait_s:  # stuck on another cube, say
                self._idle_since, self._reset_requested = now, True
            if self._resting(data) and not on_target:
                self._yaw = self._choose_yaw(data, grasp)
                self.stats.attempts += 1
                # This episode's deliberate failure, if any, happens on its first try only.
                self._failure, self._planned_failure = self._planned_failure, None
                if self._failure is not None:
                    setattr(self.stats, f"injected_{self._failure}", getattr(self.stats, f"injected_{self._failure}") + 1)
                    self._ease_s = float(self._rng.uniform(*cfg.drop_ease_s))
                self._idle_since = float("inf")
                self._go("approach", now)
        # Shift the grasp point toward the moving jaw (see PickPlaceConfig.jaw_offset).
        across = np.array([-np.sin(self._yaw), np.cos(self._yaw), 0.0])
        grasp = grasp + cfg.jaw_offset * np.array([np.cos(self._yaw), np.sin(self._yaw), 0.0])
        if self._failure == "miss":  # close beside the cube, across the jaws
            grasp = grasp + cfg.miss_offset * across
        elif self._failure == "drop":  # hold only the cube's top edge
            grasp[2] = cube[2] + cfg.drop_grasp_dz

        # Wait for the arm: a phase ends only once the real grasp point has caught up with the
        # target, and timed phases do not count the time spent waiting.
        tcp = self.ik.tcp(data)
        there = float(np.linalg.norm(tcp - self._setpoint)) < cfg.follow_tol
        jaws_still = abs(float(data.qvel[self._grip_dadr])) < 0.3  # noqa: PLR2004
        self._waiting = self._phase != "idle" and not there
        if self._waiting:
            self._since += dt

        if self._phase == "idle":
            pass
        elif self._phase == "approach":
            goal = np.array([grasp[0], grasp[1], cfg.hover_z])
            self._move_toward(goal, dt)
            lined_up = _yaw_distance(self.ik.gripper_yaw(data), self._yaw) < np.radians(5)
            if np.linalg.norm(self._setpoint - goal) < 0.003 and lined_up and there:  # noqa: PLR2004
                self._go("descend", now)
            elif now - self._since > cfg.approach_timeout_s:
                self.stats.timeouts += 1
                self._failed()
                self._go("rise", now)
        elif self._phase == "descend":
            self._move_toward(grasp, dt, cfg.descend_speed)
            if np.linalg.norm(self._setpoint - grasp) < 0.002 and np.linalg.norm(tcp - grasp) < 0.006:  # noqa: PLR2004
                self._go("close", now)
        elif self._phase == "close":
            self._grip = cfg.drop_grip if self._failure == "drop" else cfg.grip_closed
            if now - self._since > cfg.close_s and jaws_still:
                self._go("lift", now)
        elif self._phase == "lift":
            if self._raise_to(cfg.carry_z, dt, cfg.lift_speed) and there:
                if cube[2] > cfg.cube_half + cfg.lifted_dz:
                    self._carry_start = now
                    self._go("carry", now)
                else:
                    self.stats.grasp_failed += 1
                    self._failed()
                    self._failure = None
                    self._grip = cfg.grip_open
                    self._go("rise", now)
        elif self._phase == "carry":
            if cube[2] < cfg.cube_half + 0.01:  # noqa: PLR2004 - fell out of the jaws
                self.stats.dropped += 1
                self._failed()
                self._failure = None
                self._grip = cfg.grip_open
                self._go("rise", now)
            else:
                if self._failure == "drop":  # ease the shallow grip until the cube slips
                    # Real time since the carry began: waiting for the arm must not hold the easing back.
                    ease = min(1.0, (now - self._carry_start) / self._ease_s)
                    self._grip = cfg.drop_grip + (cfg.drop_slip_grip - cfg.drop_grip) * ease
                # Put the cube, not the grasp point, over the disc centre.
                offset = (self.ik.tcp(data) - cube)[:2]
                target = data.xpos[self._target]
                goal = np.array([target[0] + offset[0], target[1] + offset[1], cfg.carry_z])
                self._move_toward(goal, dt)
                if np.linalg.norm(self._setpoint - goal) < 0.003 and there:  # noqa: PLR2004
                    self._go("lower", now)
        elif self._phase == "lower":
            if self._raise_to(cfg.place_z, dt, cfg.descend_speed) and abs(tcp[2] - cfg.place_z) < 0.005:  # noqa: PLR2004
                self._grip = cfg.grip_open
                self._released = True
                self._go("release", now)
        elif self._phase == "release":
            self._grip = cfg.grip_open
            if now - self._since > cfg.release_s and jaws_still:
                self._go("rise", now)
        elif self._phase == "rise":
            self._grip = cfg.grip_open
            if self._raise_to(cfg.hover_z, dt) and there:
                if self._released and self._on_target(data):
                    self.stats.placed += 1
                    self._failures = 0
                elif self._released:
                    self.stats.missed += 1
                    self._failed()
                self._go("idle", now)

        self._q, _ = self.ik.solve(self._q, self._setpoint, self._yaw)
        targets = np.array([*self._q, self._grip], dtype=np.float64)
        if apply:
            for actuator, value in zip(self._actuators, targets, strict=True):
                data.ctrl[actuator] = value
        return targets
