"""Hook the pick-and-place demonstrator into the MuJoCo plugin, without editing the plugin.

The plugin runs the simulation in a separate owner process
(``python -m physicalai.robot.transport._owner_worker``), which rebuilds the robot
from its exported ``class_path``. So the hook lives in this importable module:

- ``PickPlaceMuJoCoRobot`` is a plain subclass of the plugin's ``MuJoCoRobot``; its
  exported ``class_path`` is ``pick_place_sim.PickPlaceMuJoCoRobot``, so the owner
  process imports this module before it builds the robot.
- Importing this module wraps ``conveyor_automation._scene_demonstrator``: the
  plugin's function that picks a scene's demonstrator. For the ``single_pick_place``
  scene (which has none), it returns ``PickPlaceDemonstrator``.

- Importing ``handson_scene`` registers the hands-on task scene, ``handson_pick_place``,
  and ``PickPlaceMuJoCoRobot`` gives that scene its episode judge (``handson_episode.py``).

``run_sim.py`` puts this directory on ``PYTHONPATH`` so the owner process finds it.
"""

from __future__ import annotations

import handson_scene  # registers the handson_pick_place scene
from physicalai_mujoco_plugin import conveyor_automation
from physicalai_mujoco_plugin.robot import MuJoCoRobot
from pick_place_demo import PickPlaceDemonstrator, is_pick_place_scene

_plugin_scene_demonstrator = conveyor_automation._scene_demonstrator  # noqa: SLF001


def _scene_demonstrator(model, conveyor, **kwargs):  # noqa: ANN001, ANN003, ANN202
    demonstrator, unavailable = _plugin_scene_demonstrator(model, conveyor, **kwargs)
    if (
        demonstrator is None
        and model is not None
        and conveyor is None
        and kwargs.get("robots", 1) == 1
        and is_pick_place_scene(model)
    ):
        return PickPlaceDemonstrator(model), None
    return demonstrator, unavailable


if getattr(conveyor_automation._scene_demonstrator, "__module__", None) != __name__:  # noqa: SLF001
    conveyor_automation._scene_demonstrator = _scene_demonstrator  # noqa: SLF001


class PickPlaceMuJoCoRobot(MuJoCoRobot):
    """The plugin's MuJoCo robot; importing its module installs the demonstrator hook.

    In the hands-on scene it also swaps the plugin's cube-on-plate auto-reset for the
    hands-on episode judge (``handson_episode.HandsOnEpisode``).
    """

    def _init_episode_auto_reset(self) -> None:
        super()._init_episode_auto_reset()
        sim = self._sim
        if getattr(getattr(sim, "scene", None), "scene_id", None) != handson_scene.SCENE_ID:
            return
        from handson_episode import HandsOnEpisode  # noqa: PLC0415

        helper = HandsOnEpisode(sim.model, self._run_scene_reset, active=self._auto_reset_active)
        with self._state_lock:
            self._episode_auto_reset = helper
