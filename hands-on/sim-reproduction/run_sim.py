"""Start the MuJoCo simulation with the cube pick-and-place demonstrator.

Same command line as ``physicalai-mujoco``, e.g.::

    python run_sim.py start --scene single_pick_place

The plugin builds the robot in a separate owner process, so a patch made only in this
process would be lost. Instead, this script swaps in ``PickPlaceMuJoCoRobot`` (see
``pick_place_sim.py``) and puts this directory on ``PYTHONPATH``; the owner process
then imports ``pick_place_sim``, which installs the demonstrator hook there too.
"""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(HERE), os.environ.get("PYTHONPATH")]))

import physicalai_mujoco_plugin.robot as plugin_robot  # noqa: E402
from physicalai_mujoco_plugin.__main__ import main  # noqa: E402
from pick_place_sim import PickPlaceMuJoCoRobot  # noqa: E402

# The plugin's `start` command imports MuJoCoRobot from this module when it runs.
plugin_robot.MuJoCoRobot = PickPlaceMuJoCoRobot

if __name__ == "__main__":
    main()
