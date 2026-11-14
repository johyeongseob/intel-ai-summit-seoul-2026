# Hands-on: Physical AI Studio — Simulation Reproduction

> **Unofficial.** Personal notes. This is not official Intel documentation; see the [Physical AI Studio repository](https://github.com/open-edge-platform/physical-ai-studio) for that.

I reproduce the Physical AI Studio hands-on session from Intel AI Summit Seoul 2026 in simulation only, without robot hardware, by implementing the cube pick-and-place task myself.

Details of the original session (time, room, presenter, table setup) are in [SESSION.md](SESSION.md).

## Implementation

**Setup:** Galaxy Book6 Pro (Intel Core Ultra X7 358H, Arc B390, NPU, 32 GB), WSL2 Ubuntu 24.04, Physical AI Studio with the MuJoCo plugin. No robot hardware: every episode comes from a scripted demonstrator.

### Milestones

| # | Milestone | What to do | Done when | Status |
|---|---|---|---|---|
| M0 | Environment | Install uv, Node.js 24, git, and ffmpeg in WSL2 Ubuntu 24.04; clone `physical-ai-studio` and `physicalai` | Tool versions confirmed | ⬜ |
| M1 | Simulator check | Install the MuJoCo plugin and start the SO-101 in the `single_pick_place` scene | Cameras near 30 fps and real-time factor near 1.00x in the browser viewer | ⬜ |
| M2 | Studio connection | Run Studio; add the MuJoCo follower, virtual leader, and two IP cameras; record a few `conveyor_sort` episodes, train ACT briefly, and run the model. Check whether the iGPU (XPU) is used for training | The whole GUI workflow runs end to end | ⬜ |
| M3 | Cube demonstrator (drive mode) | Install the plugin in editable mode; write a pick-and-place demonstrator reusing the plugin's SO-101 IK; test it in drive mode | 90%+ success from a fixed position | ⬜ |
| M4 | Cube demonstrator (leader mode) + recording | Connect the demonstrator as a virtual leader; record with **Start episode** → **Accept** in Studio | Episodes saved and replayable in a dataset | ⬜ |
| M5 | Randomization + deliberate failures | Randomize the cube position each episode; fail on purpose at a set rate and retry | Controlled by settings (zone size, failure rate, seed) | ⬜ |
| M6 | Stage 1 training and evaluation | Record about 50 episodes (one object, fixed target), train ACT, and run it in simulation | Success rate measured | ⬜ |
| M7 | Hands-on scene | Build a scene with red, blue, and yellow cubes and a square PLACE AREA; record and train Stage 2 (distractors) and Stage 3 (several target positions) | Success rates compared across stages | ⬜ |
| M8 | Write-up | Put code, configs, and results (success-rate table, videos) in `sim-reproduction/` and this README | Reproduction steps and results documented | ⬜ |

Implementation notes will be added here as I go.
