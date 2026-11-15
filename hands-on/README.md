# Hands-on: Physical AI Studio — Simulation Reproduction

> **Unofficial.** Personal notes. This is not official Intel documentation; see the [Physical AI Studio repository](https://github.com/open-edge-platform/physical-ai-studio) for that.

I reproduce the Physical AI Studio hands-on session from Intel AI Summit Seoul 2026 in simulation only, without robot hardware, by implementing the cube pick-and-place task myself.

Details of the original session (time, room, presenter, table setup) are in [SESSION.md](SESSION.md).

## Implementation

**Setup:** Galaxy Book6 Pro (Intel Core Ultra X7 358H, Arc B390, NPU, 32 GB), WSL2 Ubuntu 24.04, Physical AI Studio with the MuJoCo plugin. No robot hardware: every episode comes from a scripted demonstrator.

### Milestones

| # | Milestone | What to do | Done when | Status |
|---|---|---|---|---|
| M0 | Environment | Install uv, Node.js 24, git, and ffmpeg in WSL2 Ubuntu 24.04; clone `physical-ai-studio` and `physicalai` | Tool versions confirmed | ✅ |
| M1 | Simulator check | Install the MuJoCo plugin and start the SO-101 in the `single_pick_place` scene | Cameras near 30 fps and real-time factor near 1.00x in the browser viewer | ✅ |
| M2 | Studio connection | Run Studio; add the MuJoCo follower, virtual leader, and two IP cameras; record a few `conveyor_sort` episodes, train ACT briefly, and run the model. Check whether the iGPU (XPU) is used for training | The whole GUI workflow runs end to end | ✅ |
| M3 | Cube demonstrator (drive mode) | Install the plugin in editable mode; write a pick-and-place demonstrator reusing the plugin's SO-101 IK; test it in drive mode | 90%+ success from a fixed position | ⬜ |
| M4 | Cube demonstrator (leader mode) + recording | Connect the demonstrator as a virtual leader; record with **Start episode** → **Accept** in Studio | Episodes saved and replayable in a dataset | ⬜ |
| M5 | Randomization + deliberate failures | Randomize the cube position each episode; fail on purpose at a set rate and retry | Controlled by settings (zone size, failure rate, seed) | ⬜ |
| M6 | Stage 1 training and evaluation | Record about 50 episodes (one object, fixed target), train ACT, and run it in simulation | Success rate measured | ⬜ |
| M7 | Hands-on scene | Build a scene with red, blue, and yellow cubes and a square PLACE AREA; record and train Stage 2 (distractors) and Stage 3 (several target positions) | Success rates compared across stages | ⬜ |
| M8 | Write-up | Put code, configs, and results (success-rate table, videos) in `sim-reproduction/` and this README | Reproduction steps and results documented | ⬜ |

### Pinned versions

Verified on 2026-11-14. The setup script will pin the same versions.

| Component | Version |
|---|---|
| [physical-ai-studio](https://github.com/open-edge-platform/physical-ai-studio) | commit `8f1324161ece2b24f2d38765b6e943ed1961258c` |
| [physicalai](https://github.com/openvinotoolkit/physicalai) (MuJoCo plugin) | commit `0ec06e9f38ac69170f391cf8343d83f7c6081668` (plugin 0.5.1.dev27, physicalai 0.4.1.dev7, MuJoCo 3.16.0, OpenVINO 2026.4.1) |
| OS | WSL2, Ubuntu 24.04.5 LTS |
| Python | 3.13.18 (via uv 0.13.8) |
| Node.js / npm | v24.22.3 / 11.19.1 |
| git / git-lfs | 2.43.0 / 3.4.1 |
| ffmpeg | 6.1.1 |
| Studio backend (`uv sync --extra xpu --extra sim-e2e`) | physicalai-studio 0.3.0, PyTorch 2.11.0+xpu, MuJoCo plugin 0.5.0 |
| Intel GPU compute runtime (`ppa:kobuk-team/intel-graphics`) | libze-intel-gpu1 / intel-opencl-icd 26.40, IGC 2.42.14 |
| WSL memory (`%USERPROFILE%\.wslconfig`) | `memory=22GB`, `swap=16GB` |

### M2 result

<p align="center">
  <img src="images/sim-m2-run-model.webp" alt="ACT policy running in simulation with OpenVINO on the iGPU" width="80%">
  <br>
  <em>Run model in Studio: ACT · OpenVINO · GPU (Inference mode) on the MuJoCo SO-101 in the <code>conveyor_sort</code> scene.</em>
</p>

ACT trained on the iGPU (XPU) for 1 epoch on 3 `conveyor_sort` episodes in 6 min 9 s, then ran in simulation with OpenVINO on the GPU. As in the hands-on session, the undertrained model moves toward the items but cannot grasp them.
