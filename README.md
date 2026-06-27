# SLAM_AI

![ROS 2 Jazzy](https://img.shields.io/badge/ROS%202-Jazzy-22314E?logo=ros&logoColor=white)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A research framework that compares **classical** and **AI-assisted** localization and
mapping for a 2D mobile robot, built on **ROS 2 Jazzy**, **Gazebo**, and a 360° LiDAR.
It provides a reproducible pipeline to collect datasets in simulated worlds, train
several learned motion/odometry models, and evaluate them against classical SLAM
baselines on a separate test world.

## Methods compared

| # | Method | Description |
|---|--------|-------------|
| 1 | **Baseline SLAM** | `slam_toolbox` running on noisy odometry (`/map`). |
| 2 | **AI SLAM** | An MLP (`363 → 256 → 128 → 64 → 3`) corrects the pose estimate before it is fed to `slam_toolbox` (`/map_ai`). |
| 3 | **ScanMatcher (local map)** | Classical consecutive-scan matching — fast and lightweight reference. |
| 4 | **ScanMatcher (brute force)** | Full grid search over candidate transforms (slower, optional reference). |
| 5 | **Robak** | A Conv1D model that predicts `Δx, Δy, Δθ` from a pair of scans `(scan_{t-1}, scan_t)`. |
| 6 | **Rywak** | An MLP that predicts `v, ω` from the features `d_theta1, d_theta2, delta_scan`. |

## Repository structure

```
SLAM_AI/
├── ai_slam_ws/src/
│   ├── ai_slam_ai/           # AI models and ROS nodes (recorders, training, inference)
│   ├── ai_slam_bringup/      # Launch files and experiment configuration
│   ├── ai_slam_description/  # Robot model (diffbot.sdf)
│   ├── ai_slam_eval/         # Evaluation node (metrics and plots)
│   └── ai_slam_gazebo/       # Gazebo worlds and the ROS–Gazebo bridge
├── scripts/                  # Pipeline, dashboard, and reporting utilities
├── docs/                     # Auto-generated function index
├── real_world_results/       # Processed results from real-world mocap validation
├── requirements.txt          # Python (pip) dependencies
└── README.md
```

## Requirements

- Ubuntu 24.04 with **ROS 2 Jazzy**
- **Gazebo** (as bundled with ROS 2 Jazzy)
- Python 3.12 and the packages listed in [`requirements.txt`](requirements.txt)
  (`numpy`, `torch`, `matplotlib`, `PyYAML`, `psutil`, `scikit-image`)

ROS-provided Python packages (`rclpy`, the `*_msgs` interfaces, `tf2_ros`, `launch`,
`launch_ros`, …) come from the ROS 2 installation and are **not** installed via pip.

## Installation

```bash
git clone https://github.com/Matejko240/SLAM_AI.git
cd SLAM_AI

# Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# System / ROS dependencies (optional helper)
./scripts/install_deps.sh

# Build the ROS 2 workspace
cd ai_slam_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## Usage

### One-click full cycle

Runs the complete sequence: clean `build/install/log`, activate `.venv`,
`rosdep install`, `colcon build --symlink-install`, source the workspace, and run
the full collect → train → test → evaluate cycle.

```bash
./scripts/run_all.sh
```

### Single experiment

`run_experiment.sh` launches individual phases:

```bash
./scripts/run_experiment.sh fast    # fast smoke configuration
./scripts/run_experiment.sh full    # full experiment configuration
./scripts/run_experiment.sh train   # training phase only
./scripts/run_experiment.sh test    # test/evaluation phase only
```

### Web dashboard

A local HTTP dashboard lets you browse experiments and datasets from `out/`, launch
the pipeline scripts, train the `AI` / `robak` / `rywak` models, and generate
trajectory and error plots.

```bash
source .venv/bin/activate
./scripts/run_dashboard.sh          # default: http://127.0.0.1:8765
./scripts/kill_dashboard.sh         # stop (optionally pass a port, e.g. 8766)
```

## Configuration

- Main experiment config: `ai_slam_ws/src/ai_slam_bringup/config/experiment_config.yaml`
- Main launch file: `ai_slam_ws/src/ai_slam_bringup/launch/demo.launch.py`
- Available simulated worlds: `world_house.sdf`, `world_office.sdf`, `world_hospital.sdf`

The pipeline phases are:

1. Collect a dataset in a chosen world.
2. Train the models.
3. Test and evaluate on a separately chosen test world.
4. Save results under `out/exp_YYYYMMDD_HHMMSS`.

## Outputs

Each `out/exp_*` directory contains, among others:

- `results.json` — aggregated metrics (RMSE / IoU)
- `eval_trajectory.png`, `eval_errors.png`, `eval_maps.png`
- `dataset*.npz`, `model*.pt`, `train_history*.json`
- `train_curve_{ai,robak,rywak}.png` and dataset coverage plots

### Reports and reference maps

```bash
# Thesis-style tables and figures from a sweep
python3 scripts/generate_thesis_report.py \
  --sweep out/sweep_YYYYMMDD_HHMMSS.csv \
  --output-dir out/thesis_report

# Reference occupancy map from a world
python3 scripts/generate_reference_map.py \
  --world ai_slam_ws/src/ai_slam_gazebo/worlds/world_office.sdf \
  --output-stem reference_map_office
```

## Real-world validation

The `real_world_results/` directory holds the processed results of a real-world
validation campaign using a JetRacer platform and a motion-capture ground truth
(CSV trajectories and figures). The online inference path is provided by
`jetracer_scan_adapter` and `jetracer_online.launch.py`.

## License

Released under the [MIT License](LICENSE).
