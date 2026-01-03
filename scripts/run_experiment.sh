#!/bin/bash
# Run SLAM AI experiment with filtered output (suppresses TF_OLD_DATA spam)
# Usage:
#   ./run_experiment.sh                    # Default config
#   ./run_experiment.sh fast               # Fast test config
#   ./run_experiment.sh full               # Full experiment config
#   ./run_experiment.sh config:=my.yaml    # Custom config

cd "$(dirname "$0")/.."

# Source ROS2
source /opt/ros/jazzy/setup.bash
source ai_slam_ws/install/setup.bash

# Determine config
CONFIG=""
case "$1" in
    fast)
        CONFIG="config:=fast_test.yaml"
        shift
        ;;
    full)
        CONFIG="config:=experiment_config.yaml"
        shift
        ;;
    *)
        # Pass through any other arguments
        ;;
esac

echo "========================================"
echo "Running SLAM AI Experiment"
echo "Config: ${CONFIG:-default (experiment_config.yaml)}"
echo "Extra args: $@"
echo "========================================"
echo ""

# Run with filtered output - hide TF, SLAM sync warnings, and PyTorch GPU warnings
ros2 launch ai_slam_bringup demo.launch.py $CONFIG "$@" 2>&1 | grep -v -E "TF_OLD_DATA|wiki.ros.org/tf/Errors|jump back in time|Transform from base_link|unconnected trees|extrapolation|rcl_lifecycle|cuda capability|CUDA capability|pytorch.org/get-started|warnings.warn|UserWarning|sm_61|sm_70|Failed to compute odom"
