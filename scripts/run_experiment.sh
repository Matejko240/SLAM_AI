#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"
WS_SETUP="$ROOT_DIR/ai_slam_ws/install/setup.bash"
VENV_SITE="$ROOT_DIR/.venv/lib/python3.12/site-packages"
if [[ -d "$VENV_SITE" ]]; then
    export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$VENV_SITE"
fi

safe_source() {
    set +u
    # shellcheck disable=SC1090
    source "$1"
    set -u
}

fail_runtime_prereq() {
    cat >&2 <<EOF
ERROR: $1

This command needs the ROS 2 + workspace runtime:
  1. Install ROS dependencies: ./scripts/install_deps.sh
  2. Build the workspace:
     cd "$ROOT_DIR/ai_slam_ws"
     source "$ROS_SETUP"
     colcon build --symlink-install

After that, retry the command from:
  cd "$ROOT_DIR"
  source .venv/bin/activate
  ./scripts/run_experiment.sh ...
EOF
    exit 1
}
# Run SLAM AI experiment with filtered output (suppresses TF_OLD_DATA spam)
# Usage:
#   ./run_experiment.sh                    # Default config
#   ./run_experiment.sh fast               # Fast test config
#   ./run_experiment.sh full               # Full experiment config
#   ./run_experiment.sh dataset            # Tylko zbieranie datasetu
#   ./run_experiment.sh config:=my.yaml    # Custom config

cd "$ROOT_DIR"

if [[ ! -f "$ROS_SETUP" ]]; then
    fail_runtime_prereq "Missing ROS 2 setup: $ROS_SETUP"
fi

if [[ ! -f "$WS_SETUP" ]]; then
    fail_runtime_prereq "Workspace is not built yet: $WS_SETUP"
fi

safe_source "$ROS_SETUP"
safe_source "$WS_SETUP"

if ! command -v ros2 >/dev/null 2>&1; then
    fail_runtime_prereq "ros2 CLI is still unavailable after sourcing the environment."
fi

# Determine launch args
LAUNCH_ARGS=()
case "${1:-}" in
    fast)
        LAUNCH_ARGS+=("config:=fast_test.yaml")
        shift
        ;;
    full)
        LAUNCH_ARGS+=("config:=experiment_config.yaml")
        shift
        ;;
    train)
        LAUNCH_ARGS+=("config:=experiment_config.yaml" "phase:=train")
        shift
        ;;
    dataset)
        LAUNCH_ARGS+=("config:=experiment_config.yaml" "phase:=dataset")
        shift
        ;;
    test)
        LAUNCH_ARGS+=("config:=experiment_config.yaml" "phase:=test")
        shift
        ;;
    *)
        # Pass through any other arguments
        ;;
esac

echo "========================================"
echo "Running SLAM AI Experiment"
echo "Launch args: ${LAUNCH_ARGS[*]:-default (experiment_config.yaml)}"
echo "Extra args: $@"
echo "========================================"
echo ""

# Run with filtered output - hide TF, SLAM sync warnings, and PyTorch GPU warnings
set +e
ros2 launch ai_slam_bringup demo.launch.py "${LAUNCH_ARGS[@]}" "$@" 2>&1 | grep -v -E "TF_OLD_DATA|wiki.ros.org/tf/Errors|jump back in time|Transform from base_link|unconnected trees|extrapolation|rcl_lifecycle|cuda capability|CUDA capability|pytorch.org/get-started|warnings.warn|UserWarning|sm_61|sm_70|Failed to compute odom"
launch_rc=${PIPESTATUS[0]}
set -e
exit "$launch_rc"
