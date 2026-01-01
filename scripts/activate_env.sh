#!/usr/bin/env bash
set -euo pipefail

ROS_DISTRO="${ROS_DISTRO:-jazzy}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
  # shellcheck disable=SC1090
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
fi

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "${REPO_ROOT}/.venv/bin/activate"
fi

if [ -f "${REPO_ROOT}/ai_slam_ws/install/setup.bash" ]; then
  # shellcheck disable=SC1090
  source "${REPO_ROOT}/ai_slam_ws/install/setup.bash"
fi

exec "$@"
