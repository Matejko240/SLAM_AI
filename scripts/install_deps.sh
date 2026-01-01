#!/usr/bin/env bash
set -euo pipefail

ROS_DISTRO="${ROS_DISTRO:-jazzy}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

sudo apt-get update

sudo apt-get install -y --no-install-recommends \
  curl gnupg lsb-release ca-certificates \
  git build-essential cmake pkg-config \
  python3 python3-venv python3-dev python3-pip \
  python3-numpy python3-matplotlib \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-setuptools

if [ ! -f /usr/share/keyrings/ros-archive-keyring.gpg ]; then
  sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
fi

if [ ! -f /etc/apt/sources.list.d/ros2.list ]; then
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | \
    sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
fi

sudo apt-get update

sudo apt-get install -y --no-install-recommends \
  "ros-${ROS_DISTRO}-desktop" \
  "ros-${ROS_DISTRO}-slam-toolbox" \
  "ros-${ROS_DISTRO}-navigation2" \
  "ros-${ROS_DISTRO}-nav2-bringup" \
  "ros-${ROS_DISTRO}-robot-state-publisher" \
  "ros-${ROS_DISTRO}-joint-state-publisher" \
  "ros-${ROS_DISTRO}-xacro" \
  "ros-${ROS_DISTRO}-tf-transformations" \
  "ros-${ROS_DISTRO}-ros-gz" \
  "ros-${ROS_DISTRO}-ros-gz-sim" \
  "ros-${ROS_DISTRO}-ros-gz-bridge" \
  "ros-${ROS_DISTRO}-gz-tools-vendor" \
  "ros-${ROS_DISTRO}-gz-sim-vendor"

sudo rosdep init 2>/dev/null || true
rosdep update

if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv --system-site-packages "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install -U pip wheel setuptools

"${VENV_DIR}/bin/python" -m pip install -U \
  torch --index-url https://download.pytorch.org/whl/cpu

echo "OK: deps installed."
echo "Next:"
echo "  source /opt/ros/${ROS_DISTRO}/setup.bash"
echo "  source ${VENV_DIR}/bin/activate"
echo "  cd ai_slam_ws && rosdep install --from-paths src --ignore-src -r -y"
echo "  cd ai_slam_ws && python -m colcon build --symlink-install"
