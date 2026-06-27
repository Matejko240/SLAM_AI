#!/usr/bin/env bash
# Uruchomienie Gazebo z modelami z ai_slam_gazebo (naprawia błąd „Unable to find uri[model://…]”).
#
#   source /opt/ros/jazzy/setup.bash   # lub humble
#   source ~/SLAM_AI/ai_slam_ws/install/setup.bash
#   ~/SLAM_AI/scripts/run_gz_world.sh world_hospital.sdf
#   ~/SLAM_AI/scripts/run_gz_world.sh world_office.sdf
#
# Możesz podać pełną ścieżkę do .sdf zamiast samej nazwy pliku z worlds/.

set -euo pipefail

if [ -z "${ROS_DISTRO:-}" ]; then
  echo "Najpierw: source /opt/ros/<distro>/setup.bash" >&2
  exit 1
fi

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WS_DEFAULT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)/ai_slam_ws"
WS="${SLAM_AI_WS:-${WS_DEFAULT}}"

if [ ! -f "${WS}/install/setup.bash" ]; then
  echo "Brak ${WS}/install/setup.bash — ustaw SLAM_AI_WS lub zbuduj workspace." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${WS}/install/setup.bash"
# shellcheck source=/dev/null
source "$(ros2 pkg prefix ai_slam_gazebo)/share/ai_slam_gazebo/environment/gz_sim_resource_path.sh"

WORLD_ARG="${1:?Podaj nazwę pliku .sdf (np. world_hospital.sdf)}"
shift || true

if [[ "${WORLD_ARG}" == *.sdf ]] && [ -f "${WORLD_ARG}" ]; then
  WORLD_PATH="${WORLD_ARG}"
else
  WORLD_PATH="$(ros2 pkg prefix ai_slam_gazebo)/share/ai_slam_gazebo/worlds/${WORLD_ARG}"
fi

if [ ! -f "${WORLD_PATH}" ]; then
  echo "Nie znaleziono świata: ${WORLD_PATH}" >&2
  exit 1
fi

exec gz sim "${WORLD_PATH}" -r "$@"
