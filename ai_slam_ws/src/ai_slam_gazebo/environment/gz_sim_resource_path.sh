# Ustawia GZ_SIM_RESOURCE_PATH tak, aby Gazebo znajdowało modele z ai_slam_gazebo (model://…).
# Użycie (po colcon build + source install/setup.bash):
#   source "$(ros2 pkg prefix ai_slam_gazebo)/share/ai_slam_gazebo/environment/gz_sim_resource_path.sh"
# Albo bez instalacji (katalog źródeł pakietu):
#   source /ścieżka/do/ai_slam_gazebo/environment/gz_sim_resource_path.sh

_aisg_env_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Zainstalowany pakiet: .../install/.../share/ai_slam_gazebo/environment -> prefix = ../../..
_aisg_prefix="$(CDPATH= cd -- "${_aisg_env_dir}/../../.." && pwd)"
_aisg_models="${_aisg_prefix}/share/ai_slam_gazebo/models"
# Drzewo źródeł: environment leży w ai_slam_gazebo/environment -> modele w ../models
if [ ! -d "${_aisg_models}" ]; then
  _aisg_pkg_root="$(CDPATH= cd -- "${_aisg_env_dir}/.." && pwd)"
  if [ -d "${_aisg_pkg_root}/models" ]; then
    _aisg_models="${_aisg_pkg_root}/models"
  fi
fi
if [ -d "${_aisg_models}" ]; then
  case ":${GZ_SIM_RESOURCE_PATH:-}:" in
    *":${_aisg_models}:"*) ;;
    *) export GZ_SIM_RESOURCE_PATH="${_aisg_models}${GZ_SIM_RESOURCE_PATH:+:${GZ_SIM_RESOURCE_PATH}}" ;;
  esac
fi
unset _aisg_env_dir _aisg_prefix _aisg_models
