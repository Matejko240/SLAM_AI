#!/usr/bin/env bash
set -euo pipefail

# Konfiguracja
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
OS_CODENAME="$(. /etc/os-release && echo "${UBUNTU_CODENAME:-${VERSION_CODENAME}}")"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

echo "=== Rozpoczynam instalację zależności dla projektu SLAM AI ==="
echo "Repo root: ${REPO_ROOT}"
echo "ROS Distro: ${ROS_DISTRO}"
echo "Ubuntu codename: ${OS_CODENAME}"

# ROS apt source conflict guard:
# jeśli istnieją jednocześnie ros2.list i ros2.sources, apt może zgłaszać
# "Conflicting values set for option Signed-By".
ROS2_LIST_FILE="/etc/apt/sources.list.d/ros2.list"
ROS2_SOURCES_FILE="/etc/apt/sources.list.d/ros2.sources"
if [[ -f "${ROS2_LIST_FILE}" && -f "${ROS2_SOURCES_FILE}" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  BACKUP_FILE="${ROS2_LIST_FILE}.bak.${TS}"
  echo "Wykryto konflikt wpisów ROS APT: ${ROS2_LIST_FILE} + ${ROS2_SOURCES_FILE}"
  echo "Archiwizuję ${ROS2_LIST_FILE} -> ${BACKUP_FILE} (preferuję ros2.sources zarządzane przez ros2-apt-source)"
  sudo mv "${ROS2_LIST_FILE}" "${BACKUP_FILE}"
fi

# 1. Instalacja pakietów systemowych i ROS 2
sudo apt-get update
sudo apt-get install -f -y
sudo apt-get install -y --no-install-recommends \
  curl gnupg lsb-release ca-certificates software-properties-common \
  git build-essential cmake pkg-config \
  python3 python3-venv python3-dev python3-pip \
  python3-numpy python3-matplotlib \
  python3-setuptools

# Ubuntu 24.04 / ROS Jazzy: najstabilniej dodać repo przez ros2-apt-source.
sudo add-apt-repository -y universe

if ! dpkg -s ros2-apt-source >/dev/null 2>&1; then
  echo "=== Instalacja ros2-apt-source ==="
  ROS_APT_SOURCE_VERSION="$(
    curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest |
      grep -F '"tag_name"' |
      awk -F'"' '{print $4}'
  )"
  ROS_APT_DEB="/tmp/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${OS_CODENAME}_all.deb"
  curl -fL -o "${ROS_APT_DEB}" \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${OS_CODENAME}_all.deb"
  sudo dpkg -i "${ROS_APT_DEB}"
  rm -f "${ROS_APT_DEB}"
fi

sudo apt-get update

# Na nowszych Ubuntu rosdep/colcon są dostarczane przez ros-dev-tools.
sudo apt-get install -y --no-install-recommends ros-dev-tools

# Instalacja pakietów ROS 2 potrzebnych do symulacji i nawigacji
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

# Inicjalizacja rosdep
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    sudo rosdep init
fi
rosdep update || echo "rosdep update failed, continuing..."

# 2. Konfiguracja środowiska wirtualnego Python (VENV)
echo "=== Konfiguracja Python Virtual Environment ==="

if [ ! -d "${VENV_DIR}" ]; then
    echo "Tworzenie nowego venv w: ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
    # Tworzymy plik flagi, aby colcon ignorował ten folder
    touch "${VENV_DIR}/COLCON_IGNORE"
else
    echo "Venv już istnieje w: ${VENV_DIR}"
fi

# Aktywacja środowiska do instalacji pakietów
source "${VENV_DIR}/bin/activate"

echo "Aktualizacja pip..."
pip install --upgrade pip setuptools wheel

echo "--- Instalacja PyTorch z kanału: ${TORCH_INDEX_URL} ---"
echo "Jeśli chcesz wymusić inny build, ustaw TORCH_INDEX_URL przed uruchomieniem skryptu."
pip install --upgrade torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"

echo "--- Weryfikacja PyTorch / CUDA ---"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print("device0", torch.cuda.get_device_name(0))
    try:
        x = torch.randn(32, 32, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print("cuda_smoke_test", "ok", tuple(y.shape))
    except Exception as exc:
        print("cuda_smoke_test", "failed", exc)
PY

echo "--- Instalacja pozostałych bibliotek ---"
pip install numpy matplotlib transforms3d scipy pyyaml

# Opcjonalnie: requirements.txt jeśli istnieje
if [ -f "${REPO_ROOT}/requirements.txt" ]; then
    echo "Instalacja z requirements.txt..."
    pip install -r "${REPO_ROOT}/requirements.txt"
fi

echo ""
echo "========================================================"
echo " INSTALACJA ZAKOŃCZONA SUKCESEM"
echo "========================================================"
echo "Aby rozpocząć pracę, pamiętaj o aktywacji środowiska:"
echo "source .venv/bin/activate"
echo "========================================================"
