#!/usr/bin/env bash
set -euo pipefail

# Konfiguracja
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

echo "=== Rozpoczynam instalację zależności dla projektu SLAM AI ==="
echo "Repo root: ${REPO_ROOT}"
echo "ROS Distro: ${ROS_DISTRO}"

# 1. Instalacja pakietów systemowych i ROS 2
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  curl gnupg lsb-release ca-certificates \
  git build-essential cmake pkg-config \
  python3 python3-venv python3-dev python3-pip \
  python3-numpy python3-matplotlib \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-setuptools

# Dodanie kluczy ROS (jeśli nie istnieją)
if [ ! -f /usr/share/keyrings/ros-archive-keyring.gpg ]; then
  sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
fi

# Dodanie repozytorium ROS (jeśli nie istnieje)
if [ ! -f /etc/apt/sources.list.d/ros2.list ]; then
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | \
    sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
fi

sudo apt-get update

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
    sudo rosdep init 2>/dev/null || true
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

echo "--- Instalacja PyTorch dla GTX 1050 Ti (CUDA 11.8) ---"
# KLUCZOWE: Instalacja wersji kompatybilnej z Python 3.12 oraz Pascal GPU (GTX 10xx)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

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