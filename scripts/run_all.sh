#!/usr/bin/env bash
set -euo pipefail

# One-click pipeline:
# clean ws -> install deps -> build -> source env -> cleanup -> full cycle

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS_DIR="$ROOT_DIR/ai_slam_ws"
VENV_ACTIVATE="$ROOT_DIR/.venv/bin/activate"

safe_source() {
  # Some setup scripts are not compatible with nounset (`set -u`).
  set +u
  # shellcheck disable=SC1090
  source "$1"
  set -u
}

sanitize_path_var() {
  local var_name="$1"
  local raw="${!var_name-}"
  local out=""
  local part=""
  IFS=':' read -r -a parts <<< "$raw"
  for part in "${parts[@]}"; do
    [[ -z "$part" ]] && continue
    [[ -d "$part" ]] || continue
    if [[ -z "$out" ]]; then
      out="$part"
    else
      out="$out:$part"
    fi
  done
  export "$var_name=$out"
}

echo "========================================"
echo "SLAM_AI one-click run"
echo "ROOT: $ROOT_DIR"
echo "WS:   $WS_DIR"
echo "========================================"

if [[ ! -d "$WS_DIR" ]]; then
  echo "ERROR: Workspace not found: $WS_DIR"
  exit 1
fi

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "ERROR: Virtual environment not found: $VENV_ACTIVATE"
  exit 1
fi

echo "[0/8] Removing stray root colcon artifacts (if any)"
rm -rf "$ROOT_DIR/build" "$ROOT_DIR/install" "$ROOT_DIR/log"

cd "$WS_DIR"

echo "[1/8] Removing build artifacts (build/install/log)"
rm -rf build/ install/ log/

echo "[2/8] Activating Python venv"
safe_source "$VENV_ACTIVATE"

echo "[2.5/8] Sanitizing stale prefix paths"
sanitize_path_var AMENT_PREFIX_PATH
sanitize_path_var COLCON_PREFIX_PATH
sanitize_path_var CMAKE_PREFIX_PATH

echo "[3/8] Installing ROS dependencies with rosdep"
rosdep install --from-paths src --ignore-src -r -y --skip-keys ament_python

echo "[4/8] Building workspace"
colcon build \
  --symlink-install \
  --build-base "$WS_DIR/build" \
  --install-base "$WS_DIR/install" \
  --log-base "$WS_DIR/log"

echo "[5/8] Sourcing ROS Jazzy environment"
safe_source /opt/ros/jazzy/setup.bash

echo "[6/8] Sourcing local workspace"
safe_source install/setup.bash

echo "[7/8] Running cleanup"
"$ROOT_DIR/scripts/cleanup.sh"

echo "[8/8] Running full cycle"
"$ROOT_DIR/scripts/run_full_cycle.sh" "$@"
