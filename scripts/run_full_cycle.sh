#!/bin/bash
# Skrypt uruchamiający pełny cykl: Trening -> Test
# Z automatycznym czyszczeniem i wymuszonym zamykaniem GUI na końcu.
set -euo pipefail

# W WSL FastDDS SHM często zostawia locki portów po restarcie procesów.
# Domyślnie wymuszamy UDPv4 (bez SHM), chyba że użytkownik poda własne ustawienie.
if grep -qiE "(microsoft|wsl)" /proc/version 2>/dev/null; then
    export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"
fi

# Blokada przed równoległym uruchomieniem wielu cykli.
if command -v flock >/dev/null 2>&1; then
    LOCK_FILE="/tmp/slam_ai_run_full_cycle.lock"
    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
        echo "BŁĄD: Inny run_full_cycle.sh już działa. Poczekaj lub użyj cleanup.sh."
        exit 1
    fi
fi

# --- 0. Ustawienia wstępne ---
export PYTHONPATH=$PYTHONPATH:$HOME/SLAM_AI/.venv/lib/python3.12/site-packages
cd "$(dirname "$0")/.."

# === CZYSZCZENIE PRZED STARTEM ===
echo "--- Uruchamianie cleanup.sh przed startem ---"
~/SLAM_AI/scripts/cleanup.sh || true
sleep 2 
echo "--- Środowisko wyczyszczone ---"

CONFIG_FILE="experiment_config.yaml"
if [ $# -ge 1 ] && [ -n "${1:-}" ]; then
    CONFIG_FILE="$1"
    shift
fi

if [[ "$CONFIG_FILE" = /* ]]; then
    CONFIG_PATH="$CONFIG_FILE"
else
    CONFIG_PATH="ai_slam_ws/src/ai_slam_bringup/config/$CONFIG_FILE"
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "BŁĄD: Nie znaleziono pliku config: $CONFIG_PATH"
    exit 1
fi

# --- 1. Generowanie ID Eksperymentu ---
EXP_ID="exp_$(date +%Y%m%d_%H%M%S)"

# --- 2. Parsowanie Konfiguracji (bezpiecznie przez YAML) ---
read -r TRAIN_MAP TEST_MAP DATASET_TIME EVAL_TIME < <(
python3 - "$CONFIG_PATH" <<'PY'
import sys
import yaml

cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

sim = cfg.get("simulation", {}) or {}
timing = cfg.get("timing", {}) or {}

train_map = sim.get("train_world", "world_train_house.sdf")
test_map = sim.get("test_world", "world_test_house.sdf")
dataset_time = timing.get("dataset_duration", 30.0)
eval_time = timing.get("eval_duration", 60.0)

print(f"{train_map} {test_map} {dataset_time} {eval_time}")
PY
)

TRAIN_MAP="${TRAIN_MAP:-world_train_house.sdf}"
TEST_MAP="${TEST_MAP:-world_test_house.sdf}"
DATASET_TIME="${DATASET_TIME:-30.0}"
EVAL_TIME="${EVAL_TIME:-60.0}"

echo "=========================================================="
echo "ID EKSPERYMENTU: $EXP_ID"
echo "=========================================================="
echo "FAZA 1: TRENING"
echo "Mapa: $TRAIN_MAP"
echo "Czas datasetu: $DATASET_TIME s"
echo "=========================================================="

# Uruchamiamy trening
TRAIN_LAUNCH_RC=0
set +e
ros2 launch ai_slam_bringup demo.launch.py \
    config:=$CONFIG_FILE \
    phase:=train \
    world_sdf:=$TRAIN_MAP \
    dataset_duration_sec:=$DATASET_TIME \
    experiment_id:=$EXP_ID \
    "$@"
TRAIN_LAUNCH_RC=$?
set -e

if [ "$TRAIN_LAUNCH_RC" -ne 0 ]; then
    echo ""
    echo "OSTRZEŻENIE: Faza treningu zakończona kodem $TRAIN_LAUNCH_RC."
    echo "Sprawdzam, czy model został mimo to zapisany..."
fi

echo ""
echo "--- Trening zakończony. Weryfikacja wyników... ---"
echo ""

# Sprawdzamy model w głównym folderze out/
MODEL_PATH="out/$EXP_ID/model.pt"

if [ ! -f "$MODEL_PATH" ]; then
    # Fallback
    if [ -f "ai_slam_ws/out/$EXP_ID/model.pt" ]; then
        MODEL_PATH="ai_slam_ws/out/$EXP_ID/model.pt"
    else
        echo "BŁĄD: Plik modelu nie istnieje: $MODEL_PATH"
        # Spróbujmy posprzątać przed wyjściem
        ~/SLAM_AI/scripts/cleanup.sh || true
        exit 1
    fi
fi

echo "SUKCES: Model znaleziony ($MODEL_PATH)."
echo "Przechodzę do fazy testów..."

# Czyszczenie między fazami (ważne, żeby zamknąć poprzednie Gazebo)
echo "--- Czyszczenie przed Faza 2 ---"
~/SLAM_AI/scripts/cleanup.sh || true
sleep 5 # Dłuższa pauza, żeby Gazebo na pewno zniknęło

echo "Start FAZY 2..."

echo "=========================================================="
echo "FAZA 2: TEST / EWALUACJA"
echo "Mapa: $TEST_MAP"
echo "ID: $EXP_ID"
echo "Czas testu: $EVAL_TIME s"
echo "=========================================================="

TEST_LAUNCH_RC=0
set +e
ros2 launch ai_slam_bringup demo.launch.py \
    config:=$CONFIG_FILE \
    phase:=test \
    world_sdf:=$TEST_MAP \
    experiment_id:=$EXP_ID \
    eval_duration_sec:=$EVAL_TIME \
    "$@"
TEST_LAUNCH_RC=$?
set -e

if [ "$TEST_LAUNCH_RC" -ne 0 ]; then
    echo ""
    echo "OSTRZEŻENIE: Faza testowa zakończona kodem $TEST_LAUNCH_RC."
    echo "Sprawdzam, czy results.json został zapisany..."
fi

RESULTS_PATH="out/$EXP_ID/results.json"
if [ ! -f "$RESULTS_PATH" ]; then
    if [ -f "ai_slam_ws/out/$EXP_ID/results.json" ]; then
        RESULTS_PATH="ai_slam_ws/out/$EXP_ID/results.json"
    else
        echo "BŁĄD: Plik wyników nie istnieje: $RESULTS_PATH"
        ~/SLAM_AI/scripts/cleanup.sh || true
        exit 1
    fi
fi

echo ""
echo "PEŁNY CYKL ZAKOŃCZONY"
echo "Wyniki: out/$EXP_ID"

# === FINALNE CZYSZCZENIE ===
echo "--- Zamykanie wszystkich procesów (cleanup) ---"
~/SLAM_AI/scripts/cleanup.sh || true
echo "--- Gotowe ---"
