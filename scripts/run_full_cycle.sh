#!/bin/bash
# Skrypt uruchamiający pełny cykl: Trening -> Test
# Z automatycznym czyszczeniem i wymuszonym zamykaniem GUI na końcu.

# --- 0. Ustawienia wstępne ---
export PYTHONPATH=$PYTHONPATH:$HOME/SLAM_AI/.venv/lib/python3.12/site-packages
cd "$(dirname "$0")/.."

# === CZYSZCZENIE PRZED STARTEM ===
echo "--- Uruchamianie cleanup.sh przed startem ---"
~/SLAM_AI/scripts/cleanup.sh || true
sleep 2 
echo "--- Środowisko wyczyszczone ---"

CONFIG_FILE="experiment_config.yaml"
CONFIG_PATH="ai_slam_ws/src/ai_slam_bringup/config/$CONFIG_FILE"

# --- 1. Generowanie ID Eksperymentu ---
EXP_ID="exp_$(date +%Y%m%d_%H%M%S)"

# --- 2. Parsowanie Konfiguracji ---
TRAIN_MAP=$(grep "train_world:" $CONFIG_PATH | sed 's/#.*//' | sed 's/.*: "\(.*\)"/\1/' | tr -d ' "')
TEST_MAP=$(grep "test_world:" $CONFIG_PATH | sed 's/#.*//' | sed 's/.*: "\(.*\)"/\1/' | tr -d ' "')
DATASET_TIME=$(grep "dataset_duration:" $CONFIG_PATH | sed 's/#.*//' | head -1 | awk '{print $2}')
EVAL_TIME=$(grep "eval_duration:" $CONFIG_PATH | sed 's/#.*//' | head -1 | awk '{print $2}')

# Wartości domyślne
TRAIN_MAP=${TRAIN_MAP:-"world_train_house.sdf"}
TEST_MAP=${TEST_MAP:-"world_test_house.sdf"}
DATASET_TIME=${DATASET_TIME:-30.0}
EVAL_TIME=${EVAL_TIME:-60.0}

echo "=========================================================="
echo "ID EKSPERYMENTU: $EXP_ID"
echo "=========================================================="
echo "FAZA 1: TRENING"
echo "Mapa: $TRAIN_MAP"
echo "Czas datasetu: $DATASET_TIME s"
echo "=========================================================="

# Uruchamiamy trening
ros2 launch ai_slam_bringup demo.launch.py \
    config:=$CONFIG_FILE \
    phase:=train \
    world_sdf:=$TRAIN_MAP \
    dataset_duration_sec:=$DATASET_TIME \
    experiment_id:=$EXP_ID

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

ros2 launch ai_slam_bringup demo.launch.py \
    config:=$CONFIG_FILE \
    phase:=test \
    world_sdf:=$TEST_MAP \
    experiment_id:=$EXP_ID \
    eval_duration_sec:=$EVAL_TIME

echo ""
echo "PEŁNY CYKL ZAKOŃCZONY"
echo "Wyniki: out/$EXP_ID"

# === FINALNE CZYSZCZENIE ===
echo "--- Zamykanie wszystkich procesów (cleanup) ---"
~/SLAM_AI/scripts/cleanup.sh || true
echo "--- Gotowe ---"