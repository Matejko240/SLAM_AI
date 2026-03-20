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
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"
WS_SETUP="$ROOT_DIR/ai_slam_ws/install/setup.bash"
CLEANUP_SCRIPT="$ROOT_DIR/scripts/cleanup.sh"
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

After that, retry from:
  cd "$ROOT_DIR"
  source .venv/bin/activate
  ./scripts/run_full_cycle.sh ...
EOF
    exit 1
}

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

# === CZYSZCZENIE PRZED STARTEM ===
echo "--- Uruchamianie cleanup.sh przed startem ---"
"$CLEANUP_SCRIPT" || true
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
EXP_ID=""
for arg in "$@"; do
    if [[ "$arg" == experiment_id:=* ]]; then
        EXP_ID="${arg#experiment_id:=}"
        break
    fi
done
if [ -z "$EXP_ID" ]; then
    EXP_ID="exp_$(date +%Y%m%d_%H%M%S)"
fi

# --- 2. Parsowanie Konfiguracji (bezpiecznie przez YAML) ---
mapfile -t CONFIG_LINES < <(
python3 - "$CONFIG_PATH" <<'PY'
import json
import sys
import yaml

cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

sim = cfg.get("simulation", {}) or {}
timing = cfg.get("timing", {}) or {}
evaluation = cfg.get("evaluation", {}) or {}

train_map = str(sim.get("train_world", "world_house.sdf"))
default_test_map = str(sim.get("test_world", "world_house.sdf"))
default_ref_map = str(evaluation.get("reference_map_yaml", "reference_map.yaml"))
dataset_time = timing.get("dataset_duration", 30.0)
eval_time = timing.get("eval_duration", 60.0)

scenarios = []
for idx, item in enumerate(evaluation.get("test_scenarios", []) or [], start=1):
    if not isinstance(item, dict):
        continue
    world = str(item.get("world") or item.get("test_world") or "").strip()
    if not world:
        continue
    label = str(item.get("label") or item.get("name") or f"test_{idx:02d}").strip()
    ref_map = str(item.get("reference_map_yaml") or default_ref_map).strip()
    scenarios.append(
        {
            "label": label or f"test_{idx:02d}",
            "world": world,
            "reference_map_yaml": ref_map or default_ref_map,
        }
    )

if not scenarios:
    scenarios = [
        {
            "label": "test_primary",
            "world": default_test_map,
            "reference_map_yaml": default_ref_map,
        }
    ]

print(f"TRAIN_MAP\t{train_map}")
print(f"DATASET_TIME\t{dataset_time}")
print(f"EVAL_TIME\t{eval_time}")
for scenario in scenarios:
    print("SCENARIO\t" + json.dumps(scenario, ensure_ascii=False))
PY
)

TRAIN_MAP="world_house.sdf"
DATASET_TIME="30.0"
EVAL_TIME="60.0"
TEST_SCENARIOS=()
for line in "${CONFIG_LINES[@]}"; do
    key="${line%%$'\t'*}"
    value="${line#*$'\t'}"
    case "$key" in
        TRAIN_MAP) TRAIN_MAP="$value" ;;
        DATASET_TIME) DATASET_TIME="$value" ;;
        EVAL_TIME) EVAL_TIME="$value" ;;
        SCENARIO) TEST_SCENARIOS+=("$value") ;;
    esac
done

slugify() {
    printf '%s' "$1" \
        | tr '[:upper:]' '[:lower:]' \
        | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//; s/__+/_/g'
}

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

# Sprawdzamy komplet artefaktów treningu dla wszystkich aktywnych torów.
mapfile -t TRAIN_EXPECTED_FILES < <(
python3 - "$CONFIG_PATH" <<'PY'
import sys
import yaml

cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

experiment = cfg.get("experiment", {}) or {}
tracks = cfg.get("tracks", {}) or {}
robak = cfg.get("robak", {}) or {}
rywak = cfg.get("rywak", {}) or {}

mode = str(experiment.get("mode", "ai")).strip().lower()
if mode == "ai":
    print("model.pt")
    print("train_history.json")

if bool(tracks.get("tor5_robak", False)):
    print(str(robak.get("model_name", "model_robak.pt")))
    print(str(robak.get("history_name", "train_history_robak.json")))

if bool(tracks.get("tor6_rywak", False)):
    print(str(rywak.get("model_name", "model_rywak.pt")))
    print(str(rywak.get("history_name", "train_history_rywak.json")))
PY
)

TRAIN_DIR_PRIMARY="out/$EXP_ID"
TRAIN_DIR_FALLBACK="ai_slam_ws/out/$EXP_ID"
TRAIN_OUTPUT_DIR=""

if [ -d "$TRAIN_DIR_PRIMARY" ]; then
    TRAIN_OUTPUT_DIR="$TRAIN_DIR_PRIMARY"
elif [ -d "$TRAIN_DIR_FALLBACK" ]; then
    TRAIN_OUTPUT_DIR="$TRAIN_DIR_FALLBACK"
fi

if [ -z "$TRAIN_OUTPUT_DIR" ]; then
    echo "BŁĄD: Nie znaleziono katalogu wynikowego treningu: $TRAIN_DIR_PRIMARY"
    "$CLEANUP_SCRIPT" || true
    exit 1
fi

MISSING_TRAIN_ARTIFACTS=()
for expected_name in "${TRAIN_EXPECTED_FILES[@]}"; do
    [ -n "$expected_name" ] || continue
    expected_path="$TRAIN_OUTPUT_DIR/$expected_name"
    if [ ! -s "$expected_path" ]; then
        MISSING_TRAIN_ARTIFACTS+=("$expected_name")
    fi
done

if [ "${#MISSING_TRAIN_ARTIFACTS[@]}" -gt 0 ]; then
    echo "BŁĄD: Brakuje artefaktów treningu w $TRAIN_OUTPUT_DIR:"
    for missing_name in "${MISSING_TRAIN_ARTIFACTS[@]}"; do
        echo "  - $missing_name"
    done
    "$CLEANUP_SCRIPT" || true
    exit 1
fi

echo "SUKCES: Artefakty treningu zapisane w $TRAIN_OUTPUT_DIR."
echo "Przechodzę do fazy testów..."

# Czyszczenie między fazami (ważne, żeby zamknąć poprzednie Gazebo)
echo "--- Czyszczenie przed Faza 2 ---"
"$CLEANUP_SCRIPT" || true
sleep 5 # Dłuższa pauza, żeby Gazebo na pewno zniknęło

SCENARIO_COUNT="${#TEST_SCENARIOS[@]}"
if [ "$SCENARIO_COUNT" -le 0 ]; then
    echo "BŁĄD: Brak scenariuszy testowych w konfiguracji."
    "$CLEANUP_SCRIPT" || true
    exit 1
fi

echo "Start FAZY 2..."
echo "=========================================================="
echo "FAZA 2: TEST / EWALUACJA"
echo "ID: $EXP_ID"
echo "Liczba scenariuszy: $SCENARIO_COUNT"
echo "Czas testu na scenariusz: $EVAL_TIME s"
echo "=========================================================="

SCENARIO_RESULT_PATHS=()
for idx in "${!TEST_SCENARIOS[@]}"; do
    scenario_json="${TEST_SCENARIOS[$idx]}"
    IFS=$'\t' read -r TEST_LABEL TEST_MAP TEST_REF_MAP < <(
    python3 - "$scenario_json" <<'PY'
import json
import sys

scenario = json.loads(sys.argv[1])
print(f"{scenario.get('label', 'test')}\t{scenario.get('world', '')}\t{scenario.get('reference_map_yaml', '')}")
PY
    )

    if [ -z "$TEST_MAP" ]; then
        echo "BŁĄD: Pusty world w scenariuszu testowym #$((idx + 1))."
        "$CLEANUP_SCRIPT" || true
        exit 1
    fi

    SCENARIO_SLUG="$(slugify "$TEST_LABEL")"
    if [ -z "$SCENARIO_SLUG" ]; then
        SCENARIO_SLUG="scenario_$((idx + 1))"
    fi

    OUTPUT_SUBDIR=""
    FINALIZE_EXPERIMENT="true"
    WRITE_EVAL_METADATA="true"
    if [ "$SCENARIO_COUNT" -gt 1 ]; then
        OUTPUT_SUBDIR="evaluations/${SCENARIO_SLUG}"
        FINALIZE_EXPERIMENT="false"
        WRITE_EVAL_METADATA="false"
        if [ "$idx" -eq $((SCENARIO_COUNT - 1)) ]; then
            FINALIZE_EXPERIMENT="true"
            WRITE_EVAL_METADATA="true"
        fi
    fi

    echo ""
    echo "----------------------------------------------------------"
    echo "SCENARIUSZ $((idx + 1))/$SCENARIO_COUNT"
    echo "Etykieta: $TEST_LABEL"
    echo "Świat: $TEST_MAP"
    echo "Mapa referencyjna: $TEST_REF_MAP"
    if [ -n "$OUTPUT_SUBDIR" ]; then
        echo "Artefakty ewaluacji: out/$EXP_ID/$OUTPUT_SUBDIR"
    fi
    echo "----------------------------------------------------------"

    TEST_CMD=(
        ros2 launch ai_slam_bringup demo.launch.py
        "config:=$CONFIG_FILE"
        "phase:=test"
        "world_sdf:=$TEST_MAP"
        "reference_map_yaml:=$TEST_REF_MAP"
        "evaluation_label:=$TEST_LABEL"
        "finalize_experiment:=$FINALIZE_EXPERIMENT"
        "write_evaluation_metadata:=$WRITE_EVAL_METADATA"
        "experiment_id:=$EXP_ID"
        "eval_duration_sec:=$EVAL_TIME"
    )
    if [ -n "$OUTPUT_SUBDIR" ]; then
        TEST_CMD+=("evaluation_output_subdir:=$OUTPUT_SUBDIR")
    fi
    TEST_CMD+=("$@")

    TEST_LAUNCH_RC=0
    set +e
    "${TEST_CMD[@]}"
    TEST_LAUNCH_RC=$?
    set -e

    if [ "$TEST_LAUNCH_RC" -ne 0 ]; then
        echo ""
        echo "OSTRZEŻENIE: Scenariusz '$TEST_LABEL' zakończony kodem $TEST_LAUNCH_RC."
        echo "Sprawdzam, czy results.json został mimo to zapisany..."
    fi

    if [ -n "$OUTPUT_SUBDIR" ]; then
        SCENARIO_RESULTS_PATH="out/$EXP_ID/$OUTPUT_SUBDIR/results.json"
        SCENARIO_RESULTS_FALLBACK="ai_slam_ws/out/$EXP_ID/$OUTPUT_SUBDIR/results.json"
    else
        SCENARIO_RESULTS_PATH="out/$EXP_ID/results.json"
        SCENARIO_RESULTS_FALLBACK="ai_slam_ws/out/$EXP_ID/results.json"
    fi

    if [ -f "$SCENARIO_RESULTS_PATH" ]; then
        :
    elif [ -f "$SCENARIO_RESULTS_FALLBACK" ]; then
        SCENARIO_RESULTS_PATH="$SCENARIO_RESULTS_FALLBACK"
    else
        echo "BŁĄD: Plik wyników nie istnieje: $SCENARIO_RESULTS_PATH"
        "$CLEANUP_SCRIPT" || true
        exit 1
    fi

    SCENARIO_RESULT_PATHS+=("$SCENARIO_RESULTS_PATH")

    if [ "$idx" -lt $((SCENARIO_COUNT - 1)) ]; then
        echo "--- Czyszczenie przed kolejnym scenariuszem testowym ---"
        "$CLEANUP_SCRIPT" || true
        sleep 5
    fi
done

if [ "$SCENARIO_COUNT" -gt 1 ]; then
    echo "--- Agregowanie wyników wielu scenariuszy testowych ---"
    python3 scripts/aggregate_multi_test_results.py "out/$EXP_ID" "${SCENARIO_RESULT_PATHS[@]}"
fi

RESULTS_PATH="out/$EXP_ID/results.json"
if [ ! -f "$RESULTS_PATH" ]; then
    if [ -f "ai_slam_ws/out/$EXP_ID/results.json" ]; then
        RESULTS_PATH="ai_slam_ws/out/$EXP_ID/results.json"
    else
        echo "BŁĄD: Plik wyników nie istnieje: $RESULTS_PATH"
        "$CLEANUP_SCRIPT" || true
        exit 1
    fi
fi

echo ""
echo "PEŁNY CYKL ZAKOŃCZONY"
echo "Wyniki: out/$EXP_ID"

echo "--- Generowanie raportów datasetu i treningu ---"
python3 scripts/inspect_dataset.py "out/$EXP_ID" || true

# === FINALNE CZYSZCZENIE ===
echo "--- Zamykanie wszystkich procesów (cleanup) ---"
"$CLEANUP_SCRIPT" || true
echo "--- Gotowe ---"
