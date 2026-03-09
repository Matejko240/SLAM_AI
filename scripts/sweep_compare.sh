#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_CONFIG="$ROOT_DIR/ai_slam_ws/src/ai_slam_bringup/config/experiment_config.yaml"
RUN_FULL_SCRIPT="$ROOT_DIR/scripts/run_full_cycle.sh"
OUT_DIR="$ROOT_DIR/out"
TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

safe_source() {
  set +u
  # shellcheck disable=SC1090
  source "$1"
  set -u
}

print_usage() {
  cat <<'USAGE'
Usage:
  ./scripts/sweep_compare.sh
  ./scripts/sweep_compare.sh variant1 variant2 ...
  ./scripts/sweep_compare.sh --list

Default objective:
  min score = rmse_xy_robak + rmse_xy_rywak
USAGE
}

list_variants() {
  cat <<'EOF'
stable_lock
rywak_adapt_mild
rywak_adapt_strong
robak_strong
balanced
EOF
}

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "ERROR: Missing base config: $BASE_CONFIG"
  exit 1
fi

if [[ ! -x "$RUN_FULL_SCRIPT" ]]; then
  echo "ERROR: Missing executable: $RUN_FULL_SCRIPT"
  exit 1
fi

has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

variant_exists() {
  local needle="$1"
  local item=""
  for item in "${ALL_VARIANTS[@]}"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  print_usage
  exit 0
fi

if [[ "${1:-}" == "--list" ]]; then
  list_variants
  exit 0
fi

mapfile -t ALL_VARIANTS < <(list_variants)
if [[ $# -eq 0 ]]; then
  VARIANTS=("${ALL_VARIANTS[@]}")
else
  VARIANTS=("$@")
fi

for v in "${VARIANTS[@]}"; do
  if ! variant_exists "$v"; then
    echo "ERROR: Unknown variant '$v'. Use --list."
    exit 1
  fi
done

safe_source /opt/ros/jazzy/setup.bash
safe_source "$ROOT_DIR/ai_slam_ws/install/setup.bash"

mkdir -p "$OUT_DIR"
SWEEP_ID="sweep_$(date +%Y%m%d_%H%M%S)"
SUMMARY_CSV="$OUT_DIR/${SWEEP_ID}.csv"
LOG_DIR="$OUT_DIR/${SWEEP_ID}_logs"
mkdir -p "$LOG_DIR"

echo "variant,exp_id,status,elapsed_sec,score,rmse_xy_robak,rmse_xy_rywak,rmse_theta_robak,rmse_theta_rywak,rmse_xy_ai,iou_map_baseline,iou_map_ai,config_path,log_path" > "$SUMMARY_CSV"

make_variant_config() {
  local variant="$1"
  local out_cfg="$2"
  python3 - "$BASE_CONFIG" "$out_cfg" "$variant" <<'PY'
import sys
import yaml

base_path, out_path, variant = sys.argv[1:]
with open(base_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

profiles = {
    "stable_lock": {
        "rywak": {
            "interpolate_odom": False,
            "fuse_odom_v_weight": 0.45,
            "fuse_odom_w_weight": 0.70,
            "fuse_odom_v_gain": 0.00,
            "fuse_odom_w_gain": 0.00,
            "vel_ema_alpha": 0.55,
            "anchor_yaw_to_odom": 0.45,
            "heading_for_xy_odom_weight": 0.75,
            "xy_step_odom_weight": 0.35,
            "xy_step_odom_gain": 0.00,
            "max_integration_dt": 0.20,
        },
        "robak": {
            "infer_odom_delta_xy_alpha": 0.35,
            "infer_odom_delta_yaw_alpha": 0.45,
        },
    },
    "rywak_adapt_mild": {
        "rywak": {
            "interpolate_odom": False,
            "fuse_odom_v_weight": 0.30,
            "fuse_odom_w_weight": 0.60,
            "fuse_odom_v_gain": 0.25,
            "fuse_odom_w_gain": 0.20,
            "vel_ema_alpha": 0.55,
            "anchor_yaw_to_odom": 0.40,
            "heading_for_xy_odom_weight": 0.70,
            "xy_step_odom_weight": 0.20,
            "xy_step_odom_gain": 0.25,
            "max_integration_dt": 0.20,
        },
        "robak": {
            "infer_odom_delta_xy_alpha": 0.35,
            "infer_odom_delta_yaw_alpha": 0.45,
        },
    },
    "rywak_adapt_strong": {
        "rywak": {
            "interpolate_odom": False,
            "fuse_odom_v_weight": 0.25,
            "fuse_odom_w_weight": 0.55,
            "fuse_odom_v_gain": 0.45,
            "fuse_odom_w_gain": 0.35,
            "vel_ema_alpha": 0.55,
            "anchor_yaw_to_odom": 0.40,
            "heading_for_xy_odom_weight": 0.70,
            "xy_step_odom_weight": 0.15,
            "xy_step_odom_gain": 0.45,
            "max_integration_dt": 0.20,
        },
        "robak": {
            "infer_odom_delta_xy_alpha": 0.35,
            "infer_odom_delta_yaw_alpha": 0.45,
        },
    },
    "robak_strong": {
        "rywak": {
            "interpolate_odom": False,
            "fuse_odom_v_weight": 0.45,
            "fuse_odom_w_weight": 0.70,
            "fuse_odom_v_gain": 0.00,
            "fuse_odom_w_gain": 0.00,
            "vel_ema_alpha": 0.55,
            "anchor_yaw_to_odom": 0.45,
            "heading_for_xy_odom_weight": 0.75,
            "xy_step_odom_weight": 0.35,
            "xy_step_odom_gain": 0.00,
            "max_integration_dt": 0.20,
        },
        "robak": {
            "infer_odom_delta_xy_alpha": 0.45,
            "infer_odom_delta_yaw_alpha": 0.55,
            "infer_odom_heading_alpha": 0.25,
        },
    },
    "balanced": {
        "rywak": {
            "interpolate_odom": False,
            "fuse_odom_v_weight": 0.35,
            "fuse_odom_w_weight": 0.65,
            "fuse_odom_v_gain": 0.10,
            "fuse_odom_w_gain": 0.10,
            "vel_ema_alpha": 0.55,
            "anchor_yaw_to_odom": 0.42,
            "heading_for_xy_odom_weight": 0.72,
            "xy_step_odom_weight": 0.25,
            "xy_step_odom_gain": 0.10,
            "max_integration_dt": 0.20,
        },
        "robak": {
            "infer_odom_delta_xy_alpha": 0.30,
            "infer_odom_delta_yaw_alpha": 0.40,
            "infer_odom_heading_alpha": 0.18,
        },
    },
}

if variant not in profiles:
    raise SystemExit(f"Unknown variant: {variant}")

def deep_update(dst, src):
    for k, v in src.items():
        if isinstance(v, dict):
            if k not in dst or not isinstance(dst[k], dict):
                dst[k] = {}
            deep_update(dst[k], v)
        else:
            dst[k] = v

deep_update(cfg, profiles[variant])

with open(out_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

parse_results_row() {
  local results_path="$1"
  python3 - "$results_path" <<'PY'
import json
import math
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    m = json.load(f)["metrics"]

def fv(name):
    x = m.get(name)
    if x is None:
        return float("nan")
    return float(x)

robak_xy = fv("rmse_xy_robak")
rywak_xy = fv("rmse_xy_rywak")
score = robak_xy + rywak_xy

vals = [
    score,
    robak_xy,
    rywak_xy,
    fv("rmse_theta_robak"),
    fv("rmse_theta_rywak"),
    fv("rmse_xy_ai"),
    fv("iou_map_baseline"),
    fv("iou_map_ai"),
]

def fmt(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return ""
    return f"{v:.6f}"

print(",".join(fmt(v) for v in vals))
PY
}

total="${#VARIANTS[@]}"
idx=0

echo "========================================"
echo "Sweep ID: $SWEEP_ID"
echo "Variants: ${VARIANTS[*]}"
echo "Summary:  $SUMMARY_CSV"
echo "Logs:     $LOG_DIR"
echo "========================================"

for variant in "${VARIANTS[@]}"; do
  idx=$((idx + 1))
  cfg_path="$TMP_DIR/${SWEEP_ID}_${variant}.yaml"
  log_path="$LOG_DIR/${variant}.log"

  make_variant_config "$variant" "$cfg_path"

  echo ""
  echo "[$idx/$total] Variant: $variant"
  echo "Config: $cfg_path"

  t0="$(date +%s)"
  set +e
  "$RUN_FULL_SCRIPT" "$cfg_path" > >(tee "$log_path") 2>&1
  rc=$?
  set -e
  t1="$(date +%s)"
  elapsed="$((t1 - t0))"

  if has_cmd rg; then
    exp_id="$(rg -o 'out/exp_[0-9_]+' "$log_path" | tail -1 | sed 's|out/||' || true)"
  else
    exp_id="$(grep -Eo 'out/exp_[0-9_]+' "$log_path" | tail -1 | sed 's|out/||' || true)"
  fi
  if [[ -z "$exp_id" ]]; then
    latest="$(ls -1dt "$OUT_DIR"/exp_* 2>/dev/null | head -1 || true)"
    exp_id="$(basename "$latest" || true)"
  fi

  if [[ "$rc" -ne 0 ]]; then
    echo "WARN: run failed for $variant (rc=$rc)"
    echo "$variant,$exp_id,FAIL,$elapsed,,,,,,,,$cfg_path,$log_path" >> "$SUMMARY_CSV"
    continue
  fi

  results_path="$OUT_DIR/$exp_id/results.json"
  if [[ ! -f "$results_path" ]]; then
    echo "WARN: missing results.json for $variant (exp=$exp_id)"
    echo "$variant,$exp_id,NO_RESULTS,$elapsed,,,,,,,,$cfg_path,$log_path" >> "$SUMMARY_CSV"
    continue
  fi

  metric_row="$(parse_results_row "$results_path")"
  echo "$variant,$exp_id,OK,$elapsed,$metric_row,$cfg_path,$log_path" >> "$SUMMARY_CSV"
done

echo ""
echo "========================================"
echo "Ranking (best score first)"
python3 - "$SUMMARY_CSV" <<'PY'
import csv
import math
import sys

path = sys.argv[1]
rows = []
with open(path, "r", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        if row["status"] != "OK":
            continue
        try:
            score = float(row["score"])
        except Exception:
            score = math.inf
        rows.append((score, row))

rows.sort(key=lambda x: x[0])
if not rows:
    print("No successful runs.")
else:
    for i, (_, row) in enumerate(rows, start=1):
        print(
            f"{i}. {row['variant']} | exp={row['exp_id']} | score={row['score']} | "
            f"robak_xy={row['rmse_xy_robak']} | rywak_xy={row['rmse_xy_rywak']}"
        )
    best = rows[0][1]
    print("")
    print(f"BEST_VARIANT={best['variant']}")
    print(f"BEST_EXPERIMENT={best['exp_id']}")
    print(f"BEST_SCORE={best['score']}")
PY
echo "Summary CSV: $SUMMARY_CSV"
echo "========================================"
