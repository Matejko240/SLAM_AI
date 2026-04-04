#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS_DIR="${ROOT_DIR}/ai_slam_ws"
OUT_DIR="${ROOT_DIR}/out"
DEFAULT_CONFIG="${WS_DIR}/src/ai_slam_bringup/config/experiment_config_office_cyclic_dataset.yaml"

CONFIG_PATH="${DEFAULT_CONFIG}"
CONFIG_SEQUENCE_CSV=""
PATH_SEQUENCE_CSV=""
MAX_DATASET_RUNS=3
TARGET_BINS=24
MIN_TARGET_PER_BIN=1000
MIN_TARGET_PER_BIN_RYWAK=1000
MIN_TARGET_PER_BIN_ROBAK=10000
DATASET_DURATION_CAP_SEC=600
GUI=false
TRAIN_SEEDS_CSV="123,321,777"
SKIP_BUILD=false
ADAPTIVE_CONFIG=true
ADAPTIVE_PATH=true
DEDUP_USE_POSE_KEY=true
WARMUP_BASE_RUNS=2
FOCUS_SEQUENCE_CSV=""
APPEND_FROM_MERGE_EXP_ID=""
RUN_MODE="all" # all | dataset_only | train_only
TRAIN_ONLY_EXP_ID=""
RYWAK_V_MIN=0.0
RYWAK_V_MAX=1.2
RYWAK_W_MIN=0.0
RYWAK_W_MAX=3.0
ROBAK_T_MIN=0.0
ROBAK_T_MAX=1.0
ROBAK_R_MIN=0.0
ROBAK_R_MAX=150.0

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [options]

Options:
  --config <path>             Launch config YAML (default: ${DEFAULT_CONFIG})
  --config-sequence <csv>     Lista configów YAML (round-robin), np. "cfg1.yaml,cfg2.yaml"
  --path-sequence <csv>       Lista planned_path spec_yaml (round-robin), np. "planned_paths/office_trajectory_acyclic.yaml,planned_paths/hospital_trajectory_acyclic.yaml"
  --max-dataset-runs <N>      Max liczba rund dataset+merge (default: 3)
  --target-bins <N>           Docelowa liczba niepustych bins na histogram (default: 24)
  --dataset-duration-cap-sec <N>  Twardy limit czasu pojedynczej rundy datasetu (default: 600)
  --min-target-per-bin <N>    Ustawia wspólny próg minimalny dla Rywaka i Robaka (default: ${MIN_TARGET_PER_BIN})
  --min-target-per-bin-rywak <N>  Minimalna liczba próbek/bin dla Rywaka po strict rebalance (default: ${MIN_TARGET_PER_BIN_RYWAK})
  --min-target-per-bin-robak <N>  Minimalna liczba próbek/bin dla Robaka po strict rebalance (default: ${MIN_TARGET_PER_BIN_ROBAK})
  --train-seeds <csv>         Seedy treningu, np. "123,321,777" (default: ${TRAIN_SEEDS_CSV})
  --focus-sequence <csv>      Sekwencja fokusów rund: balanced,rotation,translation,slalom (alias: linear=translation)
  --append-from-merge <exp_id>  Dołącza nowe rundy do istniejącego merge out/<exp_id> (bez zaczynania od zera)
  --adaptive-config <bool>    Adaptacyjne strojenie configu między rundami (default: true)
  --adaptive-path <bool>      Przy adaptive-config=true: czy adaptować geometrię planned_path (default: true)
  --dedup-use-pose-key <bool> Deduplikacja po X+Y+P (gdy P dostępne), zamiast tylko X+Y (default: true)
  --warmup-base-runs <N>      Pierwsze N rund wymusza bazową trasę (safe warmup; adaptive_path=OFF, focus=balanced) (default: ${WARMUP_BASE_RUNS})
  --gui <true|false>          Czy odpalać Gazebo GUI (default: false)
  --dataset-only              Tylko zbieranie datasetów + merge + strict rebalance (bez treningu).
  --train-only <exp_id>       Tylko trening na istniejącym out/<exp_id> (musi mieć dataset_robak.npz i dataset_rywak.npz).
  --skip-build                Pomiń colcon build
  -h, --help                  Pomoc

Opis:
  tryb all (domyślny):
  1) Odpala kolejne eksperymenty datasetowe (phase:=dataset).
  2) Po każdej rundzie merguje komponentowe datasety (Rywak liniowy+kątowy, Robak translacja+rotacja).
  3) Wykonuje strict rebalance: dokładnie równe histogramy + unikalne próbki (bez replacement).
  4) Kończy rundy dopiero gdy strict target zostanie osiągnięty; inaczej kończy z błędem.
  5) Uruchamia kilka treningów (Robak + Rywak) na strict-merged dataset, po jednym na seed.

  Uwaga:
  - Czas rundy datasetu jest capowany do --dataset-duration-cap-sec (domyślnie 600s).
  - Przy adaptive-config=true zmienia sterowanie oraz geometrię planned_path
    (tymczasowy spec z dodatkowymi kotwicami slalomu dla rund rotation/slalom/balanced).
  - Przy adaptive-config=true i adaptive-path=false: focus/parametry są adaptowane, ale geometria planned_path pozostaje bez zmian.

  tryb dataset-only:
  - wykonuje kroki 1-4 i kończy bez treningu.
  - z --append-from-merge <exp_id>: startuje od istniejącego merge i dokleja nowe rundy.

  tryb train-only:
  - pomija zbieranie i merge; trenuje tylko na podanym out/<exp_id>.
EOF
}

log() {
  printf '[multi-merge-train] %s\n' "$*"
}

calc_elapsed_sec() {
  local start_ts="$1"
  local end_ts="$2"
  python3 - "${start_ts}" "${end_ts}" <<'PY'
import sys
s = float(sys.argv[1])
e = float(sys.argv[2])
print(f"{max(0.0, e - s):.3f}")
PY
}

append_dataset_timing_jsonl() {
  local jsonl_path="$1"
  local run_idx="$2"
  local experiment_id="$3"
  local wall_sec="$4"
  local status="$5"
  local launch_rc="$6"
  local run_cfg="$7"
  python3 - "${jsonl_path}" "${run_idx}" "${experiment_id}" "${wall_sec}" "${status}" "${launch_rc}" "${run_cfg}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, run_idx, exp_id, wall_sec, status, launch_rc, run_cfg = sys.argv[1:]
obj = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "run_idx": int(run_idx),
    "experiment_id": exp_id,
    "wall_sec": float(wall_sec),
    "status": status,
    "launch_rc": int(launch_rc),
    "run_config": run_cfg,
}
p = Path(path)
p.parent.mkdir(parents=True, exist_ok=True)
with p.open("a", encoding="utf-8") as f:
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
PY
}

summarize_dataset_timing_jsonl() {
  local jsonl_path="$1"
  local summary_path="$2"
  python3 - "${jsonl_path}" "${summary_path}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

jsonl_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])

rows = []
if jsonl_path.is_file():
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue

def agg(items):
    vals = []
    for i in items:
        try:
            vals.append(float(i.get("wall_sec")))
        except Exception:
            pass
    if not vals:
        return {"n": 0, "avg_sec": None, "min_sec": None, "max_sec": None}
    return {
        "n": len(vals),
        "avg_sec": float(sum(vals) / len(vals)),
        "min_sec": float(min(vals)),
        "max_sec": float(max(vals)),
    }

accepted = [r for r in rows if str(r.get("status", "")).strip().lower() == "accepted"]
status_counts = {}
for r in rows:
    k = str(r.get("status", "unknown"))
    status_counts[k] = int(status_counts.get(k, 0)) + 1

summary = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_jsonl": str(jsonl_path),
    "all_runs": agg(rows),
    "accepted_runs": agg(accepted),
    "status_counts": status_counts,
    "runs": rows,
}

summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

def fmt(v):
    return "n/a" if v is None else f"{float(v):.1f}"

out = [
    str(summary_path),
    str(summary["accepted_runs"]["n"]),
    fmt(summary["accepted_runs"]["avg_sec"]),
    fmt(summary["accepted_runs"]["min_sec"]),
    fmt(summary["accepted_runs"]["max_sec"]),
    str(summary["all_runs"]["n"]),
    fmt(summary["all_runs"]["avg_sec"]),
    fmt(summary["all_runs"]["min_sec"]),
    fmt(summary["all_runs"]["max_sec"]),
]
print("\t".join(out))
PY
}

source_ros_setup() {
  local setup_file="$1"
  # ROS setup scripts are not fully nounset-safe, so source them with set +u.
  set +u
  # shellcheck disable=SC1090
  source "${setup_file}"
  set -u
}

cleanup_ros() {
  if [[ -x "${ROOT_DIR}/scripts/cleanup.sh" ]]; then
    bash "${ROOT_DIR}/scripts/cleanup.sh" >/dev/null 2>&1 || true
  fi
}

require_file() {
  local p="$1"
  if [[ ! -f "$p" ]]; then
    echo "Brak pliku: $p" >&2
    exit 1
  fi
}

split_csv() {
  local csv="$1"
  local -n out_arr_ref="$2"
  out_arr_ref=()
  IFS=',' read -r -a _tmp_arr <<< "${csv}"
  for _item in "${_tmp_arr[@]}"; do
    _trimmed="$(echo "${_item}" | xargs)"
    if [[ -n "${_trimmed}" ]]; then
      out_arr_ref+=("${_trimmed}")
    fi
  done
}

choose_round_robin() {
  local idx="$1"
  shift
  local -a arr=("$@")
  local n="${#arr[@]}"
  if [[ "${n}" -eq 0 ]]; then
    echo ""
    return
  fi
  local k=$(( (idx - 1) % n ))
  echo "${arr[$k]}"
}

analyze_run_trajectory_health() {
  local summary_json="$1"
  local experiment_id="$2"
  python3 - "${summary_json}" "${experiment_id}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1]).expanduser().resolve()
exp_id = str(sys.argv[2]).strip()

if not summary_path.is_file() or not exp_id:
    print("0\tmissing_summary\t-1\t-1\t-1")
    raise SystemExit(0)

try:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
except Exception:
    print("0\tinvalid_summary\t-1\t-1\t-1")
    raise SystemExit(0)

run_obj = None
for group in data.get("map_groups", []):
    if not isinstance(group, dict):
        continue
    for run in group.get("runs", []):
        if not isinstance(run, dict):
            continue
        if str(run.get("experiment_id", "")).strip() == exp_id:
            run_obj = run
            break
    if run_obj is not None:
        break

if run_obj is None:
    print("0\tmissing_run\t-1\t-1\t-1")
    raise SystemExit(0)

status = str(run_obj.get("status", "unknown")).strip().lower() or "unknown"
stuck_segments = int(run_obj.get("stuck_segments", 0) or 0)
stuck_points = int(run_obj.get("stuck_points", 0) or 0)
traj_len = float(run_obj.get("trajectory_length_m", 0.0) or 0.0)

flag = 0
if status != "ok":
    flag = 1
elif stuck_points >= 120:
    flag = 1
elif stuck_segments >= 3:
    flag = 1
elif stuck_segments >= 2 and stuck_points >= 60:
    flag = 1
elif traj_len > 0.0 and traj_len < 100.0:
    flag = 1

print(f"{flag}\t{status}\t{stuck_segments}\t{stuck_points}\t{traj_len:.3f}")
PY
}

analyze_single_run_trajectory_health() {
  local out_dir="$1"
  local experiment_id="$2"
  local run_cfg="$3"
  local summary_name="run_traj_health_${experiment_id}.json"
  local summary_path="${out_dir}/${summary_name}"

  if ! python3 "${ROOT_DIR}/scripts/plot_merged_dataset_trajectories.py" \
    --out-dir "${out_dir}" \
    --experiment-ids "${experiment_id}" \
    --run-configs "${run_cfg}" \
    --summary-name "${summary_name}" >/dev/null 2>&1; then
    echo "0\thealth_check_error\t-1\t-1\t-1"
    return 0
  fi
  analyze_run_trajectory_health "${summary_path}" "${experiment_id}"
}

load_append_component_paths() {
  local merge_dir="$1"
  local component="$2"
  python3 - "${merge_dir}" "${component}" <<'PY'
import os
import sys
from pathlib import Path

import numpy as np

merge_dir = Path(sys.argv[1]).expanduser().resolve()
component = str(sys.argv[2]).strip().lower()

if component not in {"rywak_linear", "rywak_angular", "robak_translation", "robak_rotation"}:
    raise SystemExit(f"unsupported component: {component}")

def classify(path_str: str) -> str | None:
    name = os.path.basename(path_str).strip().lower()
    if not name.endswith(".npz"):
        return None
    if "rywak" in name:
        if "linear" in name:
            return "rywak_linear"
        if "angular" in name:
            return "rywak_angular"
        return None
    if "robak" in name:
        if "translation" in name:
            return "robak_translation"
        if "rotation" in name:
            return "robak_rotation"
        return None
    return None

def collect(npz_name: str) -> list[str]:
    p = merge_dir / npz_name
    if not p.is_file():
        return []
    with np.load(p, allow_pickle=True) as data:
        if "meta" not in data:
            return []
        meta = data["meta"].item()
    src = meta.get("source_paths", [])
    out: list[str] = []
    for raw in np.asarray(src, dtype=object).tolist():
        s = str(raw).strip()
        if not s:
            continue
        if classify(s) != component:
            continue
        if not Path(s).is_file():
            print(f"[WARN] append source path missing, skipped: {s}", file=sys.stderr)
            continue
        out.append(str(Path(s).resolve()))
    return out

seen = set()
for npz_name in ("dataset_rywak_merged.npz", "dataset_robak_merged.npz"):
    for s in collect(npz_name):
        if s in seen:
            continue
        seen.add(s)
        print(s)
PY
}

load_append_trajectory_runs() {
  local merge_dir="$1"
  local tmp_cfg_dir="$2"
  python3 - "${merge_dir}" "${tmp_cfg_dir}" <<'PY'
import json
import sys
from pathlib import Path

import yaml

merge_dir = Path(sys.argv[1]).expanduser().resolve()
tmp_cfg_dir = Path(sys.argv[2]).expanduser().resolve()
summary_path = merge_dir / "merged_trajectory_overview_summary.json"
if not summary_path.is_file():
    raise SystemExit(0)

try:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(0)

seen: set[str] = set()
run_idx = 0
for group in data.get("map_groups", []):
    if not isinstance(group, dict):
        continue
    ref_map = str(group.get("reference_map_yaml", "")).strip()
    if not ref_map:
        continue
    world_names = group.get("world_names", [])
    default_world = str(world_names[0]).strip() if world_names else "unknown_world"
    for run in group.get("runs", []):
        if not isinstance(run, dict):
            continue
        exp_id = str(run.get("experiment_id", "")).strip()
        if not exp_id or exp_id in seen:
            continue
        seen.add(exp_id)
        world_name = str(run.get("world_name", default_world)).strip() or default_world
        run_idx += 1
        cfg_path = tmp_cfg_dir / f"append_prev_run_{run_idx:04d}.yaml"
        cfg = {
            "simulation": {"train_world": world_name},
            "driver": {"planned_path": {"reference_map_yaml": ref_map}},
        }
        cfg_path.write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
        print(f"{exp_id}\t{cfg_path}")
PY
}

extract_rywak_weak_bins() {
  local hist_dir="$1"
  python3 - "${hist_dir}" <<'PY'
import csv
import math
import sys
from pathlib import Path

hist_dir = sys.argv[1].strip()

def emit_defaults() -> None:
    print("RYWAK_WEAK_LINEAR_BIN_INDEX=-1")
    print("RYWAK_WEAK_LINEAR_BIN_MIN=nan")
    print("RYWAK_WEAK_LINEAR_BIN_MAX=nan")
    print("RYWAK_WEAK_LINEAR_BIN_CENTER=nan")
    print("RYWAK_WEAK_LINEAR_COUNT=-1")
    print("RYWAK_WEAK_ANGULAR_BIN_INDEX=-1")
    print("RYWAK_WEAK_ANGULAR_BIN_MIN=nan")
    print("RYWAK_WEAK_ANGULAR_BIN_MAX=nan")
    print("RYWAK_WEAK_ANGULAR_BIN_CENTER=nan")
    print("RYWAK_WEAK_ANGULAR_COUNT=-1")

def parse_int(v, default=-1):
    try:
        return int(float(str(v).strip()))
    except Exception:
        return int(default)

def parse_float(v):
    try:
        x = float(str(v).strip())
        return x if math.isfinite(x) else float("nan")
    except Exception:
        return float("nan")

def weak_bin(csv_path: Path):
    if not csv_path.is_file():
        return None
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        cols = list(rd.fieldnames or [])
        if "component_count" in cols:
            count_key = "component_count"
        elif "raw_count" in cols:
            count_key = "raw_count"
        elif "final_count" in cols:
            count_key = "final_count"
        else:
            return None
        for row in rd:
            cnt = parse_int(row.get(count_key, "-1"), default=-1)
            if cnt < 0:
                continue
            idx = parse_int(row.get("bin_index", "-1"), default=-1)
            bmin = parse_float(row.get("bin_min", "nan"))
            bmax = parse_float(row.get("bin_max", "nan"))
            ctr = parse_float(row.get("bin_center", "nan"))
            rows.append((cnt, idx, bmin, bmax, ctr))
    if not rows:
        return None
    rows.sort(key=lambda x: (x[0], x[1]))
    cnt, idx, bmin, bmax, ctr = rows[0]
    return {
        "count": int(cnt),
        "idx": int(idx),
        "min": float(bmin),
        "max": float(bmax),
        "center": float(ctr),
    }

if not hist_dir:
    emit_defaults()
    raise SystemExit(0)

base = Path(hist_dir)
if not base.is_dir():
    emit_defaults()
    raise SystemExit(0)

lin = weak_bin(base / "rywak_linear_bins.csv")
ang = weak_bin(base / "rywak_angular_bins.csv")

if lin is None and ang is None:
    emit_defaults()
    raise SystemExit(0)

if lin is None:
    lin = {"count": -1, "idx": -1, "min": float("nan"), "max": float("nan"), "center": float("nan")}
if ang is None:
    ang = {"count": -1, "idx": -1, "min": float("nan"), "max": float("nan"), "center": float("nan")}

print(f"RYWAK_WEAK_LINEAR_BIN_INDEX={lin['idx']}")
print(f"RYWAK_WEAK_LINEAR_BIN_MIN={lin['min']}")
print(f"RYWAK_WEAK_LINEAR_BIN_MAX={lin['max']}")
print(f"RYWAK_WEAK_LINEAR_BIN_CENTER={lin['center']}")
print(f"RYWAK_WEAK_LINEAR_COUNT={lin['count']}")
print(f"RYWAK_WEAK_ANGULAR_BIN_INDEX={ang['idx']}")
print(f"RYWAK_WEAK_ANGULAR_BIN_MIN={ang['min']}")
print(f"RYWAK_WEAK_ANGULAR_BIN_MAX={ang['max']}")
print(f"RYWAK_WEAK_ANGULAR_BIN_CENTER={ang['center']}")
print(f"RYWAK_WEAK_ANGULAR_COUNT={ang['count']}")
PY
}

select_focus_mode() {
  local run_idx="$1"
  local prev_summary="$2"
  local focus_override="$3"
  local min_target_rywak="$4"
  local min_target_robak="$5"
  local prev_hist_dir="$6"
  python3 - "${run_idx}" "${prev_summary}" "${focus_override}" "${min_target_rywak}" "${min_target_robak}" "${prev_hist_dir}" <<'PY'
import json
import csv
import sys
from pathlib import Path

run_idx = int(sys.argv[1])
prev_summary = sys.argv[2].strip()
focus_override = sys.argv[3].strip().lower()
min_target_rywak = int(sys.argv[4])
min_target_robak = int(sys.argv[5])
prev_hist_dir = sys.argv[6].strip()

if focus_override == "linear":
    focus_override = "translation"

valid = {"balanced", "rotation", "translation", "slalom"}
if focus_override in valid:
    print(focus_override)
    raise SystemExit(0)

cycle = ["translation", "rotation", "slalom", "balanced"]

if not prev_summary:
    print(cycle[(run_idx - 1) % len(cycle)])
    raise SystemExit(0)

p = Path(prev_summary)
if not p.is_file():
    print(cycle[(run_idx - 1) % len(cycle)])
    raise SystemExit(0)

try:
    d = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    print(cycle[(run_idx - 1) % len(cycle)])
    raise SystemExit(0)

ry = d.get("rywak", {}) if isinstance(d.get("rywak", {}), dict) else {}
rb = d.get("robak", {}) if isinstance(d.get("robak", {}), dict) else {}
ry_reason = str(ry.get("reason", "")).strip().lower()
rb_reason = str(rb.get("reason", "")).strip().lower()
ry_tgt = int(ry.get("target_per_bin", 0))
rb_tgt = int(rb.get("target_per_bin", 0))
ry_row_min = int(ry.get("row_count_min_before", 0))
ry_col_min = int(ry.get("col_count_min_before", 0))
rb_row_min = int(rb.get("row_count_min_before", 0))
rb_col_min = int(rb.get("col_count_min_before", 0))

def _weak_component_count(csv_path: Path) -> int | None:
    if not csv_path.is_file():
        return None
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rd = csv.DictReader(f)
            cols = set(rd.fieldnames or [])
            key = "component_count" if "component_count" in cols else ("raw_count" if "raw_count" in cols else "")
            if not key:
                return None
            vals = []
            for row in rd:
                try:
                    vals.append(int(float(str(row.get(key, "")).strip())))
                except Exception:
                    continue
            if not vals:
                return None
            return int(min(vals))
    except Exception:
        return None

weak_lin = None
weak_ang = None
if prev_hist_dir:
    hd = Path(prev_hist_dir)
    weak_lin = _weak_component_count(hd / "rywak_linear_bins.csv")
    weak_ang = _weak_component_count(hd / "rywak_angular_bins.csv")

# Jeśli target/bin nadal za niski, priorytet: Rywak (zgodnie z celem dobijania jego słabych koszyków).
if ry_tgt > 0 and ry_tgt < min_target_rywak:
    if weak_lin is not None and weak_ang is not None:
        if weak_lin <= weak_ang:
            print("translation")
        else:
            print("rotation")
    elif ry_row_min <= ry_col_min:
        print("translation")
    elif ry_col_min < ry_row_min:
        print("rotation")
    else:
        print("slalom")
    raise SystemExit(0)
if rb_tgt > 0 and rb_tgt < min_target_robak:
    if rb_col_min <= rb_row_min:
        print("rotation")
    else:
        print("translation")
    raise SystemExit(0)

if rb_reason == "missing_bins":
    print("rotation")
elif ry_reason == "no_feasible_target":
    print("slalom")
elif rb_reason == "no_feasible_target":
    print("translation")
else:
    print(cycle[(run_idx - 1) % len(cycle)])
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="${2:?missing value for --config}"
      shift 2
      ;;
    --config-sequence)
      CONFIG_SEQUENCE_CSV="${2:?missing value for --config-sequence}"
      shift 2
      ;;
    --path-sequence)
      PATH_SEQUENCE_CSV="${2:?missing value for --path-sequence}"
      shift 2
      ;;
    --max-dataset-runs)
      MAX_DATASET_RUNS="${2:?missing value for --max-dataset-runs}"
      shift 2
      ;;
    --target-bins)
      TARGET_BINS="${2:?missing value for --target-bins}"
      shift 2
      ;;
    --dataset-duration-cap-sec)
      DATASET_DURATION_CAP_SEC="${2:?missing value for --dataset-duration-cap-sec}"
      shift 2
      ;;
    --min-target-per-bin)
      MIN_TARGET_PER_BIN="${2:?missing value for --min-target-per-bin}"
      MIN_TARGET_PER_BIN_RYWAK="${2}"
      MIN_TARGET_PER_BIN_ROBAK="${2}"
      shift 2
      ;;
    --min-target-per-bin-rywak)
      MIN_TARGET_PER_BIN_RYWAK="${2:?missing value for --min-target-per-bin-rywak}"
      shift 2
      ;;
    --min-target-per-bin-robak)
      MIN_TARGET_PER_BIN_ROBAK="${2:?missing value for --min-target-per-bin-robak}"
      shift 2
      ;;
    --train-seeds)
      TRAIN_SEEDS_CSV="${2:?missing value for --train-seeds}"
      shift 2
      ;;
    --focus-sequence)
      FOCUS_SEQUENCE_CSV="${2:?missing value for --focus-sequence}"
      shift 2
      ;;
    --append-from-merge)
      APPEND_FROM_MERGE_EXP_ID="${2:?missing value for --append-from-merge}"
      shift 2
      ;;
    --adaptive-config)
      ADAPTIVE_CONFIG="${2:?missing value for --adaptive-config}"
      shift 2
      ;;
    --adaptive-path)
      ADAPTIVE_PATH="${2:?missing value for --adaptive-path}"
      shift 2
      ;;
    --dedup-use-pose-key)
      DEDUP_USE_POSE_KEY="${2:?missing value for --dedup-use-pose-key}"
      shift 2
      ;;
    --warmup-base-runs)
      WARMUP_BASE_RUNS="${2:?missing value for --warmup-base-runs}"
      shift 2
      ;;
    --gui)
      GUI="${2:?missing value for --gui}"
      shift 2
      ;;
    --dataset-only)
      RUN_MODE="dataset_only"
      shift
      ;;
    --train-only)
      RUN_MODE="train_only"
      TRAIN_ONLY_EXP_ID="${2:?missing value for --train-only}"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Nieznana opcja: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! "${MAX_DATASET_RUNS}" =~ ^[0-9]+$ ]] || [[ "${MAX_DATASET_RUNS}" -lt 1 ]]; then
  echo "--max-dataset-runs musi być >= 1" >&2
  exit 1
fi
if [[ ! "${TARGET_BINS}" =~ ^[0-9]+$ ]] || [[ "${TARGET_BINS}" -lt 1 ]]; then
  echo "--target-bins musi być >= 1" >&2
  exit 1
fi
if [[ ! "${DATASET_DURATION_CAP_SEC}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "--dataset-duration-cap-sec musi być dodatnią liczbą" >&2
  exit 1
fi
if python3 - "${DATASET_DURATION_CAP_SEC}" <<'PY'
import sys
v = float(sys.argv[1])
raise SystemExit(0 if v > 0.0 else 1)
PY
then
  :
else
  echo "--dataset-duration-cap-sec musi być > 0" >&2
  exit 1
fi
if [[ ! "${MIN_TARGET_PER_BIN}" =~ ^[0-9]+$ ]] || [[ "${MIN_TARGET_PER_BIN}" -lt 1 ]]; then
  echo "--min-target-per-bin musi być >= 1" >&2
  exit 1
fi
if [[ ! "${MIN_TARGET_PER_BIN_RYWAK}" =~ ^[0-9]+$ ]] || [[ "${MIN_TARGET_PER_BIN_RYWAK}" -lt 1 ]]; then
  echo "--min-target-per-bin-rywak musi być >= 1" >&2
  exit 1
fi
if [[ ! "${MIN_TARGET_PER_BIN_ROBAK}" =~ ^[0-9]+$ ]] || [[ "${MIN_TARGET_PER_BIN_ROBAK}" -lt 1 ]]; then
  echo "--min-target-per-bin-robak musi być >= 1" >&2
  exit 1
fi
if [[ ! "${WARMUP_BASE_RUNS}" =~ ^[0-9]+$ ]] || [[ "${WARMUP_BASE_RUNS}" -lt 0 ]]; then
  echo "--warmup-base-runs musi być >= 0" >&2
  exit 1
fi

declare -a CONFIG_SEQUENCE=()
declare -a PATH_SEQUENCE=()
declare -a FOCUS_SEQUENCE=()

if [[ -n "${CONFIG_SEQUENCE_CSV}" ]]; then
  split_csv "${CONFIG_SEQUENCE_CSV}" CONFIG_SEQUENCE
else
  CONFIG_SEQUENCE=("${CONFIG_PATH}")
fi
if [[ "${#CONFIG_SEQUENCE[@]}" -eq 0 ]]; then
  echo "Pusta sekwencja configów." >&2
  exit 1
fi
for cfg_item in "${CONFIG_SEQUENCE[@]}"; do
  require_file "${cfg_item}"
done

if [[ -n "${PATH_SEQUENCE_CSV}" ]]; then
  split_csv "${PATH_SEQUENCE_CSV}" PATH_SEQUENCE
fi
if [[ -n "${FOCUS_SEQUENCE_CSV}" ]]; then
  split_csv "${FOCUS_SEQUENCE_CSV}" FOCUS_SEQUENCE
fi

if [[ "${ADAPTIVE_CONFIG}" != "true" && "${ADAPTIVE_CONFIG}" != "false" ]]; then
  echo "--adaptive-config musi mieć wartość true/false" >&2
  exit 1
fi
if [[ "${ADAPTIVE_PATH}" != "true" && "${ADAPTIVE_PATH}" != "false" ]]; then
  echo "--adaptive-path musi mieć wartość true/false" >&2
  exit 1
fi
if [[ "${DEDUP_USE_POSE_KEY}" != "true" && "${DEDUP_USE_POSE_KEY}" != "false" ]]; then
  echo "--dedup-use-pose-key musi mieć wartość true/false" >&2
  exit 1
fi
if [[ "${RUN_MODE}" == "train_only" && -z "${TRAIN_ONLY_EXP_ID}" ]]; then
  echo "--train-only wymaga podania experiment_id" >&2
  exit 1
fi
if [[ "${RUN_MODE}" == "train_only" && -n "${CONFIG_SEQUENCE_CSV}" ]]; then
  echo "--train-only nie używa --config-sequence" >&2
fi
if [[ "${RUN_MODE}" == "train_only" && -n "${PATH_SEQUENCE_CSV}" ]]; then
  echo "--train-only nie używa --path-sequence" >&2
fi
if [[ "${RUN_MODE}" == "train_only" && -n "${APPEND_FROM_MERGE_EXP_ID}" ]]; then
  echo "--train-only nie używa --append-from-merge" >&2
  exit 1
fi

require_file "${ROOT_DIR}/scripts/merge_component_datasets.py"
require_file "${ROOT_DIR}/scripts/report_dataset_histogram_balance.py"
require_file "${ROOT_DIR}/scripts/rebalance_unique_histograms.py"
require_file "${ROOT_DIR}/scripts/adapt_planned_path_for_focus.py"
require_file "${ROOT_DIR}/scripts/plot_merged_dataset_trajectories.py"

eval "$(
  python3 - "${CONFIG_SEQUENCE[0]}" <<'PY'
import os
import sys
from pathlib import Path
import yaml

p = sys.argv[1]
with open(p, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

ry = cfg.get("rywak", {}) if isinstance(cfg.get("rywak", {}), dict) else {}
rb = cfg.get("robak", {}) if isinstance(cfg.get("robak", {}), dict) else {}

def fv(dct, key, default):
    try:
        return float(dct.get(key, default))
    except Exception:
        return float(default)

print(f"RYWAK_V_MIN={fv(ry, 'balance_linear_hist_min_mps', 0.0):.12g}")
print(f"RYWAK_V_MAX={fv(ry, 'balance_linear_hist_max_mps', 1.2):.12g}")
print(f"RYWAK_W_MIN={fv(ry, 'balance_angular_hist_min_radps', 0.0):.12g}")
print(f"RYWAK_W_MAX={fv(ry, 'balance_angular_hist_max_radps', 3.0):.12g}")
print(f"ROBAK_T_MIN={fv(rb, 'balance_translation_hist_min_m', 0.0):.12g}")
print(f"ROBAK_T_MAX={fv(rb, 'balance_translation_hist_max_m', 1.0):.12g}")
print(f"ROBAK_R_MIN={fv(rb, 'balance_rotation_hist_min_deg', 0.0):.12g}")
print(f"ROBAK_R_MAX={fv(rb, 'balance_rotation_hist_max_deg', 150.0):.12g}")
PY
)"

cd "${ROOT_DIR}"
source_ros_setup /opt/ros/jazzy/setup.bash

if [[ "${SKIP_BUILD}" != "true" ]]; then
  log "colcon build --symlink-install"
  (
    cd "${WS_DIR}"
    colcon build --symlink-install
  )
fi

source_ros_setup "${WS_DIR}/install/setup.bash"
trap cleanup_ros EXIT

rywak_linear_paths=()
rywak_angular_paths=()
robak_translation_paths=()
robak_rotation_paths=()
dataset_experiment_ids=()
dataset_run_cfg_paths=()

MERGE_EXP_ID=""
MERGE_DIR=""
BINS_OK=false
LAST_RY_TGT=0
LAST_RB_TGT=0
PREV_REBALANCE_SUMMARY=""
PREV_HIST_REPORT_DIR=""
TMP_CONFIG_DIR="$(mktemp -d /tmp/slam_multi_cfg_XXXXXX)"
DATASET_TIMING_JSONL="${TMP_CONFIG_DIR}/dataset_run_timing.jsonl"
DATASET_TIMING_SUMMARY_PATH=""
touch "${DATASET_TIMING_JSONL}"
DEDUP_POSE_ARGS=()
if [[ "${DEDUP_USE_POSE_KEY}" == "true" ]]; then
  DEDUP_POSE_ARGS+=(--dedup-use-pose-key)
fi
FORCE_BASE_PATH_NEXT=false
FORCE_BASE_PATH_REASON=""
SAFE_MODE_AFTER_BAD_RUNS=2
SAFE_MODE_RUNS_LEFT=0
SAFE_MODE_REASON=""

if [[ "${RUN_MODE}" != "train_only" && -n "${APPEND_FROM_MERGE_EXP_ID}" ]]; then
  APPEND_MERGE_DIR="${OUT_DIR}/${APPEND_FROM_MERGE_EXP_ID}"
  if [[ ! -d "${APPEND_MERGE_DIR}" ]]; then
    echo "--append-from-merge: brak katalogu ${APPEND_MERGE_DIR}" >&2
    exit 1
  fi
  require_file "${APPEND_MERGE_DIR}/dataset_rywak_merged.npz"
  require_file "${APPEND_MERGE_DIR}/dataset_robak_merged.npz"
  # Memory-safe append:
  # zaczynamy od snapshotu już zmergowanego zbioru (1 plik/metoda), a nie od pełnej listy
  # historycznych komponentów. To ogranicza zużycie RAM i zapobiega OOM ("Killed") przy dużych merge.
  rywak_linear_paths+=("${APPEND_MERGE_DIR}/dataset_rywak_merged.npz")
  robak_translation_paths+=("${APPEND_MERGE_DIR}/dataset_robak_merged.npz")

  if [[ -f "${APPEND_MERGE_DIR}/rebalance_unique_summary.json" ]]; then
    PREV_REBALANCE_SUMMARY="${APPEND_MERGE_DIR}/rebalance_unique_summary.json"
  fi
  if [[ -d "${APPEND_MERGE_DIR}/hist_balance_report" ]]; then
    PREV_HIST_REPORT_DIR="${APPEND_MERGE_DIR}/hist_balance_report"
  fi

  append_prev_runs_loaded=0
  while IFS=$'\t' read -r prev_exp_id prev_run_cfg; do
    if [[ -z "${prev_exp_id}" || -z "${prev_run_cfg}" ]]; then
      continue
    fi
    dataset_experiment_ids+=("${prev_exp_id}")
    dataset_run_cfg_paths+=("${prev_run_cfg}")
    append_prev_runs_loaded=$((append_prev_runs_loaded + 1))
  done < <(load_append_trajectory_runs "${APPEND_MERGE_DIR}" "${TMP_CONFIG_DIR}")

  log "Append mode: base merge=${APPEND_FROM_MERGE_EXP_ID}"
  log "Append seed sources (snapshot): rywak=${APPEND_MERGE_DIR}/dataset_rywak_merged.npz, robak=${APPEND_MERGE_DIR}/dataset_robak_merged.npz"
  if [[ "${append_prev_runs_loaded}" -gt 0 ]]; then
    log "Append mode: preloaded previous trajectory runs for overlay=${append_prev_runs_loaded}"
  else
    log "Append mode: no previous trajectory summary found; overlay will include only runs from this invocation."
  fi
  if [[ -n "${PREV_REBALANCE_SUMMARY}" ]]; then
    log "Append mode: using previous rebalance summary=${PREV_REBALANCE_SUMMARY}"
  fi
  if [[ -n "${PREV_HIST_REPORT_DIR}" ]]; then
    log "Append mode: using previous hist report dir=${PREV_HIST_REPORT_DIR}"
  fi
fi

if [[ "${RUN_MODE}" != "train_only" ]]; then
for run_idx in $(seq 1 "${MAX_DATASET_RUNS}"); do
  cleanup_ros
  base_cfg_run="$(choose_round_robin "${run_idx}" "${CONFIG_SEQUENCE[@]}")"
  selected_path=""
  if [[ "${#PATH_SEQUENCE[@]}" -gt 0 ]]; then
    selected_path="$(choose_round_robin "${run_idx}" "${PATH_SEQUENCE[@]}")"
  fi
  focus_override=""
  if [[ "${#FOCUS_SEQUENCE[@]}" -gt 0 ]]; then
    focus_override="$(choose_round_robin "${run_idx}" "${FOCUS_SEQUENCE[@]}")"
  fi
  if [[ "${ADAPTIVE_CONFIG}" == "true" ]]; then
    focus_mode="$(
      select_focus_mode \
        "${run_idx}" \
        "${PREV_REBALANCE_SUMMARY}" \
        "${focus_override}" \
        "${MIN_TARGET_PER_BIN_RYWAK}" \
        "${MIN_TARGET_PER_BIN_ROBAK}" \
        "${PREV_HIST_REPORT_DIR}"
    )"
  else
    if [[ -n "${focus_override}" ]]; then
      focus_mode="${focus_override}"
    else
      focus_mode="balanced"
    fi
  fi

  run_selected_path="${selected_path}"
  run_adaptive_path="${ADAPTIVE_PATH}"
  run_safe_mode=false
  if [[ "${run_idx}" -le "${WARMUP_BASE_RUNS}" ]]; then
    run_safe_mode=true
    SAFE_MODE_REASON="warmup_base_runs"
  fi
  if [[ "${SAFE_MODE_RUNS_LEFT}" -gt 0 ]]; then
    run_safe_mode=true
    SAFE_MODE_RUNS_LEFT=$((SAFE_MODE_RUNS_LEFT - 1))
  fi
  if [[ "${FORCE_BASE_PATH_NEXT}" == "true" ]]; then
    log "Fallback safety: run ${run_idx} wymusza bazową trasę (adaptive-path=OFF) | reason=${FORCE_BASE_PATH_REASON}"
    run_selected_path=""
    run_adaptive_path="false"
    run_safe_mode=true
    FORCE_BASE_PATH_NEXT=false
    FORCE_BASE_PATH_REASON=""
  fi
  if [[ "${run_safe_mode}" == "true" ]]; then
    if [[ -n "${SAFE_MODE_REASON}" ]]; then
      log "Safe mode: run ${run_idx} -> baza + focus=balanced (remaining_safe_runs=${SAFE_MODE_RUNS_LEFT}) | reason=${SAFE_MODE_REASON}"
    else
      log "Safe mode: run ${run_idx} -> baza + focus=balanced (remaining_safe_runs=${SAFE_MODE_RUNS_LEFT})"
    fi
    run_selected_path=""
    run_adaptive_path="false"
    focus_mode="balanced"
  fi

  ts="$(date +%Y%m%d_%H%M%S)"
  dataset_exp_id="exp_multi_dataset_${ts}_r${run_idx}"

  eval "$(extract_rywak_weak_bins "${PREV_HIST_REPORT_DIR}")"

  run_cfg="${TMP_CONFIG_DIR}/run_${run_idx}_${ts}.yaml"
  python3 - "${base_cfg_run}" "${run_cfg}" "${focus_mode}" "${run_selected_path}" "${run_idx}" "${ADAPTIVE_CONFIG}" "${run_adaptive_path}" "${DATASET_DURATION_CAP_SEC}" \
    "${RYWAK_WEAK_LINEAR_BIN_CENTER}" "${RYWAK_WEAK_LINEAR_BIN_MIN}" "${RYWAK_WEAK_LINEAR_BIN_MAX}" "${RYWAK_WEAK_LINEAR_COUNT}" \
    "${RYWAK_WEAK_ANGULAR_BIN_CENTER}" "${RYWAK_WEAK_ANGULAR_BIN_MIN}" "${RYWAK_WEAK_ANGULAR_BIN_MAX}" "${RYWAK_WEAK_ANGULAR_COUNT}" \
    "${RYWAK_WEAK_LINEAR_BIN_INDEX}" "${RYWAK_WEAK_ANGULAR_BIN_INDEX}" \
    "${RYWAK_V_MIN}" "${RYWAK_V_MAX}" "${RYWAK_W_MIN}" "${RYWAK_W_MAX}" <<'PY'
import os
import sys
import math
from pathlib import Path
import yaml

(
    base_cfg,
    out_cfg,
    focus_mode,
    selected_path,
    run_idx_s,
    adaptive_cfg_s,
    adaptive_path_s,
    dataset_cap_s,
    weak_lin_center_s,
    weak_lin_min_s,
    weak_lin_max_s,
    weak_lin_count_s,
    weak_ang_center_s,
    weak_ang_min_s,
    weak_ang_max_s,
    weak_ang_count_s,
    weak_lin_idx_s,
    weak_ang_idx_s,
    rywak_v_min_s,
    rywak_v_max_s,
    rywak_w_min_s,
    rywak_w_max_s,
) = sys.argv[1:]
run_idx = int(run_idx_s)
adaptive_cfg = adaptive_cfg_s.strip().lower() == "true"
adaptive_path = adaptive_path_s.strip().lower() == "true"
dataset_cap = max(1.0, float(dataset_cap_s))

def _to_float(v: str, default: float = float("nan")) -> float:
    try:
        x = float(str(v).strip())
        return x if math.isfinite(x) else float(default)
    except Exception:
        return float(default)

def _to_int(v: str, default: int = -1) -> int:
    try:
        return int(float(str(v).strip()))
    except Exception:
        return int(default)

weak_lin_center = _to_float(weak_lin_center_s)
weak_lin_min = _to_float(weak_lin_min_s)
weak_lin_max = _to_float(weak_lin_max_s)
weak_lin_count = _to_int(weak_lin_count_s, -1)
weak_ang_center = _to_float(weak_ang_center_s)
weak_ang_min = _to_float(weak_ang_min_s)
weak_ang_max = _to_float(weak_ang_max_s)
weak_ang_count = _to_int(weak_ang_count_s, -1)
weak_lin_idx = _to_int(weak_lin_idx_s, -1)
weak_ang_idx = _to_int(weak_ang_idx_s, -1)
rywak_v_min = _to_float(rywak_v_min_s, 0.0)
rywak_v_max = _to_float(rywak_v_max_s, 1.2)
rywak_w_min = _to_float(rywak_w_min_s, 0.0)
rywak_w_max = _to_float(rywak_w_max_s, 3.0)

with open(base_cfg, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

driver = cfg.setdefault("driver", {})
pp = driver.setdefault("planned_path", {})
simulation = cfg.setdefault("simulation", {})
dataset = cfg.setdefault("dataset", {})
tracks = cfg.setdefault("tracks", {})
rywak = cfg.setdefault("rywak", {})
robak = cfg.setdefault("robak", {})
experiment = cfg.setdefault("experiment", {})
pipeline = cfg.setdefault("pipeline", {})
timing = cfg.setdefault("timing", {})
base_seed = int(experiment.get("seed", 123))
dataset_seed = int(base_seed + (run_idx * 1009))
# Usuń stare pola adaptive_* z configu bazowego, żeby każda runda była liczona od zera.
for k in list(experiment.keys()):
    if str(k).startswith("adaptive_"):
        experiment.pop(k, None)
experiment["seed"] = dataset_seed
experiment["adaptive_dataset_seed"] = dataset_seed
experiment["adaptive_focus_enabled"] = bool(adaptive_cfg)
experiment["adaptive_path_enabled"] = bool(adaptive_path)
# Wyłączamy tor AI-SLAM; zbieramy datasety tylko baseline/Robak/Rywak.
tracks["tor2_ai_slam"] = False

def _default_spec_for_world(world_name: str) -> str:
    wn = str(world_name or "").strip().lower()
    if "hospital" in wn:
        return "planned_paths/hospital_trajectory_acyclic.yaml"
    return "planned_paths/office_trajectory_acyclic.yaml"

# Twardy cap czasu rundy datasetu: nawet jeśli bazowy config ma dłuższy czas (np. 1800s),
# w multi-run nie przekraczamy dataset_cap.
pipe_ds = pipeline.get("dataset_collection_sec", None)
if pipe_ds is None:
    pipeline["dataset_collection_sec"] = dataset_cap
else:
    try:
        pipeline["dataset_collection_sec"] = min(float(pipe_ds), dataset_cap)
    except Exception:
        pipeline["dataset_collection_sec"] = dataset_cap

timing_ds = timing.get("dataset_duration", None)
if timing_ds is None:
    timing["dataset_duration"] = dataset_cap
else:
    try:
        timing["dataset_duration"] = min(float(timing_ds), dataset_cap)
    except Exception:
        timing["dataset_duration"] = dataset_cap

for dct in (rywak, robak):
    cur = dct.get("dataset_duration", None)
    if cur is None:
        dct["dataset_duration"] = float(pipeline["dataset_collection_sec"])
    else:
        try:
            dct["dataset_duration"] = min(float(cur), float(pipeline["dataset_collection_sec"]))
        except Exception:
            dct["dataset_duration"] = float(pipeline["dataset_collection_sec"])

# Fail-fast na utknięcie robota: kończ rundę i przechodź do kolejnej adaptacji.
dataset["motion_stall_watchdog_enabled"] = True
dataset.setdefault("motion_stall_pose_topic", "/ground_truth_pose")
dataset.setdefault("motion_stall_min_delta_m", 0.035)
dataset.setdefault("motion_stall_timeout_sec", 35.0)
dataset.setdefault("motion_stall_startup_grace_sec", 18.0)
dataset.setdefault("motion_stall_no_pose_timeout_sec", 20.0)
dataset.setdefault("motion_stall_check_hz", 4.0)
dataset.setdefault("motion_stall_enable_window_guard", True)
dataset.setdefault("motion_stall_min_window_progress_m", 0.12)
dataset.setdefault("motion_stall_window_span_ratio", 1.8)
dataset.setdefault("motion_stall_enable_circling_guard", True)
dataset.setdefault("motion_stall_circling_min_window_path_m", 1.6)
dataset.setdefault("motion_stall_circling_max_net_path_ratio", 0.25)
dataset.setdefault("motion_stall_circling_max_net_m", 1.2)
dataset.setdefault("motion_stall_circling_max_span_m", 2.5)
# Twarde zakończenie rundy po ukończeniu planned path.
dataset["stop_on_planned_path_done"] = True
dataset.setdefault("planned_path_done_topic", "/planned_path_done")
dataset.setdefault("planned_path_done_min_elapsed_sec", 5.0)
pp["publish_completion_topic"] = True
pp.setdefault("completion_topic", "/planned_path_done")

if selected_path.strip():
    path = selected_path.strip()
    if not path.startswith("/") and not path.startswith("planned_paths/"):
        path = f"planned_paths/{path}"
    pp["spec_yaml"] = path
    if "cyclic" in path:
        pp["loop_path"] = True
    if "acyclic" in path:
        pp["loop_path"] = False
    # Avoid old hard overrides from base config when explicitly cycling paths.
    pp["world_overrides"] = {}
    if "hospital" in path:
        simulation["train_world"] = "world_hospital.sdf"
        pp["reference_map_yaml"] = "reference_map_hospital.yaml"
    elif "office" in path:
        simulation["train_world"] = "world_office.sdf"
        pp["reference_map_yaml"] = "reference_map_office.yaml"
else:
    spec_raw = str(pp.get("spec_yaml", "")).strip()
    # Ochrona przed "skażeniem" configu: jeśli bazowy config wskazuje tymczasowy
    # adaptive_path z /tmp, resetujemy na stałą trasę dla danego świata.
    if spec_raw and os.path.isabs(spec_raw):
        spec_name = Path(spec_raw).name.lower()
        is_tmp_adaptive = (
            ("adaptive_path_" in spec_name)
            and (spec_raw.startswith("/tmp/") or "/tmp/" in spec_raw)
        )
        if is_tmp_adaptive:
            fallback_spec = _default_spec_for_world(str(simulation.get("train_world", "")))
            pp["spec_yaml"] = fallback_spec
            experiment["adaptive_tmp_spec_reset_from"] = spec_raw
            experiment["adaptive_tmp_spec_reset_to"] = fallback_spec
            spec_raw = fallback_spec
    # Safe mode / fallback: jeżeli zostaje stary tymczasowy adaptive_path z /tmp,
    # wracamy do bazowej trajektorii dla świata (office/hospital), żeby nie mielić
    # tej samej problematycznej geometrii.
    if (not adaptive_path) and spec_raw:
        spec_name = Path(spec_raw).name.lower()
        looks_tmp = os.path.isabs(spec_raw) and (spec_raw.startswith("/tmp/") or "/tmp/" in spec_raw)
        looks_adaptive = ("adaptive_path_" in spec_name) or ("adaptive_path_" in spec_raw.lower())
        if looks_tmp or looks_adaptive:
            fallback_spec = _default_spec_for_world(str(simulation.get("train_world", "")))
            pp["spec_yaml"] = fallback_spec
            experiment["adaptive_safe_fallback_from"] = spec_raw
            experiment["adaptive_safe_fallback_to"] = fallback_spec
            spec_raw = fallback_spec
    # Częsty przypadek: stary tymczasowy spec z /tmp po poprzednim uruchomieniu.
    # Jeśli plik nie istnieje, wracamy do stałego specu zależnego od świata.
    if spec_raw and os.path.isabs(spec_raw) and (not Path(spec_raw).is_file()):
        fallback_spec = _default_spec_for_world(str(simulation.get("train_world", "")))
        pp["spec_yaml"] = fallback_spec
        experiment["adaptive_spec_missing_fallback_from"] = spec_raw
        experiment["adaptive_spec_missing_fallback_to"] = fallback_spec

# Deterministyczny multi-run przy adaptacji: unikamy cichego nadpisania spec_yaml przez world_overrides.
if adaptive_cfg:
    pp["world_overrides"] = {}

pp["dataset_excitation_enabled"] = True
pp.setdefault("linear_vel_max", 1.25)
pp.setdefault("angular_vel_max", 3.0)
pp.setdefault("lookahead_m", 0.22)
pp.setdefault("heading_gain", 2.4)
pp.setdefault("heading_stop_deg", 52.0)
pp.setdefault("heading_resume_deg", 32.0)
pp.setdefault("alignment_cos_power", 2.4)
pp.setdefault("excitation_period_sec", 3.8)
pp.setdefault("excitation_v_min_scale", 0.02)
pp.setdefault("excitation_v_max_scale", 1.6)
pp.setdefault("excitation_heading_bias_deg", 30.0)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def unit_pos(v: float, lo: float, hi: float):
    if not (math.isfinite(v) and math.isfinite(lo) and math.isfinite(hi)):
        return None
    if hi <= lo:
        return None
    return clamp((v - lo) / (hi - lo), 0.0, 1.0)

mode = focus_mode.strip().lower()
if mode == "rotation":
    pp["linear_vel_max"] = clamp(float(pp.get("linear_vel_max", 1.0)), 0.6, 0.95)
    pp["angular_vel_max"] = 3.0
    pp["lookahead_m"] = clamp(float(pp.get("lookahead_m", 0.18)), 0.12, 0.18)
    pp["heading_gain"] = clamp(float(pp.get("heading_gain", 3.2)), 3.0, 4.0)
    pp["heading_stop_deg"] = clamp(float(pp.get("heading_stop_deg", 68.0)), 60.0, 80.0)
    pp["heading_resume_deg"] = clamp(float(pp.get("heading_resume_deg", 42.0)), 35.0, 55.0)
    pp["excitation_period_sec"] = 2.1
    pp["excitation_v_min_scale"] = 0.0
    pp["excitation_v_max_scale"] = 0.95
    pp["excitation_heading_bias_deg"] = 72.0
    robak["offsets"] = [2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40]
    robak["max_pair_dist"] = max(2.5, float(robak.get("max_pair_dist", 1.0)))
    robak["max_pair_dyaw"] = 3.141592653589793
    robak["pair_filter_mode"] = "any"
    robak["min_pair_dyaw"] = min(0.02, float(robak.get("min_pair_dyaw", 0.05)))
    rywak["min_sample_dist"] = 0.005
    rywak["min_sample_dyaw"] = 0.005
    rywak["min_delta_scan_rms"] = 0.0
    rywak["sample_filter_mode"] = "any"
    # Normalize against the full configured histogram range (not weak-bin edges),
    # otherwise norm would be ~0.5 for every bin center.
    weak_ang_norm = unit_pos(weak_ang_center, rywak_w_min, rywak_w_max)
    if weak_ang_norm is not None and weak_ang_count >= 0:
        # Dla najsłabszych wysokich binów angular (blisko max zakresu) wymuszamy
        # wyraźnie mocniejszą rotację; inaczej angular_vel_max bywa tuż poniżej progu binu.
        pp["excitation_heading_bias_deg"] = clamp(30.0 + 62.0 * float(weak_ang_norm), 24.0, 88.0)
        pp["angular_vel_max"] = clamp(1.45 + 1.75 * float(weak_ang_norm), 1.3, 3.0)
        if weak_ang_norm >= 0.90:
            pp["angular_vel_max"] = 3.0
            pp["linear_vel_max"] = clamp(float(pp.get("linear_vel_max", 0.70)), 0.45, 0.72)
            pp["lookahead_m"] = clamp(float(pp.get("lookahead_m", 0.14)), 0.10, 0.15)
            pp["excitation_period_sec"] = 1.55
            pp["excitation_heading_bias_deg"] = clamp(float(pp.get("excitation_heading_bias_deg", 80.0)), 72.0, 88.0)
        elif weak_ang_norm >= 0.70:
            pp["linear_vel_max"] = clamp(float(pp.get("linear_vel_max", 0.75)), 0.50, 0.80)
            pp["lookahead_m"] = clamp(float(pp.get("lookahead_m", 0.16)), 0.11, 0.16)
            pp["excitation_period_sec"] = 1.75
        elif weak_ang_norm <= 0.25:
            pp["linear_vel_max"] = clamp(float(pp.get("linear_vel_max", 1.0)), 0.88, 1.18)
            pp["excitation_period_sec"] = 2.40
        experiment["adaptive_rywak_weak_angular_bin_index"] = int(weak_ang_idx)
        experiment["adaptive_rywak_weak_angular_bin_center_radps"] = float(weak_ang_center)
        experiment["adaptive_rywak_weak_angular_bin_count"] = int(weak_ang_count)
elif mode == "translation":
    t_phase = (run_idx - 1) % 6
    lin_targets = [1.85, 1.98, 2.12, 2.24, 2.05, 1.92]
    period_targets = [1.45, 1.75, 2.05, 2.30, 1.60, 2.20]
    vmin_targets = [0.03, 0.10, 0.18, 0.26, 0.06, 0.14]
    vmax_targets = [1.45, 1.70, 1.95, 2.20, 1.55, 1.85]
    heading_bias_targets = [1.0, 3.0, 0.0, 5.0, 2.0, 4.0]
    lin_target = float(lin_targets[t_phase])
    pp["linear_vel_max"] = clamp(max(float(pp.get("linear_vel_max", lin_target)), lin_target), 1.8, 2.3)
    # Ogranicz ekstremalną rotację, która powoduje długie okresy v~0.
    pp["angular_vel_max"] = clamp(float(pp.get("angular_vel_max", 1.7)), 1.3, 2.0)
    pp["lookahead_m"] = clamp(float(pp.get("lookahead_m", 0.46)), 0.38, 0.64)
    pp["heading_gain"] = clamp(float(pp.get("heading_gain", 1.2)), 0.9, 1.6)
    # W translation-focus unikamy turn-in-place: podnosimy progi histerezy.
    pp["heading_stop_deg"] = clamp(float(pp.get("heading_stop_deg", 82.0)), 72.0, 89.0)
    pp["heading_resume_deg"] = clamp(float(pp.get("heading_resume_deg", 60.0)), 48.0, 75.0)
    pp["alignment_cos_power"] = clamp(float(pp.get("alignment_cos_power", 1.1)), 1.0, 1.5)
    # Zmieniamy profil excitation między rundami, żeby uzupełniać różne biny v.
    pp["excitation_period_sec"] = float(period_targets[t_phase])
    pp["excitation_v_min_scale"] = float(vmin_targets[t_phase])
    pp["excitation_v_max_scale"] = float(vmax_targets[t_phase])
    pp["excitation_heading_bias_deg"] = float(heading_bias_targets[t_phase])
    robak["offsets"] = [4, 6, 8, 10, 12, 16, 20, 24, 32, 40]
    robak["max_pair_dist"] = max(2.5, float(robak.get("max_pair_dist", 1.0)))
    rywak["min_sample_dist"] = 0.005
    rywak["min_sample_dyaw"] = 0.005
    rywak["min_delta_scan_rms"] = 0.0
    rywak["sample_filter_mode"] = "any"
    # Normalize against the full configured histogram range (not weak-bin edges),
    # otherwise norm would be ~0.5 for every bin center.
    weak_lin_norm = unit_pos(weak_lin_center, rywak_v_min, rywak_v_max)
    if weak_lin_norm is not None and weak_lin_count >= 0:
        if weak_lin_norm < 0.20:
            lin_target = 1.28
            pp["excitation_v_min_scale"] = 0.00
            pp["excitation_v_max_scale"] = 0.60
            pp["excitation_period_sec"] = 1.35
            pp["excitation_heading_bias_deg"] = 26.0
        elif weak_lin_norm < 0.45:
            lin_target = 1.58
            pp["excitation_v_min_scale"] = 0.10
            pp["excitation_v_max_scale"] = 1.00
            pp["excitation_period_sec"] = 1.60
            pp["excitation_heading_bias_deg"] = 18.0
        elif weak_lin_norm < 0.75:
            lin_target = 1.95
            pp["excitation_v_min_scale"] = 0.28
            pp["excitation_v_max_scale"] = 1.45
            pp["excitation_period_sec"] = 1.85
            pp["excitation_heading_bias_deg"] = 10.0
        else:
            lin_target = 2.28
            pp["excitation_v_min_scale"] = 0.45
            pp["excitation_v_max_scale"] = 2.10
            pp["excitation_period_sec"] = 2.10
            pp["excitation_heading_bias_deg"] = 2.0
        pp["linear_vel_max"] = clamp(max(float(pp.get("linear_vel_max", lin_target)), lin_target), 1.2, 2.4)
        experiment["adaptive_rywak_weak_linear_bin_index"] = int(weak_lin_idx)
        experiment["adaptive_rywak_weak_linear_bin_center_mps"] = float(weak_lin_center)
        experiment["adaptive_rywak_weak_linear_bin_count"] = int(weak_lin_count)
elif mode == "slalom":
    pp["linear_vel_max"] = clamp(float(pp.get("linear_vel_max", 1.45)), 1.3, 2.0)
    pp["angular_vel_max"] = 3.0
    pp["lookahead_m"] = clamp(float(pp.get("lookahead_m", 0.16)), 0.12, 0.20)
    pp["heading_gain"] = clamp(float(pp.get("heading_gain", 3.0)), 2.8, 4.0)
    pp["alignment_cos_power"] = clamp(float(pp.get("alignment_cos_power", 3.0)), 2.4, 4.0)
    pp["excitation_period_sec"] = 2.0
    pp["excitation_v_min_scale"] = 0.02
    pp["excitation_v_max_scale"] = 2.0
    pp["excitation_heading_bias_deg"] = 58.0
    robak["offsets"] = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24]
    robak["max_pair_dist"] = max(2.0, float(robak.get("max_pair_dist", 1.0)))
    rywak["min_sample_dist"] = 0.02
    rywak["min_sample_dyaw"] = 0.03
else:
    # balanced
    pp["linear_vel_max"] = clamp(float(pp.get("linear_vel_max", 1.55)), 1.3, 1.95)
    pp["angular_vel_max"] = 3.0
    pp["lookahead_m"] = clamp(float(pp.get("lookahead_m", 0.28)), 0.22, 0.34)
    pp["heading_gain"] = clamp(float(pp.get("heading_gain", 1.9)), 1.6, 2.4)
    pp["heading_stop_deg"] = clamp(float(pp.get("heading_stop_deg", 74.0)), 66.0, 86.0)
    pp["heading_resume_deg"] = clamp(float(pp.get("heading_resume_deg", 50.0)), 40.0, 68.0)
    pp["excitation_period_sec"] = 2.4
    pp["excitation_v_min_scale"] = 0.12
    pp["excitation_v_max_scale"] = 2.0
    pp["excitation_heading_bias_deg"] = 20.0
    robak["offsets"] = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
    robak["max_pair_dist"] = max(1.5, float(robak.get("max_pair_dist", 1.0)))
    rywak["min_sample_dist"] = 0.005
    rywak["min_sample_dyaw"] = 0.01
    rywak["min_delta_scan_rms"] = 0.0
    rywak["sample_filter_mode"] = "any"

if bool(pp.get("loop_path", True)):
    robak["trajectory_mode"] = "cycle"
    rywak["trajectory_mode"] = "cycle"
else:
    robak["trajectory_mode"] = "no_cycle"
    rywak["trajectory_mode"] = "no_cycle"
    # Mniej odrzuceń "trajectory_repeats" przy no_cycle bez łamania zasady unikalności.
    robak["trajectory_cell_size_m"] = 0.02
    rywak["trajectory_cell_size_m"] = 0.005
    if mode == "slalom":
        robak["trajectory_cell_size_m"] = 0.025
        rywak["trajectory_cell_size_m"] = 0.008

experiment["adaptive_focus_mode"] = mode
experiment["adaptive_rywak_weak_linear_bin_index"] = int(weak_lin_idx)
experiment["adaptive_rywak_weak_angular_bin_index"] = int(weak_ang_idx)
if selected_path.strip():
    experiment["adaptive_selected_path"] = str(pp.get("spec_yaml", selected_path.strip()))
else:
    experiment["adaptive_selected_path"] = str(pp.get("spec_yaml", ""))

with open(out_cfg, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
PY

  if [[ "${ADAPTIVE_CONFIG}" == "true" && "${run_adaptive_path}" == "true" ]]; then
    python3 "${ROOT_DIR}/scripts/adapt_planned_path_for_focus.py" \
      --config "${run_cfg}" \
      --focus-mode "${focus_mode}" \
      --run-idx "${run_idx}" \
      --work-dir "${TMP_CONFIG_DIR}"
  elif [[ "${ADAPTIVE_CONFIG}" == "true" && "${run_adaptive_path}" == "false" ]]; then
    log "Adaptive focus: ON | adaptive path geometry: OFF (brak modyfikacji planned_path)"
  fi

  log "=== RUN ${run_idx}/${MAX_DATASET_RUNS}: dataset collection (${dataset_exp_id}) ==="
  log "Config base=${base_cfg_run} | focus=${focus_mode} | path=${run_selected_path:-<from-config>} | adaptive_path=${run_adaptive_path} | run_cfg=${run_cfg}"
  run_wall_start="$(date +%s.%N)"
  set +e
  ros2 launch ai_slam_bringup demo.launch.py \
    config:="${run_cfg}" \
    phase:=dataset \
    gui:="${GUI}" \
    experiment_id:="${dataset_exp_id}"
  launch_rc=$?
  set -e
  run_wall_end="$(date +%s.%N)"
  run_wall_sec="$(calc_elapsed_sec "${run_wall_start}" "${run_wall_end}")"

  if [[ "${launch_rc}" -ne 0 ]]; then
    append_dataset_timing_jsonl \
      "${DATASET_TIMING_JSONL}" \
      "${run_idx}" \
      "${dataset_exp_id}" \
      "${run_wall_sec}" \
      "launch_failed" \
      "${launch_rc}" \
      "${run_cfg}"
    log "WARN: dataset run ${dataset_exp_id} zakończony błędem rc=${launch_rc}; pomijam rundę i jadę dalej."
    FORCE_BASE_PATH_NEXT=true
    FORCE_BASE_PATH_REASON="launch_rc_${launch_rc}"
    if [[ "${SAFE_MODE_RUNS_LEFT}" -lt "${SAFE_MODE_AFTER_BAD_RUNS}" ]]; then
      SAFE_MODE_RUNS_LEFT="${SAFE_MODE_AFTER_BAD_RUNS}"
    fi
    SAFE_MODE_REASON="launch_rc_${launch_rc}"
    continue
  fi

  exp_dir="${OUT_DIR}/${dataset_exp_id}"
  expected_files=(
    "${exp_dir}/dataset_rywak_linear_balanced.npz"
    "${exp_dir}/dataset_rywak_angular_balanced.npz"
    "${exp_dir}/dataset_robak_translation_balanced.npz"
    "${exp_dir}/dataset_robak_rotation_balanced.npz"
  )
  missing_artifacts=0
  for f in "${expected_files[@]}"; do
    if [[ ! -f "${f}" ]]; then
      log "WARN: brak artefaktu po rundzie ${dataset_exp_id}: ${f}"
      missing_artifacts=1
    fi
  done
  if [[ "${missing_artifacts}" -ne 0 ]]; then
    append_dataset_timing_jsonl \
      "${DATASET_TIMING_JSONL}" \
      "${run_idx}" \
      "${dataset_exp_id}" \
      "${run_wall_sec}" \
      "incomplete_artifacts" \
      "${launch_rc}" \
      "${run_cfg}"
    log "WARN: runda ${dataset_exp_id} przerwana lub niepełna (np. watchdog/no-motion). Pomijam i przechodzę dalej."
    FORCE_BASE_PATH_NEXT=true
    FORCE_BASE_PATH_REASON="incomplete_artifacts"
    if [[ "${SAFE_MODE_RUNS_LEFT}" -lt "${SAFE_MODE_AFTER_BAD_RUNS}" ]]; then
      SAFE_MODE_RUNS_LEFT="${SAFE_MODE_AFTER_BAD_RUNS}"
    fi
    SAFE_MODE_REASON="incomplete_artifacts"
    continue
  fi

  run_health_line="$(
    analyze_single_run_trajectory_health "${TMP_CONFIG_DIR}" "${dataset_exp_id}" "${run_cfg}"
  )"
  IFS=$'\t' read -r run_health_flag run_health_status run_health_segs run_health_pts run_health_len <<< "${run_health_line}"
  if [[ "${run_health_flag}" == "1" ]]; then
    append_dataset_timing_jsonl \
      "${DATASET_TIMING_JSONL}" \
      "${run_idx}" \
      "${dataset_exp_id}" \
      "${run_wall_sec}" \
      "rejected_trajectory_health" \
      "${launch_rc}" \
      "${run_cfg}"
    log "WARN: odrzucam rundę ${dataset_exp_id} (traj health: status=${run_health_status}, stuck_segs=${run_health_segs}, stuck_pts=${run_health_pts}, len=${run_health_len}m)."
    FORCE_BASE_PATH_NEXT=true
    FORCE_BASE_PATH_REASON="traj_health_${run_health_status}_segs${run_health_segs}_pts${run_health_pts}_len${run_health_len}"
    if [[ "${SAFE_MODE_RUNS_LEFT}" -lt "${SAFE_MODE_AFTER_BAD_RUNS}" ]]; then
      SAFE_MODE_RUNS_LEFT="${SAFE_MODE_AFTER_BAD_RUNS}"
    fi
    SAFE_MODE_REASON="${FORCE_BASE_PATH_REASON}"
    continue
  fi

  append_dataset_timing_jsonl \
    "${DATASET_TIMING_JSONL}" \
    "${run_idx}" \
    "${dataset_exp_id}" \
    "${run_wall_sec}" \
    "accepted" \
    "${launch_rc}" \
    "${run_cfg}"

  dataset_experiment_ids+=("${dataset_exp_id}")
  dataset_run_cfg_paths+=("${run_cfg}")

  SAFE_MODE_REASON=""

  python3 "${ROOT_DIR}/scripts/report_dataset_histogram_balance.py" --experiment-dir "${exp_dir}"

  rywak_linear_paths+=("${exp_dir}/dataset_rywak_linear_balanced.npz")
  rywak_angular_paths+=("${exp_dir}/dataset_rywak_angular_balanced.npz")
  robak_translation_paths+=("${exp_dir}/dataset_robak_translation_balanced.npz")
  robak_rotation_paths+=("${exp_dir}/dataset_robak_rotation_balanced.npz")

  merge_ts="$(date +%Y%m%d_%H%M%S)"
  MERGE_EXP_ID="exp_multi_merged_${merge_ts}"
  MERGE_DIR="${OUT_DIR}/${MERGE_EXP_ID}"

  log "Merging component datasets -> ${MERGE_DIR}"
  python3 "${ROOT_DIR}/scripts/merge_component_datasets.py" \
    --out-dir "${MERGE_DIR}" \
    --rywak-linear "${rywak_linear_paths[@]}" \
    --rywak-angular "${rywak_angular_paths[@]}" \
    --robak-translation "${robak_translation_paths[@]}" \
    --robak-rotation "${robak_rotation_paths[@]}" \
    --deduplicate \
    "${DEDUP_POSE_ARGS[@]}"

  log "Trajectory overview (all merged dataset runs)"
  if ! python3 "${ROOT_DIR}/scripts/plot_merged_dataset_trajectories.py" \
    --out-dir "${MERGE_DIR}" \
    --experiment-ids "${dataset_experiment_ids[@]}" \
    --run-configs "${dataset_run_cfg_paths[@]}"; then
    log "WARN: trajectory overview generation failed for ${MERGE_DIR}"
  fi
  traj_summary_json="${MERGE_DIR}/merged_trajectory_overview_summary.json"
  if [[ -f "${traj_summary_json}" ]]; then
    health_line="$(
      analyze_run_trajectory_health "${traj_summary_json}" "${dataset_exp_id}"
    )"
    IFS=$'\t' read -r run_stuck_flag run_traj_status run_stuck_segs run_stuck_pts run_traj_len <<< "${health_line}"
    if [[ "${run_stuck_flag}" == "1" ]]; then
      FORCE_BASE_PATH_NEXT=true
      FORCE_BASE_PATH_REASON="traj_health_${run_traj_status}_segs${run_stuck_segs}_pts${run_stuck_pts}_len${run_traj_len}"
      if [[ "${SAFE_MODE_RUNS_LEFT}" -lt "${SAFE_MODE_AFTER_BAD_RUNS}" ]]; then
        SAFE_MODE_RUNS_LEFT="${SAFE_MODE_AFTER_BAD_RUNS}"
      fi
      SAFE_MODE_REASON="${FORCE_BASE_PATH_REASON}"
      log "WARN: wykryto możliwe utknięcie/niski progres w ${dataset_exp_id} (status=${run_traj_status}, stuck_segs=${run_stuck_segs}, stuck_pts=${run_stuck_pts}, len=${run_traj_len}m). Następne runy: safe mode (baza + focus=balanced, adaptive_path=OFF)."
    fi
  fi

  rebalance_summary="${MERGE_DIR}/rebalance_unique_summary.json"
  log "Strict rebalance (equal+unique) -> ${rebalance_summary}"
  if python3 "${ROOT_DIR}/scripts/rebalance_unique_histograms.py" \
    --rywak-npz "${MERGE_DIR}/dataset_rywak_merged.npz" \
    --robak-npz "${MERGE_DIR}/dataset_robak_merged.npz" \
    --rywak-out "${MERGE_DIR}/dataset_rywak.npz" \
    --robak-out "${MERGE_DIR}/dataset_robak.npz" \
    --bins "${TARGET_BINS}" \
    --seed 123 \
    --require-all-bins \
    --summary-json "${rebalance_summary}" \
    --rywak-v-min "${RYWAK_V_MIN}" \
    --rywak-v-max "${RYWAK_V_MAX}" \
    --rywak-w-min "${RYWAK_W_MIN}" \
    --rywak-w-max "${RYWAK_W_MAX}" \
    --robak-t-min "${ROBAK_T_MIN}" \
    --robak-t-max "${ROBAK_T_MAX}" \
    --robak-r-min "${ROBAK_R_MIN}" \
    --robak-r-max "${ROBAK_R_MAX}" \
    "${DEDUP_POSE_ARGS[@]}"; then
    strict_ok=true
  else
    strict_ok=false
  fi

  if [[ "${strict_ok}" != "true" ]]; then
    PREV_REBALANCE_SUMMARY="${rebalance_summary}"
    log "Strict equal+unique nieosiągnięty po rundzie ${run_idx}. Kontynuuję kolejną rundę datasetu..."
    continue
  fi
  PREV_REBALANCE_SUMMARY="${rebalance_summary}"

  python3 "${ROOT_DIR}/scripts/report_dataset_histogram_balance.py" --experiment-dir "${MERGE_DIR}"
  if [[ -d "${MERGE_DIR}/hist_balance_report" ]]; then
    PREV_HIST_REPORT_DIR="${MERGE_DIR}/hist_balance_report"
  fi

  strict_line="$(
    python3 - "${rebalance_summary}" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text(encoding="utf-8"))
ry = d.get("rywak", {}) if isinstance(d.get("rywak", {}), dict) else {}
rb = d.get("robak", {}) if isinstance(d.get("robak", {}), dict) else {}
print(
    f"{int(ry.get('target_per_bin', 0))} "
    f"{int(rb.get('target_per_bin', 0))} "
    f"{int(ry.get('n_output', 0))} "
    f"{int(rb.get('n_output', 0))}"
)
PY
  )"
  read -r ry_tgt rb_tgt ry_n rb_n <<< "${strict_line}"
  LAST_RY_TGT="${ry_tgt}"
  LAST_RB_TGT="${rb_tgt}"

  log "Strict balanced (unique) target/bin: rywak=${ry_tgt}, robak=${rb_tgt}; sizes: rywak=${ry_n}, robak=${rb_n}"

  if [[ "${ry_tgt}" -lt "${MIN_TARGET_PER_BIN_RYWAK}" || "${rb_tgt}" -lt "${MIN_TARGET_PER_BIN_ROBAK}" ]]; then
    log "Target/bin za niski (wymagane: rywak>=${MIN_TARGET_PER_BIN_RYWAK}, robak>=${MIN_TARGET_PER_BIN_ROBAK}). Kontynuuję kolejną rundę datasetu..."
    continue
  fi

  BINS_OK=true
  log "Target bins osiągnięty po rundzie ${run_idx}."
  break
done
fi

if [[ "${RUN_MODE}" != "train_only" ]]; then
  if [[ -n "${MERGE_DIR}" ]]; then
    DATASET_TIMING_SUMMARY_PATH="${MERGE_DIR}/dataset_run_timing_summary.json"
  else
    DATASET_TIMING_SUMMARY_PATH="${TMP_CONFIG_DIR}/dataset_run_timing_summary.json"
  fi
  timing_line="$(
    summarize_dataset_timing_jsonl \
      "${DATASET_TIMING_JSONL}" \
      "${DATASET_TIMING_SUMMARY_PATH}"
  )"
  IFS=$'\t' read -r timing_path acc_n acc_avg acc_min acc_max all_n all_avg all_min all_max <<< "${timing_line}"
  log "Czas rund dataset (zaakceptowane): n=${acc_n}, avg=${acc_avg}s, min=${acc_min}s, max=${acc_max}s"
  log "Czas rund dataset (wszystkie próby): n=${all_n}, avg=${all_avg}s, min=${all_min}s, max=${all_max}s"
  log "Timing summary: ${timing_path}"
fi

if [[ -z "${MERGE_EXP_ID}" ]]; then
  if [[ "${RUN_MODE}" == "train_only" ]]; then
    MERGE_EXP_ID="${TRAIN_ONLY_EXP_ID}"
    MERGE_DIR="${OUT_DIR}/${MERGE_EXP_ID}"
  else
    echo "Nie udało się utworzyć merged datasetu." >&2
    exit 1
  fi
fi

if [[ "${RUN_MODE}" == "train_only" ]]; then
  if [[ ! -d "${MERGE_DIR}" ]]; then
    echo "Brak katalogu merged experiment: ${MERGE_DIR}" >&2
    exit 1
  fi
  require_file "${MERGE_DIR}/dataset_robak.npz"
  require_file "${MERGE_DIR}/dataset_rywak.npz"
  log "TRAIN-ONLY mode: pomijam zbieranie datasetu i merge, używam: ${MERGE_DIR}"
  BINS_OK=true
fi

if [[ "${RUN_MODE}" != "train_only" && "${BINS_OK}" != "true" ]]; then
  echo "Nie osiągnięto strict equal+unique histogram target (bins=${TARGET_BINS}, min_target_per_bin_rywak=${MIN_TARGET_PER_BIN_RYWAK}, min_target_per_bin_robak=${MIN_TARGET_PER_BIN_ROBAK}) w limicie rund=${MAX_DATASET_RUNS}." >&2
  echo "Ostatnie target/bin: rywak=${LAST_RY_TGT}, robak=${LAST_RB_TGT}" >&2
  echo "Sprawdź ostatni merge: ${MERGE_DIR}" >&2
  exit 2
fi

if [[ "${RUN_MODE}" == "dataset_only" ]]; then
  log "=== DONE (DATASET-ONLY) ==="
  log "Dataset runs: ${dataset_experiment_ids[*]}"
  log "Merged experiment: ${MERGE_EXP_ID}"
  log "Histogram target osiągnięty (strict equal+unique, bins=${TARGET_BINS}, min_target_per_bin_rywak=${MIN_TARGET_PER_BIN_RYWAK}, min_target_per_bin_robak=${MIN_TARGET_PER_BIN_ROBAK})."
  log "Artifacts: ${MERGE_DIR}"
  if [[ -n "${DATASET_TIMING_SUMMARY_PATH}" ]]; then
    log "Dataset timing summary: ${DATASET_TIMING_SUMMARY_PATH}"
  fi
  log "Generated run configs: ${TMP_CONFIG_DIR}"
  exit 0
fi

IFS=',' read -r -a TRAIN_SEEDS <<< "${TRAIN_SEEDS_CSV}"
if [[ "${#TRAIN_SEEDS[@]}" -eq 0 ]]; then
  echo "Pusta lista --train-seeds" >&2
  exit 1
fi

log "=== TRAINING on merged dataset: ${MERGE_EXP_ID} ==="
for raw_seed in "${TRAIN_SEEDS[@]}"; do
  seed="$(echo "${raw_seed}" | xargs)"
  if [[ -z "${seed}" ]]; then
    continue
  fi
  if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
    echo "Nieprawidłowy seed: ${seed}" >&2
    exit 1
  fi

  log "Training seed=${seed} (Robak)"
  ros2 run ai_slam_ai train_model_robak --ros-args \
    -p out_dir:="${OUT_DIR}" \
    -p experiment_id:="${MERGE_EXP_ID}" \
    -p dataset_name:=dataset_robak.npz \
    -p model_name:="model_robak_seed${seed}.pt" \
    -p history_name:="train_history_robak_seed${seed}.json" \
    -p skip_if_model_exists:=false \
    -p seed:="${seed}" \
    -p split_strategy:=tail_holdout_no_shuffle \
    -p torch_deterministic:=true \
    -p write_experiment_metadata:=true

  log "Training seed=${seed} (Rywak)"
  ros2 run ai_slam_ai train_model_rywak --ros-args \
    -p out_dir:="${OUT_DIR}" \
    -p experiment_id:="${MERGE_EXP_ID}" \
    -p dataset_name:=dataset_rywak.npz \
    -p model_name:="model_rywak_seed${seed}.pt" \
    -p history_name:="train_history_rywak_seed${seed}.json" \
    -p skip_if_model_exists:=false \
    -p seed:="${seed}" \
    -p split_strategy:=tail_holdout_no_shuffle \
    -p torch_deterministic:=true \
    -p write_experiment_metadata:=true
done

log "=== DONE ==="
log "Dataset runs: ${dataset_experiment_ids[*]}"
log "Merged experiment: ${MERGE_EXP_ID}"
log "Histogram target osiągnięty (strict equal+unique, bins=${TARGET_BINS}, min_target_per_bin_rywak=${MIN_TARGET_PER_BIN_RYWAK}, min_target_per_bin_robak=${MIN_TARGET_PER_BIN_ROBAK})."
log "Artifacts: ${MERGE_DIR}"
if [[ -n "${DATASET_TIMING_SUMMARY_PATH}" ]]; then
  log "Dataset timing summary: ${DATASET_TIMING_SUMMARY_PATH}"
fi
log "Generated run configs: ${TMP_CONFIG_DIR}"
