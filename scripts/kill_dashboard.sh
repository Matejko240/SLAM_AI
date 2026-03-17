#!/bin/bash
set -euo pipefail

PORT="${1:-8765}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

collect_pids() {
    {
        pgrep -f "scripts/slam_dashboard.py" || true
        pgrep -f "$REPO_ROOT/scripts/slam_dashboard.py" || true
        if command -v fuser >/dev/null 2>&1; then
            fuser -n tcp "$PORT" 2>/dev/null | tr ' ' '\n'
        fi
    } | awk 'NF { print $1 }' | sort -u
}

mapfile -t PIDS < <(collect_pids)

if [[ "${#PIDS[@]}" -eq 0 ]]; then
    echo "Dashboard nie działa na porcie $PORT."
    exit 0
fi

echo "Zatrzymuję dashboard (PID: ${PIDS[*]})..."
kill "${PIDS[@]}" 2>/dev/null || true
sleep 1

STILL_RUNNING=()
for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
        STILL_RUNNING+=("$pid")
    fi
done

if [[ "${#STILL_RUNNING[@]}" -gt 0 ]]; then
    echo "Wymuszam zatrzymanie pozostałych procesów: ${STILL_RUNNING[*]}"
    kill -9 "${STILL_RUNNING[@]}" 2>/dev/null || true
fi

echo "Dashboard zatrzymany."
