#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
python3 scripts/slam_dashboard.py "$@"
