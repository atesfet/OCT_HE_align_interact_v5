#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python src/coregistration_app.py --host 127.0.0.1 --port 8766
