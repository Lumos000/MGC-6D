#!/usr/bin/env bash
set -euo pipefail

# Query-only entry point for a registry produced by
# run_ho3d_toyl_real275_full.sh (typically with RUN_QUERY=0).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_ho3d_query.sh"
