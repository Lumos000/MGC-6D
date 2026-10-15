#!/usr/bin/env bash
set -euo pipefail

# HO3D data/evaluation with the TOYL/REAL275 fused-candidate policy.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FUSION_ADMISSION_POLICY=toyl_real275
export ALIGN_STRICT_BOTH=0
exec bash "$SCRIPT_DIR/run_ho3d_full.sh"
