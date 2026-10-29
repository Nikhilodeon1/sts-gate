#!/usr/bin/env bash
# Lightweight pilot gate. Usage: bash scripts/pod_pilot.sh smoke|probe
# Needs the pilot bundle + ECG-FM checkpoint configured via scripts/pod_setup.sh.
# Does NOT run Exp 1 / Test A and does not use pod_run.sh.
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TIER="${1:-}"
if [[ "$TIER" != "smoke" && "$TIER" != "probe" ]]; then
    printf 'Usage: bash scripts/pod_pilot.sh smoke|probe\n'
    exit 2
fi

LOG_DIR="${HOME}/ecg-gate-logs"
mkdir -p "$LOG_DIR"
ENV_FILE="${HOME}/.ecg-gate.env"
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi
SCRATCH_DIR="${ECG_SCRATCH_DIR:-/tmp/ecg-gate}"
PYTHON="${ECG_VENV_DIR:-${SCRATCH_DIR}/venv}/bin/python"
export ECG_SCRATCH_DIR="$SCRATCH_DIR"
export ECG_CACHE_DIR="${ECG_CACHE_DIR:-${SCRATCH_DIR}/cache}"
export FAIRSEQ_SIGNALS_REPO="${FAIRSEQ_SIGNALS_REPO:-${SCRATCH_DIR}/vendor/fairseq-signals-src}"
export ECG_PROBE_N_JOBS="${ECG_PROBE_N_JOBS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

fail() { printf '[ERROR] %s\n' "$*"; exit 2; }

[[ -x "$PYTHON" ]] || fail "Venv Python not found: $PYTHON. /tmp may have been cleared; run scripts/pod_setup.sh."
[[ -n "${ECG_BUNDLE_ROOT:-}" && -f "$ECG_BUNDLE_ROOT/bundle.json" && -f "$ECG_BUNDLE_ROOT/manifest.parquet" ]] \
    || fail "Pilot bundle not configured (ECG_BUNDLE_ROOT). Run scripts/pod_setup.sh with the pilot ZIP."
export ECG_BUNDLE_ROOT
[[ -n "${ECG_FM_CKPT:-}" && -f "$ECG_FM_CKPT" ]] || fail "ECG-FM checkpoint unset or missing: ${ECG_FM_CKPT:-<unset>}"
export ECG_FM_CKPT

"$PYTHON" -c '
import torch, fairseq_signals
assert torch.cuda.is_available(), "CUDA is unavailable"
print("torch:", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
'

printf '[%s] Pilot tier: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$TIER"
"$PYTHON" -u "$PROJECT_ROOT/src/pilot.py" --tier "$TIER" --device cuda "${@:2}" 2>&1 | tee -a "${LOG_DIR}/pilot-${TIER}.log"
