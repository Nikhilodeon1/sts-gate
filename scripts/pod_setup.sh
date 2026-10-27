#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${HOME}/ecg-gate-logs"
LOG_FILE="${LOG_DIR}/setup.log"
ENV_FILE="${HOME}/.ecg-gate.env"
SCRATCH_DIR="${ECG_SCRATCH_DIR:-/tmp/ecg-gate}"
VENV_DIR="${ECG_VENV_DIR:-${SCRATCH_DIR}/venv}"

mkdir -p "$LOG_DIR"
touch "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

on_error() {
    local status=$?
    printf '\n[ERROR] Setup failed at line %s (exit %s): %s\n' \
        "${BASH_LINENO[0]:-unknown}" "$status" "${BASH_COMMAND:-unknown}"
    printf '[ERROR] Full setup log: %s\n' "$LOG_FILE"
    exit "$status"
}
trap on_error ERR

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

python_has_cuda() {
    "$1" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' \
        >/dev/null 2>&1
}

prompt_asset_path() {
    local variable="$1"
    local prompt="$2"
    local candidate
    local entered=""

    candidate="${!variable:-}"
    if [[ -n "$candidate" ]]; then
        read -r -p "${prompt} [${candidate}]: " entered
        candidate="${entered:-$candidate}"
    else
        read -r -p "${prompt}: " candidate || candidate=""
    fi
    printf -v "$variable" '%s' "$candidate"
}

save_asset_config() {
    local medalcare_root="${MEDALCARE_ROOT:-}"
    local checkpoint="${ECG_FM_CKPT:-}"
    local temp_file

    if [[ -n "$medalcare_root" ]] &&
       [[ ! -d "$medalcare_root/WP2_largeDataset_Noise" ||
          ! -d "$medalcare_root/WP2_largeDataset_ParameterFiles" ]]; then
        log "[WARN] MedalCare root is not valid; it will not be saved: $medalcare_root"
        medalcare_root=""
    fi
    if [[ -n "$checkpoint" && ! -f "$checkpoint" ]]; then
        log "[WARN] Checkpoint file is not valid; it will not be saved: $checkpoint"
        checkpoint=""
    fi

    temp_file="$(mktemp "${HOME}/.ecg-gate.env.XXXXXX")"
    {
        printf 'export ECG_SCRATCH_DIR=%q\n' "$SCRATCH_DIR"
        printf 'export ECG_CACHE_DIR=%q\n' "${ECG_CACHE_DIR:-${SCRATCH_DIR}/cache}"
        printf 'export ECG_VENV_DIR=%q\n' "$VENV_DIR"
        printf 'export FAIRSEQ_SIGNALS_REPO=%q\n' \
            "${FAIRSEQ_SIGNALS_REPO:-${SCRATCH_DIR}/vendor/fairseq-signals-src}"
        printf 'export ECG_PROBE_N_JOBS=%q\n' "${ECG_PROBE_N_JOBS:-4}"
        if [[ -n "$medalcare_root" ]]; then
            printf 'export MEDALCARE_ROOT=%q\n' "$medalcare_root"
        fi
        if [[ -n "$checkpoint" ]]; then
            printf 'export ECG_FM_CKPT=%q\n' "$checkpoint"
        fi
    } > "$temp_file"
    chmod 600 "$temp_file"
    mv -f "$temp_file" "$ENV_FILE"
    log "Saved verified paths and scratch settings to $ENV_FILE"
}

log "ECG Gate setup starting"
log "Project: $PROJECT_ROOT"
log "Persistent setup log: $LOG_FILE"
log "Temporary scratch: $SCRATCH_DIR (cleared when the Pod restarts)"

if [[ ! -f "$PROJECT_ROOT/requirements.txt" || ! -f "$PROJECT_ROOT/src/install.py" ]]; then
    printf '[ERROR] Run this script from a cloned ECG Gate repository.\n'
    exit 2
fi

for command_name in df tee find; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        printf '[ERROR] Required command not found: %s\n' "$command_name"
        exit 2
    fi
done

log "Storage snapshot:"
df -h "$HOME" /tmp || true
if [[ -r /sys/fs/cgroup/memory.max ]]; then
    log "Cgroup memory.max: $(cat /sys/fs/cgroup/memory.max)"
elif [[ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
    log "Cgroup memory limit: $(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)"
else
    log "[WARN] Could not read a cgroup memory limit."
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv || true
else
    log "[WARN] nvidia-smi is unavailable."
fi

if [[ -f "$ENV_FILE" ]]; then
    log "Loading saved Pod configuration."
    # This file is created by this script with shell-escaped values and mode 600.
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

SCRATCH_DIR="${ECG_SCRATCH_DIR:-$SCRATCH_DIR}"
VENV_DIR="${ECG_VENV_DIR:-${SCRATCH_DIR}/venv}"
export ECG_SCRATCH_DIR="$SCRATCH_DIR"
export ECG_CACHE_DIR="${ECG_CACHE_DIR:-${SCRATCH_DIR}/cache}"
export ECG_VENV_DIR="$VENV_DIR"
export FAIRSEQ_SIGNALS_REPO="${FAIRSEQ_SIGNALS_REPO:-${SCRATCH_DIR}/vendor/fairseq-signals-src}"
export ECG_PROBE_N_JOBS="${ECG_PROBE_N_JOBS:-4}"
export PIP_NO_CACHE_DIR=1
export UV_NO_CACHE=1

mkdir -p "$SCRATCH_DIR" "$ECG_CACHE_DIR"
if [[ ! -w "$SCRATCH_DIR" ]]; then
    printf '[ERROR] Scratch directory is not writable: %s\n' "$SCRATCH_DIR"
    exit 2
fi
if [[ -d "$VENV_DIR" ]]; then
    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        printf '[ERROR] An incomplete venv exists at %s. Inspect it, then remove only that venv directory before rerunning setup.\n' "$VENV_DIR"
        exit 2
    fi
    if ! python_has_cuda "$VENV_DIR/bin/python"; then
        printf '[ERROR] Existing venv lacks usable CUDA PyTorch: %s\n' "$VENV_DIR"
        printf 'Do not delete it automatically. Check the Pod base Python and available disk before rebuilding.\n'
        exit 2
    fi
    log "Reusing CUDA-enabled temporary venv: $VENV_DIR"
else
    BASE_PYTHON="${ECG_BASE_PYTHON:-$(command -v python3 || command -v python || true)}"
    if [[ -z "$BASE_PYTHON" ]]; then
        printf '[ERROR] Could not find python3/python. Activate the Pod Python environment first.\n'
        exit 2
    fi
    if ! python_has_cuda "$BASE_PYTHON"; then
        printf '[ERROR] Base Python does not have CUDA-enabled PyTorch: %s\n' "$BASE_PYTHON"
        "$BASE_PYTHON" -c 'import torch; print("torch:", torch.__version__); print("cuda:", torch.cuda.is_available())' || true
        printf 'Stop here rather than installing a multi-GB Torch wheel automatically. See README.md for the Pod-tested fallback.\n'
        exit 2
    fi
    log "Creating venv that reuses the Pod CUDA PyTorch: $VENV_DIR"
    "$BASE_PYTHON" -m venv --system-site-packages "$VENV_DIR"
fi

PYTHON="${VENV_DIR}/bin/python"
if ! python_has_cuda "$PYTHON"; then
    printf '[ERROR] The temporary venv cannot access CUDA-enabled PyTorch.\n'
    exit 2
fi
log "Using Python: $("$PYTHON" --version 2>&1)"
"$PYTHON" -c 'import torch; print("torch:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0))'

export MEDALCARE_ROOT="${MEDALCARE_ROOT:-}"
export ECG_FM_CKPT="${ECG_FM_CKPT:-}"
if [[ -t 0 ]]; then
    prompt_asset_path MEDALCARE_ROOT \
        "MedalCare root (parent of WP2_largeDataset_Noise; blank if not mounted yet)"
    prompt_asset_path ECG_FM_CKPT \
        "Full ECG-FM checkpoint path (blank if not mounted yet)"
fi
save_asset_config

log "Installing dependencies with pip cache disabled."
"$PYTHON" "$PROJECT_ROOT/src/install.py" --python "$PYTHON"

log "Verifying imports and GPU after dependency installation."
"$PYTHON" -c '
import torch, numpy, pandas, sklearn, scipy, matplotlib, pyarrow, fairseq_signals
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable")
print("fairseq_signals:", fairseq_signals.__file__)
print("numpy/pandas/sklearn/scipy/matplotlib/pyarrow imports: OK")
assert torch.cuda.is_available(), "CUDA became unavailable after setup"
'

if [[ -d "${MEDALCARE_ROOT:-}/WP2_largeDataset_Noise" &&
      -d "${MEDALCARE_ROOT:-}/WP2_largeDataset_ParameterFiles" &&
      -f "${ECG_FM_CKPT:-}" ]]; then
    log "Dataset and checkpoint paths are configured and valid."
else
    log "[WARN] Python environment is ready, but MedalCare data and/or checkpoint paths are not configured."
    log "Run scripts/pod_status.sh for diagnostics. Set real paths, then rerun this setup or save them in $ENV_FILE."
fi

log "Setup completed successfully."
log "Next: run bash scripts/pod_status.sh, then bash scripts/pod_run.sh after all preflight checks pass."
