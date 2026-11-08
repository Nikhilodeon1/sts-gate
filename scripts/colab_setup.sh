#!/usr/bin/env bash
# Google Colab setup script for ECG Gate.
# Leaves the JupyterHub pod_* scripts untouched.
#
# Usage (in Colab terminal or notebook cell with !):
#   bash scripts/colab_setup.sh [options]
#
# Options:
#   --bundle /path/to/MedalCare-XL-pod-bundle.zip   Path to uploaded/Drive bundle ZIP
#   --ckpt   /path/to/mimic_iv_ecg_physionet_pretrained.pt   Path to checkpoint
#   --download-ckpt                                 Download checkpoint from HuggingFace if missing
#   --persist-dir /content/drive/MyDrive/ecg-gate   Persist cache, logs, and runs to Google Drive
#   --venv /tmp/ecg-gate/venv                       Optional virtualenv path (default: Colab system Python)
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${HOME}/.ecg-gate-colab.env"
SCRATCH_DIR="${ECG_SCRATCH_DIR:-/tmp/ecg-gate}"

# Load previous saved Colab config if present
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

BUNDLE_ARG="${ECG_BUNDLE_ARCHIVE:-}"
CKPT_ARG="${ECG_FM_CKPT:-}"
MEDALCARE_ARG="${MEDALCARE_ROOT:-}"
PERSIST_DIR="${ECG_PERSIST_DIR:-}"
VENV_DIR="${ECG_VENV_DIR:-}"
DOWNLOAD_CKPT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bundle)
            BUNDLE_ARG="$2"; shift 2 ;;
        --ckpt)
            CKPT_ARG="$2"; shift 2 ;;
        --medalcare-root)
            MEDALCARE_ARG="$2"; shift 2 ;;
        --persist-dir)
            PERSIST_DIR="$2"; shift 2 ;;
        --venv)
            VENV_DIR="$2"; shift 2 ;;
        --download-ckpt)
            DOWNLOAD_CKPT=1; shift ;;
        -h|--help)
            sed -n '2,14p' "${BASH_SOURCE[0]}"
            exit 0 ;;
        *)
            printf '[ERROR] Unknown argument: %s\n' "$1"
            exit 2 ;;
    esac
done

# Auto-detect mounted Google Drive if --persist-dir was not passed
if [[ -z "$PERSIST_DIR" && -d "/content/drive/MyDrive" ]]; then
    PERSIST_DIR="/content/drive/MyDrive/ecg-gate"
fi

if [[ -n "$PERSIST_DIR" ]]; then
    mkdir -p "$PERSIST_DIR/logs" "$PERSIST_DIR/cache" "$PERSIST_DIR/runs" "$PERSIST_DIR/ckpt"
    LOG_DIR="${PERSIST_DIR}/logs"
    CACHE_DIR="${ECG_CACHE_DIR:-${PERSIST_DIR}/cache}"
    RUNS_DIR="${ECG_RUNS_DIR:-${PERSIST_DIR}/runs}"
else
    LOG_DIR="${HOME}/ecg-gate-logs"
    mkdir -p "$LOG_DIR"
    CACHE_DIR="${ECG_CACHE_DIR:-${SCRATCH_DIR}/cache}"
    RUNS_DIR="${ECG_RUNS_DIR:-${PROJECT_ROOT}/runs}"
fi

LOG_FILE="${LOG_DIR}/colab_setup.log"
mkdir -p "$LOG_DIR" "$SCRATCH_DIR" "$CACHE_DIR" "$RUNS_DIR"
touch "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

on_error() {
    local status=$?
    printf '\n[ERROR] Colab setup failed at line %s (exit %s): %s\n' \
        "${BASH_LINENO[0]:-unknown}" "$status" "${BASH_COMMAND:-unknown}"
    printf '[ERROR] Full setup log: %s\n' "$LOG_FILE"
    exit "$status"
}
trap on_error ERR

log "ECG Gate Colab setup starting"
log "Project root: $PROJECT_ROOT"
log "Scratch dir:  $SCRATCH_DIR"
log "Cache dir:    $CACHE_DIR"
log "Runs dir:     $RUNS_DIR"
if [[ -n "$PERSIST_DIR" ]]; then
    log "Drive persistence enabled at: $PERSIST_DIR"
else
    log "[INFO] Google Drive not detected at /content/drive/MyDrive; using local runtime storage."
fi

log "Storage & GPU snapshot:"
df -h / /tmp "${PERSIST_DIR:-/tmp}" 2>/dev/null | uniq || true
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
else
    log "[WARN] nvidia-smi not found. Make sure Colab Runtime -> Change runtime type is set to a GPU (T4 / L4 / G4 / A100)."
fi

# Select Python (Colab system Python by default, or optional venv)
if [[ -n "$VENV_DIR" ]]; then
    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        BASE_PY="$(command -v python3 || command -v python)"
        log "Creating virtualenv with --system-site-packages at $VENV_DIR"
        "$BASE_PY" -m venv --system-site-packages "$VENV_DIR"
    fi
    PYTHON="$VENV_DIR/bin/python"
else
    PYTHON="$(command -v python3 || command -v python)"
fi

log "Using Python: $("$PYTHON" --version 2>&1) ($PYTHON)"
"$PYTHON" -c '
import torch
print("torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
'

export ECG_SCRATCH_DIR="$SCRATCH_DIR"
export ECG_CACHE_DIR="$CACHE_DIR"
export ECG_RUNS_DIR="$RUNS_DIR"
export FAIRSEQ_SIGNALS_REPO="${FAIRSEQ_SIGNALS_REPO:-${SCRATCH_DIR}/vendor/fairseq-signals-src}"
export PYTHONPATH="${FAIRSEQ_SIGNALS_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export ECG_PROBE_N_JOBS="${ECG_PROBE_N_JOBS:-4}"
export PIP_NO_CACHE_DIR=1

# Auto-discover bundle ZIP if not explicitly given
if [[ -z "$BUNDLE_ARG" ]]; then
    for candidate in \
        "/content/MedalCare-XL-pod-bundle.zip" \
        "/content/drive/MyDrive/MedalCare-XL-pod-bundle.zip" \
        "/content/drive/MyDrive/ecg-gate/MedalCare-XL-pod-bundle.zip" \
        "/content/MedalCare-XL-pilot.zip" \
        "/content/drive/MyDrive/MedalCare-XL-pilot.zip" \
        "/content/drive/MyDrive/ecg-gate/MedalCare-XL-pilot.zip"; do
        if [[ -f "$candidate" ]]; then
            BUNDLE_ARG="$candidate"
            log "Auto-detected bundle archive: $BUNDLE_ARG"
            break
        fi
    done
fi

# Auto-discover checkpoint if not explicitly given
if [[ -z "$CKPT_ARG" ]]; then
    for candidate in \
        "/content/mimic_iv_ecg_physionet_pretrained.pt" \
        "${PROJECT_ROOT}/ckpt/mimic_iv_ecg_physionet_pretrained.pt" \
        "/content/drive/MyDrive/mimic_iv_ecg_physionet_pretrained.pt" \
        "/content/drive/MyDrive/ecg-gate/ckpt/mimic_iv_ecg_physionet_pretrained.pt" \
        "/content/drive/MyDrive/ecg-gate/mimic_iv_ecg_physionet_pretrained.pt" \
        "${SCRATCH_DIR}/ckpt/mimic_iv_ecg_physionet_pretrained.pt"; do
        if [[ -f "$candidate" ]]; then
            CKPT_ARG="$candidate"
            log "Auto-detected checkpoint: $CKPT_ARG"
            break
        fi
    done
fi

# Optionally download checkpoint directly from HuggingFace (fast on Colab network)
EXPECTED_CKPT_SHA="4d0142bcb485eb9f0c7845e0c19ff3463f6ae9d0e458eab69136efe90ceb9b7e"
HF_CKPT_URL="https://huggingface.co/wanglab/ecg-fm/resolve/main/mimic_iv_ecg_physionet_pretrained.pt"

if [[ "$DOWNLOAD_CKPT" -eq 1 && ( -z "$CKPT_ARG" || ! -f "$CKPT_ARG" ) ]]; then
    if [[ -n "$PERSIST_DIR" ]]; then
        CKPT_ARG="${PERSIST_DIR}/ckpt/mimic_iv_ecg_physionet_pretrained.pt"
    else
        mkdir -p "${SCRATCH_DIR}/ckpt"
        CKPT_ARG="${SCRATCH_DIR}/ckpt/mimic_iv_ecg_physionet_pretrained.pt"
    fi
    log "Downloading ECG-FM checkpoint from HuggingFace to $CKPT_ARG ..."
    curl -L --fail -C - -o "$CKPT_ARG" "$HF_CKPT_URL"
fi

if [[ -n "$CKPT_ARG" && -f "$CKPT_ARG" ]]; then
    if command -v sha256sum >/dev/null 2>&1; then
        actual_sha="$(sha256sum "$CKPT_ARG" | awk '{print $1}')"
        log "Checkpoint SHA-256: $actual_sha"
        if [[ "$actual_sha" != "$EXPECTED_CKPT_SHA" ]]; then
            printf '[ERROR] Checkpoint SHA-256 mismatch! Expected %s, got %s (file may be truncated).\n' \
                "$EXPECTED_CKPT_SHA" "$actual_sha"
            exit 2
        fi
    fi
fi

# Extract dataset bundle to fast local /tmp scratch
ECG_BUNDLE_ROOT="${ECG_BUNDLE_ROOT:-}"
if [[ -n "$BUNDLE_ARG" ]]; then
    if [[ ! -f "$BUNDLE_ARG" ]]; then
        printf '[ERROR] Specified bundle archive not found: %s\n' "$BUNDLE_ARG"
        exit 2
    fi
    bundle_dest="${SCRATCH_DIR}/medalcare_bundle"
    archive_sha="$(sha256sum "$BUNDLE_ARG" | awk '{print $1}')"
    log "Bundle archive: $BUNDLE_ARG (SHA-256: $archive_sha)"

    if [[ -f "$bundle_dest/bundle.json" && -f "$bundle_dest/manifest.parquet" && -f "$bundle_dest/archive.sha256" ]] &&
       [[ "$(cat "$bundle_dest/archive.sha256")" == "$archive_sha" ]]; then
        log "Using already-extracted bundle at $bundle_dest"
    else
        rm -rf "$bundle_dest"
        staging_dir="$(mktemp -d "${SCRATCH_DIR}/medalcare_bundle.XXXXXX")"
        log "Extracting bundle to $bundle_dest ..."
        "$PYTHON" - "$BUNDLE_ARG" "$staging_dir" "$archive_sha" <<'PY'
import pathlib, sys, zipfile
archive_path = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2]).resolve()
sha = sys.argv[3]
with zipfile.ZipFile(archive_path) as bundle:
    for member in bundle.infolist():
        target = (destination / member.filename).resolve()
        if target != destination and destination not in target.parents:
            raise ValueError(f"Unsafe bundle path: {member.filename!r}")
    bundle.extractall(destination)
required = (destination / "bundle.json", destination / "manifest.parquet")
if not all(path.is_file() for path in required):
    raise ValueError("Bundle archive must contain bundle.json and manifest.parquet.")
(destination / "archive.sha256").write_text(sha + "\n", encoding="ascii")
PY
        mv "$staging_dir" "$bundle_dest"
        log "Bundle extracted and verified."
    fi
    ECG_BUNDLE_ROOT="$bundle_dest"
fi

log "Installing Python dependencies and fairseq-signals..."
"$PYTHON" "$PROJECT_ROOT/src/install.py" --python "$PYTHON"

# Save configuration
{
    printf 'export ECG_PYTHON=%q\n' "$PYTHON"
    printf 'export ECG_SCRATCH_DIR=%q\n' "$SCRATCH_DIR"
    printf 'export ECG_CACHE_DIR=%q\n' "$CACHE_DIR"
    printf 'export ECG_RUNS_DIR=%q\n' "$RUNS_DIR"
    printf 'export ECG_LOG_DIR=%q\n' "$LOG_DIR"
    printf 'export FAIRSEQ_SIGNALS_REPO=%q\n' "$FAIRSEQ_SIGNALS_REPO"
    printf 'export PYTHONPATH=%q\n' "${FAIRSEQ_SIGNALS_REPO}"
    printf 'export ECG_PROBE_N_JOBS=%q\n' "$ECG_PROBE_N_JOBS"
    if [[ -n "$PERSIST_DIR" ]]; then
        printf 'export ECG_PERSIST_DIR=%q\n' "$PERSIST_DIR"
    fi
    if [[ -n "$VENV_DIR" ]]; then
        printf 'export ECG_VENV_DIR=%q\n' "$VENV_DIR"
    fi
    if [[ -n "$BUNDLE_ARG" && -f "$BUNDLE_ARG" ]]; then
        printf 'export ECG_BUNDLE_ARCHIVE=%q\n' "$BUNDLE_ARG"
        printf 'export ECG_BUNDLE_ROOT=%q\n' "$ECG_BUNDLE_ROOT"
    fi
    if [[ -n "$MEDALCARE_ARG" && -d "$MEDALCARE_ARG" ]]; then
        printf 'export MEDALCARE_ROOT=%q\n' "$MEDALCARE_ARG"
    fi
    if [[ -n "$CKPT_ARG" && -f "$CKPT_ARG" ]]; then
        printf 'export ECG_FM_CKPT=%q\n' "$CKPT_ARG"
    fi
} > "${ENV_FILE}.tmp"
chmod 600 "${ENV_FILE}.tmp"
mv -f "${ENV_FILE}.tmp" "$ENV_FILE"
log "Saved Colab configuration to $ENV_FILE"

log "Setup complete! Run 'bash scripts/colab_status.sh' to verify or 'bash scripts/colab_run.sh' to start."
