#!/usr/bin/env bash
# Google Colab runner script for ECG Gate.
# Runs in the foreground by default (so Colab cells stay alive and stream progress),
# or in the background with --background.
#
# Usage:
#   bash scripts/colab_run.sh [mode] [options]
#
# Modes:
#   full-sensitivity   (Default) Exp 0 -> Exp 1 -> Test A sensitivity (excludes run_S64, run_S67)
#   full-primary       Exp 0 -> Exp 1 -> Test A primary (all 16,842 ECGs, no exclusion)
#   probe-only         Re-run Test A sensitivity directly (reuses cached embeddings if present)
#   pilot-smoke        Run pilot smoke tier (24 ECGs) on a pilot bundle
#   pilot-probe        Run pilot probe tier (560 ECGs) on a pilot bundle
#
# Options:
#   --background       Detach with nohup (useful if running from Colab terminal)
#   --force-overwrite  Move existing result JSON to .bak before running
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${HOME}/.ecg-gate-colab.env"

if [[ ! -f "$ENV_FILE" ]]; then
    printf '[ERROR] Colab config not found at %s. Run bash scripts/colab_setup.sh first.\n' "$ENV_FILE"
    exit 2
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

MODE="full-sensitivity"
BACKGROUND=0
FORCE_OVERWRITE=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        full-sensitivity|full-primary|probe-only|pilot-smoke|pilot-probe)
            MODE="$1"; shift ;;
        --background)
            BACKGROUND=1; shift ;;
        --foreground)
            BACKGROUND=0; shift ;;
        --force-overwrite)
            FORCE_OVERWRITE=1; shift ;;
        --worker)
            BACKGROUND=0; shift ;;
        -h|--help)
            sed -n '2,18p' "${BASH_SOURCE[0]}"
            exit 0 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

PYTHON="${ECG_PYTHON:-$(command -v python3 || command -v python)}"
SCRATCH_DIR="${ECG_SCRATCH_DIR:-/tmp/ecg-gate}"
LOG_DIR="${ECG_LOG_DIR:-${HOME}/ecg-gate-logs}"
RUNS_DIR="${ECG_RUNS_DIR:-${PROJECT_ROOT}/runs}"
mkdir -p "$LOG_DIR" "$RUNS_DIR"

export ECG_SCRATCH_DIR="$SCRATCH_DIR"
export ECG_CACHE_DIR="${ECG_CACHE_DIR:-${SCRATCH_DIR}/cache}"
export ECG_RUNS_DIR="$RUNS_DIR"
export FAIRSEQ_SIGNALS_REPO="${FAIRSEQ_SIGNALS_REPO:-${SCRATCH_DIR}/vendor/fairseq-signals-src}"
export ECG_PROBE_N_JOBS="${ECG_PROBE_N_JOBS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

LOG_FILE="${LOG_DIR}/colab-${MODE}.log"
PID_FILE="${LOG_DIR}/colab-${MODE}.pid"
EXIT_FILE="${LOG_DIR}/colab-${MODE}.exit"

if [[ "$BACKGROUND" -eq 1 ]]; then
    nohup bash "${BASH_SOURCE[0]}" "$MODE" --worker \
        $([[ "$FORCE_OVERWRITE" -eq 1 ]] && printf -- '--force-overwrite') \
        "${EXTRA_ARGS[@]}" >> "$LOG_FILE" 2>&1 < /dev/null &
    bg_pid=$!
    printf '%s\n' "$bg_pid" > "$PID_FILE"
    printf 'Started %s in background (PID %s).\nLog: %s\n' "$MODE" "$bg_pid" "$LOG_FILE"
    printf 'Monitor live with: tail -f %s\n' "$LOG_FILE"
    exit 0
fi

exec > >(tee -a "$LOG_FILE") 2>&1
printf '%s\n' "$$" > "$PID_FILE"
rm -f "$EXIT_FILE"

on_exit() {
    local status=$?
    printf '%s\n' "$status" > "$EXIT_FILE"
    if [[ "$status" -eq 0 ]]; then
        printf '\n[OK] Colab workflow (%s) finished successfully.\n' "$MODE"
    else
        printf '\n[ERROR] Colab workflow (%s) failed with exit code %s. Log: %s\n' "$MODE" "$status" "$LOG_FILE"
    fi
}
trap on_exit EXIT

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

# Preflight checks
if [[ -z "${ECG_FM_CKPT:-}" || ! -f "$ECG_FM_CKPT" ]]; then
    printf '[ERROR] ECG_FM_CKPT is missing or invalid: %s\n' "${ECG_FM_CKPT:-<unset>}"
    exit 2
fi
if [[ -z "${ECG_BUNDLE_ROOT:-}" || ! -f "${ECG_BUNDLE_ROOT}/manifest.parquet" ]] &&
   [[ -z "${MEDALCARE_ROOT:-}" || ! -d "${MEDALCARE_ROOT}/WP2_largeDataset_Noise" ]]; then
    printf '[ERROR] No valid dataset bundle or raw MedalCare directory configured.\n'
    exit 2
fi

log "Preflight: verifying PyTorch CUDA and fairseq_signals..."
"$PYTHON" -c '
import torch, fairseq_signals
print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Switch Colab runtime to a GPU.")
print("GPU:", torch.cuda.get_device_name(0))
'

maybe_backup_result() {
    local target="$1"
    if [[ -f "$target" ]]; then
        if [[ "$FORCE_OVERWRITE" -eq 1 ]]; then
            local bak="${target}.bak.$(date '+%Y%m%d_%H%M%S')"
            mv "$target" "$bak"
            log "Backed up existing result to $bak"
        else
            printf '[ERROR] Result file already exists: %s\n' "$target"
            printf 'Pass --force-overwrite or move it manually before rerunning.\n'
            exit 2
        fi
    fi
}

case "$MODE" in
    pilot-smoke)
        maybe_backup_result "${RUNS_DIR}/pilot_smoke.json"
        log "Running pilot smoke test..."
        "$PYTHON" -u "$PROJECT_ROOT/src/pilot.py" --tier smoke --device cuda "${EXTRA_ARGS[@]}"
        ;;
    pilot-probe)
        maybe_backup_result "${RUNS_DIR}/pilot_probe.json"
        log "Running pilot probe test..."
        "$PYTHON" -u "$PROJECT_ROOT/src/pilot.py" --tier probe --device cuda "${EXTRA_ARGS[@]}"
        ;;
    probe-only)
        RESULT_FILE="${RUNS_DIR}/A_results_sensitivity_without_S64_S67.json"
        maybe_backup_result "$RESULT_FILE"
        log "Running Test A sensitivity (excluding run_S64, run_S67)..."
        "$PYTHON" -u "$PROJECT_ROOT/src/probe.py" \
            --device cuda --ctrl-seeds 3 --exclude-run-ids run_S64 run_S67 "${EXTRA_ARGS[@]}"
        ;;
    full-sensitivity)
        RESULT_FILE="${RUNS_DIR}/A_results_sensitivity_without_S64_S67.json"
        maybe_backup_result "$RESULT_FILE"
        log "Step 1/3: Exp 0 smoke test"
        "$PYTHON" -u "$PROJECT_ROOT/src/exp0_smoke.py" --device cuda
        log "Step 2/3: Exp 1 index and mapping validation"
        "$PYTHON" -u "$PROJECT_ROOT/src/exp1_index.py"
        if ! grep -Fq -- '- **Exp 1 verdict: PASS**' "$PROJECT_ROOT/mapping_report.md"; then
            printf '[ERROR] Exp 1 did not report PASS in mapping_report.md.\n'
            exit 3
        fi
        log "Step 3/3: Test A sensitivity analysis (excluding run_S64, run_S67)"
        "$PYTHON" -u "$PROJECT_ROOT/src/probe.py" \
            --device cuda --ctrl-seeds 3 --exclude-run-ids run_S64 run_S67 "${EXTRA_ARGS[@]}"
        ;;
    full-primary)
        RESULT_FILE="${RUNS_DIR}/A_results.json"
        maybe_backup_result "$RESULT_FILE"
        log "Step 1/3: Exp 0 smoke test"
        "$PYTHON" -u "$PROJECT_ROOT/src/exp0_smoke.py" --device cuda
        log "Step 2/3: Exp 1 index and mapping validation"
        "$PYTHON" -u "$PROJECT_ROOT/src/exp1_index.py"
        if ! grep -Fq -- '- **Exp 1 verdict: PASS**' "$PROJECT_ROOT/mapping_report.md"; then
            printf '[ERROR] Exp 1 did not report PASS in mapping_report.md.\n'
            exit 3
        fi
        log "Step 3/3: Test A primary analysis (all indexed ECGs)"
        "$PYTHON" -u "$PROJECT_ROOT/src/probe.py" \
            --device cuda --ctrl-seeds 3 "${EXTRA_ARGS[@]}"
        ;;
esac

# Copy updated reports to Drive persist dir if active
if [[ -n "${ECG_PERSIST_DIR:-}" && -d "${ECG_PERSIST_DIR}" ]]; then
    for doc in "$PROJECT_ROOT/mapping_report.md" "$PROJECT_ROOT/notebook.md"; do
        if [[ -f "$doc" ]]; then
            cp -f "$doc" "${ECG_PERSIST_DIR}/"
        fi
    done
    log "Synced reports and results to ${ECG_PERSIST_DIR}"
fi
