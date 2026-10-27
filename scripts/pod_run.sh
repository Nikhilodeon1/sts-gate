#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="${PROJECT_ROOT}/scripts/pod_run.sh"
LOG_DIR="${HOME}/ecg-gate-logs"
LOG_FILE="${LOG_DIR}/test-a-sensitivity.log"
PID_FILE="${LOG_DIR}/test-a-sensitivity.pid"
EXIT_FILE="${LOG_DIR}/test-a-sensitivity.exit"
LOCK_DIR="${LOG_DIR}/test-a-sensitivity.lock"
ENV_FILE="${HOME}/.ecg-gate.env"
RESULT_FILE="${PROJECT_ROOT}/runs/A_results_sensitivity_without_S64_S67.json"
MODE="${1:-}"
LOCK_OWNED=0

mkdir -p "$LOG_DIR"
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi
SCRATCH_DIR="${ECG_SCRATCH_DIR:-/tmp/ecg-gate}"
VENV_DIR="${ECG_VENV_DIR:-${SCRATCH_DIR}/venv}"
MEDALCARE_ROOT="${MEDALCARE_ROOT:-}"
ECG_FM_CKPT="${ECG_FM_CKPT:-}"

if [[ "$MODE" != "--worker" ]]; then
    if [[ "$MODE" != "" && "$MODE" != "--foreground" ]]; then
        printf 'Usage: bash scripts/pod_run.sh [--foreground]\n'
        exit 2
    fi

    if [[ "$MODE" == "--foreground" ]]; then
        exec > >(tee -a "$LOG_FILE") 2>&1
        MODE="--worker"
    else
        if [[ -f "$PID_FILE" ]]; then
            old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
            if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
                printf '[ERROR] Recorded run PID %s is still alive. Check it with scripts/pod_status.sh.\n' "$old_pid"
                exit 2
            fi
        fi
        if [[ -f "$RESULT_FILE" ]]; then
            printf '[ERROR] A sensitivity result already exists: %s\n' "$RESULT_FILE"
            printf 'Inspect it with scripts/pod_status.sh. Move/copy it before any intentional rerun; this script will not overwrite it.\n'
            exit 2
        fi
        if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
            printf '[ERROR] Venv Python not found: %s/bin/python. Run scripts/pod_setup.sh first.\n' "$VENV_DIR"
            exit 2
        fi
        if [[ -z "$MEDALCARE_ROOT" ||
              ! -d "$MEDALCARE_ROOT/WP2_largeDataset_Noise" ||
              ! -d "$MEDALCARE_ROOT/WP2_largeDataset_ParameterFiles" ]]; then
            printf '[ERROR] MedalCare paths are unset or invalid. Configure them with scripts/pod_setup.sh first.\n'
            exit 2
        fi
        if [[ -z "$ECG_FM_CKPT" || ! -f "$ECG_FM_CKPT" ]]; then
            printf '[ERROR] ECG-FM checkpoint is unset or invalid. Configure it with scripts/pod_setup.sh first.\n'
            exit 2
        fi
        if ! "${VENV_DIR}/bin/python" -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
            printf '[ERROR] Configured Python cannot access CUDA. Run scripts/pod_status.sh and inspect setup before launching.\n'
            exit 2
        fi
        nohup bash "$SCRIPT_PATH" --worker >> "$LOG_FILE" 2>&1 < /dev/null &
        run_pid=$!
        printf '%s\n' "$run_pid" > "${PID_FILE}.tmp"
        mv -f "${PID_FILE}.tmp" "$PID_FILE"
        printf 'Started guarded Test A sensitivity workflow.\n'
        printf 'PID: %s\nLog: %s\n' "$run_pid" "$LOG_FILE"
        printf 'Check progress with: bash %s/scripts/pod_status.sh\n' "$PROJECT_ROOT"
        exit 0
    fi
fi

if [[ "$MODE" != "--worker" ]]; then
    printf 'Internal invocation error.\n'
    exit 2
fi

on_error() {
    local status=$?
    printf '\n[ERROR] Workflow failed at line %s (exit %s): %s\n' \
        "${BASH_LINENO[0]:-unknown}" "$status" "${BASH_COMMAND:-unknown}"
    printf '[ERROR] See complete log: %s\n' "$LOG_FILE"
    exit "$status"
}
on_exit() {
    local status=$?
    if [[ "$LOCK_OWNED" -eq 1 ]]; then
        printf '%s\n' "$status" > "${EXIT_FILE}.tmp"
        mv -f "${EXIT_FILE}.tmp" "$EXIT_FILE"
        if [[ -d "$LOCK_DIR" ]]; then
            rm -f "$LOCK_DIR/pid"
            rmdir "$LOCK_DIR" 2>/dev/null || true
        fi
    fi
    if [[ "$status" -eq 0 ]]; then
        printf '\n[OK] Workflow finished successfully.\n'
    else
        printf '\n[ERROR] Workflow ended with exit code %s. Review %s\n' "$status" "$LOG_FILE"
    fi
    rm -f "$step_log"
}
trap on_error ERR
trap on_exit EXIT

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
step_log="${TMPDIR:-/tmp}/ecg-gate-step-$$.log"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    existing_pid="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
    printf '[ERROR] Another workflow holds %s (PID %s).\n' "$LOCK_DIR" "${existing_pid:-unknown}"
    printf 'Inspect with scripts/pod_status.sh; stale locks require manual review, not automatic deletion.\n'
    exit 2
fi
LOCK_OWNED=1
printf '%s\n' "$$" > "$LOCK_DIR/pid"
printf '%s\n' "$$" > "$PID_FILE"
rm -f "$EXIT_FILE"

if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi
SCRATCH_DIR="${ECG_SCRATCH_DIR:-/tmp/ecg-gate}"
VENV_DIR="${ECG_VENV_DIR:-${SCRATCH_DIR}/venv}"
PYTHON="${VENV_DIR}/bin/python"
MEDALCARE_ROOT="${MEDALCARE_ROOT:-}"
ECG_FM_CKPT="${ECG_FM_CKPT:-}"
export ECG_SCRATCH_DIR="$SCRATCH_DIR"
export ECG_CACHE_DIR="${ECG_CACHE_DIR:-${SCRATCH_DIR}/cache}"
export FAIRSEQ_SIGNALS_REPO="${FAIRSEQ_SIGNALS_REPO:-${SCRATCH_DIR}/vendor/fairseq-signals-src}"
export ECG_PROBE_N_JOBS="${ECG_PROBE_N_JOBS:-4}"
export MEDALCARE_ROOT ECG_FM_CKPT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

run_step() {
    local title="$1"
    shift
    log "START: $title"
    : > "$step_log"
    if "$@" > "$step_log" 2>&1; then
        cat "$step_log"
        log "PASS: $title"
    else
        local status=$?
        cat "$step_log"
        printf '[ERROR] %s failed with exit code %s. Captured command: ' "$title" "$status"
        printf '%q ' "$@"
        printf '\n'
        return "$status"
    fi
}

log "Guarded Test A workflow started."
log "Repository: $PROJECT_ROOT"
log "Python: $PYTHON"
log "Persistent log: $LOG_FILE"
log "Scratch: $SCRATCH_DIR"

if [[ -f "$RESULT_FILE" ]]; then
    printf '[ERROR] Result already exists; refusing to overwrite: %s\n' "$RESULT_FILE"
    exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
    printf '[ERROR] Venv Python not found: %s. /tmp may have been cleared; run scripts/pod_setup.sh.\n' "$PYTHON"
    exit 2
fi
if [[ -z "$MEDALCARE_ROOT" || ! -d "$MEDALCARE_ROOT/WP2_largeDataset_Noise" ||
      ! -d "$MEDALCARE_ROOT/WP2_largeDataset_ParameterFiles" ]]; then
    printf '[ERROR] MedalCare dataset paths are unset or invalid: %s\n' "${MEDALCARE_ROOT:-<unset>}"
    printf 'Run scripts/pod_setup.sh and configure the real dataset path. Test A was not started.\n'
    exit 2
fi
if [[ -z "$ECG_FM_CKPT" || ! -f "$ECG_FM_CKPT" ]]; then
    printf '[ERROR] ECG-FM checkpoint path is unset or invalid: %s\n' "${ECG_FM_CKPT:-<unset>}"
    printf 'Run scripts/pod_setup.sh and configure the real checkpoint path. Test A was not started.\n'
    exit 2
fi

run_step "Python/CUDA/fairseq preflight" "$PYTHON" -c '
import torch, fairseq_signals
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
print("GPU:", torch.cuda.get_device_name(0))
print("fairseq_signals:", fairseq_signals.__file__)
'

run_step "Exp 0 smoke test" "$PYTHON" -u "$PROJECT_ROOT/src/exp0_smoke.py" --device cuda
if ! grep -Fq '"verdict": "PASS"' "$step_log"; then
    printf '[ERROR] Exp 0 did not report a PASS verdict. Test A was not started.\n'
    exit 3
fi

run_step "Exp 1 index and mapping validation" "$PYTHON" -u "$PROJECT_ROOT/src/exp1_index.py"
if ! grep -Fq -- '- **Exp 1 verdict: PASS**' "$PROJECT_ROOT/mapping_report.md"; then
    printf '[ERROR] mapping_report.md does not report Exp 1 PASS. Test A was not started.\n'
    printf 'Inspect %s/mapping_report.md and the log above before proceeding.\n' "$PROJECT_ROOT"
    exit 3
fi

run_step "Test A sensitivity analysis" "$PYTHON" -u "$PROJECT_ROOT/src/probe.py" \
    --device cuda --ctrl-seeds 3 --exclude-run-ids run_S64 run_S67

if [[ ! -s "$RESULT_FILE" ]]; then
    printf '[ERROR] Probe exited without creating a non-empty result file: %s\n' "$RESULT_FILE"
    exit 4
fi
"$PYTHON" - "$RESULT_FILE" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    result = json.load(f)
if result.get("excluded_run_ids") != ["run_S64", "run_S67"]:
    raise SystemExit(f"Unexpected excluded_run_ids in result: {result.get('excluded_run_ids')!r}")
if result.get("verdict") not in {"PASS", "FAIL_STOP", "MARGINAL_CONTINUE", "MARGINAL_REPORT"}:
    raise SystemExit(f"Missing/invalid Test A verdict: {result.get('verdict')!r}")
print("Result JSON verified.")
print("Test A verdict:", result["verdict"])
print("Excluded run IDs:", result["excluded_run_ids"])
print("Indexed records:", result.get("n_index_records"))
print("Excluded records:", result.get("n_excluded_records"))
PY

log "Test A sensitivity result: $RESULT_FILE"
log "Workflow completed. Check the verdict and metrics in the result JSON."
