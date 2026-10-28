#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${HOME}/ecg-gate-logs"
ENV_FILE="${HOME}/.ecg-gate.env"
PID_FILE="${LOG_DIR}/test-a-sensitivity.pid"
EXIT_FILE="${LOG_DIR}/test-a-sensitivity.exit"
RUN_LOG="${LOG_DIR}/test-a-sensitivity.log"
RESULT_FILE="${PROJECT_ROOT}/runs/A_results_sensitivity_without_S64_S67.json"
SCRATCH_DIR="${ECG_SCRATCH_DIR:-/tmp/ecg-gate}"
VENV_DIR="${ECG_VENV_DIR:-${SCRATCH_DIR}/venv}"
PYTHON="${VENV_DIR}/bin/python"
ERRORS=0

section() { printf '\n===== %s =====\n' "$*"; }
ok() { printf '[OK] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }
bad() { printf '[MISSING/ERROR] %s\n' "$*"; ERRORS=$((ERRORS + 1)); }

if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi
SCRATCH_DIR="${ECG_SCRATCH_DIR:-$SCRATCH_DIR}"
VENV_DIR="${ECG_VENV_DIR:-${SCRATCH_DIR}/venv}"
PYTHON="${VENV_DIR}/bin/python"
MEDALCARE_ROOT="${MEDALCARE_ROOT:-}"
ECG_FM_CKPT="${ECG_FM_CKPT:-}"
ECG_BUNDLE_ROOT="${ECG_BUNDLE_ROOT:-}"
ECG_BUNDLE_ARCHIVE="${ECG_BUNDLE_ARCHIVE:-}"

section "ECG Gate status"
printf 'Time: %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
printf 'Repository: %s\n' "$PROJECT_ROOT"
printf 'Host: %s\n' "$(hostname 2>/dev/null || printf unknown)"

section "Storage and resource limits"
df -h "$HOME" /tmp 2>&1 || warn "Could not read disk usage."
if [[ -r /sys/fs/cgroup/memory.max ]]; then
    printf 'cgroup memory.max: %s\n' "$(cat /sys/fs/cgroup/memory.max)"
elif [[ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
    printf 'cgroup memory limit: %s\n' "$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)"
else
    warn "Cgroup memory limit is not readable."
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,utilization.gpu,memory.used,power.draw --format=csv 2>&1 ||
        warn "nvidia-smi query failed."
else
    warn "nvidia-smi is not installed or not on PATH."
fi

section "Saved configuration and assets"
printf 'Config file: %s\n' "$ENV_FILE"
[[ -f "$ENV_FILE" ]] && ok "Saved configuration exists." || warn "No saved configuration; run scripts/pod_setup.sh."
printf 'Scratch: %s\n' "$SCRATCH_DIR"
printf 'Venv: %s\n' "$VENV_DIR"
printf 'MEDALCARE_ROOT: %s\n' "${MEDALCARE_ROOT:-<unset>}"
printf 'ECG_BUNDLE_ROOT: %s\n' "${ECG_BUNDLE_ROOT:-<unset>}"
printf 'ECG_BUNDLE_ARCHIVE: %s\n' "${ECG_BUNDLE_ARCHIVE:-<unset>}"
printf 'ECG_FM_CKPT: %s\n' "${ECG_FM_CKPT:-<unset>}"
raw_noise_ready=0
raw_params_ready=0
bundle_dataset_ready=0
if [[ -n "$MEDALCARE_ROOT" && -d "$MEDALCARE_ROOT/WP2_largeDataset_Noise" ]]; then
    ok "Raw MedalCare noise directory exists."
    raw_noise_ready=1
else
    warn "Raw MedalCare noise directory is not configured/found."
fi
if [[ -n "$MEDALCARE_ROOT" && -d "$MEDALCARE_ROOT/WP2_largeDataset_ParameterFiles" ]]; then
    ok "Raw MedalCare parameter directory exists."
    raw_params_ready=1
else
    warn "Raw MedalCare parameter directory is not configured/found."
fi
if [[ -n "$ECG_BUNDLE_ARCHIVE" && -f "$ECG_BUNDLE_ARCHIVE" ]]; then
    stat -c 'Bundle archive: %n | size=%s bytes | modified=%y' "$ECG_BUNDLE_ARCHIVE" 2>/dev/null ||
        ls -lh "$ECG_BUNDLE_ARCHIVE"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$ECG_BUNDLE_ARCHIVE"
    fi
elif [[ -n "$ECG_BUNDLE_ARCHIVE" ]]; then
    warn "Saved bundle archive path does not exist: $ECG_BUNDLE_ARCHIVE"
fi
if [[ -n "$ECG_BUNDLE_ROOT" &&
      -f "$ECG_BUNDLE_ROOT/bundle.json" &&
      -f "$ECG_BUNDLE_ROOT/manifest.parquet" ]]; then
    ok "Extracted ECG bundle metadata and manifest exist."
    if [[ -x "$PYTHON" ]]; then
        "$PYTHON" - "$ECG_BUNDLE_ROOT/bundle.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)
print("Bundle records:", data.get("record_count"))
print("Bundle quantization:", data.get("waveform", {}).get("quantization"))
print("Bundle estimated source-preprocess RMSE:", data.get("quantization_stats", {}).get("rmse"))
PY
        if [[ $? -eq 0 ]]; then
            (
                cd "$PROJECT_ROOT"
                ECG_BUNDLE_ROOT="$ECG_BUNDLE_ROOT" \
                    "$PYTHON" - <<'PY'
import sys
sys.path.insert(0, "src")
from data import build_index
index = build_index()
print("Resolved bundle index rows:", len(index))
print("Resolved bundle groups:", sorted(index["group"].unique()))
PY
            )
            [[ "$?" -eq 0 ]] || bad "Bundle shards/index validation failed."
        else
            bad "Bundle metadata JSON is invalid."
        fi
    fi
    bundle_dataset_ready=1
elif [[ -n "$ECG_BUNDLE_ROOT" ]]; then
    warn "Configured extracted bundle is missing bundle.json or manifest.parquet."
fi
if { [[ "$raw_noise_ready" -eq 1 && "$raw_params_ready" -eq 1 ]]; } ||
   [[ "$bundle_dataset_ready" -eq 1 ]]; then
    ok "At least one complete dataset source is configured."
else
    bad "Neither a complete raw dataset nor a compact ECG bundle is available."
fi
if [[ -n "$ECG_FM_CKPT" && -f "$ECG_FM_CKPT" ]]; then
    ok "ECG-FM checkpoint exists."
else
    bad "ECG-FM checkpoint is not configured/found."
fi

section "Python environment"
if [[ -x "$PYTHON" ]]; then
    "$PYTHON" - <<'PY'
try:
    import torch
    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except Exception as exc:
    print(f"torch import failed: {type(exc).__name__}: {exc}")
    raise SystemExit(1)
try:
    import fairseq_signals
    print(f"fairseq_signals: {fairseq_signals.__file__}")
except Exception as exc:
    print(f"fairseq_signals import failed: {type(exc).__name__}: {exc}")
    raise SystemExit(2)
PY
    py_status=$?
    if [[ "$py_status" -eq 0 ]]; then
        ok "Python, CUDA Torch, and fairseq-signals imports passed."
    else
        bad "Python environment check failed (exit $py_status). See error above."
    fi
else
    bad "Venv Python not found: $PYTHON. /tmp may have been cleared; run scripts/pod_setup.sh."
fi

section "Index and reports"
if [[ -f "${ECG_CACHE_DIR:-${SCRATCH_DIR}/cache}/index.parquet" ]]; then
    stat -c 'Index: %n | size=%s bytes | modified=%y' "${ECG_CACHE_DIR:-${SCRATCH_DIR}/cache}/index.parquet" 2>/dev/null ||
        ls -lh "${ECG_CACHE_DIR:-${SCRATCH_DIR}/cache}/index.parquet"
else
    warn "No index cache found. Exp 1 will rebuild it."
fi
for report in "$PROJECT_ROOT/mapping_report.md" "$PROJECT_ROOT/notebook.md"; do
    if [[ -f "$report" ]]; then
        stat -c '%n | modified=%y' "$report" 2>/dev/null || ls -l "$report"
    else
        warn "Report not present: $report"
    fi
done
if [[ -f "$PROJECT_ROOT/mapping_report.md" ]]; then
    grep -E 'Generated:|Exp 1 verdict:|Minimum expected group/parameter coverage:|Total ECGs indexed:' \
        "$PROJECT_ROOT/mapping_report.md" || warn "Could not find expected summary lines in mapping report."
fi
for prior_log in "${HOME}/ecg-gate-exp0.log" "${LOG_DIR}/setup.log"; do
    if [[ -f "$prior_log" ]]; then
    printf '\nRecent lines from %s:\n' "$prior_log"
    tail -n 12 "$prior_log" || true
    fi
done
if [[ -f "${HOME}/ecg-gate-exp0.exit" ]]; then
    printf 'Previous Exp 0 exit status: '
    cat "${HOME}/ecg-gate-exp0.exit"
fi

section "Test A sensitivity run"
if [[ -d "${LOG_DIR}/test-a-sensitivity.lock" ]]; then
    lock_pid="$(cat "${LOG_DIR}/test-a-sensitivity.lock/pid" 2>/dev/null || true)"
    warn "Run lock exists (PID ${lock_pid:-unknown}). If no matching job is alive, inspect the log and lock before removing it."
fi
if [[ -f "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        printf 'PID %s is alive.\n' "$pid"
        ps -p "$pid" -o pid,etime,stat,cmd 2>&1 || true
    else
        warn "Recorded PID is not running: ${pid:-<empty>}"
    fi
else
    printf 'No run PID file found.\n'
fi
if [[ -f "$EXIT_FILE" ]]; then
    printf 'Recorded run exit: '
    cat "$EXIT_FILE"
elif [[ -f "$PID_FILE" ]]; then
    warn "No completion status file; run may be active, interrupted, or from an older script."
fi
if [[ -f "$RESULT_FILE" ]]; then
    stat -c 'Result: %n | size=%s bytes | modified=%y' "$RESULT_FILE" 2>/dev/null ||
        ls -lh "$RESULT_FILE"
    if [[ -x "$PYTHON" ]]; then
        "$PYTHON" - "$RESULT_FILE" <<'PY'
import json, sys
path = sys.argv[1]
try:
    with open(path, encoding="utf-8") as f:
        result = json.load(f)
    for key in ("verdict", "n_index_records", "n_excluded_records",
                "excluded_run_ids", "best_layer", "best_val_auc", "test_auc",
                "gap_vs_ctrl", "ctrl_val_aucs"):
        if key in result:
            print(f"{key}: {result[key]}")
except Exception as exc:
    print(f"Could not parse result JSON: {type(exc).__name__}: {exc}")
    raise SystemExit(1)
PY
        [[ "$?" -eq 0 ]] || bad "Test A result JSON is invalid."
    fi
else
    warn "Sensitivity result JSON is not present."
fi
if [[ -f "$RUN_LOG" ]]; then
    printf '\nLast 30 lines of run log (%s):\n' "$RUN_LOG"
    tail -n 30 "$RUN_LOG" || true
else
    warn "No sensitivity run log found."
fi

section "Status summary"
printf 'Diagnostic checks requiring attention: %s\n' "$ERRORS"
if [[ "$ERRORS" -eq 0 ]]; then
    printf 'No missing assets or Python import failures detected.\n'
else
    printf 'Resolve the [MISSING/ERROR] items above before starting or resuming Test A.\n'
fi
