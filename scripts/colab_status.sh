#!/usr/bin/env bash
# Read-only status and diagnostics for ECG Gate on Google Colab.
set -uo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${HOME}/.ecg-gate-colab.env"

section() { printf '\n===== %s =====\n' "$*"; }
ok() { printf '[OK] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }

if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

PYTHON="${ECG_PYTHON:-$(command -v python3 || command -v python)}"
SCRATCH_DIR="${ECG_SCRATCH_DIR:-/tmp/ecg-gate}"
CACHE_DIR="${ECG_CACHE_DIR:-${SCRATCH_DIR}/cache}"
RUNS_DIR="${ECG_RUNS_DIR:-${PROJECT_ROOT}/runs}"
LOG_DIR="${ECG_LOG_DIR:-${HOME}/ecg-gate-logs}"

section "Colab Environment & Hardware"
printf 'Time:       %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
printf 'Repository: %s\n' "$PROJECT_ROOT"
printf 'Config:     %s (%s)\n' "$ENV_FILE" "$([[ -f "$ENV_FILE" ]] && echo "found" || echo "missing")"
printf 'Persist:    %s\n' "${ECG_PERSIST_DIR:-<none (local runtime only)>}"
printf 'Cache dir:  %s\n' "$CACHE_DIR"
printf 'Runs dir:   %s\n' "$RUNS_DIR"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
else
    warn "nvidia-smi unavailable (CPU runtime?)"
fi
df -h / /tmp "${ECG_PERSIST_DIR:-/tmp}" 2>/dev/null | uniq || true

section "Assets & Cached Embeddings"
printf 'ECG_BUNDLE_ARCHIVE: %s\n' "${ECG_BUNDLE_ARCHIVE:-<unset>}"
printf 'ECG_BUNDLE_ROOT:    %s\n' "${ECG_BUNDLE_ROOT:-<unset>}"
printf 'ECG_FM_CKPT:        %s\n' "${ECG_FM_CKPT:-<unset>}"

if [[ -n "${ECG_BUNDLE_ROOT:-}" && -f "${ECG_BUNDLE_ROOT}/bundle.json" ]]; then
    "$PYTHON" - "${ECG_BUNDLE_ROOT}/bundle.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    b = json.load(f)
print(f"Bundle records: {b.get('record_count')} | pilot: {'yes' if 'pilot' in b else 'no (full)'}")
PY
else
    warn "No extracted bundle at ECG_BUNDLE_ROOT."
fi

if [[ -d "${CACHE_DIR}/embeddings" ]]; then
    printf 'Cached embedding sets under %s/embeddings:\n' "$CACHE_DIR"
    ls -1 "${CACHE_DIR}/embeddings"
else
    printf 'No cached embeddings yet.\n'
fi

section "Results Summary"
for res in \
    "${RUNS_DIR}/pilot_smoke.json" \
    "${RUNS_DIR}/pilot_probe.json" \
    "${RUNS_DIR}/A_results_sensitivity_without_S64_S67.json" \
    "${RUNS_DIR}/A_results.json" \
    "${RUNS_DIR}/B_results_sensitivity_without_S64_S67.json" \
    "${RUNS_DIR}/B_results_ctrl_sensitivity_without_S64_S67.json" \
    "${RUNS_DIR}/C_results_sensitivity_without_S64_S67.json" \
    "${RUNS_DIR}/C_results_sensitivity_without_S64_S67_with_run_id_covariate.json"; do
    if [[ -f "$res" ]]; then
        ok "Found $res"
        "$PYTHON" - "$res" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    d = json.load(f)
for k in ("tier", "verdict", "n_ecgs", "n_index_records", "n_excluded_records",
          "best_layer", "best_val_auc", "test_auc", "gap_vs_ctrl", "ctrl_mean_auc",
          "scale", "checks", "n_confirmed_total", "n_physiology_atoms", "n_morphology_atoms",
          "perm_null_mean", "perm_null_p95", "ctrl_confirmed_atoms", "moderate_thr_015", "total_seconds"):
    if k in d:
        print(f"  {k}: {d[k]}")
PY
    fi
done
