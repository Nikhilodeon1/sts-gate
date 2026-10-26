# notebook.md – Run log and observations

All experiment runs are logged here. For each run record: date, command, git hash, seed, output file.

## Format

```
Date: YYYY-MM-DD HH:MM
Command: python src/<script>.py <args>
Git hash: (git rev-parse --short HEAD)
Seed: N/A or 42
Output: runs/<file>
Notes: ...
```

---

<!-- Exp 0 entry will be appended by exp0_smoke.py -->
<!-- Exp 1 entry will be appended by exp1_index.py -->

## Exp 0 � Smoke Test (2026-10-25 20:59)

| Metric | Value |
|--------|-------|
| Transformer layers | 12 |
| Hidden dim | 768 |
| Tokens per 5s window | 156 |
| Forward pass (10 windows) | 2.45s |
| Pretrained vs ctrl cosine similarity | 0.0139 |
| Model parameters | 90,883,072 |
| **Verdict** | **PASS** |

## Exp 1 - Index validation (superseded; 2026-10-26 02:54)

Date: 2026-10-26 02:54
Command: python src/exp1_index.py
Git hash: N/A
Seed: 42 (plot sampling)
Output: mapping_report.md; runs/exp1_ecg_plots/
Notes: Superseded by the 03:20 rerun below. Used provisional `a_geo.atria` grouping; its split-overlap result was not the selected model-ID definition.

## Exp 1 - Index validation (2026-10-26 03:20)

Date: 2026-10-26 03:20
Command: python src/exp1_index.py
Git hash: N/A
Seed: 42 (plot sampling)
Output: mapping_report.md; runs/exp1_ecg_plots/
Notes: 16,842/16,900 expected ECGs indexed (99.7% overall); the 58 absent records are 56 IAB and 2 RBBB rows with null siginfo. Coverage is computed against expected counts and passes the 90% threshold. Selected `group::source_shape_id` model key has 144 IDs and none cross predefined splits. Splits remain unchanged. `run_S64` and `run_S67` are logged for sensitivity analysis. Test A has not been run.
