# ECG Gate: A-to-Z Project Status Report

**Report date:** 2026-10-28  
**Project:** `Nikhilodeon1/sts-gate`  
**Purpose:** Record the original experiment request, verified work, current
files and Pod state, and the corrected next step: a lightweight pilot rather
than a full-dataset transfer or run.

## 1. Executive summary

The project is an ECG-FM research workflow with three planned gates:

- **Exp 0:** verify the pretrained ECG-FM model and input pipeline.
- **Exp 1:** audit the MedalCare-XL index, expected coverage, parameters, and
  model/split assignments.
- **Test A:** measure an eight-class diagnosis probe, grouped by the selected
  anatomical model identity, with random-initialization controls.
- **Test B / Test C:** later SAE and association analyses; these have not been
  reached.

The local Windows environment previously completed Exp 0 successfully and
rebuilt Exp 1 against the full local dataset. The Pod environment was
successfully prepared with CUDA PyTorch and `fairseq_signals`, but the Pod did
not have the MedalCare data or ECG-FM checkpoint mounted. The earlier Pod
attempts therefore failed before a real model/data run. **Test A has not been
run.**

The user clarified that the goal is a **lightweight initial gate**, not
processing or transferring the full 30 GB dataset. A full-data quantized ZIP
was created while measuring compression, but that is not the right next step:
it contains all 16,842 indexed signals and uses lossy quantization. It should
not be uploaded or treated as the requested pilot.

**Current recommendation:** implement a deterministic, small, split-preserving
pilot subset locally, transfer only that subset and the required checkpoint,
then run a pipeline/latency pilot on the Pod. Treat pilot AUROC as exploratory;
do not use it to claim Test A passed or to lock the model layer.

## 2. What the user originally asked for

The work began as continuation of an unfinished ECG-FM gate workflow. Exp 0
reported a PASS on Windows:

- ECG-FM: 12 transformer layers, hidden dimension 768.
- Parameters: 90,883,072.
- Forward pass: 10 windows in about 2.45 seconds.
- Tokens per window: 156.
- No NaNs in captured layers.
- Pretrained/random-init cosine similarity at layer 12: 0.0139.
- PTB-XL was not locally available in the partial download; the test continued
  with MedalCare-only validation.

An initial Exp 1 report was challenged because it called five-group coverage
“100%” while omitting `fam`, `iab`, and `lae`, and used a geometry key that did
not represent the selected model grouping. The user explicitly said **do not
start Test A** until the coverage and geometry/split issues were addressed.

The user selected:

- Model identity/grouping key: **`group::source_shape_id`**.
- Keep the dataset-provided train/validation/test assignments unchanged.
- Log `run_S64` and `run_S67`, which cross provided splits.
- Run a separately named Test A sensitivity analysis excluding those run IDs.

Most recently the user corrected the transfer strategy: **do not transfer or
initially test on the full dataset**. The first Pod run should be a lightweight
pilot.

## 3. Verified local dataset facts

Local Windows dataset root:

`C:\Users\nikhi\Codes\RES\datasets\MedalCare-XL\MedalCare-XL`

It contains:

- `WP2_largeDataset_Noise`
- `WP2_largeDataset_ParameterFiles`

Measured local storage:

- Noise tree: 50,733 files, about **26.245 GiB**.
- Parameter-file tree: 33,688 files, about **0.042 GiB**.
- Filtered ECG files found in the noise tree: 16,848.
- Index rows available from the local cache: **16,842**.
- Therefore six files from the filtered-file scan do not appear in the index;
  their individual causes were not established in this audit.

The local index contains 16,842 rows and includes ECG path, group, split,
`run_id`, `source_shape_id`, selected `model_id`, and flattened parameter
columns. All indexed ECG paths existed at the time of validation.

### Exp 1 expected-count report

The expected full dataset count is 16,900. The most recent local index rebuild
reported:

- Indexed: **16,842 / 16,900 = 99.7% overall**.
- Minimum group/parameter expected-count coverage: **95.7%**, passing the
  configured 90% threshold.
- All eight groups are present.
- Missing expected rows: **56 IAB** and **2 RBBB**.
- The missing rows were reported as null `siginfo` entries; those corresponding
  records could not be mapped.
- Key parameter coverage: 16,842 / 16,900 = **99.6568%** for each reported
  parameter.
- Parameter range checks passed.
- All indexed rows have a selected model ID.
- Selected group-scoped IDs: **144 unique**, none span splits.
- Raw source shape tokens: 143 unique; one token appears in two groups, which
  is why the group-qualified model key matters.
- `run_S64` and `run_S67` each cross predefined split boundaries.

The generated [mapping_report.md](./mapping_report.md) was last regenerated
2026-10-26 03:20. It is the full-data local report, not a Pod run report and not
a report for a reduced pilot bundle.

### Model counts by group and split

Model-ID counts from the local index:

| Group | Train | Validation | Test |
|---|---:|---:|---:|
| avblock | 9 | 2 | 2 |
| fam | 8 | 1 | 1 |
| iab | 48 | 5 | 6 |
| lae | 8 | 1 | 1 |
| lbbb | 9 | 2 | 2 |
| mi | 9 | 2 | 2 |
| rbbb | 9 | 2 | 2 |
| sinus | 9 | 2 | 2 |

After excluding all rows with `run_S64` and `run_S67`, the local index showed
14,065 records remaining. Every group still has at least one model in every
split, but several groups have only **one** model in validation and test.
Therefore that exclusion subset can support a plumbing/representation pilot,
but the resulting small held-out AUROCs would be weak evidence of
generalization.

## 4. Corrected lightweight pilot scope

Compression and subsampling are different:

- Compression makes the same full dataset smaller.
- A pilot selects fewer examples and therefore reduces both transfer and
  compute.

The previous full bundle did the first, not the second. The intended first Pod
gate is a **small deterministic subset**, selected locally from the already
validated index and preserving its labels, model IDs, and original splits.

Suggested engineering-pilot design:

1. Include all eight diagnostic groups.
2. Keep the predefined train/validation/test assignment for every selected row.
3. Use only model IDs belonging to their existing split; do not move models
   between splits.
4. Include enough distinct train model IDs to exercise grouped CV if CV is
   included. The current Test A uses five-fold `GroupKFold`, so at least five
   training model IDs are required overall; a safer balanced pilot selects five
   training model IDs per group.
5. Select one validation and one test model per group where available after the
   run-ID exclusions. Several groups only have one model in either held-out
   split after exclusions.
6. Cap the pilot at a small fixed number of ECGs per selected model (for example
   10), selected deterministically, to keep transfer/compute bounded.
7. Label the output as **pilot-only**. Do not interpret its AUROC as the final
   Test A decision or use it to set a locked layer.

The example design above is at most 560 ECGs (8 groups × (5 train + 1
validation + 1 test models) × 10 ECGs/model), before accounting for any
undersized model. That is an upper-bound design sketch, not a measured final
bundle size or an already implemented sampling rule.

The best first gate may be even smaller: a few ECGs per class to verify model
loading, signal decoding, inference, and elapsed time, with **no AUROC claim**.
Only after that succeeds should the larger pilot test the grouped-probe code.

### Required code changes before this pilot is runnable

These are not yet implemented:

- A subset exporter that selects the declared model/ECG limits deterministically
  from the local index, records its seed and selection rules, and exports
  original split/group/model/parameter metadata with the waveforms.
- A pilot mode in the Pod runner. Current `pod_run.sh` runs full Exp 1 and
  requires its full-dataset PASS report before it will run Test A. Exp 1 compares
  counts with expectations for all 16,900 records, so a true subset should not
  be mislabeled as a full-index Exp 1 pass.
- A pilot report with explicit subset size and purpose; it must distinguish
  technical readiness from scientific gate results.
- If grouped CV is used with the reduced sample, verify each fold contains the
  necessary classes; the current `GroupKFold` does not itself promise
  stratification.

Thus, do **not** use the current `pod_run.sh` unchanged for a subset pilot: its
full-data coverage gate is intentionally incompatible with a reduced manifest.

## 5. Full-data ZIP created during compression feasibility work

This archive was created locally to measure whether compression could fit the
full index into less than 1 GiB:

- Local file:
  `C:\Users\nikhi\Codes\TEST\MedalCare-XL-pod-bundle.zip`
- Size: **483,246,591 bytes** (460.86 MiB; 0.450 GiB).
- Records: 16,842 (the entire current index).
- Shards: 132, 128 ECGs per shard except the last.
- SHA-256:
  `67e859382757fc3f87e3e528055cb80ac9352212d943374aa13e1cf1e3bdfc47`
- Encoding: per-lead z-score preprocessing, then adaptive int8 quantization
  with per-lead float32 scales.
- Signal reconstruction RMSE: 0.0091926.
- Maximum absolute normalized-signal error: 0.0696624.
- ZIP integrity, bundle manifest loading, full 16,842-row Pod-style index
  loading, and one-record-per-group waveform round trips were verified locally.

Important caveats:

- This archive contains essentially the **whole indexed dataset**, so it is not
  the requested lightweight pilot.
- Quantization is **lossy**. Signal round-trip checks do not prove ECG-FM
  embedding or AUROC equivalence.
- It was not uploaded to the Pod and was not used for a Pod inference run.
- **Do not upload or use it for the first gate.** It is retained locally only as
  a feasibility artifact unless the user later chooses otherwise.

The model checkpoint is separate:

- Local path:
  `C:\Users\nikhi\Codes\TEST\sts-gate\ckpt\mimic_iv_ecg_physionet_pretrained.pt`
- Size: **1,090,825,421 bytes** (about 1.016 GiB).
- It was not present on the Pod in the last status output.

## 6. Pod environment and run history

Last user-provided Pod status was from 2026-10-27. It reported:

- JupyterHub user: `jovyan`.
- GPU: Tesla V100-PCIE-16GB, 16,384 MiB.
- cgroup memory limit: 34,359,738,368 bytes (32 GiB).
- Home filesystem: 5.0 GiB total, about 4.0 GiB free.
- `/tmp` overlay showed substantial node disk capacity; actual pod quota was not
  independently established.
- Python 3.12.8 in `/tmp/ecg-gate/venv`.
- PyTorch 2.5.1+cu124; `torch.cuda.is_available()` returned true.
- `fairseq_signals` imported from
  `/tmp/ecg-gate/vendor/fairseq-signals-src/fairseq_signals`.
- Setup verified scientific dependencies and fairseq imports.

The following remain true **as of that last supplied output only**:

- `MEDALCARE_ROOT` and `ECG_FM_CKPT` were unset.
- Searches under `/home/jovyan`, `/tmp`, `/mnt`, `/media`, `/data`, and
  `/workspace` found neither dataset nor checkpoint.
- A previous Exp 0 failed because that earlier environment had no
  `fairseq_signals`; the subsequent setup fixed that dependency issue.
- A previous Exp 0 failed to load its checkpoint because a literal placeholder
  path was used.
- A previous Test A attempt failed immediately because the dataset path was a
  literal placeholder.
- No Test A result JSON existed; no successful Pod Test A run was reported.
- `/tmp` is wiped on Pod restart; the Python environment and caches must be
  rebuilt.

Do not assume the Pod is still alive or that its `/tmp` environment survived.
The next Pod session should be checked with the scripts; do not infer job
completion from the old PID, old logs, or the old local mapping report.

## 7. Repository and directory layout

Repository path:

`C:\Users\nikhi\Codes\TEST\sts-gate`

The current local checkout is on branch `master`, has an `origin` remote, and
was clean at inspection. HEAD was `41693dc` (subject: `aster d**g ate my meat`).
That commit contains the Pod scripts and full bundle support. No uncommitted
changes were present when this report was prepared.

Relevant files:

| Path | Purpose | Current notes |
|---|---|---|
| `src/config.py` | Paths, classes, expected counts, model/test settings. | Supports `MEDALCARE_ROOT`, `ECG_BUNDLE_ROOT`, cache/scratch envs, and four probe CPU workers by default. |
| `src/data.py` | Raw data parsing, parameter ingestion, preprocessing, index construction, bundle decoding. | Bundle mode expects `bundle.json` plus `manifest.parquet`; raw mode walks the dataset tree. |
| `src/package_medalcare.py` | Local full-index bundle exporter. | Currently exports the whole validated index; it does not yet support the intended small pilot selection. |
| `src/model.py` | ECG-FM checkpoint loading, transformer hooks, pooling. | Requires installed `fairseq_signals` and the pretrained checkpoint for Exp 0/Test A. |
| `src/exp0_smoke.py` | Model/data smoke test. | Samples up to five indexed records after calling `build_index`; missing raw assets/bundle prevent it from running. |
| `src/exp1_index.py` | Full expected-count coverage, parameter ranges, model/split checks, plots, mapping report. | A subset is not expected to pass its full-dataset coverage test. |
| `src/probe.py` | Test A embeddings, grouped CV, AUROC, controls, sensitivity exclusion. | Currently processes all records in its index; has `--exclude-run-ids`, not a pilot row/model limit. |
| `src/sae.py` | Test B sparse autoencoder. | Not started. |
| `src/assoc.py` | Test C association analysis. | Not started. |
| `scripts/pod_setup.sh` | Pod resource checks, temporary venv, dependency setup, path prompts/config. | Can extract a configured bundle; current bundle support was designed for the full ZIP. |
| `scripts/pod_status.sh` | Pod resource, asset, env, report, PID, log, and result diagnostics. | Read-only with respect to jobs; may report historical old errors. |
| `scripts/pod_run.sh` | Full Exp 0 -> Exp 1 PASS -> Test A sensitivity. | Do not use as-is for a reduced pilot; the full Exp 1 gate blocks subsets. |
| `README.md` | Setup and workflow overview. | Now says full bundle is not the intended first gate; see this report. |
| `mapping_report.md` | Generated full-dataset Exp 1 report. | Local report dated 2026-10-26; not a current Pod report. |
| `notebook.md` | Run log. | Contains previous Exp 0/Exp 1 entries; old Exp 1 entry marked superseded. |
| `requirements.txt` | Scientific Python package list. | Does not install a CUDA PyTorch build; Pod setup reuses Pod CUDA Torch. |

Directories:

- `cache/`: local index and caches; ignored by Git.
- `ckpt/`: local checkpoint; ignored by Git.
- `data/`: optional local/default dataset location; ignored by Git.
- `runs/`: generated plots/results; plots from the local Exp 1 run are tracked.
- `scripts/`: the three Pod workflow scripts.
- `src/`: experiment/data/model implementation.
- `logs/`: runtime logs; ignored by Git.
- `vendor/`: fairseq-signals checkout; ignored by Git and intended for Pod
  scratch.

## 8. Prior implementation and validation completed

### Data and index fixes

- Added support for direct and `siginfo` row-ordinal ECG/parameter filename
  mapping.
- Added loud errors for malformed `siginfo` headers instead of silently
  skipping incompatible layouts.
- Parsed parameter files with `=` or whitespace separators, numeric values with
  units, and null-like `NA`.
- Added `source_shape_id`, including LAE `pwave_NNN` handling.
- Rebuilt the cached index when the stored model-ID scheme was stale.
- Changed the selected model grouping to `group::source_shape_id`.
- Added expected-count coverage by dataset folder, 16,900 total expected ECGs.
- Added the full Exp 1 missing-record, group/split, parameter, model/split, and
  run-ID report.
- Kept provided split assignments unchanged.

### Pod setup

- Added configurable dataset, checkpoint, scratch, cache, and fairseq paths.
- Added Linux Pod setup/status/run scripts with persistent home logs and `/tmp`
  scratch.
- Avoided automatically installing a multi-gigabyte CUDA Torch wheel when
  Pod-provided CUDA Torch works.
- Added preflight and safety checks so the run script refuses to start on
  missing paths and checks Exp 0/Exp 1 before full Test A.
- Limited grouped-CV joblib workers to a configurable four by default rather
  than all host-reported CPUs.

### Validation evidence

- Local Exp 0: PASS, facts listed in section 2.
- Local Exp 1: PASS under the corrected expected-count threshold and selected
  model key.
- Local Python syntax checks and Bash `bash -n`: passed for modified modules and
  helper scripts.
- Bundle test: ZIP integrity, manifest counts, 16,842-row index loading,
  per-group waveform decode shape/dtype, and sampled reconstruction passed.
- No end-to-end Pod experiment has passed.
- No Test A, Test B, or Test C result is available.

## 9. Current blockers and non-blockers

### Blockers for any Pod model run

1. The Pod must have the ECG-FM checkpoint at an accessible path.
2. The Pod must have a dataset source. Under the corrected lightweight plan,
   this should be a small pilot subset generated locally, not the full raw
   tree or the full-data ZIP.
3. The exporter/runner must implement pilot selection/reporting and must not
   mistake subset coverage for full-dataset Exp 1 coverage.
4. A fresh Pod restart may have erased its `/tmp` venv and caches; check status
   and rerun setup as needed.

### Not blockers for the first pilot

- PTB-XL: optional; the requested MedalCare-only smoke/pilot can run without it.
- Test B/Test C dependencies and datasets: later stages, not required for
  initial Test A pipeline readiness.
- The 58 missing source rows: known, documented; the indexed dataset still
  exceeds the 90% expected coverage gate.

## 10. Next actions in order

1. **Do not upload the full-data ZIP.** It is not the lightweight gate requested.
2. Add a local deterministic pilot-export mode. Preserve original split and
   `model_id`; report the selection seed, counts, selected model IDs, and
   per-group/split counts.
3. Add a Pod pilot-run mode that:
   - loads the pilot manifest directly,
   - runs a small inference smoke check,
   - reports the number of waveforms, GPU, elapsed time, memory where available,
     and any decoding/model errors,
   - optionally exercises a pilot grouped probe,
   - explicitly labels any pilot metric exploratory, and
   - does not invoke or require full-data Exp 1 PASS.
4. Confirm the checkpoint can be transferred/mounted separately. Its local
   size is about 1.016 GiB; it is larger than the proposed compact pilot
   dataset package would be.
5. Build and verify the pilot locally, including archive size, SHA-256, all
   labels/splits/model IDs, archive integrity, and signal decode.
6. Transfer only the pilot package and checkpoint. Run Pod status, then the
   pilot—not full Test A.
7. Review pilot runtime and numerical behavior. If satisfactory, decide the
   final data/compute plan before running full Test A.
8. Run final Test A only after the user has reviewed the pilot and explicitly
   authorized moving from pilot to full evaluation.

## 11. One-screen status

| Item | Status |
|---|---|
| Local MedalCare data | Available; full raw tree about 26.245 GiB |
| Corrected local index | 16,842 / 16,900; 99.7% overall; minimum group/parameter coverage 95.7% |
| Selected model key | `group::source_shape_id`; 144 IDs, zero cross-split |
| Known source omissions | 56 IAB + 2 RBBB null-siginfo rows |
| Run IDs needing sensitivity exclusion | `run_S64`, `run_S67` |
| Local ECG-FM checkpoint | Available; 1,090,825,421 bytes |
| Pod CUDA/fairseq | Worked in last reported Pod session |
| Pod dataset/checkpoint | Not found/configured in last reported Pod status |
| Full compressed ZIP | Built (0.450 GiB), but contains all indexed records and is not the intended pilot |
| Lightweight pilot exporter | Not implemented |
| Pilot-specific runner | Not implemented |
| Exp 0 | Local PASS; Pod attempts failed before successful model/data validation |
| Exp 1 | Local PASS; Pod run not completed |
| Test A | Not run |
| Test B / Test C | Not run |
| Next safe action | Implement and validate a small pilot export/runner; do not upload the full ZIP or launch `pod_run.sh` |
