# ECG Gate: handoff to the next agent

Written 2026-11-07 at the end of a long working session. Read this first, then
`PROJECT_STATUS_REPORT.md` (the user's own record, partly out of date) and
`README.md`. Where this file and the code disagree, the code wins; verify before
relying on anything marked "unverified".

## 1. The task, in one paragraph

ECG Gate is an ECG-FM research workflow with gated experiments: Exp 0 (model and
pipeline smoke test), Exp 1 (audit of the MedalCare-XL index), Test A (8-class
diagnosis probe on frozen ECG-FM embeddings, grouped by anatomical model, with
random-init controls), then Test B (sparse autoencoder) and Test C (association
analysis), which are not started. Repo: `Nikhilodeon1/sts-gate`, local path
`C:\Users\nikhi\Codes\TEST\sts-gate`, Pod clone at `/home/jovyan/sts-gate`.
Local HEAD at handoff is `ad55e5a`, in sync with `origin/master`
(`git log` shows `ad55e5a hello`, `ceadfe8 s`, `841ab3c Add lightweight pilot export and Pod pilot runner`). Those
last two commit messages are poor; nobody has asked to rewrite them.

## 2. Decisions the user already made (do not relitigate)

- Model identity / grouping key for CV: `group::source_shape_id`.
- Keep the dataset's own train/validation/test assignments unchanged.
- `run_S64` and `run_S67` each cross provided splits. Log them, and run a
  separately named Test A sensitivity analysis excluding those run IDs.
- Do not transfer or first-test on the full 30 GB dataset. A lightweight pilot
  came first (done, section 5). The user then chose to run full Test A on the
  lossy 460 MiB quantized bundle, accepting the quantization caveat.
- The user wants you to reuse the existing code, not reinvent it, and not wander
  beyond the task. Ask before widening scope (see section 8).

## 3. How to work with this user

- The user runs every Pod command themselves and pastes output back. You cannot
  reach the Pod. Give commands in their own fenced ```bash block, one command per
  block, no `$` prompt. Use full absolute paths. Chained one-liners are fine.
- Their environment has a "caveman" terse-reply hook; replies are short
  fragments. Files, commit messages and docs are written in normal prose.
- They paste very long logs. Tell them what to `grep`/`tail` instead of asking for
  the whole thing.
- Never run destructive commands without asking. The user decides on anything
  irreversible, and on moving from pilot to full evaluation.
- Commits end with the `Co-Authored-By` attribution line the harness gives you.
- Windows quirks on the local machine: Git Bash plus PowerShell. Only
  `C:\Python314\python` has pandas, pyarrow, sklearn and torch (do not pass `-I`;
  it hides the user site). Windows Python cannot take `/c/...` paths; convert with
  `cygpath -w`. There is no `hydra` locally, so `fairseq_signals` cannot import
  and the real model cannot run locally. A fairseq checkout exists at
  `C:\Users\nikhi\Codes\RES\datasets\fairseq-signals-src` but needs `hydra`;
  the user did not ask for packages to be installed into their global Python.
  `core.autocrlf=true`, so git warns about CRLF; committed blobs are LF (checked
  for `scripts/pod_pilot.sh`: zero `\r`).

## 4. Code map

| Path | Purpose |
|---|---|
| `src/config.py` | Paths, classes, expected counts, `PROBE_LAYERS=[4,6,8,10,12]`, `PROBE_C_GRID=[0.01,0.1,1,10]`, `PROBE_N_JOBS` (default 4), env overrides |
| `src/data.py` | Raw parsing, `build_index()` (raw mode or bundle mode), `load_medalcare()` (decodes raw CSV or `shard.npz::key` bundle records) |
| `src/package_medalcare.py` | Bundle exporter. Full export, or `--pilot` (new this session) |
| `src/model.py` | `load_ecgfm()`, `extract_layers()`, `pool_tokens()` |
| `src/exp0_smoke.py`, `src/exp1_index.py` | Exp 0 and Exp 1; Exp 1 rewrites `mapping_report.md` |
| `src/probe.py` | Test A: `embed_medalcare`, `fit_probe_layer`, `eval_probe`, `run_test_a`; caches embeddings under `ECG_CACHE_DIR/embeddings/<tag>/` |
| `src/pilot.py` | New this session. Pilot runner with `smoke` and `probe` tiers |
| `src/sae.py`, `src/assoc.py` | Test B and Test C, not started |
| `scripts/pod_setup.sh` | Pod setup: venv, deps, vendors fairseq-signals into `/tmp/ecg-gate/vendor`, extracts the bundle ZIP to `/tmp/ecg-gate/medalcare_bundle`, saves paths to `~/.ecg-gate.env` |
| `scripts/pod_status.sh` | Read-only diagnostics (but it rebuilds the cached index each run) |
| `scripts/pod_run.sh` | Guarded full workflow: Exp 0, then Exp 1 (must PASS), then Test A sensitivity (`--exclude-run-ids run_S64 run_S67`, 3 control seeds). Runs under `nohup`; refuses to start if its result JSON already exists |
| `scripts/pod_pilot.sh` | New this session. `bash scripts/pod_pilot.sh smoke|probe [extra pilot.py args]` |

Environment variables: `ECG_BUNDLE_ROOT`, `ECG_CACHE_DIR`, `ECG_FM_CKPT`,
`ECG_SCRATCH_DIR`, `FAIRSEQ_SIGNALS_REPO`, `ECG_RUNS_DIR`, `ECG_PROBE_N_JOBS`,
`MEDALCARE_ROOT` (raw data; unused on the Pod).

Test A pass rule in `src/probe.py`: `PASS` if test AUROC at the best-by-validation
layer is at least 0.80 and `gap >= 0.10`; `FAIL_STOP` if test AUROC is below 0.70;
otherwise `MARGINAL_CONTINUE` if at least 5 groups have per-group AUROC at least
0.75, else `MARGINAL_REPORT`. Note the quirk: `gap = test AUROC (pretrained) - mean
VALIDATION AUROC (control)`, which mixes splits. The control's validation score is
the higher one, so this is the stricter comparison. `LOCKED_LAYER` in config is
`None` and must not be set from pilot data.

## 5. What was built this session

1. `package_medalcare.py --pilot`: deterministic, seeded, split-preserving subset.
   Defaults: 5 train + 1 validation + 1 test model per group, at most 10 ECGs per
   model, `run_S64`/`run_S67` dropped, seed 0. Models are never moved between
   splits. Prefers models with at least 10 ECGs. Selection rules, seed and counts
   are written to `bundle.json` under `"pilot"`. Same bundle format as the full
   export, so `pod_setup.sh` and `build_index()` work unchanged.
2. `src/pilot.py`: `smoke` tier (24 ECGs, 3 per group: decode, inference, timing,
   no AUROC) and `probe` tier (all pilot ECGs through the existing probe code, plus
   a random-init control with `--ctrl-seeds`, default 3; prints a per-layer gap
   table, writes everything to `runs/pilot_<tier>.json`, no verdict). Guards: refuses
   a bundle without `"pilot"` metadata, requires `ECG_CACHE_DIR` to be set (because
   bundle-mode `build_index()` overwrites `<cache>/index.parquet`), refuses to
   overwrite an existing result file, aborts on decode failure or non-finite
   embeddings, and checks that every `GroupKFold(5)` fold on the pilot train set has
   all 8 classes.
3. `scripts/pod_pilot.sh` (wrapper with CUDA/checkpoint/bundle checks, tees to
   `~/ecg-gate-logs/pilot-<tier>.log`).
4. `src/probe.py` fix: `LogisticRegression(multi_class="ovr")` was removed in
   scikit-learn 1.8 and `requirements.txt` does not pin sklearn, so Test A would
   have crashed on the Pod. Replaced with `OneVsRestClassifier(LogisticRegression)`
   and grid key `estimator__C` (behaviour-preserving).
5. README pilot section.

## 6. Artifacts and checksums (verify, never assume)

| File | Bytes | SHA-256 |
|---|---|---|
| `C:\Users\nikhi\Codes\TEST\MedalCare-XL-pilot.zip` (560 ECGs) | 16,112,720 | `ee380798aab4c6c4e56d6793b5eee38ceb76c552c6fcbe4f92341b39e19205e2` |
| `C:\Users\nikhi\Codes\TEST\MedalCare-XL-pod-bundle.zip` (16,842 ECGs, int8 quantized, lossy, per-lead scales) | 483,246,591 | `67e859382757fc3f87e3e528055cb80ac9352212d943374aa13e1cf1e3bdfc47` |
| `ckpt/mimic_iv_ecg_physionet_pretrained.pt` | 1,090,825,421 | `4d0142bcb485eb9f0c7845e0c19ff3463f6ae9d0e458eab69136efe90ceb9b7e` |

The checkpoint's public source is `huggingface.co/wanglab/ecg-fm`
(`.../resolve/main/mimic_iv_ecg_physionet_pretrained.pt`). A browser upload of it
to the Pod was once truncated (`PytorchStreamReader failed reading zip archive:
failed finding central directory`); the user re-downloaded it on the Pod with
`curl -L -C -`. Always compare SHA-256 after a transfer. `ckpt/`, `cache/`, `data/`
and `logs/` are git-ignored.

Quantization: RMSE 0.0092 on the z-scored signal for the full bundle (0.0096 for
the pilot). Whether it changes embeddings or AUROC is untested.

## 7. Results so far

**Pilot (560 ECGs, seed 0), real model on the Pod.**

- Smoke: 24 ECGs, 12 layers, 768 dim, `[24, 768]` embeddings per probed layer,
  0.86 s inference on a V100, 1.17 GiB peak GPU memory.
- Probe, first run (no control): validation macro AUROC 0.963-0.980, test
  0.964-0.974, accuracy about 0.68-0.83.
- Probe with 3 random-init control seeds: pretrained validation 0.953-0.976, test
  0.939-0.963; control mean validation 0.797-0.847, test 0.658-0.693. Gaps are
  +0.12 to +0.16 on validation and +0.245 to +0.296 on test.
- Applying the real pass-rule formula to pilot numbers: best-by-validation layer
  8 gives 0.963 - 0.840 = +0.123; layer 10 gives +0.099 (misses 0.10). A thin
  margin on 80 test ECGs, not a verdict.
- Unexplained: pretrained AUROC drifted 0.01-0.03 between two identical pilot
  runs (e.g. layer 4 test 0.968 then 0.939). Cause not established.
- Weak classes: `fam`, `sinus`, `lae`; `lbbb` is often confused with `rbbb`.
- Result files on the Pod: `runs/pilot_smoke.json`, `runs/pilot_probe_nocontrol.json`,
  `runs/pilot_probe.json`.

**Full run on the quantized full bundle** (Pod: NVIDIA A10 23 GiB, launched about
07:40 UTC on 2026-11-07 Pod clock, worker PID 1497; the final outcome was unknown
at handoff).

- Exp 0: PASS. 90,883,072 parameters, 12 layers, 768 dim, 156 tokens per window,
  no NaNs, pretrained-vs-random cosine similarity 0.0062.
- Exp 1: PASS. 16,842 / 16,900 indexed (99.7%), minimum group/parameter coverage
  95.7% (`iab` 1,244/1,300; `rbbb` 1,298/1,300), 144 model IDs, none in more than
  one split, range checks passed. Its "Could not plot ... not found" warnings are
  cosmetic: the plotting code does not understand bundle record paths.
- Test A sensitivity: 2,777 ECGs excluded, 14,065 remain. Probe train subsample
  3,000 (375 per group), validation 1,434, test 1,386. Layer 4: validation AUROC
  0.992, test 0.990, `best_C=0.01`. Remaining layers, controls and verdict were
  not seen. Expected output: `runs/A_results_sensitivity_without_S64_S67.json`.

## 8. Open issues, in priority order

1. **CV scoring bug in `fit_probe_layer` (found, not fixed).** `GridSearchCV` with
   `GroupKFold(5)` is not stratified. In the full run, at least one fold lacks
   some of the 8 classes, so `roc_auc_ovr` raises `Number of classes in y_true not
   equal to the number of columns in 'y_score'`, sklearn sets those scores to NaN,
   every `C` ties at NaN, and `best_params_` silently falls back to the first grid
   value (`C=0.01`). So the full run's probes all use an untuned `C=0.01`. Final
   validation/test AUROCs are still real held-out numbers, and pretrained and
   control probes share the same `C`, so the comparison is like-for-like, but this
   departs from the intended protocol and must be disclosed or fixed. Proposed fix
   (untested, needs user approval): a stratified, model-grouped splitter such as
   `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)`, then rerun
   only the probe fits. Embeddings are cached on the Pod under
   `/tmp/ecg-gate/cache/embeddings/` (tags `pretrained_sensitivity_without_S64_S67`
   and `ctrl_seed{0,1,2}_sensitivity_without_S64_S67`, reused only if
   `_cache_matches_index` passes), so a rerun should be minutes, not hours.
   `pod_run.sh` refuses to run if the result JSON exists, so move it aside first.
   A read-only diagnostic to confirm the cause (which classes each fold misses)
   was given to the user; its output was not seen.
2. **No primary Test A run.** `pod_run.sh` runs only the sensitivity analysis
   (excluding `run_S64`/`run_S67`). A primary run on all 16,842 ECGs
   (`python src/probe.py` with no exclusion) is not part of the scripts. Ask the
   user before adding it.
3. **Quantization effect untested.** Option offered, not chosen: export the same
   560 pilot ECGs unquantized and compare embeddings/AUROC. Roughly 0.8-1 GB of
   raw CSV upload (estimate).
4. **Pilot-to-pilot drift** (item in section 7) unexplained.
5. **Test B, Test C** not started. `LOCKED_LAYER` stays `None`. Do not set a layer
   from the pilot.
6. **Stale docs.** `PROJECT_STATUS_REPORT.md` still says the pilot exporter and
   runner are unimplemented and describes the full-ZIP as "do not upload". Update
   it once the full run is understood. `mapping_report.md` and `notebook.md` on the
   Pod clone are rewritten by Exp 0/1 (the Pod's copy is dirty); copy results back
   rather than `git pull` over them. Local `mapping_report.md` is the older
   2026-10-26 local run.
7. **Estimates, not measurements.** Pilot embedding speed was about 0.018 s/ECG on
   a V100; the full run's duration on the A10 was never measured.
8. **Housekeeping.** There is a stray `C:\c` folder (a mistaken extraction of
   session scratch files; the delete was blocked by the safety check, so the user
   must remove it). I did not read `prompts.md` in the repo root.

## 9. Pod facts and gotchas

- JupyterHub user `jovyan`. Home (`/home/jovyan`) persists, 5 GiB quota, about
  2.5 GiB used. `/tmp` is wiped on Pod restart, taking the venv
  (`/tmp/ecg-gate/venv`), vendored `fairseq-signals`, the extracted bundle and all
  embedding caches. GPUs vary by Pod (V100-16GB, then A10-23GB). `/` is not
  writable; "root directory" in the Jupyter file browser means `/home/jovyan`.
- Persistent files in `~`: `sts-gate/`, `MedalCare-XL-pilot.zip`,
  `MedalCare-XL-pod-bundle.zip`, `mimic_iv_ecg_physionet_pretrained.pt`,
  `.ecg-gate.env` (saved asset paths), `ecg-gate-logs/`. Other projects live there
  (`paperML`, `iclr-agent-paper`): do not touch them.
- `pod_setup.sh` prompts in this order: bundle ZIP, MedalCare root (leave blank),
  checkpoint. It shows the previously saved value in brackets as a default; the
  user once accepted the pilot ZIP default by mistake and ran the full workflow on
  the 560-ECG pilot bundle (Exp 1 then failed harmlessly). When switching bundles,
  delete `/tmp/ecg-gate/medalcare_bundle` first or setup reuses/rejects it. After
  switching to the full bundle, `pod_pilot.sh` correctly refuses to run.
- `pod_setup.sh` needs internet (GitHub, PyPI) and a base Python with CUDA PyTorch
  (the Pod image has `/opt/conda` Python 3.12.8, torch 2.5.1+cu124). Its pip step
  upgrades numpy to 2.4.x and prints dependency-conflict errors for unrelated
  preinstalled packages (pennylane, cirq, numba, cupy, qsimcirq); imports for this
  project still pass.
- `pod_run.sh` buffers each step's output into `/tmp/ecg-gate-step-<PID>.log` and
  only appends it to `~/ecg-gate-logs/test-a-sensitivity.log` when the step ends,
  so `pod_status.sh` shows nothing mid-step. Peek with
  `tr '\r' '\n' < /tmp/ecg-gate-step-*.log | grep -E "^\[Test A\]|val AUROC|Ctrl seed|Best layer|verdict"`.
  The persistent log is appended across runs, so old tracebacks (a missing
  `fairseq_signals`, a placeholder `/actual/path/...`, a failed Exp 1 on the pilot
  bundle) still appear in status output and are stale. `Ctrl-C` on a `tail -f` does
  not stop the `nohup` job.
- `pod_status.sh` rebuilds `/tmp/ecg-gate/cache/index.parquet` on every call;
  harmless but avoid running it in a tight loop.

## 10. Suggested first moves for you

1. Ask the user whether the full run finished. If so, get
   `runs/A_results_sensitivity_without_S64_S67.json` (or the `grep` output above)
   and report the best layer, test AUROC, control AUROCs, gap and verdict, always
   with the quantization and untuned-`C` caveats.
2. Get approval for the CV fix, implement it in `src/probe.py` (keep default
   behaviour changes minimal and note them), verify locally with a stub model if
   the real one is unavailable, commit, have the user push, and rerun the probe
   fits on the cached embeddings.
3. Decide with the user whether a primary no-exclusion Test A run and the
   quantization check are wanted.
4. Only after that, update `PROJECT_STATUS_REPORT.md`.
