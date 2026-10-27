# ECG Gate

Experiments for evaluating ECG-FM representations on the MedalCare-XL dataset.
Dataset files, model checkpoints, cached embeddings, and the fairseq-signals
checkout are not stored in Git.

## Pod workflow

Use the three scripts under `scripts/` in order:

1. `pod_setup.sh` checks Pod resources, reuses CUDA PyTorch from the active Pod
   Python when possible, creates the environment under `/tmp`, installs
   dependencies, and prompts for mounted dataset/checkpoint paths. Only valid
   paths are saved in the private file `~/.ecg-gate.env`.
2. `pod_status.sh` reports environment/assets, index and report timestamps,
   prior run PID/exit status, any Test A result metrics, and the end of the run
   log. It does not start or stop jobs.
3. `pod_run.sh` validates configuration, runs Exp 0 and Exp 1, checks both
   PASS verdicts, then launches Test A sensitivity excluding `run_S64` and
   `run_S67`. It refuses to overwrite an existing sensitivity result.

```bash
bash scripts/pod_setup.sh
bash scripts/pod_status.sh
bash scripts/pod_run.sh
```

By default, the run script detaches the workflow and logs to
`~/ecg-gate-logs/test-a-sensitivity.log`. Use
`bash scripts/pod_run.sh --foreground` to keep it attached to the terminal.
Check progress and previous results with `bash scripts/pod_status.sh`.

## Pod storage and restart behavior

The JupyterHub home directory is persistent but small: keep the clone, saved
path config, and logs there. The Python environment, fairseq-signals checkout,
and embedding/index caches go under `/tmp/ecg-gate`, which is wiped on a Pod
restart. After restart, run setup and status again; caches may need rebuilding.
The scripts do not assume `/workspace` or a particular data mount. Setup prompts
for the real MedalCare root and checkpoint path; leave a prompt empty if the
asset is not mounted yet. Test A will refuse to start until both are valid.

Exp 1 is expected to index 16,842 of 16,900 records from the current source
dataset. The 56 IAB and 2 RBBB records with null `siginfo` are absent from that
source. Test A keeps the predefined splits unchanged, groups cross-validation
by `group::source_shape_id`, and writes sensitivity outputs separately from
primary-run results.
