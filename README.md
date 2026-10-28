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

## Create and transfer a compact dataset bundle

The raw local MedalCare tree is about 26.2 GiB and contains many files Test A
does not need. The indexed table already contains the group/split/model IDs and
parameter values; Test A needs the indexed ECG waveforms. The packaging script
stores all indexed waveforms after the existing per-lead z-score preprocessing,
using adaptive per-lead int8 quantization in compressed shards, and includes the
complete index metadata. A 128-record sample measured about 0.44 GiB projected
for the 16,842 indexed records. The generated full bundle was 483,246,591 bytes
(0.450 GiB). Quantization is lossy: the full bundle's signal-level RMSE was
0.00919 and maximum absolute normalized-signal error was 0.0697. This checks
waveform reconstruction, not equivalence of final model embeddings or AUROC;
interpret the Pod result as using the documented quantized input.

From PowerShell on the local Windows machine:

```powershell
cd C:\Users\nikhi\Codes\TEST\sts-gate
C:\Users\nikhi\Codes\TEST\venv\Scripts\python.exe `
  src\package_medalcare.py `
  --output C:\Users\nikhi\Codes\TEST\MedalCare-XL-pod-bundle.zip
```

The script validates that indexed source files exist, all eight groups and
splits are represented, and model IDs do not cross predefined splits. It
refuses to overwrite an existing output archive. Keep the printed SHA-256 and
final size for verification. The archive contains only indexed ECG inputs,
metadata, and bundle-format information—not raw text CSVs or parameter files
that are already represented in the manifest.

Upload `MedalCare-XL-pod-bundle.zip` to persistent Pod home using the Jupyter
file browser. In a separate transfer, make the pretrained
`mimic_iv_ecg_physionet_pretrained.pt` checkpoint available to the Pod. That
checkpoint is about 1.02 GiB locally, so the dataset bundle can be under 1 GiB
but the bundle plus checkpoint cannot. Do not add either asset to Git.

Then run `bash scripts/pod_setup.sh`; at its prompts, give the uploaded ZIP path
and real checkpoint path. Setup extracts the bundle under `/tmp/ecg-gate`,
validates its manifest, and saves the config. Run `bash scripts/pod_status.sh`
and proceed only if it reports the extracted bundle, checkpoint, CUDA, and
fairseq imports valid. On restart, setup re-extracts from the ZIP in persistent
home because `/tmp` is ephemeral.
