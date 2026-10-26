# ECG Gate

Experiments for evaluating ECG-FM representations on the MedalCare-XL dataset.
Dataset files, model checkpoints, cached embeddings, and the fairseq-signals
checkout are not stored in Git.

## Pod setup (JupyterHub GPU container)

The Pod home directory is persistent but small. Keep the Git clone and logs
there; put the Python environment, fairseq-signals checkout, and caches under
`/tmp`. `/tmp` is wiped on a Pod restart, so rerun setup and rebuild caches
after a restart. Do not assume `/workspace` exists or that the hardware/storage
matches prior Pods.

### 1. Clone and check the Pod

```bash
cd ~
git clone <your-repository-url> sts-gate
cd ~/sts-gate

df -h ~ /tmp
cat /sys/fs/cgroup/memory.max
nvidia-smi --query-gpu=name,memory.total --format=csv
ls /workspace 2>&1 | head -3
command -v python3.10 || true
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

Confirm the actual mounted MedalCare-XL root and checkpoint path before
continuing. The MedalCare root must contain both
`WP2_largeDataset_Noise/` and `WP2_largeDataset_ParameterFiles/`. The checkpoint
must be the pretrained `mimic_iv_ecg_physionet_pretrained.pt` file. Do not
download another copy if the assets are already mounted.

### 2. Create a temporary environment and install dependencies

First use the Pod's Python/Torch if its CUDA check above succeeds. Avoid
reinstalling Torch in that case: the CUDA wheel is large and has caused Jupyter
servers to drop during installation on some Pods.

If a temporary venv is needed, use the available `uv` workflow and expose the
Pod's existing packages. When the current Python already has usable CUDA
PyTorch, create a venv from that same interpreter so it can reuse the existing
packages:

```bash
export PIP_NO_CACHE_DIR=1
mkdir -p /tmp/ecg-gate
python -m venv --system-site-packages /tmp/ecg-gate/venv
source /tmp/ecg-gate/venv/bin/activate

python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

If the base Python does not have working CUDA PyTorch, use `uv` to install
Python 3.12 and make a venv under `/tmp` instead. Keep uv's Python downloads
out of home:

```bash
export UV_NO_CACHE=1
export UV_PYTHON_INSTALL_DIR=/tmp/ecg-gate/python
mkdir -p /tmp/ecg-gate
command -v uv >/dev/null || python3 -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
uv venv --python 3.12 /tmp/ecg-gate/venv
source /tmp/ecg-gate/venv/bin/activate
```

Recheck `torch.cuda.is_available()` in this venv. Only if no usable CUDA Torch
is present in the Pod or venv, install the Pod-tested build (this can use
several GB and may disrupt the Jupyter server):

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
```

Set scratch paths and install the remaining project packages. Setup output is
tee'd to the persistent home directory so it survives a server restart:

```bash
cd ~/sts-gate
export ECG_SCRATCH_DIR=/tmp/ecg-gate
export ECG_CACHE_DIR=/tmp/ecg-gate/cache
export FAIRSEQ_SIGNALS_REPO=/tmp/ecg-gate/vendor/fairseq-signals-src
export ECG_PROBE_N_JOBS=4
export UV_NO_CACHE=1
export PIP_NO_CACHE_DIR=1

set -o pipefail
python src/install.py 2>&1 | tee ~/ecg-gate-setup.log
```

The installer installs the packages in `requirements.txt` and the ECG-FM
fairseq-signals fork. Run the same setup again after a Pod restart because the
environment and checkout live under `/tmp`. In a new terminal session,
reactivate `/tmp/ecg-gate/venv` and re-export the environment variables before
running commands.

### 3. Point to confirmed data and checkpoint paths

Set these to the locations you confirmed in the Pod file browser or terminal;
the example values below are placeholders, not assumed mount paths:

```bash
export MEDALCARE_ROOT=/actual/path/to/MedalCare-XL/MedalCare-XL
export ECG_FM_CKPT=/actual/path/to/mimic_iv_ecg_physionet_pretrained.pt

test -d "$MEDALCARE_ROOT/WP2_largeDataset_Noise"
test -d "$MEDALCARE_ROOT/WP2_largeDataset_ParameterFiles"
test -f "$ECG_FM_CKPT"
```

If any `test` command fails, correct the path before proceeding. Keep these
exports in the same terminal session used for validation and execution.

### 4. Validate before the long run

```bash
cd ~/sts-gate
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python src/exp0_smoke.py --device cuda
python src/exp1_index.py
```

Exp 1 should report 16,842 indexed records out of 16,900 expected (99.7%
overall). The 56 IAB and 2 RBBB records missing from the current source data
have null `siginfo` entries.

### 5. Launch Test A sensitivity analysis

Run the eight-group Test A sensitivity analysis excluding the two `run_id`s that
span the provided splits. The predefined split assignments are not modified.
The run excludes 2,777 indexed ECGs, retains all eight groups in every split,
and groups cross-validation by `group::source_shape_id`. Cross-validation uses
four CPU workers by default rather than trusting the host's reported CPU count;
adjust `ECG_PROBE_N_JOBS` only after checking the Pod's actual CPU and memory
limits.

Launch it detached from the terminal and write output to persistent home:

```bash
cd ~/sts-gate
mkdir -p ~/ecg-gate-logs
nohup env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u src/probe.py \
    --device cuda \
    --ctrl-seeds 3 \
    --exclude-run-ids run_S64 run_S67 \
  > ~/ecg-gate-logs/test-a-sensitivity.log 2>&1 < /dev/null &
echo $! > ~/ecg-gate-logs/test-a-sensitivity.pid
```

Monitor the run:

```bash
tail -f ~/ecg-gate-logs/test-a-sensitivity.log
ps -p "$(cat ~/ecg-gate-logs/test-a-sensitivity.pid)" -o pid,etime,stat,cmd
nvidia-smi
```

`nvidia-smi` may say no processes are found even while the job is running; use
GPU utilization, memory, and power alongside the log and process status.

On successful completion, results are saved under
`runs/A_results_sensitivity_without_S64_S67.json`; layer plots are saved under
`runs/`, and completed embedding caches under
`/tmp/ecg-gate/cache/embeddings/`. The pretrained and each control embedding
cache are saved when that embedding pass finishes. `/tmp` caches do not survive
a Pod restart, so a restarted run must rebuild them. The log remains in home.
