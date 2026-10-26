# ECG Gate

Experiments for evaluating ECG-FM representations on the MedalCare-XL dataset.
Dataset files, model checkpoints, cached embeddings, and the fairseq-signals
checkout are deliberately excluded from Git; configure or provide those assets
after cloning.

## Pod setup (Linux, NVIDIA GPU)

The recommended workflow uses Python 3.10 and a CUDA-enabled PyTorch build.
From the Pod terminal:

```bash
git clone <your-repository-url> sts-gate
cd sts-gate

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install a CUDA-enabled PyTorch build compatible with the Pod's NVIDIA driver.
# This cu124 wheel is suitable when the driver supports CUDA 12.4.
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124

# Installs scientific/data dependencies and the ECG-FM fairseq-signals fork.
python src/install.py
```

If the Pod already supplies a compatible CUDA PyTorch environment, activate that
environment instead of creating `.venv` and installing torch again. Check it
before proceeding:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

The output must report CUDA available and identify the A10 GPU.

## Dataset and checkpoint

Provide the complete MedalCare-XL directory tree and the pretrained ECG-FM
checkpoint separately (they are not stored in this repository). The default
dataset path is `data/MedalCare-XL/MedalCare-XL`; alternatively set
`MEDALCARE_ROOT` to the directory containing `WP2_largeDataset_Noise` and
`WP2_largeDataset_ParameterFiles`. Set `ECG_FM_CKPT` if the checkpoint is not
at `ckpt/mimic_iv_ecg_physionet_pretrained.pt`.

Example when the dataset and checkpoint are mounted elsewhere on the Pod:

```bash
export MEDALCARE_ROOT=/workspace/datasets/MedalCare-XL/MedalCare-XL
export ECG_FM_CKPT=/workspace/checkpoints/mimic_iv_ecg_physionet_pretrained.pt
```

The expected dataset structure includes:

```text
$MEDALCARE_ROOT/
  WP2_largeDataset_Noise/
  WP2_largeDataset_ParameterFiles/
```

## Validate and run the requested sensitivity analysis

First build/check the index. This should report 16,842 indexed records out of
16,900 expected with 99.7% overall coverage; 56 IAB and 2 RBBB records are
unavailable in the current source data.

```bash
python src/exp1_index.py
```

Then run the eight-group Test A sensitivity analysis on the A10, excluding all
records from the two run IDs that span the provided splits:

```bash
python src/probe.py \
  --device cuda \
  --ctrl-seeds 3 \
  --exclude-run-ids run_S64 run_S67
```

The provided train/validation/test assignments are not changed. This run
excludes 2,777 indexed ECGs, retains all eight groups in every split, and groups
cross-validation by `group::source_shape_id`. Its embedding caches and results
use separate `_sensitivity_without_S64_S67` names, leaving primary-run outputs
untouched. Results are written under `runs/` and embeddings under
`cache/embeddings/`.

For a shorter initial GPU/checkpoint smoke test before the full run:

```bash
python src/exp0_smoke.py --device cuda
```
