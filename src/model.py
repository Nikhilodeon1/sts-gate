"""
model.py – ECG-FM loading, layer hooks, and random-init control.

Key facts verified from infer_quickstart.ipynb:
  - Model class: ECGTransformerClassificationModel (fairseq_signals)
  - Loaded with: build_model_from_checkpoint(checkpoint_path=...)
  - Layer path:  model.encoder.encoder.layers   (list of transformer layers)
  - Encoder output: out['encoder_out']  shape (batch, tokens, dim)
  - The notebook uses out['encoder_out'] directly – so it IS batch-first.

Unverified (marked *verify* in guide):
  - Exact layer count and hidden dim (expect 12 / 768).
  - Whether layers return (out,) or (out, attn) tuples.

Usage
-----
  model, n_layers, d_model = load_ecgfm(C.ECG_FM_CKPT)
  ctrl,  _,       _        = load_ecgfm(C.ECG_FM_CKPT, random_init=True, seed=0)
  feats = extract_layers(model, batch_tensor, layers=[4,6,8,10,12])
  # feats: dict  layer_int -> (batch, tokens, dim)  cpu float32
"""
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_ecgfm(ckpt_path: str | Path | None = None,
               random_init: bool = False,
               seed: int = 0) -> tuple:
    """
    Load the ECG-FM wav2vec-2 model from a fairseq_signals checkpoint.

    Parameters
    ----------
    ckpt_path    : path to .pt checkpoint.  Defaults to config.ECG_FM_CKPT.
    random_init  : if True, load the architecture but DO NOT restore weights.
    seed         : random seed used when random_init=True.

    Returns
    -------
    (model, n_layers, d_model)
    """
    # fairseq_signals must be on sys.path; add ECG-FM repo if needed
    _add_fairseq_to_path()

    from fairseq_signals.models import build_model_from_checkpoint  # type: ignore

    ckpt_path = Path(ckpt_path or C.ECG_FM_CKPT)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Download mimic_iv_ecg_physionet_pretrained.pt from "
            f"huggingface.co/wanglab/ecg-fm and place it in {C.CKPT_DIR}/"
        )

    model = build_model_from_checkpoint(checkpoint_path=str(ckpt_path))

    if random_init:
        torch.manual_seed(seed)
        for p in model.parameters():
            if p.dim() >= 2:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p, -0.1, 0.1)

    model.eval()

    # Inspect layer count and hidden dim
    layers = _get_layers(model)
    n_layers = len(layers)

    # d_model from first encoder weight that has >=2 dims
    d_model = None
    for p in model.parameters():
        if p.dim() >= 2:
            d_model = max(p.shape)
            break
    d_model = d_model or C.SAE_D_IN

    print(f"ECG-FM loaded: n_layers={n_layers}, d_model={d_model}, "
          f"random_init={random_init}")
    return model, n_layers, d_model


def _add_fairseq_to_path():
    """Add the fairseq_signals checkout used by ECG-FM to sys.path."""
    candidates = [
        C.FAIRSEQ_SIGNALS_REPO,
        C.ECG_FM_REPO,
        C.ECG_FM_REPO / "fairseq-signals",
        C.DATASETS_ROOT / "fairseq-signals",
        C.DATASETS_ROOT,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        fairseq_dir = candidate / "fairseq_signals"
        if fairseq_dir.is_dir():
            p = str(candidate)
            if p not in sys.path:
                sys.path.insert(0, p)
            return
    # Fallback: maybe it is installed in the venv
    pass


def _get_layers(model) -> list:
    """
    Return the list of transformer encoder layers from the model.
    Tries common attribute paths; prints a warning if none found.
    """
    for attr_path in [
        "encoder.encoder.layers",
        "encoder.layers",
        "model.encoder.encoder.layers",
    ]:
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__len__") and len(obj) > 0:
                return list(obj)
        except AttributeError:
            continue
    print("[WARNING] Could not locate transformer layers; hook extraction will fail.")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Layer hook extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_layers(model, x: torch.Tensor,
                   layers: list[int] | None = None,
                   device: str = "cpu") -> dict[int, torch.Tensor]:
    """
    Run a forward pass and return intermediate layer outputs.

    Parameters
    ----------
    model   : loaded ECG-FM model
    x       : (batch, 12, 2500) float32 tensor
    layers  : 1-indexed layer numbers to capture.  None -> all.
    device  : 'cpu' or 'cuda'

    Returns
    -------
    dict  layer_int -> (batch, tokens, dim)  cpu float32
    """
    model_layers = _get_layers(model)
    n = len(model_layers)
    if layers is None:
        layers = list(range(1, n + 1))

    feats: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(module, inp, out):
            # Out may be a tuple (hidden_state, ...) or just a tensor
            h = out[0] if isinstance(out, tuple) else out
            h = h.detach().float().cpu()
            # Normalise to (batch, tokens, dim)
            if h.dim() == 3 and h.shape[0] != x.shape[0]:
                # Likely (tokens, batch, dim) – transpose
                h = h.permute(1, 0, 2).contiguous()
            feats[layer_idx] = h
        return hook

    for i, layer in enumerate(model_layers, start=1):
        if i in layers:
            h = layer.register_forward_hook(make_hook(i))
            handles.append(h)

    model = model.to(device)
    x_dev = x.to(device)
    with torch.no_grad():
        model(source=x_dev)

    for h in handles:
        h.remove()

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Embedding pooling helper
# ─────────────────────────────────────────────────────────────────────────────

def pool_tokens(hidden: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool token dimension: (batch, tokens, dim) -> (batch, dim).
    Uses non-zero mask matching ECG-FM's own aggregation:
        sum(h) / count(h != 0)
    """
    mask = (hidden != 0).any(dim=-1, keepdim=True).float()   # (B, T, 1)
    summed = (hidden * mask).sum(dim=1)
    denom  = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom   # (B, dim)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test helper
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test(ckpt_path: str | Path | None = None, device: str = "cpu"):
    """
    Exp 0 smoke test: load model, run 10 random windows, check shapes,
    confirm random-init differs.
    """
    print("=" * 60)
    print("ECG-FM Smoke Test")
    print("=" * 60)

    model, n_layers, d_model = load_ecgfm(ckpt_path)
    ctrl,  _,       _        = load_ecgfm(ckpt_path, random_init=True, seed=42)

    # Dummy batch: 10 windows of (12, 2500)
    x = torch.randn(10, 12, C.WIN_SAMPLES)

    import time
    t0 = time.time()
    feats = extract_layers(model, x, layers=C.PROBE_LAYERS, device=device)
    elapsed = time.time() - t0

    print(f"\nForward pass time (10 windows): {elapsed:.2f}s")
    for layer, h in sorted(feats.items()):
        print(f"  Layer {layer:2d}: {tuple(h.shape)}  "
              f"NaNs={h.isnan().sum().item()}")

    # Check random-init differs
    ctrl_feats = extract_layers(ctrl, x, layers=[n_layers], device=device)
    pt_feats   = extract_layers(model, x, layers=[n_layers], device=device)
    h_pt   = pool_tokens(pt_feats[n_layers])
    h_ctrl = pool_tokens(ctrl_feats[n_layers])
    cos = torch.nn.functional.cosine_similarity(h_pt, h_ctrl, dim=-1).mean().item()
    print(f"\nPretrained vs random-init cosine similarity (layer {n_layers}): {cos:.4f}")
    print("PASS" if abs(cos) < 0.5 else "WARNING: similarity unexpectedly high")

    tokens_per_window = feats[list(feats.keys())[0]].shape[1]
    print(f"\nTokens per 5s window: {tokens_per_window}")
    print(f"Hidden dim (d_model):  {d_model}")
    print(f"Transformer layers:    {n_layers}")

    return {
        "n_layers": n_layers,
        "d_model":  d_model,
        "tokens":   tokens_per_window,
        "cos_sim":  cos,
        "time_s":   elapsed,
    }


if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    result = smoke_test(args.ckpt, args.device)
    print("\nSmoke test result:", json.dumps(result, indent=2))
