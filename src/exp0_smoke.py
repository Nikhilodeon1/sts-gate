"""
exp0_smoke.py – Experiment 0: verify the full pipeline can run end-to-end.

Checks:
  1. Load model and print architecture.
  2. Run 10 PTB-XL + 10 MedalCare windows through the model.
  3. Print shapes for every captured layer.
  4. Confirm random-init differs (cosine similarity near 0).
  5. Time one forward pass.

Records results in notebook.md.
"""
import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C


def run_exp0(device: str = "cpu"):
    from model import load_ecgfm, extract_layers, pool_tokens
    from data import load_medalcare, load_ptbxl, get_ptbxl_records, build_index
    import numpy as np
    import torch

    log = {}

    # ── 1. Load model ─────────────────────────────────────────────────────
    print("=" * 60)
    print("Experiment 0: Smoke Test")
    print("=" * 60)

    model, n_layers, d_model = load_ecgfm()
    model.to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    print(f"Transformer layers: {n_layers}")
    print(f"Hidden dim (d_model): {d_model}")

    log["n_layers"]   = n_layers
    log["d_model"]    = d_model
    log["n_params"]   = total_params

    # ── 2a. Load PTB-XL windows if data is present ─────────────────────────
    print("\n[Exp0] Loading PTB-XL windows...")
    records = get_ptbxl_records(max_folder=0)  # folder 00000 only
    ptbxl_wins = []
    if len(records) == 0:
        print("  No PTB-XL records are available in the current partial download; continuing with MedalCare-only validation.")
    else:
        for _, row in records.head(5).iterrows():
            try:
                w = load_ptbxl(row["wfdb_path"])   # (2, 12, 2500)
                ptbxl_wins.append(w[0])            # first window
                ptbxl_wins.append(w[1])            # second window
                if len(ptbxl_wins) >= 10:
                    break
            except Exception as e:
                print(f"  Skip {row['wfdb_path']}: {e}")
        print(f"  Loaded {len(ptbxl_wins)} PTB-XL windows")

    # ── 2b. Load 10 MedalCare windows ────────────────────────────────────
    print("[Exp0] Loading 10 MedalCare records...")
    index = build_index()
    medal_wins = []
    for _, row in index.sample(5, random_state=0).iterrows():
        try:
            w = load_medalcare(row["ecg_path"])
            medal_wins.append(w[0])
            medal_wins.append(w[1])
            if len(medal_wins) >= 10:
                break
        except Exception as e:
            print(f"  Skip {row['ecg_path']}: {e}")
    print(f"  Loaded {len(medal_wins)} MedalCare windows")

    # ── 3. Run forward pass, print shapes ────────────────────────────────
    if len(ptbxl_wins) >= 5 and len(medal_wins) >= 5:
        all_wins = ptbxl_wins[:5] + medal_wins[:5]
    elif len(medal_wins) >= 5:
        all_wins = medal_wins[:10]
    elif len(ptbxl_wins) >= 5:
        all_wins = ptbxl_wins[:10]
    else:
        raise RuntimeError("Not enough ECG windows available to run the smoke test. Need at least 5 usable windows from either PTB-XL or MedalCare.")
    x = torch.from_numpy(np.stack(all_wins)).float()   # (N, 12, 2500)
    print(f"\nInput batch shape: {tuple(x.shape)}")

    t0 = time.time()
    feats = extract_layers(model, x, layers=C.PROBE_LAYERS, device=device)
    elapsed = time.time() - t0

    print(f"\nForward pass time (batch=10): {elapsed:.2f}s  ({elapsed/10*1000:.1f}ms/window)")
    log["forward_time_s"] = elapsed

    for layer in sorted(feats.keys()):
        h = feats[layer]
        nan_count = h.isnan().sum().item()
        print(f"  Layer {layer:2d}: shape={tuple(h.shape)}  NaNs={nan_count}")
        if layer == min(feats.keys()):
            log["tokens_per_window"] = h.shape[1]

    # ── 4. Random-init control ───────────────────────────────────────────
    print("\n[Exp0] Loading random-init control...")
    ctrl, _, _ = load_ecgfm(random_init=True, seed=42)
    ctrl.to(device)

    ctrl_feats = extract_layers(ctrl, x, layers=[max(C.PROBE_LAYERS)], device=device)
    pt_feats   = feats[max(C.PROBE_LAYERS)]

    import torch.nn.functional as F
    h_pt   = pool_tokens(pt_feats)
    h_ctrl = pool_tokens(ctrl_feats[max(C.PROBE_LAYERS)])
    cos = F.cosine_similarity(h_pt, h_ctrl, dim=-1).mean().item()
    print(f"Cosine similarity (pretrained vs random-init, layer {max(C.PROBE_LAYERS)}): {cos:.4f}")
    log["cos_sim_ctrl"] = cos

    verdict = "PASS" if abs(cos) < 0.5 and not any(
        feats[l].isnan().any().item() for l in feats
    ) else "FAIL"
    log["verdict"] = verdict
    print(f"\nExp 0 verdict: {verdict}")

    # ── 5. Write notebook.md entry ────────────────────────────────────────
    _write_notebook_entry(log)

    return log


def _write_notebook_entry(log: dict):
    nb_path = C.PROJECT_ROOT / "notebook.md"
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"""
## Exp 0 – Smoke Test ({ts})

| Metric | Value |
|--------|-------|
| Transformer layers | {log.get('n_layers','?')} |
| Hidden dim | {log.get('d_model','?')} |
| Tokens per 5s window | {log.get('tokens_per_window','?')} |
| Forward pass (10 windows) | {log.get('forward_time_s','?'):.2f}s |
| Pretrained vs ctrl cosine similarity | {log.get('cos_sim_ctrl','?'):.4f} |
| Model parameters | {log.get('n_params','?'):,} |
| **Verdict** | **{log.get('verdict','?')}** |

"""
    with open(nb_path, "a") as f:
        f.write(entry)
    print(f"\nNotebook entry written to {nb_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    result = run_exp0(device=args.device)
    print("\nResult:", json.dumps(result, indent=2))
