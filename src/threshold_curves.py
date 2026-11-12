"""
threshold_curves.py – Confirmation Threshold Sweep (0.10 to 0.50).

Evaluates confirmed atom counts and generative parameter recall across a fine grid
of partial rank correlation thresholds (|rho| in 0.10..0.50) comparing:
  - Pretrained ECG-FM SAE
  - Centered Random-Init ECG-FM SAE
  - Dense Layer 8 Neurons
  - Random 6,144 Expansion

Run:
    python src/threshold_curves.py [--device cpu] [--layer 8]
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from baselines import build_ctx, discover_confirm, MECHANISTIC_PARAMS, random_expansion, pooled_dense

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
GEN_PARAMS = [
    "v_APD.min", "v_APD.max", "v_APD.rho_d", "v_APD.z_d", "v_APD.v_d",
    "v_stim[0].z", "v_stim[0].phi", "v_stim[0].thr",
    "v_stim[1].z", "v_stim[1].phi",
    "v_stim[2].z", "v_stim[2].phi",
    "v_stim[3].z", "v_stim[3].phi",
    "v_stim[4].time", "v_stim[4].z", "v_stim[4].phi",
    "a_rot.x", "a_rot.y", "a_rot.z"
]


def run_threshold_sweep(ctx, features: dict, params: list[str]) -> dict:
    out = {}
    for name, F in features.items():
        if F is None:
            continue
        print(f"\n[Sweep] Evaluating {name} ({F.shape[1]} features)...")
        # Run discovery once on train split to get all candidate pairs with q < 0.05
        # Then evaluate confirmation at each threshold
        curve = {}
        for thr in THRESHOLDS:
            res = discover_confirm(F, ctx, params, rho_thr=thr)
            df = res["confirmed"]
            conf_atoms = int(df["feature"].nunique()) if len(df) else 0
            conf_pairs = len(df)
            conf_params = sorted(df["param"].unique()) if len(df) else []
            recall = len(conf_params) / len(params)
            curve[f"{thr:.2f}"] = {
                "confirmed_features": conf_atoms,
                "confirmed_pairs": conf_pairs,
                "n_confirmed_params": len(conf_params),
                "parameter_recall": float(recall),
                "confirmed_params": conf_params
            }
            print(f"  thr={thr:.2f} | atoms={conf_atoms:4d} | pairs={conf_pairs:4d} | recall={recall*100:.1f}% ({len(conf_params)}/{len(params)})")
        out[name] = curve
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--layer", type=int, default=8)
    args = ap.parse_args()

    ctx = build_ctx(args.layer)
    if ctx.A_sae is None or ctx.A_dense is None:
        print("[WARN] A_sae or A_dense not cached locally. Run on Colab with data bundle.")
        return

    features = {
        "pretrained_sae_6144": ctx.A_sae,
        "dense_layer8_768": ctx.A_dense,
        "random_expansion_6144": random_expansion(ctx.A_dense, ctx.tr, ctx.A_sae.shape[1], float((ctx.A_sae > 0).mean()))
    }
    ctrl_dense = pooled_dense(ctx, True, args.device, layer=args.layer)
    if ctrl_dense is not None:
        features["random_init_dense_768"] = ctrl_dense

    res = run_threshold_sweep(ctx, features, GEN_PARAMS)
    out_path = C.RUNS_DIR / "R4_threshold_curves.json"
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n[DONE] Saved threshold curves to {out_path}")


if __name__ == "__main__":
    main()
