"""
pilot.py – lightweight pilot gate on a pilot-only bundle (see package_medalcare.py --pilot).

Tiers:
  smoke  few ECGs per class: model load, decode, inference, timing. No AUROC.
  probe  all pilot ECGs through the existing Test A probe code. Exploratory only.

This is NOT Exp 1 and NOT Test A. It does not require the full-dataset coverage gate and its
metrics must not be used to lock a layer or claim a Test A verdict.

Run (needs ECG_BUNDLE_ROOT, ECG_CACHE_DIR, ECG_FM_CKPT):
    python src/pilot.py --tier smoke --device cuda
    python src/pilot.py --tier probe --device cuda
"""
import sys, os, json, time, argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from data import build_index, load_medalcare

SMOKE_PER_GROUP = 3


def _load_pilot_index():
    if C.ECG_BUNDLE_ROOT is None:
        raise SystemExit("ECG_BUNDLE_ROOT is unset; the pilot runs only on a pilot bundle.")
    if "ECG_CACHE_DIR" not in os.environ:
        # build_index() in bundle mode overwrites <cache>/index.parquet
        raise SystemExit("Set ECG_CACHE_DIR to a scratch dir so the full local index is not overwritten.")
    meta = json.loads((C.ECG_BUNDLE_ROOT / "bundle.json").read_text(encoding="utf-8"))
    if "pilot" not in meta:
        raise SystemExit("Bundle has no 'pilot' metadata: refusing to run the pilot on a non-pilot bundle.")
    return build_index(), meta["pilot"]


def _check_decode(index):
    bad = []
    for path in index["ecg_path"]:
        try:
            w = load_medalcare(path)
            if w.shape != (2, 12, C.WIN_SAMPLES) or not np.isfinite(w).all():
                bad.append(f"{path}: bad shape/values")
        except Exception as e:
            bad.append(f"{path}: {e}")
    if bad:
        raise SystemExit(f"{len(bad)} ECGs failed to decode, e.g. {bad[:3]}")


def _gpu_info(device):
    import torch
    if not device.startswith("cuda"):
        return {"device": device}
    return {
        "device": device,
        "gpu": torch.cuda.get_device_name(0),
        "peak_gpu_mem_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
    }


def run_smoke(index, device):
    import torch
    from model import load_ecgfm
    from probe import embed_medalcare

    sub = index.groupby("group", group_keys=False).head(SMOKE_PER_GROUP).reset_index(drop=True)
    _check_decode(sub)
    model, n_layers, d_model = load_ecgfm()
    model.to(device)
    t0 = time.time()
    out = embed_medalcare(model, sub, C.PROBE_LAYERS, device=device)
    elapsed = time.time() - t0
    nan_layers = [l for l, e in out["embeddings"].items() if not np.isfinite(e).all()]
    if nan_layers:
        raise SystemExit(f"Non-finite embeddings in layers {nan_layers}")
    return {
        "tier": "smoke",
        "n_ecgs": len(sub),
        "n_windows": 2 * len(sub),
        "groups": sorted(sub["group"].unique()),
        "n_layers": n_layers,
        "d_model": d_model,
        "embedding_shapes": {l: list(e.shape) for l, e in out["embeddings"].items()},
        "inference_seconds": round(elapsed, 2),
        "seconds_per_ecg": round(elapsed / len(sub), 3),
        "torch": torch.__version__,
        **_gpu_info(device),
    }


def _probe_layers(cache, label):
    from probe import fit_probe_layer, eval_probe

    labels, groups, splits = cache["labels"], cache["groups"], cache["splits"]
    tr, va, te = splits == "train", splits == "validation", splits == "test"
    layers = {}
    for layer, X in sorted(cache["embeddings"].items()):
        t1 = time.time()
        clf, sc = fit_probe_layer(X[tr], labels[tr], groups[tr])
        layers[layer] = {
            "val": eval_probe(clf, sc, X[va], labels[va]),
            "test": eval_probe(clf, sc, X[te], labels[te]),
            "fit_seconds": round(time.time() - t1, 1),
        }
        print(f"  {label} layer {layer}: val AUROC={layers[layer]['val']['macro_auc']:.3f} "
              f"test AUROC={layers[layer]['test']['macro_auc']:.3f}")
    return layers


def run_probe(index, device, ctrl_seeds=3):
    from sklearn.model_selection import GroupKFold
    from model import load_ecgfm
    from probe import embed_medalcare

    _check_decode(index)
    train = index[index["split"] == "train"]
    # GroupKFold does not stratify; probe scoring breaks if a fold lacks a class.
    n_classes = len(C.GROUP_LABELS)
    for k, (a, b) in enumerate(GroupKFold(5).split(train, train["group"], train["model_id"])):
        if train.iloc[a]["group"].nunique() < n_classes or train.iloc[b]["group"].nunique() < n_classes:
            raise SystemExit(f"CV fold {k} is missing a class; pilot probe not run. Change --pilot-seed.")

    model, _, _ = load_ecgfm()
    model.to(device)
    t0 = time.time()
    cache = embed_medalcare(model, index, C.PROBE_LAYERS, device=device)
    embed_s = time.time() - t0
    splits = cache["splits"]
    layers = _probe_layers(cache, "pretrained")
    result = {
        "tier": "probe",
        "n_ecgs": len(index),
        "n_train": int((splits == "train").sum()),
        "n_val": int((splits == "validation").sum()),
        "n_test": int((splits == "test").sum()),
        "embed_seconds": round(embed_s, 2),
        "layer_results": layers,
        "note": "EXPLORATORY pilot metric; no verdict; not Test A; do not lock a layer from this.",
    }

    if ctrl_seeds > 0:
        ctrl = []
        for seed in range(ctrl_seeds):
            ctrl_model, _, _ = load_ecgfm(random_init=True, seed=seed)
            ctrl_model.to(device)
            ctrl_cache = embed_medalcare(ctrl_model, index, C.PROBE_LAYERS, device=device)
            ctrl.append(_probe_layers(ctrl_cache, f"ctrl seed{seed}"))
            del ctrl_model
        gap = {}
        for layer in layers:
            row = {}
            for split in ("val", "test"):
                mean_ctrl = float(np.mean([c[layer][split]["macro_auc"] for c in ctrl]))
                row[split] = {
                    "pretrained": layers[layer][split]["macro_auc"],
                    "ctrl_mean": mean_ctrl,
                    "ctrl_per_seed": [c[layer][split]["macro_auc"] for c in ctrl],
                    "gap": layers[layer][split]["macro_auc"] - mean_ctrl,
                }
            gap[layer] = row
        result["ctrl_seeds"] = ctrl_seeds
        result["ctrl_layer_results"] = {f"seed{s}": c for s, c in enumerate(ctrl)}
        result["pretrained_vs_ctrl"] = gap
        print("\n  layer  split  pretrained  ctrl_mean   gap")
        for layer, row in gap.items():
            for split, r in row.items():
                print(f"  {layer:>5}  {split:<5}  {r['pretrained']:.3f}       {r['ctrl_mean']:.3f}      {r['gap']:+.3f}")
    return {**result, **_gpu_info(device)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["smoke", "probe"], required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--ctrl-seeds", type=int, default=3,
                    help="probe tier: random-init control seeds (0 disables)")
    args = ap.parse_args()

    out_path = args.out or C.RUNS_DIR / f"pilot_{args.tier}.json"
    if out_path.exists():
        raise SystemExit(f"Result exists; move it before rerunning: {out_path}")
    index, pilot_meta = _load_pilot_index()
    print(f"[pilot] {len(index)} ECGs; bundle purpose: {pilot_meta['purpose']}")
    t_start = time.time()
    if args.tier == "smoke":
        result = run_smoke(index, args.device)
    else:
        result = run_probe(index, args.device, args.ctrl_seeds)
    result["pilot_selection"] = {k: pilot_meta[k] for k in
                                 ("seed", "train_models_per_group", "heldout_models_per_group",
                                  "max_ecgs_per_model", "excluded_run_ids", "records")}
    result["total_seconds"] = round(time.time() - t_start, 2)
    C.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved {out_path}")
