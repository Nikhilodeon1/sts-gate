"""
probe.py – Test A: linear probe on frozen ECG-FM embeddings.

Steps:
  1. Load (or build) embedding cache for MedalCare ECGs.
  2. Fit multinomial logistic regression per layer.
  3. Evaluate macro one-vs-rest AUROC on held-out anatomical model IDs.
  4. Run random-init control (3 seeds).
  5. Save A_results.json and figures.

Run:
    python src/probe.py [--device cuda] [--rebuild-cache]
"""
import sys, json, time, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from data import build_index, load_medalcare

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ─────────────────────────────────────────────────────────────────────────────
# Embedding cache
# ─────────────────────────────────────────────────────────────────────────────

def embed_medalcare(model, index: pd.DataFrame, layers: list[int],
                    device: str = "cpu", batch_size: int = 32) -> dict:
    """
    Run all MedalCare ECGs through the model and return per-layer embeddings.

    Returns dict:
        "embeddings": {layer: np.ndarray (N, d)}
        "labels":     np.ndarray (N,) int  group index
        "groups":     np.ndarray (N,) str  anatomical model ID
        "splits":     np.ndarray (N,) str  train/validation/test
    """
    from model import extract_layers, pool_tokens

    all_embs = {l: [] for l in layers}
    all_labels = []
    all_groups = []
    all_splits = []

    N = len(index)
    for start in range(0, N, batch_size):
        batch_idx = index.iloc[start:start + batch_size]
        batch_wins = []
        for _, row in batch_idx.iterrows():
            try:
                w = load_medalcare(row["ecg_path"])  # (2, 12, 2500)
                batch_wins.append(w)
            except Exception as e:
                print(f"[WARN] Skip {row['ecg_path']}: {e}")
                batch_wins.append(np.zeros((2, 12, C.WIN_SAMPLES), dtype=np.float32))

        # Stack windows: (B, 2, 12, 2500) -> process each window separately
        # then average the two windows
        win0 = np.stack([w[0] for w in batch_wins])  # (B, 12, 2500)
        win1 = np.stack([w[1] for w in batch_wins])  # (B, 12, 2500)

        x0 = torch.from_numpy(win0).float()
        x1 = torch.from_numpy(win1).float()

        feats0 = extract_layers(model, x0, layers=layers, device=device)
        feats1 = extract_layers(model, x1, layers=layers, device=device)

        for layer in layers:
            emb0 = pool_tokens(feats0[layer]).numpy()  # (B, d)
            emb1 = pool_tokens(feats1[layer]).numpy()
            avg  = (emb0 + emb1) / 2.0
            all_embs[layer].append(avg)

        all_labels.extend(C.GROUP_TO_INT[g] for g in batch_idx["group"])
        if batch_idx["model_id"].isna().any():
            raise ValueError("Index contains rows without an anatomical model ID.")
        all_groups.extend(batch_idx["model_id"].astype(str).tolist())
        all_splits.extend(batch_idx["split"].tolist())

        print(f"  Embedded {min(start+batch_size, N)}/{N}", end="\r", flush=True)

    print()
    for layer in layers:
        all_embs[layer] = np.concatenate(all_embs[layer], axis=0)

    return {
        "embeddings": all_embs,
        "labels":     np.array(all_labels),
        "groups":     np.array(all_groups, dtype=str),
        "splits":     np.array(all_splits),
    }


def save_emb_cache(cache: dict, tag: str = "pretrained"):
    """Save embedding arrays to cache/embeddings/<tag>/."""
    out_dir = C.CACHE_EMBEDDINGS / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "labels.npy",  cache["labels"])
    np.save(out_dir / "groups.npy",  cache["groups"])
    np.save(out_dir / "splits.npy",  cache["splits"])
    for layer, arr in cache["embeddings"].items():
        np.save(out_dir / f"layer{layer}.npy", arr)
    print(f"Saved embedding cache to {out_dir}")


def load_emb_cache(tag: str = "pretrained") -> dict | None:
    out_dir = C.CACHE_EMBEDDINGS / tag
    if not out_dir.exists():
        return None
    labels = np.load(out_dir / "labels.npy")
    groups = np.load(out_dir / "groups.npy")
    splits = np.load(out_dir / "splits.npy", allow_pickle=True)
    embeddings = {}
    for path in out_dir.glob("layer*.npy"):
        layer = int(path.stem.replace("layer", ""))
        embeddings[layer] = np.load(path)
    return {"embeddings": embeddings, "labels": labels,
            "groups": groups, "splits": splits}


def _cache_matches_index(cache: dict, index: pd.DataFrame) -> bool:
    """Reject embeddings whose row metadata no longer matches the current index."""
    if "model_id" not in index or index["model_id"].isna().any():
        return False
    expected_labels = index["group"].map(C.GROUP_TO_INT).to_numpy()
    expected_groups = index["model_id"].astype(str).to_numpy()
    expected_splits = index["split"].astype(str).to_numpy()
    return (
        np.array_equal(cache["labels"], expected_labels)
        and np.array_equal(cache["groups"].astype(str), expected_groups)
        and np.array_equal(cache["splits"].astype(str), expected_splits)
        and all(len(emb) == len(index) for emb in cache["embeddings"].values())
    )


# ─────────────────────────────────────────────────────────────────────────────
# Probe fitting and evaluation
# ─────────────────────────────────────────────────────────────────────────────

def fit_probe_layer(Xtr, ytr, geom_tr):
    """Fit logistic regression with geometry-grouped CV. Returns fitted clf and scaler."""
    sc  = StandardScaler().fit(Xtr)
    clf = GridSearchCV(
        # OneVsRestClassifier == the old multi_class="ovr" (removed in scikit-learn 1.8)
        OneVsRestClassifier(LogisticRegression(max_iter=2000)),
        {"estimator__C": C.PROBE_C_GRID},
        cv=GroupKFold(n_splits=5),
        scoring="roc_auc_ovr",
        n_jobs=C.PROBE_N_JOBS,
        refit=True,
    )
    clf.fit(sc.transform(Xtr), ytr, groups=geom_tr)
    return clf, sc


def eval_probe(clf, sc, Xte, yte):
    """Evaluate probe on test split. Returns dict of metrics."""
    Xt = sc.transform(Xte)
    P  = clf.predict_proba(Xt)
    y_pred = clf.predict(Xt)
    n_classes = len(C.GROUP_LABELS)

    # Macro AUROC (one-vs-rest)
    macro_auc = roc_auc_score(yte, P, multi_class="ovr", average="macro",
                              labels=list(range(n_classes)))
    per_group_auc = {}
    for i, g in enumerate(C.GROUP_LABELS):
        mask = (yte == i)
        if mask.sum() == 0:
            continue
        y_bin = (yte == i).astype(int)
        try:
            per_group_auc[g] = roc_auc_score(y_bin, P[:, i])
        except Exception:
            per_group_auc[g] = float("nan")

    acc = accuracy_score(yte, y_pred)
    cm  = confusion_matrix(yte, y_pred, labels=list(range(n_classes)))

    return {
        "macro_auc":    float(macro_auc),
        "accuracy":     float(acc),
        "per_group_auc": {k: float(v) for k, v in per_group_auc.items()},
        "confusion_matrix": cm.tolist(),
        "best_C":       float(clf.best_params_["estimator__C"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Test A pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_test_a(device: str = "cpu",
               rebuild_cache: bool = False,
               n_ctrl_seeds: int = 3,
               max_train: int = 3000,
               exclude_run_ids: list[str] | None = None) -> dict:

    results = {}

    # ── Load data index ──────────────────────────────────────────────────────
    print("\n[Test A] Loading dataset index...")
    full_index = build_index()
    index = full_index
    excluded_run_ids = sorted(set(exclude_run_ids or []))
    unknown_run_ids = set(excluded_run_ids) - set(full_index["run_id"])
    if unknown_run_ids:
        raise ValueError(f"Unknown run IDs requested for exclusion: {sorted(unknown_run_ids)}")
    excluded_records = int(full_index["run_id"].isin(excluded_run_ids).sum())
    if excluded_run_ids:
        index = full_index.loc[~full_index["run_id"].isin(excluded_run_ids)].reset_index(drop=True)
        print(f"  Excluding {excluded_records} ECGs from run IDs: {excluded_run_ids}")
    print(f"  Total ECGs: {len(index)}")
    if "model_id" not in index or index["model_id"].isna().any():
        raise ValueError("Dataset index must contain a model_id for every ECG before Test A.")
    for split in ("train", "validation", "test"):
        labels_in_split = set(index.loc[index["split"] == split, "group"])
        missing_labels = set(C.GROUP_LABELS) - labels_in_split
        if missing_labels:
            raise ValueError(
                f"Test A requires all groups in {split}; missing {sorted(missing_labels)}."
            )
    split_counts = index.groupby("model_id")["split"].nunique()
    overlapping_models = split_counts[split_counts > 1]
    if not overlapping_models.empty:
        raise RuntimeError(
            f"Test A is blocked: {len(overlapping_models)} model IDs occur in multiple "
            "predefined splits. Resolve the model-level split assignment before evaluation."
        )
    run_split_counts = full_index.groupby("run_id")["split"].nunique()
    run_ids_crossing_splits = sorted(run_split_counts[run_split_counts > 1].index.tolist())
    results["n_index_records"] = len(index)
    results["excluded_run_ids"] = excluded_run_ids
    results["n_excluded_records"] = excluded_records
    results["run_ids_crossing_splits"] = run_ids_crossing_splits

    cache_suffix = ""
    output_suffix = ""
    if excluded_run_ids:
        id_suffix = "_".join(run_id.removeprefix("run_") for run_id in excluded_run_ids)
        cache_suffix = f"_sensitivity_without_{id_suffix}"
        output_suffix = cache_suffix

    # ── Load or build pretrained embeddings ──────────────────────────────────
    if not rebuild_cache:
        cache = load_emb_cache(f"pretrained{cache_suffix}")
    else:
        cache = None
    if cache is not None and not _cache_matches_index(cache, index):
        print("[Test A] Cached embeddings do not match the current index; rebuilding.")
        cache = None

    if cache is None:
        print("[Test A] Building pretrained embeddings...")
        from model import load_ecgfm
        model, _, _ = load_ecgfm()
        model.to(device)
        cache = embed_medalcare(model, index, C.PROBE_LAYERS, device=device)
        save_emb_cache(cache, f"pretrained{cache_suffix}")

    labels = cache["labels"]
    groups = cache["groups"]
    splits = cache["splits"]
    embeddings = cache["embeddings"]

    # ── Split masks ──────────────────────────────────────────────────────────
    tr_mask = (splits == "train")
    va_mask = (splits == "validation")
    te_mask = (splits == "test")
    held_mask = va_mask | te_mask

    # Subsample train (stratified, max 3000)
    tr_idx = np.where(tr_mask)[0]
    if len(tr_idx) > max_train:
        rng = np.random.default_rng(42)
        # Stratified subsample
        chosen = []
        for g_int in range(len(C.GROUP_LABELS)):
            g_idx = tr_idx[labels[tr_idx] == g_int]
            n_each = max_train // len(C.GROUP_LABELS)
            if len(g_idx) > n_each:
                g_idx = rng.choice(g_idx, n_each, replace=False)
            chosen.append(g_idx)
        tr_idx = np.concatenate(chosen)

    # ── Per-layer probe ──────────────────────────────────────────────────────
    layer_results = {}
    best_layer, best_val_auc = None, -1.0

    for layer in sorted(embeddings.keys()):
        X = embeddings[layer]
        Xtr = X[tr_idx];   ytr  = labels[tr_idx];  gtr = groups[tr_idx]
        Xva = X[va_mask];  yva  = labels[va_mask]
        Xte = X[te_mask];  yte  = labels[te_mask]

        print(f"\n[Test A] Layer {layer}: train={len(Xtr)}, val={len(Xva)}, test={len(Xte)}")
        clf, sc = fit_probe_layer(Xtr, ytr, gtr)
        val_metrics = eval_probe(clf, sc, Xva, yva)
        te_metrics  = eval_probe(clf, sc, Xte, yte)

        print(f"  val AUROC={val_metrics['macro_auc']:.3f}  "
              f"test AUROC={te_metrics['macro_auc']:.3f}  "
              f"best_C={val_metrics['best_C']}")

        layer_results[layer] = {
            "val":  val_metrics,
            "test": te_metrics,
        }
        if val_metrics["macro_auc"] > best_val_auc:
            best_val_auc = val_metrics["macro_auc"]
            best_layer   = layer

    print(f"\n[Test A] Best layer (by val AUROC): {best_layer}  ({best_val_auc:.3f})")
    results["layer_results"] = layer_results
    results["best_layer"]    = best_layer
    results["best_val_auc"]  = best_val_auc

    # ── Random-init control ──────────────────────────────────────────────────
    ctrl_aucs = []
    for seed in range(n_ctrl_seeds):
        ctrl_tag = f"ctrl_seed{seed}{cache_suffix}"
        ctrl_cache = load_emb_cache(ctrl_tag)
        if ctrl_cache is not None and not _cache_matches_index(ctrl_cache, index):
            print(f"[Test A] Cached control embeddings do not match the current index; rebuilding seed {seed}.")
            ctrl_cache = None
        if ctrl_cache is None:
            print(f"\n[Test A] Building random-init embeddings (seed {seed})...")
            from model import load_ecgfm
            ctrl_model, _, _ = load_ecgfm(random_init=True, seed=seed)
            ctrl_cache = embed_medalcare(ctrl_model, index, C.PROBE_LAYERS, device=device)
            save_emb_cache(ctrl_cache, ctrl_tag)

        ctrl_X = ctrl_cache["embeddings"][best_layer]
        Xtr_c  = ctrl_X[tr_idx]; Xva_c = ctrl_X[va_mask]
        clf_c, sc_c = fit_probe_layer(Xtr_c, labels[tr_idx], groups[tr_idx])
        ctrl_val = eval_probe(clf_c, sc_c, Xva_c, labels[va_mask])
        ctrl_aucs.append(ctrl_val["macro_auc"])
        print(f"  Ctrl seed {seed} val AUROC: {ctrl_val['macro_auc']:.3f}")

    results["ctrl_val_aucs"]   = ctrl_aucs
    results["ctrl_mean_auc"]   = float(np.mean(ctrl_aucs))
    results["pretrained_minus_ctrl"] = best_val_auc - float(np.mean(ctrl_aucs))

    # ── Pass / fail decision ─────────────────────────────────────────────────
    te_auc = layer_results[best_layer]["test"]["macro_auc"]
    gap    = te_auc - float(np.mean(ctrl_aucs))
    per_group = layer_results[best_layer]["test"]["per_group_auc"]
    n_above_75 = sum(1 for v in per_group.values() if v >= 0.75)

    if te_auc >= 0.80 and gap >= 0.10:
        verdict = "PASS"
    elif te_auc < 0.70:
        verdict = "FAIL_STOP"
    else:
        if n_above_75 >= 5:
            verdict = "MARGINAL_CONTINUE"
        else:
            verdict = "MARGINAL_REPORT"

    results["test_auc"]     = float(te_auc)
    results["gap_vs_ctrl"]  = float(gap)
    results["n_groups_above_75"] = n_above_75
    results["verdict"]      = verdict

    print(f"\n[Test A] Test AUROC: {te_auc:.3f}  "
          f"gap vs ctrl: {gap:.3f}  verdict: {verdict}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = C.RUNS_DIR / f"A_results{output_suffix}.json"
    C.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")

    _plot_layer_auc(layer_results, output_suffix)

    return results


def _plot_layer_auc(layer_results: dict, suffix: str = ""):
    """Save layer-by-layer AUROC figure."""
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        layers = sorted(layer_results.keys())
        val_aucs  = [layer_results[l]["val"]["test"]["macro_auc"]
                     if "test" in layer_results[l]["val"]
                     else layer_results[l]["val"]["macro_auc"]
                     for l in layers]
        test_aucs = [layer_results[l]["test"]["macro_auc"] for l in layers]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(layers, [layer_results[l]["val"]["macro_auc"] for l in layers],
                "o-", label="Validation AUROC")
        ax.plot(layers, test_aucs, "s--", label="Test AUROC")
        ax.axhline(0.80, color="green",  linestyle=":", alpha=0.7, label="Pass threshold (0.80)")
        ax.axhline(0.70, color="orange", linestyle=":", alpha=0.7, label="Stop threshold (0.70)")
        ax.set_xlabel("Transformer layer"); ax.set_ylabel("Macro AUROC (8 groups)")
        ax.set_title("Test A: ECG-FM layer-by-layer probe")
        ax.legend(); ax.grid(True, alpha=0.4)
        out = C.RUNS_DIR / f"A_layer_auc{suffix}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved AUROC figure to {out}")
    except Exception as e:
        print(f"[WARN] Could not plot: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--ctrl-seeds", type=int, default=3)
    ap.add_argument("--exclude-run-ids", nargs="+", default=None)
    args = ap.parse_args()
    run_test_a(device=args.device,
               rebuild_cache=args.rebuild_cache,
               n_ctrl_seeds=args.ctrl_seeds,
               exclude_run_ids=args.exclude_run_ids)
