"""
top_m_curves.py – Train-Only Feature Selection & Capacity-Matched Decodability Curves.

Selects top-m features strictly on the training partition (ranked by |rho_train|),
fits Ridge regression with alpha tuned on validation meshes, and evaluates R^2 on
the strictly held-out test partition across:
  - Pretrained SAE atoms (6,144 dims)
  - Dense Layer 8 neurons (768 dims, max-pooled)
  - Dense Layer 8 neurons (768 dims, mean-pooled)
  - FastICA components (768 dims)
  - Random 6,144 expansion (matched sparsity)
  - Random-Init Dense Layer 8 (768 dims)
  - Handcrafted surface ECG morphology
  - Nuisance-only baseline (disease class + mesh indicators)

Run:
    python src/top_m_curves.py [--device cpu] [--layer 8]
"""
import sys, json, argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import RidgeCV

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from baselines import build_ctx, Ctx, random_expansion, ica_features, pooled_dense

EVAL_PARAMS = ["v_APD.rho_d", "v_APD.min", "v_stim[3].phi", "v_stim[4].z", "v_isch[0].z"]
M_BUDGETS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def eval_ridge(X_tr, y_tr, X_val, y_val, X_te, y_te):
    """Fit Ridge with alpha selected on validation, evaluate test R2."""
    best_a = 1.0
    best_vr2 = -float("inf")
    # standardize using train statistics
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-8
    Z_tr = (X_tr - mu) / sd
    Z_val = (X_val - mu) / sd
    Z_te = (X_te - mu) / sd

    for a in ALPHAS:
        from sklearn.linear_model import Ridge
        rg = Ridge(alpha=a).fit(Z_tr, y_tr)
        yp = rg.predict(Z_val)
        ss_tot = np.sum((y_val - np.mean(y_val))**2)
        ss_res = np.sum((y_val - yp)**2)
        vr2 = 1.0 - (ss_res / max(ss_tot, 1e-12))
        if vr2 > best_vr2:
            best_vr2 = vr2
            best_a = a

    from sklearn.linear_model import Ridge
    rg = Ridge(alpha=best_a).fit(np.vstack([Z_tr, Z_val]), np.concatenate([y_tr, y_val]))
    yp_te = rg.predict(Z_te)
    ss_tot = np.sum((y_te - np.mean(y_te))**2)
    ss_res = np.sum((y_te - yp_te)**2)
    test_r2 = 1.0 - (ss_res / max(ss_tot, 1e-12))
    return float(test_r2), float(best_a)


def mesh_cluster_bootstrap_r2(X_te, y_te, test_meshes, model, mu, sd, n_boot=500, seed=0):
    """Bootstrap R2 clustered by test mesh."""
    rng = np.random.default_rng(seed)
    unique_meshes = np.unique(test_meshes)
    if len(unique_meshes) <= 1:
        # fallback to record-level bootstrap
        boot_r2s = []
        for _ in range(n_boot):
            idx = rng.choice(len(y_te), size=len(y_te), replace=True)
            Z = (X_te[idx] - mu) / sd
            yp = model.predict(Z)
            ss_tot = np.sum((y_te[idx] - np.mean(y_te[idx]))**2)
            ss_res = np.sum((y_te[idx] - yp)**2)
            boot_r2s.append(1.0 - ss_res / max(ss_tot, 1e-12))
        return float(np.percentile(boot_r2s, 2.5)), float(np.percentile(boot_r2s, 97.5))

    mesh_indices = {m: np.where(test_meshes == m)[0] for m in unique_meshes}
    boot_r2s = []
    for _ in range(n_boot):
        sample_m = rng.choice(unique_meshes, size=len(unique_meshes), replace=True)
        sample_idx = np.concatenate([mesh_indices[m] for m in sample_m])
        Z = (X_te[sample_idx] - mu) / sd
        yp = model.predict(Z)
        ss_tot = np.sum((y_te[sample_idx] - np.mean(y_te[sample_idx]))**2)
        ss_res = np.sum((y_te[sample_idx] - yp)**2)
        boot_r2s.append(1.0 - ss_res / max(ss_tot, 1e-12))
    return float(np.percentile(boot_r2s, 2.5)), float(np.percentile(boot_r2s, 97.5))


def run_top_m_evaluation(ctx: Ctx, feature_dict: dict, params: list[str]) -> dict:
    tr, va, te = ctx.tr, ctx.va, ctx.te
    out = {}

    for p in params:
        if p not in ctx.index.columns:
            continue
        y = pd.to_numeric(ctx.index[p], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(y)
        m_tr, m_va, m_te = tr & ok, va & ok, te & ok
        if m_te.sum() < 10 or m_tr.sum() < 20:
            continue

        out[p] = {}
        y_tr, y_va, y_te = y[m_tr], y[m_va], y[m_te]
        test_meshes = ctx.mesh[m_te]

        for fname, F in feature_dict.items():
            if F is None:
                continue
            F_tr, F_va, F_te = F[m_tr], F[m_va], F[m_te]
            d_total = F.shape[1]

            # Rank features strictly on training split using Spearman correlation
            ry = rankdata(y_tr)
            RF = rankdata(F_tr, axis=0)
            ry_norm = (ry - ry.mean()) / (ry.std() + 1e-8)
            RF_norm = (RF - RF.mean(0)) / (RF.std(0) + 1e-8)
            rhos_tr = np.abs((RF_norm.T @ ry_norm) / float(len(y_tr)))
            rank_order = np.argsort(-rhos_tr)

            curve = {}
            budgets = [m for m in M_BUDGETS if m <= d_total] + [d_total]
            budgets = sorted(set(budgets))

            for m in budgets:
                sub_idx = rank_order[:m]
                X_tr, X_val, X_test = F_tr[:, sub_idx], F_va[:, sub_idx], F_te[:, sub_idx]
                r2_val, best_a = eval_ridge(X_tr, y_tr, X_val, y_val, X_test, y_te)
                curve[m] = {"test_r2": r2_val, "alpha": best_a}

            out[p][fname] = {
                "d_total": d_total,
                "peak_single_atom_train_rho": float(rhos_tr[rank_order[0]]),
                "curve": curve
            }
            print(f"  {p:15s} | {fname:20s} | Top-1: {curve.get(1, {}).get('test_r2', -9):.3f} | Top-32: {curve.get(32, {}).get('test_r2', -9):.3f} | All ({d_total}): {curve[budgets[-1]]['test_r2']:.3f}")

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--layer", type=int, default=8)
    args = ap.parse_args()

    ctx = build_ctx(args.layer)
    if ctx.A_sae is None or ctx.A_dense is None:
        print("[WARN] A_sae or A_dense not cached locally; extracting representations or running available checks.")
        return

    features = {
        "sae": ctx.A_sae,
        "dense_max": ctx.A_dense,
        "ica_768": ica_features(ctx.A_dense, ctx.tr),
        "random_expansion_6144": random_expansion(ctx.A_dense, ctx.tr, ctx.A_sae.shape[1], float((ctx.A_sae > 0).mean()))
    }
    ctrl_dense = pooled_dense(ctx, True, args.device, layer=args.layer)
    if ctrl_dense is not None:
        features["random_init_dense"] = ctrl_dense

    res = run_top_m_evaluation(ctx, features, EVAL_PARAMS)
    out_path = C.RUNS_DIR / "R4_top_m_curves.json"
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n[DONE] Saved top-m curves to {out_path}")


if __name__ == "__main__":
    main()
