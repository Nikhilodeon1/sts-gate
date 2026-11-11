"""
baselines.py – Round-3 validation suite.

Addresses the open reviewer items with experiments that write their own JSON outputs
(nothing in the dossier should cite a number that is not in runs/R3_*.json):

  null_diag      Within-mesh variance of every parameter; per-parameter breakdown of the
                 within-stratum permutation null (the saved null was constant 47 over 1,000 perms);
                 real decoys that vary within mesh (iid noise, mesh-offset noise, auto-found index columns).
  features       Identical Test C (discovery on train, confirmation on held-out) on SAE atoms, dense L8,
                 ICA, random 6144-d expansion at matched sparsity, random-init dense; best-on-train feature
                 per parameter; ridge R^2 on held-out meshes with mesh-cluster bootstrap CIs and
                 paired bootstrap SAE-minus-baseline differences.
  selectivity    Association-count distribution incl. the share of features with zero associations,
                 two thresholds, and by activation-frequency bin.
  partialout     APD/stimulus associations after partialling out measured QT / T-amplitude / QRS / HR.
  repeated       Repeated mesh-grouped discovery/confirmation splits (per-atom confirmation frequency).
  ablation       Ablation on a linear readout: many frequency-matched random sets, targeted-minus-random,
                 logit-margin metric (non-saturating), dense-direction ablation, Holm adjustment.
  stability      Cross-seed decoder cosine with frequency-matched background, in-subspace random baseline,
                 Wilson CIs, and functional replication of (atom, parameter) associations in seeds 1 and 2.
  centered_sae   Retrain SAEs on mean-centred, variance-normalised tokens (pretrained and random-init) and
                 rerun Test C on them (the previous random-init SAE failed to train: global EV < 0).

Not implemented here (requires model-internal patching or new training code, listed in the dossier):
spliced-in loss recovered through layers 9-12, from-scratch CNN upper bound, extended T-wave/ST
handcrafted set, TopK/JumpReLU variants.

Run (Colab):  python src/baselines.py --device cuda --all
              python src/baselines.py --only null_diag features
"""
import sys, json, argparse, traceback, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from data import build_index
from assoc import (MECHANISTIC_PARAMS, MORPHOLOGY_PARAMS, ALL_PARAMS,
                   partial_spearman, bh_qvalues, rho_to_pval)

EXCLUDE = ["run_S64", "run_S67"]
SUFFIX = "_sensitivity_without_S64_S67"
PHYS_PARAMS = [p for p in MECHANISTIC_PARAMS]
APD_STIM_PREFIX = ("v_APD", "v_stim")


def _save(name: str, obj) -> Path:
    C.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    p = C.RUNS_DIR / name
    p.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                 encoding="utf-8")
    print(f"  saved {p}")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Context
# ─────────────────────────────────────────────────────────────────────────────

class Ctx:
    """Everything the analyses need, built from an index + feature matrices."""

    def __init__(self, index: pd.DataFrame, A_sae: np.ndarray | None = None,
                 A_dense: np.ndarray | None = None, include_run_id: bool = False):
        self.index = index.reset_index(drop=True)
        ix = self.index
        self.labels = ix["group"].map(C.GROUP_TO_INT).to_numpy()
        self.mesh = ix["model_id"].astype(str).to_numpy()
        sp = ix["split"].to_numpy()
        self.tr, self.va, self.te = sp == "train", sp == "validation", sp == "test"
        self.held = self.va | self.te
        cov = [pd.get_dummies(ix["folder"], prefix="f", drop_first=True).astype(float).values,
               pd.get_dummies(ix["model_id"].astype(str), prefix="m", drop_first=True).astype(float).values]
        if include_run_id:
            cov.append(pd.get_dummies(ix["run_id"].astype(str), prefix="r", drop_first=True).astype(float).values)
        self.C_mat = np.concatenate(cov, axis=1)
        self.A_sae, self.A_dense = A_sae, A_dense


def build_ctx(layer: int = 8, exclude=EXCLUDE) -> Ctx:
    index = build_index()
    if exclude:
        index = index.loc[~index["run_id"].isin(exclude)].reset_index(drop=True)
    morph = C.CACHE_DIR / f"morphology{SUFFIX}.parquet"
    if morph.exists():
        m = pd.read_parquet(morph)
        if len(m) == len(index):
            for col in MORPHOLOGY_PARAMS:
                if col in m.columns:
                    index[col] = m[col].values
    else:
        print("[WARN] morphology cache missing: morphology parameters unavailable.")
    A_sae = A_dense = None
    p = C.CACHE_DIR / f"atom_acts{SUFFIX}_layer{layer}.npy"
    if p.exists():
        A_sae = np.load(p)
    p = C.CACHE_EMBEDDINGS / f"pretrained{SUFFIX}" / f"layer{layer}.npy"
    if p.exists():
        A_dense = np.load(p)
    for name, arr in [("A_sae", A_sae), ("A_dense", A_dense)]:
        if arr is not None and len(arr) != len(index):
            raise RuntimeError(f"{name} has {len(arr)} rows, index has {len(index)}")
    return Ctx(index, A_sae, A_dense)


# ─────────────────────────────────────────────────────────────────────────────
# Association engine (same maths as assoc.py, but with an explicit parameter list)
# ─────────────────────────────────────────────────────────────────────────────

def assoc(A: np.ndarray, idx: pd.DataFrame, Cm: np.ndarray, params: list[str],
          candidate_keys: set | None = None) -> dict:
    rho_d, p_d, n_d = {}, {}, {}
    for param in params:
        if param not in idx.columns:
            continue
        pv = pd.to_numeric(idx[param], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(pv)
        n = int(ok.sum())
        if n < 20 or float(np.std(pv[ok])) < 1e-8:
            continue
        rhos, va = partial_spearman(A[ok], pv[ok], Cm[ok])
        if not va.any():
            continue
        pvs = rho_to_pval(rhos, n)
        for i in np.where(va)[0]:
            rho_d[(int(i), param)], p_d[(int(i), param)], n_d[(int(i), param)] = float(rhos[i]), float(pvs[i]), n
    if not p_d:
        return {}
    keys = list(p_d)
    pv = np.array([p_d[k] for k in keys])
    q_full = bh_qvalues(pv)
    q = q_full
    if candidate_keys:
        cm = np.array([k in candidate_keys for k in keys])
        q = np.ones_like(q_full)
        if cm.any():
            q[cm] = bh_qvalues(pv[cm])
    return {k: {"rho": rho_d[k], "p": p_d[k], "q": float(q[i]), "q_full": float(q_full[i]), "n": n_d[k]}
            for i, k in enumerate(keys)}


def discover_confirm(F: np.ndarray, ctx: Ctx, params: list[str], rho_thr: float = C.ASSOC_RHO_THR,
                     tr_mask=None, held_mask=None, Cm=None) -> dict:
    tr_mask = ctx.tr if tr_mask is None else tr_mask
    held_mask = ctx.held if held_mask is None else held_mask
    Cm = ctx.C_mat if Cm is None else Cm
    disc = assoc(F[tr_mask], ctx.index[tr_mask], Cm[tr_mask], params)
    cands = {k for k, v in disc.items() if v["q"] < C.ASSOC_Q_THR and abs(v["rho"]) >= rho_thr}
    conf = assoc(F[held_mask], ctx.index[held_mask], Cm[held_mask], params, cands or None)
    rows = []
    for k in cands:
        c = conf.get(k)
        if c and c["q"] < C.ASSOC_Q_THR and abs(c["rho"]) >= rho_thr and np.sign(c["rho"]) == np.sign(disc[k]["rho"]):
            rows.append({"feature": k[0], "param": k[1], "rho_disc": disc[k]["rho"], "rho_conf": c["rho"]})
    return {"disc": disc, "conf": conf, "confirmed": pd.DataFrame(rows), "n_tests": len(disc),
            "n_candidates": len(cands)}


def summarize_test_c(res: dict, params: list[str]) -> dict:
    df = res["confirmed"]
    out = {"n_tests_discovery": res["n_tests"], "n_candidates": res["n_candidates"],
           "n_confirmed_pairs": int(len(df)),
           "n_confirmed_features": int(df["feature"].nunique()) if len(df) else 0, "per_param": {}}
    for p in params:
        keys = [k for k in res["disc"] if k[1] == p]
        if not keys:
            continue
        best = max(keys, key=lambda k: abs(res["disc"][k]["rho"]))
        c = res["conf"].get(best)
        sub = df[df["param"] == p] if len(df) else df
        out["per_param"][p] = {
            "n_confirmed_features": int(sub["feature"].nunique()) if len(sub) else 0,
            "best_on_train_feature": best[0],
            "best_on_train_abs_rho_disc": abs(res["disc"][best]["rho"]),
            "same_feature_rho_heldout": None if c is None else c["rho"],
        }
    return out


def feature_family_counts(df: pd.DataFrame) -> dict:
    if not len(df):
        return {}
    fam = {"APD": df["param"].str.startswith("v_APD"), "stim": df["param"].str.startswith("v_stim"),
           "isch": df["param"].str.startswith("v_isch"), "morph": df["param"].str.startswith("morph_")}
    return {k: int(df.loc[m, "feature"].nunique()) for k, m in fam.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Null / decoy diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def _shuffle_within(values: np.ndarray, groups: list[np.ndarray], rng) -> np.ndarray:
    out, orig = values.copy(), values.copy()
    for g in groups:
        if len(g) > 1:
            out[g] = orig[rng.permutation(g)]
    return out


def within_mesh_variance_table(ctx: Ctx, params: list[str]) -> dict:
    ix = ctx.index
    strata = ix.groupby(["folder", "model_id"]).indices
    groups = [np.asarray(v) for v in strata.values()]
    out = {}
    for p in params:
        if p not in ix.columns:
            continue
        v = pd.to_numeric(ix[p], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(v)
        if ok.sum() < 20:
            continue
        tot = float(np.nanvar(v[ok]))
        within = 0.0
        n = 0
        for g in groups:
            gv = v[g][np.isfinite(v[g])]
            if len(gv) > 1:
                within += float(np.var(gv)) * len(gv)
                n += len(gv)
        wfrac = (within / max(n, 1)) / tot if tot > 0 else float("nan")
        out[p] = {"n": int(ok.sum()), "total_var": tot, "within_stratum_var_fraction": wfrac,
                  "n_strata_with_variation": int(sum(1 for g in groups if np.nanstd(v[g]) > 1e-9))}
    return out


def null_diagnostics(ctx: Ctx, F: np.ndarray, n_perm: int = 5, seed: int = 0) -> dict:
    """Break down which parameters produce passes under within-(folder,mesh) shuffles on the held-out set."""
    rng = np.random.default_rng(seed)
    held_idx = ctx.index[ctx.held].reset_index(drop=True)
    groups = [np.asarray(v) for v in held_idx.groupby(["folder", "model_id"]).indices.values()]
    Fh, Ch = F[ctx.held], ctx.C_mat[ctx.held]
    params = [p for p in MECHANISTIC_PARAMS if p in held_idx.columns]
    per_perm = []
    for _ in range(n_perm):
        shuf = held_idx.copy()
        for p in params:
            shuf[p] = _shuffle_within(pd.to_numeric(held_idx[p], errors="coerce").to_numpy(dtype=float), groups, rng)
        res = assoc(Fh, shuf, Ch, params)
        passing = [k for k, v in res.items() if v["q"] < C.ASSOC_Q_THR and abs(v["rho"]) >= C.ASSOC_RHO_THR]
        by_param = {}
        for _, p in passing:
            by_param[p] = by_param.get(p, 0) + 1
        per_perm.append({"n_pass": len(passing), "by_param": by_param,
                         "max_abs_rho": max([abs(v["rho"]) for v in res.values()] or [0.0])})
    return {"n_perm": n_perm, "per_perm": per_perm,
            "unique_pass_counts": sorted({r["n_pass"] for r in per_perm}),
            "within_mesh_variance": within_mesh_variance_table(ctx, MECHANISTIC_PARAMS + MORPHOLOGY_PARAMS)}


def decoy_controls(ctx: Ctx, F: np.ndarray, seed: int = 0) -> dict:
    """Decoys that (a) vary within mesh and (b) are unrelated to the ECG by construction."""
    rng = np.random.default_rng(seed)
    ix = ctx.index.copy()
    n = len(ix)
    mesh_codes = pd.factorize(ix["model_id"].astype(str))[0]
    mesh_off = rng.normal(size=mesh_codes.max() + 1)
    decoys = {
        "decoy_iid_noise": rng.normal(size=n),
        "decoy_mesh_offset_plus_noise": mesh_off[mesh_codes] + rng.normal(size=n),
    }
    # auto-discovered numeric index columns that are not analysed parameters
    meta = {"split", "ecg_path", "folder", "group", "run_id", "model_id"}
    auto = []
    wtab = within_mesh_variance_table(ctx, [c for c in ix.columns if c not in meta and c not in ALL_PARAMS])
    for c, d in wtab.items():
        if d["within_stratum_var_fraction"] > 0.05 and pd.to_numeric(ix[c], errors="coerce").nunique() > 5:
            auto.append(c)
            decoys[f"auto::{c}"] = pd.to_numeric(ix[c], errors="coerce").to_numpy(dtype=float)
    for k, v in decoys.items():
        ix[k] = v
    dctx = Ctx(ix, include_run_id=False)
    res = discover_confirm(F, dctx, list(decoys))
    df = res["confirmed"]
    return {"decoys_tested": list(decoys), "auto_columns_found": auto,
            "n_confirmed_pairs_by_decoy": df.groupby("param").size().to_dict() if len(df) else {},
            "n_candidates_discovery": res["n_candidates"], "n_tests": res["n_tests"],
            "within_mesh_var_fraction_auto": {c: wtab[c]["within_stratum_var_fraction"] for c in auto}}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature sets, ridge R^2 with CIs
# ─────────────────────────────────────────────────────────────────────────────

def random_expansion(A_dense: np.ndarray, tr_mask: np.ndarray, d_out: int, active_frac: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    mu, sd = A_dense[tr_mask].mean(0), A_dense[tr_mask].std(0) + 1e-8
    Z = (A_dense - mu) / sd
    W = rng.normal(size=(Z.shape[1], d_out)).astype(np.float32) / math.sqrt(Z.shape[1])
    H = np.maximum(Z.astype(np.float32) @ W, 0.0)
    k = max(1, int(round(active_frac * d_out)))
    thr = np.partition(H, -k, axis=1)[:, -k][:, None]
    return np.log1p(np.where(H >= thr, H, 0.0)).astype(np.float32)


def ica_features(A_dense: np.ndarray, tr_mask: np.ndarray, n_components: int = 768, seed: int = 0):
    from sklearn.decomposition import FastICA
    mu, sd = A_dense[tr_mask].mean(0), A_dense[tr_mask].std(0) + 1e-8
    ica = FastICA(n_components=min(n_components, A_dense.shape[1]), whiten="unit-variance",
                  max_iter=300, tol=1e-3, random_state=seed)
    ica.fit((A_dense[tr_mask] - mu) / sd)
    return ica.transform((A_dense - mu) / sd).astype(np.float32)


class RidgeBank:
    """Ridge via a cached eigendecomposition of the train Gram matrix per row-mask signature."""

    def __init__(self, X: np.ndarray, alphas=(1.0, 10.0, 100.0, 1e3, 1e4, 1e5)):
        self.X, self.alphas, self.cache = X, alphas, {}

    def _eig(self, rows: np.ndarray):
        key = hash(rows.tobytes())
        if key not in self.cache:
            Xr = self.X[rows].astype(np.float64)
            mu, sd = Xr.mean(0), Xr.std(0) + 1e-8
            Xs = (Xr - mu) / sd
            lam, V = np.linalg.eigh(Xs.T @ Xs)
            self.cache[key] = (mu, sd, Xs, np.clip(lam, 0, None), V)
        return self.cache[key]

    def fit_predict(self, y: np.ndarray, tr_rows: np.ndarray, val_rows: np.ndarray, te_rows: np.ndarray):
        mu, sd, Xs, lam, V = self._eig(tr_rows)
        ytr = y[tr_rows]
        ym = ytr.mean()
        b = V.T @ (Xs.T @ (ytr - ym))
        def pred(rows, a):
            Xe = (self.X[rows].astype(np.float64) - mu) / sd
            return Xe @ (V @ (b / (lam + a))) + ym
        best_a, best_r2 = self.alphas[0], -np.inf
        for a in self.alphas:
            r2 = r2_score(y[val_rows], pred(val_rows, a))
            if r2 > best_r2:
                best_a, best_r2 = a, r2
        return pred(te_rows, best_a), best_a, best_r2


def r2_score(y, p) -> float:
    ss = float(((y - y.mean()) ** 2).sum())
    return float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else float("nan")


def mesh_bootstrap(y, preds: dict[str, np.ndarray], mesh: np.ndarray, n_boot=500, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    um = np.unique(mesh)
    where = {m: np.where(mesh == m)[0] for m in um}
    boots = {k: [] for k in preds}
    for _ in range(n_boot):
        ids = np.concatenate([where[m] for m in rng.choice(um, len(um), replace=True)])
        if np.std(y[ids]) < 1e-12:
            continue
        for k, p in preds.items():
            boots[k].append(r2_score(y[ids], p[ids]))
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] if v else [None, None]
            for k, v in boots.items()} | {"_boots": boots}


def decodability(ctx: Ctx, feature_sets: dict[str, np.ndarray], params: list[str], n_boot: int = 500) -> dict:
    out = {}
    banks = {k: RidgeBank(v) for k, v in feature_sets.items()}
    for p in params:
        if p not in ctx.index.columns:
            continue
        y = pd.to_numeric(ctx.index[p], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(y)
        tr_rows, va_rows, te_rows = (np.where(ok & m)[0] for m in (ctx.tr, ctx.va, ctx.te))
        if len(te_rows) < 20 or np.std(y[te_rows]) < 1e-8 or len(tr_rows) < 50:
            continue
        preds, rec = {}, {}
        for name, bank in banks.items():
            pr, a, vr2 = bank.fit_predict(y, tr_rows, va_rows, te_rows)
            preds[name] = pr
            rec[name] = {"r2_test": r2_score(y[te_rows], pr), "alpha": a, "r2_val": vr2}
        mesh_te = ctx.mesh[te_rows]
        bs = mesh_bootstrap(y[te_rows], preds, mesh_te, n_boot)
        boots = bs.pop("_boots")
        for name in rec:
            rec[name]["r2_test_ci95_mesh_bootstrap"] = bs[name]
        base = "sae" if "sae" in preds else next(iter(preds))
        rec["_paired_diff_vs_" + base] = {}
        for name in preds:
            if name != base and boots[name] and boots[base]:
                m = min(len(boots[name]), len(boots[base]))
                d = np.array(boots[base][:m]) - np.array(boots[name][:m])
                rec["_paired_diff_vs_" + base][name] = {
                    "mean": float(np.mean(d)), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
        rec["_n_test_records"], rec["_n_test_meshes"] = int(len(te_rows)), int(len(np.unique(mesh_te)))
        out[p] = rec
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Selectivity with explicit denominators
# ─────────────────────────────────────────────────────────────────────────────

def selectivity(F: np.ndarray, ctx: Ctx, thresholds=(0.20, 0.25), params=None) -> dict:
    params = params or (MECHANISTIC_PARAMS + MORPHOLOGY_PARAMS)
    res = assoc(F[ctx.held], ctx.index[ctx.held], ctx.C_mat[ctx.held], params)
    nfeat = F.shape[1]
    freq = (F[ctx.held] > 0).mean(0)
    out = {"n_features": nfeat, "n_heldout_ecgs": int(ctx.held.sum()), "thresholds": {}}
    for thr in thresholds:
        cnt = np.zeros(nfeat, dtype=int)
        for (f, _), v in res.items():
            if v["q_full"] < C.ASSOC_Q_THR and abs(v["rho"]) >= thr:
                cnt[f] += 1
        d = {"share_any": float((cnt >= 1).mean()), "share_zero": float((cnt == 0).mean()),
             "share_exactly_one_of_all": float((cnt == 1).mean()), "share_ge3_of_all": float((cnt >= 3).mean()),
             "among_active": {"n": int((cnt >= 1).sum()),
                              "share_exactly_one": float((cnt[cnt >= 1] == 1).mean()) if (cnt >= 1).any() else None,
                              "share_ge3": float((cnt[cnt >= 1] >= 3).mean()) if (cnt >= 1).any() else None,
                              "mean_params": float(cnt[cnt >= 1].mean()) if (cnt >= 1).any() else None}}
        bins = [(0, 0.02), (0.02, 0.1), (0.1, 0.3), (0.3, 0.7), (0.7, 1.01)]
        d["by_frequency_bin"] = {}
        for lo, hi in bins:
            m = (freq > lo) & (freq <= hi) if lo > 0 else (freq <= hi)
            if m.sum():
                d["by_frequency_bin"][f"({lo},{hi}]"] = {
                    "n_features": int(m.sum()), "share_any": float((cnt[m] >= 1).mean()),
                    "share_ge3_of_all": float((cnt[m] >= 3).mean())}
        out["thresholds"][str(thr)] = d
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Partial-out measured QT / T-wave features
# ─────────────────────────────────────────────────────────────────────────────

def partial_out_morphology(ctx: Ctx, F: np.ndarray,
                           covs=("morph_qt_interval", "morph_t_amplitude", "morph_qrs_duration", "morph_hr")) -> dict:
    ix = ctx.index
    avail = [c for c in covs if c in ix.columns]
    if not avail:
        return {"status": "morphology features unavailable"}
    cols = []
    for c in avail:
        v = pd.to_numeric(ix[c], errors="coerce").to_numpy(dtype=float)
        v = np.where(np.isfinite(v), v, np.nanmedian(v))
        cols.append(((rankdata(v) - rankdata(v).mean()) / (rankdata(v).std() + 1e-8))[:, None])
    Cm2 = np.concatenate([ctx.C_mat] + cols, axis=1)
    params = [p for p in MECHANISTIC_PARAMS if p.startswith(APD_STIM_PREFIX)]
    base = discover_confirm(F, ctx, params)
    adj = discover_confirm(F, ctx, params, Cm=Cm2)
    b = {(r.feature, r.param) for r in base["confirmed"].itertuples()} if len(base["confirmed"]) else set()
    a = {(r.feature, r.param) for r in adj["confirmed"].itertuples()} if len(adj["confirmed"]) else set()
    by_param = {}
    for p in params:
        bp, ap = {k for k in b if k[1] == p}, {k for k in a if k[1] == p}
        if bp:
            by_param[p] = {"baseline_pairs": len(bp), "after_partialling_pairs": len(ap),
                           "retained": len(bp & ap), "retained_fraction": len(bp & ap) / len(bp)}
    return {"covariates_partialled": avail, "baseline_pairs": len(b), "after_pairs": len(a),
            "retained_pairs": len(b & a), "retained_fraction": len(b & a) / max(len(b), 1),
            "baseline_atoms": len({k[0] for k in b}), "after_atoms": len({k[0] for k in a}), "per_param": by_param}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Repeated mesh-grouped discovery / confirmation
# ─────────────────────────────────────────────────────────────────────────────

def mesh_grouped_masks(ctx: Ctx, confirm_frac: float, rng):
    disc = np.zeros(len(ctx.index), dtype=bool)
    conf = np.zeros(len(ctx.index), dtype=bool)
    for g in sorted(ctx.index["group"].unique()):
        meshes = np.unique(ctx.mesh[(ctx.index["group"] == g).to_numpy()])
        rng.shuffle(meshes)
        n_conf = max(1, int(round(confirm_frac * len(meshes))))
        n_conf = min(n_conf, len(meshes) - 1) if len(meshes) > 1 else 0
        cm = set(meshes[:n_conf])
        m_conf = np.isin(ctx.mesh, list(cm)) & (ctx.index["group"] == g).to_numpy()
        m_disc = (ctx.index["group"] == g).to_numpy() & ~m_conf
        conf |= m_conf
        disc |= m_disc
    return disc, conf


def repeated_confirmation(ctx: Ctx, F: np.ndarray, n_repeats: int = 5, confirm_frac: float = 0.3,
                          params=None, seed: int = 0) -> dict:
    params = params or PHYS_PARAMS
    rng = np.random.default_rng(seed)
    pair_hits: dict = {}
    per_repeat = []
    for r in range(n_repeats):
        d, c = mesh_grouped_masks(ctx, confirm_frac, rng)
        res = discover_confirm(F, ctx, params, tr_mask=d, held_mask=c)
        df = res["confirmed"]
        per_repeat.append({"n_discovery_meshes": int(len(np.unique(ctx.mesh[d]))),
                           "n_confirm_meshes": int(len(np.unique(ctx.mesh[c]))),
                           "n_confirmed_pairs": int(len(df)),
                           "n_confirmed_features": int(df["feature"].nunique()) if len(df) else 0})
        for row in df.itertuples():
            pair_hits[(row.feature, row.param)] = pair_hits.get((row.feature, row.param), 0) + 1
        print(f"  repeat {r + 1}/{n_repeats}: {per_repeat[-1]}")
    freqs = np.array(list(pair_hits.values())) / n_repeats if pair_hits else np.array([])
    return {"n_repeats": n_repeats, "confirm_mesh_fraction": confirm_frac, "per_repeat": per_repeat,
            "pairs_confirmed_in_any": len(pair_hits),
            "pairs_confirmed_in_all": int((freqs == 1.0).sum()),
            "median_confirmation_frequency": float(np.median(freqs)) if len(freqs) else None,
            "confirmed_pairs_mean_count": float(np.mean([p["n_confirmed_pairs"] for p in per_repeat]))}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Ablation on a linear readout with proper baselines
# ─────────────────────────────────────────────────────────────────────────────

def _fit_ovr(Z_tr, y_tr):
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Z_tr)
    clf = OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=1000)).fit(sc.transform(Z_tr), y_tr)
    W = np.stack([e.coef_[0] for e in clf.estimators_])
    b = np.array([e.intercept_[0] for e in clf.estimators_])
    return sc, W, b


def _auc(y_bin, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_bin, s)) if 0 < y_bin.sum() < len(y_bin) else float("nan")


def _holm(p: dict) -> dict:
    items = sorted(p.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, v) in enumerate(items):
        run = max(run, min(1.0, (m - i) * v))
        out[k] = run
    return out


def ablation_suite(ctx: Ctx, k_main: int = 16, doses=(1, 2, 4, 8, 16, 32), n_random: int = 200,
                   n_boot: int = 500, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {"k_main": k_main, "n_random_sets": n_random, "selection_split": "train",
           "readouts": {}}
    classes = C.GROUP_LABELS
    for name, Z, zero_is_mean in [("sae_atoms", ctx.A_sae, False), ("dense_neurons", ctx.A_dense, True)]:
        if Z is None:
            continue
        y = ctx.labels
        Ztr, Zte, ytr, yte = Z[ctx.tr], Z[ctx.te], y[ctx.tr], y[ctx.te]
        sc, W, b = _fit_ovr(Ztr, ytr)
        Zs_te = sc.transform(Zte)
        dec_base = Zs_te @ W.T + b
        if zero_is_mean:
            repl = np.zeros(Z.shape[1])
        else:
            repl = (0.0 - sc.mean_) / sc.scale_
        # frequency bins for matched random sets (SAE only); dense: uniform random
        freq = (Ztr > 0).mean(0)
        order = np.argsort(freq)
        rank_of = np.empty_like(order)
        rank_of[order] = np.arange(len(order))
        n_feat = Z.shape[1]
        n_bins = 10
        bin_of = np.minimum((rank_of * n_bins) // n_feat, n_bins - 1)
        bin_members = [np.where(bin_of == i)[0] for i in range(n_bins)]
        mesh_te = ctx.mesh[ctx.te]

        def ablate(atoms):
            return dec_base + (repl[atoms][None, :] - Zs_te[:, atoms]) @ W[:, atoms].T

        def matched_random(atoms, exclude):
            chosen = []
            for a in atoms:
                pool = bin_members[bin_of[a]] if name == "sae_atoms" else np.arange(n_feat)
                pool = np.setdiff1d(pool, np.concatenate([exclude, np.array(chosen, dtype=int)]))
                chosen.append(int(rng.choice(pool)))
            return np.array(chosen)

        res_cls = {}
        sd = Ztr.std(0) + 1e-8
        for ci, cname in enumerate(classes):
            bin_tr = (ytr == ci).astype(float)
            corr = np.abs(((Ztr - Ztr.mean(0)).T @ (bin_tr - bin_tr.mean())) / (len(Ztr) * sd * (bin_tr.std() + 1e-12)))
            ranked = np.argsort(-corr)
            yb = (yte == ci).astype(int)
            base_auc = _auc(yb, dec_base[:, ci])
            marg_base = dec_base[yb == 1, ci].mean() if yb.sum() else float("nan")
            per_k = {}
            for k in sorted(set(list(doses) + [k_main])):
                tgt = ranked[:k]
                dec_t = ablate(tgt)
                d_auc_t = base_auc - _auc(yb, dec_t[:, ci])
                d_marg_t = marg_base - dec_t[yb == 1, ci].mean()
                rd_auc, rd_marg = [], []
                for _ in range(n_random if k in (k_main, 1, 4, 32) else max(20, n_random // 5)):
                    ra = matched_random(tgt, tgt)
                    dec_r = ablate(ra)
                    rd_auc.append(base_auc - _auc(yb, dec_r[:, ci]))
                    rd_marg.append(marg_base - dec_r[yb == 1, ci].mean())
                rd_auc, rd_marg = np.array(rd_auc), np.array(rd_marg)
                # record-level paired bootstrap on targeted drop (anti-conservative: records within mesh are not independent)
                bt = []
                for _ in range(n_boot if k == k_main else 0):
                    ii = rng.integers(0, len(yb), len(yb))
                    if 0 < yb[ii].sum() < len(ii):
                        bt.append(_auc(yb[ii], dec_base[ii, ci]) - _auc(yb[ii], dec_t[ii, ci]))
                other = [c for c in range(len(classes)) if c != ci]
                off = np.nanmean([_auc((yte == c).astype(int), dec_base[:, c]) - _auc((yte == c).astype(int), dec_t[:, c])
                                  for c in other])
                per_k[str(k)] = {
                    "targeted_auc_drop": d_auc_t, "random_auc_drop_mean": float(np.mean(rd_auc)),
                    "random_auc_drop_sd": float(np.std(rd_auc)),
                    "targeted_minus_random_mean": float(d_auc_t - rd_auc.mean()),
                    "p_empirical_random_ge_targeted": float((1 + (rd_auc >= d_auc_t).sum()) / (1 + len(rd_auc))),
                    "off_target_mean_auc_drop": float(off),
                    "margin_drop_targeted": float(d_marg_t), "margin_drop_random_mean": float(rd_marg.mean()),
                    "margin_drop_random_sd": float(rd_marg.std()),
                    "p_empirical_margin": float((1 + (rd_marg >= d_marg_t).sum()) / (1 + len(rd_marg))),
                    "targeted_drop_record_bootstrap_ci95_anticonservative":
                        [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))] if bt else None,
                }
            res_cls[cname] = {"baseline_auc": base_auc, "n_test_records": int(len(yb)),
                              "n_test_meshes_for_class": int(len(np.unique(mesh_te[yb == 1]))), "per_k": per_k}
        p_main = {c: res_cls[c]["per_k"][str(k_main)]["p_empirical_random_ge_targeted"] for c in classes}
        holm = _holm(p_main)
        for c in classes:
            res_cls[c]["per_k"][str(k_main)]["p_holm_over_8_classes"] = holm[c]
        out["readouts"][name] = {"replacement": "0 (absence)" if not zero_is_mean else "train mean",
                                 "per_class": res_cls}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 7. Stability: frequency-matched background, in-subspace random baseline, functional replication
# ─────────────────────────────────────────────────────────────────────────────

def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [c - h, c + h]


def _freq_bins(freq: np.ndarray, n_bins: int = 10):
    order = np.argsort(freq)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return np.minimum((rank * n_bins) // len(freq), n_bins - 1)


def decoder_max_cos(D0: np.ndarray, D1: np.ndarray) -> np.ndarray:
    a = D0 / np.clip(np.linalg.norm(D0, axis=1, keepdims=True), 1e-8, None)
    b = D1 / np.clip(np.linalg.norm(D1, axis=1, keepdims=True), 1e-8, None)
    return np.abs(a @ b.T).max(axis=1)


def stability_matched(D0: np.ndarray, Ds: dict[int, np.ndarray], freq: np.ndarray, confirmed: np.ndarray,
                      token_cov_sample: np.ndarray | None = None, n_draw: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    alive = freq > 0
    bins = _freq_bins(np.where(alive, freq, -1.0))
    conf = np.unique(confirmed)
    bg = np.setdiff1d(np.where(alive)[0], conf)
    out = {"n_confirmed": int(len(conf)), "n_background_alive": int(len(bg)), "seeds": {}}
    for s, Ds_ in Ds.items():
        mc = decoder_max_cos(D0, Ds_)
        r = {}
        for thr in (0.5, 0.7, 0.8):
            k = int((mc[conf] >= thr).sum())
            draws = []
            for _ in range(n_draw):
                pick = []
                for a in conf:
                    pool = bg[bins[bg] == bins[a]]
                    pool = pool if len(pool) else bg
                    pick.append(rng.choice(pool))
                draws.append(float((mc[np.array(pick)] >= thr).mean()))
            r[f"thr_{thr}"] = {
                "confirmed_rate": k / len(conf), "confirmed_k": k, "wilson_ci95": wilson(k, len(conf)),
                "freq_matched_background_rate_mean": float(np.mean(draws)),
                "freq_matched_background_ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
                "enrichment_vs_matched": float((k / len(conf)) / max(np.mean(draws), 1e-9)),
                "all_alive_rate": float((mc[alive] >= thr).mean())}
        r["mean_max_cos_confirmed"] = float(mc[conf].mean())
        r["mean_max_cos_freq_matched_background"] = float(np.mean(
            [mc[np.array([rng.choice(bg[bins[bg] == bins[a]] if (bins[bg] == bins[a]).any() else bg) for a in conf])].mean()
             for _ in range(200)]))
        r["mean_max_cos_all_alive"] = float(mc[alive].mean())
        if token_cov_sample is not None:
            X = token_cov_sample - token_cov_sample.mean(0)
            lam, U = np.linalg.eigh(np.cov(X.T))
            lam = np.clip(lam, 0, None)
            Z = rng.normal(size=(D0.shape[0], D0.shape[1])) * np.sqrt(lam)[None, :]
            rand_dirs = Z @ U.T
            rc = decoder_max_cos(rand_dirs, Ds_)
            r["in_subspace_random_mean_max_cos"] = float(rc.mean())
        gauss = rng.normal(size=D0.shape)
        r["isotropic_gaussian_mean_max_cos"] = float(decoder_max_cos(gauss, Ds_).mean())
        out["seeds"][str(s)] = r
    return out


def functional_replication(D0, Ds, rho_seed0: dict, acts_by_seed: dict, ctx: Ctx, freq0: np.ndarray,
                           params: list[str], n_draw: int = 200, seed: int = 0) -> dict:
    """For each confirmed (atom, param) in seed 0, does the cosine-matched atom in seed s show the same association?"""
    rng = np.random.default_rng(seed)
    out = {}
    for s, Ds_ in Ds.items():
        A = acts_by_seed[s]
        res = assoc(A[ctx.held], ctx.index[ctx.held], ctx.C_mat[ctx.held], params)
        a0 = D0 / np.clip(np.linalg.norm(D0, axis=1, keepdims=True), 1e-8, None)
        b = Ds_ / np.clip(np.linalg.norm(Ds_, axis=1, keepdims=True), 1e-8, None)
        sim = np.abs(a0 @ b.T)
        match, cosm = sim.argmax(1), sim.max(1)
        freq_s = (A[ctx.held] > 0).mean(0)
        bins = _freq_bins(freq_s)
        rows = []
        for (atom, param), r0 in rho_seed0.items():
            m = int(match[atom])
            v = res.get((m, param))
            rows.append({"cos": float(cosm[atom]), "agree_sign": bool(v and np.sign(v["rho"]) == np.sign(r0)),
                         "replicates": bool(v and np.sign(v["rho"]) == np.sign(r0)
                                            and abs(v["rho"]) >= C.ASSOC_RHO_THR and v["q_full"] < C.ASSOC_Q_THR)})
        df = pd.DataFrame(rows)
        rand_rates = []
        for _ in range(n_draw):
            hits = 0
            for (atom, param), r0 in rho_seed0.items():
                m = int(match[atom])
                pool = np.where(bins == bins[m])[0]
                rm = int(rng.choice(pool))
                v = res.get((rm, param))
                hits += bool(v and np.sign(v["rho"]) == np.sign(r0) and abs(v["rho"]) >= C.ASSOC_RHO_THR
                             and v["q_full"] < C.ASSOC_Q_THR)
            rand_rates.append(hits / max(len(rho_seed0), 1))
        out[str(s)] = {
            "n_pairs": int(len(df)),
            "replicates_all_pairs": float(df["replicates"].mean()),
            "replicates_pairs_with_cos_ge_0.7": float(df.loc[df["cos"] >= 0.7, "replicates"].mean())
            if (df["cos"] >= 0.7).any() else None,
            "n_pairs_cos_ge_0.7": int((df["cos"] >= 0.7).sum()),
            "sign_agreement_all_pairs": float(df["agree_sign"].mean()),
            "freq_matched_random_atom_replication_rate_mean": float(np.mean(rand_rates)),
            "freq_matched_random_atom_replication_ci95": [float(np.percentile(rand_rates, 2.5)),
                                                           float(np.percentile(rand_rates, 97.5))]}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 8. Centred / variance-normalised SAE (fixes random-init SAE training failure)
# ─────────────────────────────────────────────────────────────────────────────

def _train_centered(shard_dir: Path, mu: np.ndarray, s: float, device: str, epochs: int, seed: int,
                    lr: float = C.SAE_LR, batch: int = C.SAE_BATCH):
    import torch, torch.nn.functional as Fnn
    from sae import BatchTopKSAE, _shard_paths
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sae = BatchTopKSAE().to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    shards = _shard_paths(shard_dir, "train")
    for ep in range(epochs):
        sae.train()
        tot, nb = 0.0, 0
        for si in rng.permutation(len(shards)):
            toks = (np.load(shards[si]).astype(np.float32) - mu) / s
            toks = toks[rng.permutation(len(toks))]
            for i in range(0, len(toks), batch):
                x = torch.from_numpy(toks[i:i + batch]).to(device)
                xhat, _ = sae(x)
                loss = Fnn.mse_loss(xhat, x)
                opt.zero_grad(); loss.backward(); opt.step(); sae.normalize_decoder()
                tot += loss.item(); nb += 1
        print(f"  [centred SAE] epoch {ep + 1}/{epochs} mse={tot / max(nb, 1):.5f} thr={sae.thr.item():.4f}")
    return sae.eval()


def _eval_centered(sae, shard_dir: Path, mu: np.ndarray, s: float, device: str) -> dict:
    import torch
    from sae import _shard_paths
    shards = _shard_paths(shard_dir, "held")
    se, tv, n, l0, fired = 0.0, 0.0, 0, 0.0, None
    sx, sx2 = np.zeros(mu.shape[0]), np.zeros(mu.shape[0])
    for p in shards:
        x = (np.load(p).astype(np.float32) - mu) / s
        sx += x.sum(0, dtype=np.float64); sx2 += (x.astype(np.float64) ** 2).sum(0); n += len(x)
        for i in range(0, len(x), 8192):
            xb = torch.from_numpy(x[i:i + 8192]).to(device)
            with torch.no_grad():
                xhat, z = sae(xb)
            se += float(((xb - xhat) ** 2).sum()); l0 += float((z > 0).sum())
            f = (z > 0).any(dim=0).cpu().numpy()
            fired = f if fired is None else (fired | f)
    var_tot = float((sx2 - sx ** 2 / n).sum())
    return {"ev_feature_wise_global": 1.0 - se / var_tot, "mean_l0": l0 / n,
            "dead_fraction_heldout": float(1.0 - fired.mean()), "n_heldout_tokens": int(n)}


class _CenteredWrapper:
    def __init__(self, sae, mu, s, device):
        import torch
        self.sae, self.device = sae, device
        self.mu = torch.from_numpy(mu.astype(np.float32)).to(device)
        self.s = float(s)

    def eval(self):
        self.sae.eval(); return self

    def __call__(self, x):
        return self.sae((x - self.mu) / self.s)


def centered_sae_pipeline(ctx: Ctx, tag: str, ctrl: bool, device: str, layer: int = 8, epochs: int = C.SAE_EPOCHS,
                          seed: int = 0) -> dict:
    import torch
    from sae import TOKEN_SCRATCH_ROOT, _shard_paths, extract_tokens
    from model import load_ecgfm
    from assoc import compute_atom_activations
    shard_dir = TOKEN_SCRATCH_ROOT / f"{('ctrl_seed' + str(seed)) if ctrl else 'pretrained'}{SUFFIX}_layer{layer}"
    model, _, _ = load_ecgfm(random_init=ctrl, seed=seed)
    model.to(device)
    if not (_shard_paths(shard_dir, "train") and _shard_paths(shard_dir, "held")):
        print(f"  token shards missing at {shard_dir}; extracting...")
        extract_tokens(layer, model, shard_dir, device=device, exclude_run_ids=EXCLUDE)
    s1, s2, n = np.zeros(C.SAE_D_IN), np.zeros(C.SAE_D_IN), 0
    for p in _shard_paths(shard_dir, "train"):
        x = np.load(p).astype(np.float64)
        s1 += x.sum(0); s2 += (x ** 2).sum(0); n += len(x)
    mu = (s1 / n).astype(np.float32)
    var = s2 / n - (s1 / n) ** 2
    s = float(np.sqrt(var.mean()))
    print(f"  [{tag}] token mean-norm={np.linalg.norm(mu):.3f}  per-dim std (rms)={s:.4f}")
    sae = _train_centered(shard_dir, mu, s, device, epochs, seed + 42)
    metrics = _eval_centered(sae, shard_dir, mu, s, device)
    wrapper = _CenteredWrapper(sae, mu, s, device)
    acts_path = C.CACHE_DIR / f"atom_acts_centered_{tag}{SUFFIX}_layer{layer}.npy"
    acts = compute_atom_activations(ctx.index, wrapper, model, layer, 1.0, device)
    np.save(acts_path, acts)
    torch.save({"state": sae.state_dict(), "mu": mu, "s": s}, C.CACHE_DIR / f"sae_centered_{tag}{SUFFIX}_layer{layer}.pt")
    tc = discover_confirm(acts, ctx, MECHANISTIC_PARAMS + MORPHOLOGY_PARAMS)
    summ = summarize_test_c(tc, MECHANISTIC_PARAMS + MORPHOLOGY_PARAMS)
    summ["families"] = feature_family_counts(tc["confirmed"])
    return {"tag": tag, "random_init": ctrl, "token_mean_norm": float(np.linalg.norm(mu)), "token_rms_std": s,
            "sae_metrics": metrics, "test_c": summ}


def pooled_dense(ctx: Ctx, ctrl: bool, device: str, layer: int = 8, seed: int = 0) -> np.ndarray:
    import torch
    from model import load_ecgfm, extract_layers
    from data import load_medalcare
    tag = f"ctrl_seed{seed}{SUFFIX}" if ctrl else f"pretrained{SUFFIX}"
    path = C.CACHE_EMBEDDINGS / tag / f"layer{layer}.npy"
    if path.exists():
        arr = np.load(path)
        if len(arr) == len(ctx.index):
            return arr
    model, _, _ = load_ecgfm(random_init=ctrl, seed=seed)
    model.to(device).eval()
    outs = []
    for i in range(0, len(ctx.index), 32):
        rows = ctx.index.iloc[i:i + 32]
        w0, w1 = [], []
        for _, r in rows.iterrows():
            w = load_medalcare(r["ecg_path"])
            w0.append(w[0]); w1.append(w[1])
        with torch.no_grad():
            f0 = extract_layers(model, torch.from_numpy(np.stack(w0)).float(), layers=[layer], device=device)[layer].mean(1)
            f1 = extract_layers(model, torch.from_numpy(np.stack(w1)).float(), layers=[layer], device=device)[layer].mean(1)
        outs.append(((f0 + f1) / 2).cpu().numpy())
    arr = np.concatenate(outs)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def _load_decoders(layer: int, seeds=(0, 1, 2)) -> dict:
    import torch
    out = {}
    for s in seeds:
        tag = f"_seed{s}" if s else ""
        for p in [C.CACHE_DIR / f"sae_pretrained{SUFFIX}_layer{layer}{tag}.pt"] + ([C.SAE_WEIGHTS] if s == 0 else []):
            if p.exists():
                st = torch.load(p, map_location="cpu")
                st = st.get("model_state", st) if isinstance(st, dict) else st
                out[s] = st["W_dec"].float().numpy()
                break
    return out


def _seed_acts(ctx: Ctx, layer: int, seed: int, device: str) -> np.ndarray:
    import torch
    from sae import BatchTopKSAE
    from model import load_ecgfm
    from assoc import compute_atom_activations
    path = C.CACHE_DIR / f"atom_acts{SUFFIX}_layer{layer}_seed{seed}.npy"
    if path.exists():
        arr = np.load(path)
        if len(arr) == len(ctx.index):
            return arr
    sae = BatchTopKSAE()
    sae.load_state_dict(torch.load(C.CACHE_DIR / f"sae_pretrained{SUFFIX}_layer{layer}_seed{seed}.pt", map_location="cpu"))
    sae.to(device)
    scale = float(np.load(C.CACHE_DIR / f"scale_pretrained{SUFFIX}_layer{layer}.npy"))
    model, _, _ = load_ecgfm(random_init=False)
    model.to(device)
    arr = compute_atom_activations(ctx.index, sae, model, layer, scale, device)
    np.save(path, arr)
    return arr


def run_null_diag(ctx, args):
    out = {"within_mesh_variance": within_mesh_variance_table(ctx, MECHANISTIC_PARAMS + MORPHOLOGY_PARAMS)}
    if ctx.A_sae is not None:
        out["null_diagnostics"] = null_diagnostics(ctx, ctx.A_sae, n_perm=args.n_perm)
        out["decoys"] = decoy_controls(ctx, ctx.A_sae)
    else:
        print("[INFO] A_sae not loaded: ran within-mesh variance analysis; full feature nulls require cached activations.")
    return out


def run_features(ctx, args):
    phys = MECHANISTIC_PARAMS + MORPHOLOGY_PARAMS
    sets = {"sae": ctx.A_sae, "dense": ctx.A_dense}
    frac = float((ctx.A_sae > 0).mean())
    sets["random_expansion_6144"] = random_expansion(ctx.A_dense, ctx.tr, ctx.A_sae.shape[1], frac)
    sets["ica_768"] = ica_features(ctx.A_dense, ctx.tr)
    ctrl_dense = pooled_dense(ctx, True, args.device)
    sets["dense_random_init"] = ctrl_dense
    out = {"sae_active_fraction_used_for_matched_sparsity": frac, "test_c": {}}
    for name, F in sets.items():
        print(f"[features] Test C on {name} {F.shape}")
        tc = discover_confirm(F, ctx, phys)
        s = summarize_test_c(tc, phys)
        s["families"] = feature_family_counts(tc["confirmed"])
        out["test_c"][name] = s
    r2_params = [p for p in ["v_APD.min", "v_APD.rho_d", "v_stim[3].z", "v_stim[3].phi", "v_stim[4].z",
                             "morph_qt_interval", "morph_t_amplitude"] if p in ctx.index.columns]
    out["r2_note"] = "ridge; alpha chosen on validation meshes; CIs are mesh-cluster bootstrap over test meshes"
    out["r2"] = decodability(ctx, sets, r2_params)
    return out


def run_selectivity(ctx, args):
    out = {"sae": selectivity(ctx.A_sae, ctx, params=None), "dense": selectivity(ctx.A_dense, ctx)}
    out["ica_768"] = selectivity(ica_features(ctx.A_dense, ctx.tr), ctx)
    return out


def run_partialout(ctx, args):
    return partial_out_morphology(ctx, ctx.A_sae)


def run_repeated(ctx, args):
    return repeated_confirmation(ctx, ctx.A_sae, n_repeats=args.n_repeats)


def run_ablation(ctx, args):
    return ablation_suite(ctx, n_random=args.n_random)


def run_stability(ctx, args):
    dec = _load_decoders(8)
    if 0 not in dec or len(dec) < 2:
        return {"status": "need seed 0 plus at least one more seed", "found": list(dec)}
    csv = C.RUNS_DIR / f"C_confirmed_atoms{SUFFIX}.csv"
    df = pd.read_csv(csv)
    conf_atoms = df["atom"].unique()
    freq = (ctx.A_sae[ctx.held] > 0).mean(0)
    tok = None
    shard_dir = None
    try:
        from sae import TOKEN_SCRATCH_ROOT, _shard_paths
        shards = _shard_paths(TOKEN_SCRATCH_ROOT / f"pretrained{SUFFIX}_layer8", "train")
        if shards:
            tok = np.load(shards[0]).astype(np.float32)
            tok = tok[np.random.default_rng(0).choice(len(tok), min(100000, len(tok)), replace=False)]
            tok = tok / float(np.load(C.CACHE_DIR / f"scale_pretrained{SUFFIX}_layer8.npy"))
    except Exception as e:
        print(f"  [warn] token sample unavailable for in-subspace baseline: {e}")
    Ds = {s: d for s, d in dec.items() if s != 0}
    out = {"matched": stability_matched(dec[0], Ds, freq, conf_atoms, tok)}
    rho0 = {(int(r.atom), r.param): float(r.rho_conf) for r in df.itertuples()}
    acts = {s: _seed_acts(ctx, 8, s, args.device) for s in Ds}
    params = sorted({p for _, p in rho0})
    out["functional_replication"] = functional_replication(dec[0], Ds, rho0, acts, ctx, freq, params)
    return out


def run_centered(ctx, args):
    out = {}
    for tag, ctrl in [("pretrained", False), ("random_init", True)]:
        try:
            out[tag] = centered_sae_pipeline(ctx, tag, ctrl, args.device, epochs=args.epochs)
        except Exception:
            traceback.print_exc()
            out[tag] = {"status": "failed", "traceback": traceback.format_exc()}
    return out


ANALYSES = {
    "null_diag": (run_null_diag, "R3_null_and_decoys.json", []),
    "features": (run_features, "R3_feature_baselines.json", ["A_sae", "A_dense"]),
    "selectivity": (run_selectivity, "R3_selectivity.json", ["A_sae", "A_dense"]),
    "partialout": (run_partialout, "R3_partial_out_morphology.json", ["A_sae"]),
    "repeated": (run_repeated, "R3_repeated_mesh_confirmation.json", ["A_sae"]),
    "ablation": (run_ablation, "R3_ablation.json", ["A_sae", "A_dense"]),
    "stability": (run_stability, "R3_stability.json", ["A_sae"]),
    "centered_sae": (run_centered, "R3_centered_sae.json", []),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--only", nargs="*", choices=list(ANALYSES))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n-perm", type=int, default=5)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=C.SAE_EPOCHS)
    args = ap.parse_args()
    todo = list(ANALYSES) if (args.all or not args.only) else args.only
    ctx = build_ctx(args.layer)
    status = {}
    for name in todo:
        fn, fname, needs = ANALYSES[name]
        print("\n" + "=" * 70 + f"\n{name}\n" + "=" * 70)
        try:
            missing = [n for n in needs if getattr(ctx, n) is None]
            if missing:
                raise FileNotFoundError(f"missing cached inputs {missing}")
            _save(fname, fn(ctx, args))
            status[name] = "ok"
        except Exception as e:
            traceback.print_exc()
            status[name] = f"FAILED {type(e).__name__}: {e}"
    print("\nSUMMARY")
    for k, v in status.items():
        print(f"  {k:14s} {v}")


if __name__ == "__main__":
    main()
