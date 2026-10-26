"""
assoc.py – Test C: partial Spearman associations between SAE atoms and sim parameters.

Steps:
  1. Compute atom activations on all MedalCare ECGs: log1p(max over tokens).
  2. Load pre-locked parameter list from config (max 20 mechanistic + 6 morphology).
  3. Compute partial Spearman on train split. Identify candidates.
  4. Recompute on val+test (confirmation). Count passing atoms (BH q < 0.05, |rho| >= 0.25).
  5. Run permutation nulls (1000 permutations) and random-init null.
  6. Bin atoms: physiology / morphology / label / unexplained.
  7. Save confirmed-atom table and null distributions.

Run:
    python src/assoc.py --layer 8 [--device cuda]
"""
import sys, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from data import build_index, load_medalcare


# ─────────────────────────────────────────────────────────────────────────────
# Parameter list (LOCKED before looking at results)
# ─────────────────────────────────────────────────────────────────────────────

# Mechanistic parameters (column names as they appear in index.parquet after prefix)
MECHANISTIC_PARAMS = [
    "v_APD.min",
    "v_APD.max",
    "v_APD.rho_d",          # APD gradient
    "v_stim[0].z",          # LV fascicle site Z (height)
    "v_stim[0].phi",        # LV fascicle site angle
    "v_stim[0].rho",        # LV fascicle site radius
    "v_stim[4].time",       # RV moderator-band delay (stim[4] is RV endocardial)
    "a_cv_t.BulkTissue",    # Atrial bulk conduction velocity
    "a_cv_t.BachmannsBundle",
    "a_cv_t.CristaTerminalis",
    "a_cv_t.PectinateMuscles",
    "a_cv_t.InferiorIsthmus",
    "a_fibperc",            # Fibrosis fraction (FAM only)
    "a_rot.y",              # Atrial rotation
    "a_transl.x",           # Atrial translation
    "a_G.torso",            # Torso shape coefficient
]

# Morphology features computed by neurokit2 (computed in compute_morphology)
MORPHOLOGY_PARAMS = [
    "morph_hr",
    "morph_pr_interval",
    "morph_qrs_duration",
    "morph_qt_interval",
    "morph_r_amplitude",
    "morph_t_amplitude",
]

ALL_PARAMS = MECHANISTIC_PARAMS + MORPHOLOGY_PARAMS   # max 20+6 = 26


# ─────────────────────────────────────────────────────────────────────────────
# Partial Spearman
# ─────────────────────────────────────────────────────────────────────────────

def partial_spearman(A: np.ndarray, p: np.ndarray,
                     C_mat: np.ndarray) -> np.ndarray:
    """
    Compute partial Spearman correlation between each atom and parameter p,
    removing the shared variance explained by covariates C_mat.

    A     : (n, n_atoms)  atom activations
    p     : (n,)          parameter values
    C_mat : (n, m)        covariate matrix (e.g. one-hot subfolder + geometry)

    Returns rho : (n_atoms,)
    """
    n = len(p)
    R = np.apply_along_axis(rankdata, 0, A).astype(float)   # (n, n_atoms)
    r = rankdata(p).astype(float)                            # (n,)

    X = np.c_[np.ones(n), C_mat]   # (n, 1 + m)
    beta_A, *_ = np.linalg.lstsq(X, R, rcond=None)          # (1+m, n_atoms)
    beta_p, *_ = np.linalg.lstsq(X, r, rcond=None)          # (1+m,)

    Ra = R - X @ beta_A    # (n, n_atoms)
    rp = r - X @ beta_p    # (n,)

    Ra = (Ra - Ra.mean(0)) / (Ra.std(0) + 1e-12)
    rp = (rp - rp.mean())  / (rp.std()  + 1e-12)

    return (Ra * rp[:, None]).mean(0)   # (n_atoms,)


def bh_fdr(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns boolean mask of discoveries."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    threshold = ranks / n * q
    # Find largest rank where pval <= threshold
    sorted_pvals = pvals[order]
    sorted_thresh = threshold[order]
    below = sorted_pvals <= sorted_thresh
    if not below.any():
        return np.zeros(n, dtype=bool)
    last_sig = np.where(below)[0].max()
    sig_mask = np.zeros(n, dtype=bool)
    sig_mask[order[:last_sig+1]] = True
    return sig_mask


def rho_to_pval(rho: np.ndarray, n: int) -> np.ndarray:
    """Approximate two-sided p-value for Spearman rho via t-distribution."""
    from scipy.stats import t as t_dist
    t_stat = rho * np.sqrt((n - 2) / np.maximum(1 - rho**2, 1e-12))
    return 2.0 * t_dist.sf(np.abs(t_stat), df=n - 2)


# ─────────────────────────────────────────────────────────────────────────────
# Atom activation matrix
# ─────────────────────────────────────────────────────────────────────────────

def compute_atom_activations(index: pd.DataFrame, sae, model,
                              layer: int, scale: float,
                              device: str = "cpu") -> np.ndarray:
    """
    For each ECG in index: run through model and SAE, take log1p(max_token).
    Returns (N, d_sae) float32.
    """
    from model import extract_layers
    from sae import BatchTopKSAE

    sae.eval()
    model.eval()

    all_acts = []
    N = len(index)

    for start in range(0, N, 32):
        batch = index.iloc[start:start+32]
        for _, row in batch.iterrows():
            try:
                w = load_medalcare(row["ecg_path"])
            except Exception:
                all_acts.append(np.zeros(sae.d_sae, dtype=np.float32))
                continue
            max_acts = np.zeros(sae.d_sae, dtype=np.float32)
            for win_i in range(2):
                x = torch.from_numpy(w[win_i:win_i+1]).float()  # (1,12,2500)
                feats = extract_layers(model, x, layers=[layer], device=device)
                toks  = feats[layer].squeeze(0).float().to(device) / scale
                with torch.no_grad():
                    import torch
                    _, z = sae(toks)
                win_max = z.max(dim=0).values.cpu().numpy()  # (d_sae,)
                max_acts = np.maximum(max_acts, win_max)
            all_acts.append(np.log1p(max_acts))
        print(f"  Atom activations: {min(start+32, N)}/{N}", end="\r", flush=True)

    print()
    return np.stack(all_acts, axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Morphology feature computation (neurokit2)
# ─────────────────────────────────────────────────────────────────────────────

def compute_morphology(ecg_path: str) -> dict:
    """
    Compute ECG morphology features from lead II using neurokit2.
    Returns dict with morph_* keys or NaNs on failure.
    """
    nan_row = {k: float("nan") for k in MORPHOLOGY_PARAMS}
    try:
        import neurokit2 as nk
        arr = np.loadtxt(ecg_path, delimiter=",")  # (12, 5000)
        lead_ii = arr[1, :]  # row 1 = lead II
        signals, info = nk.ecg_process(lead_ii, sampling_rate=C.SAMPLE_RATE)
        hrv = nk.ecg_intervalrelated(signals)
        peaks = info.get("ECG_R_Peaks", [])
        r_amp = float(lead_ii[peaks].mean()) if len(peaks) > 0 else float("nan")

        # T amplitude: rough max after R peak
        t_amps = []
        for pk in peaks[:5]:
            seg = lead_ii[pk: min(pk+250, len(lead_ii))]
            if len(seg) > 0:
                t_amps.append(seg.max())
        t_amp = float(np.mean(t_amps)) if t_amps else float("nan")

        return {
            "morph_hr":          float(hrv.get("ECG_Rate_Mean", [float("nan")])[0]),
            "morph_pr_interval": float(signals.get("ECG_P_Onsets", pd.Series([float("nan")])).mean()),
            "morph_qrs_duration": float(info.get("ECG_QRS_Duration", float("nan"))),
            "morph_qt_interval":  float(info.get("ECG_QT_Interval", float("nan"))),
            "morph_r_amplitude":  r_amp,
            "morph_t_amplitude":  t_amp,
        }
    except Exception:
        return nan_row


# ─────────────────────────────────────────────────────────────────────────────
# Main Test C pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_test_c(layer: int, device: str = "cpu",
               n_permutations: int = 1000,
               include_run_id_covariate: bool = False) -> dict:

    import torch

    results = {}

    # ── Load index ────────────────────────────────────────────────────────
    print("\n[Test C] Loading index...")
    index = build_index()
    if "model_id" not in index or index["model_id"].isna().any():
        raise ValueError("Dataset index must contain a model_id for every ECG before Test C.")
    split_counts = index.groupby("model_id")["split"].nunique()
    overlapping_models = split_counts[split_counts > 1]
    if not overlapping_models.empty:
        raise RuntimeError(
            f"Test C is blocked: {len(overlapping_models)} model IDs occur in multiple "
            "predefined splits. Resolve the model-level split assignment before association testing."
        )

    # ── Load SAE and model ────────────────────────────────────────────────
    print("[Test C] Loading SAE and ECG-FM...")
    from sae import BatchTopKSAE
    from model import load_ecgfm

    sae = BatchTopKSAE()
    sae.load_state_dict(torch.load(C.SAE_WEIGHTS, map_location="cpu"))
    sae.to(device)

    model, _, _ = load_ecgfm()
    model.to(device)

    scale = float(np.load(C.CACHE_DIR / f"scale_pretrained_layer{layer}.npy"))

    # ── Atom activations (all ECGs) ───────────────────────────────────────
    act_cache = C.CACHE_DIR / f"atom_acts_layer{layer}.npy"
    if act_cache.exists():
        print(f"[Test C] Loading cached atom activations from {act_cache}")
        A_all = np.load(act_cache)
    else:
        print("[Test C] Computing atom activations...")
        A_all = compute_atom_activations(index, sae, model, layer, scale, device)
        np.save(act_cache, A_all)

    # ── Add morphology features to index ─────────────────────────────────
    morph_cache = C.CACHE_DIR / "morphology.parquet"
    if morph_cache.exists():
        morph_df = pd.read_parquet(morph_cache)
        index = index.reset_index(drop=True)
        for col in MORPHOLOGY_PARAMS:
            if col in morph_df.columns:
                index[col] = morph_df[col].values
    else:
        print("[Test C] Computing morphology features (this may take a while)...")
        morph_rows = []
        for _, row in index.iterrows():
            morph_rows.append(compute_morphology(row["ecg_path"]))
        morph_df = pd.DataFrame(morph_rows)
        morph_df.to_parquet(morph_cache, index=False)
        for col in MORPHOLOGY_PARAMS:
            index[col] = morph_df[col].values

    # ── Build covariate matrix ────────────────────────────────────────────
    # Covariates: one-hot subfolder + one-hot anatomical model ID
    subfolder_dummies = pd.get_dummies(index["folder"], prefix="f").astype(float)
    if "model_id" not in index or index["model_id"].isna().any():
        raise ValueError("Dataset index must contain a model_id for every ECG before Test C.")
    model_dummies     = pd.get_dummies(index["model_id"].astype(str), prefix="model").astype(float)
    covariates = [subfolder_dummies.values, model_dummies.values]
    if include_run_id_covariate:
        run_dummies = pd.get_dummies(index["run_id"].astype(str), prefix="run").astype(float)
        covariates.append(run_dummies.values)
    C_mat = np.concatenate(covariates, axis=1)
    results["run_id_covariate_sensitivity"] = include_run_id_covariate

    # ── Split masks ───────────────────────────────────────────────────────
    tr_mask   = (index["split"] == "train").values
    held_mask = (index["split"].isin(["validation","test"])).values

    n_atoms   = A_all.shape[1]
    n_params  = len(ALL_PARAMS)

    # ── Discovery pass (train) ────────────────────────────────────────────
    print(f"\n[Test C] Discovery pass ({tr_mask.sum()} train ECGs)...")
    discovery = _run_associations(A_all[tr_mask], index[tr_mask], C_mat[tr_mask])

    # ── Confirmation pass (held-out) ──────────────────────────────────────
    print(f"[Test C] Confirmation pass ({held_mask.sum()} held-out ECGs)...")
    confirmation = _run_associations(A_all[held_mask], index[held_mask], C_mat[held_mask])

    # ── Confirmed atoms ───────────────────────────────────────────────────
    confirmed_rows = []
    for (atom_i, param), disc_row in discovery.items():
        conf_row = confirmation.get((atom_i, param))
        if conf_row is None:
            continue
        if conf_row["q"] < C.ASSOC_Q_THR and abs(conf_row["rho"]) >= C.ASSOC_RHO_THR:
            confirmed_rows.append({
                "atom":      atom_i,
                "param":     param,
                "rho_disc":  disc_row["rho"],
                "rho_conf":  conf_row["rho"],
                "q_conf":    conf_row["q"],
                "n_samples": conf_row["n"],
            })
    confirmed_df = pd.DataFrame(confirmed_rows)
    n_confirmed  = len(confirmed_df)
    print(f"\nConfirmed atoms: {n_confirmed} (threshold: >= {C.ASSOC_MIN_ATOMS})")

    # ── Permutation null ──────────────────────────────────────────────────
    print(f"[Test C] Running {n_permutations} permutations...")
    perm_counts = _permutation_null(
        A_all[held_mask], index[held_mask], C_mat[held_mask],
        n_permutations, include_run_id_covariate,
    )
    results["perm_null_mean"] = float(np.mean(perm_counts))
    results["perm_null_p95"]  = float(np.percentile(perm_counts, 95))

    # ── Random-init null ──────────────────────────────────────────────────
    ctrl_act_cache = C.CACHE_DIR / f"atom_acts_ctrl_layer{layer}.npy"
    if ctrl_act_cache.exists():
        A_ctrl = np.load(ctrl_act_cache)
        ctrl_conf = _run_associations(A_ctrl[held_mask], index[held_mask], C_mat[held_mask])
        n_ctrl_pass = sum(1 for v in ctrl_conf.values()
                         if v["q"] < C.ASSOC_Q_THR and abs(v["rho"]) >= C.ASSOC_RHO_THR)
        results["ctrl_confirmed_atoms"] = n_ctrl_pass
        print(f"Random-init confirmed atoms: {n_ctrl_pass}")
    else:
        results["ctrl_confirmed_atoms"] = None
        print("[WARN] No control atom activations found. Run sae.py --ctrl first.")

    # ── Atom binning ──────────────────────────────────────────────────────
    if not confirmed_df.empty:
        phys_atoms  = set(confirmed_df[confirmed_df["param"].isin(MECHANISTIC_PARAMS)]["atom"])
        morph_atoms = set(confirmed_df[~confirmed_df["param"].isin(MECHANISTIC_PARAMS)]["atom"]) - phys_atoms
    else:
        phys_atoms = morph_atoms = set()

    results["n_physiology_atoms"] = len(phys_atoms)
    results["n_morphology_atoms"] = len(morph_atoms)
    results["n_confirmed_total"]  = n_confirmed

    # ── Pass / fail ───────────────────────────────────────────────────────
    ctrl_count = results.get("ctrl_confirmed_atoms") or 0
    ratio_ok   = (n_confirmed >= 2 * max(ctrl_count, 1))
    if (n_confirmed >= C.ASSOC_MIN_ATOMS
            and ratio_ok
            and results["n_physiology_atoms"] > 0):
        verdict = "PASS"
    elif (results["n_morphology_atoms"] > 0 and results["n_physiology_atoms"] == 0):
        verdict = "MORPHOLOGY_ONLY"
    elif n_confirmed == 0:
        verdict = "NO_ATOMS"
    else:
        verdict = "MARGINAL"
    results["verdict"] = verdict
    print(f"\n[Test C] Verdict: {verdict}")

    # ── Save ──────────────────────────────────────────────────────────────
    C.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    output_suffix = "_with_run_id_covariate" if include_run_id_covariate else ""
    if not confirmed_df.empty:
        confirmed_df.to_csv(C.RUNS_DIR / f"C_confirmed_atoms{output_suffix}.csv", index=False)
    with open(C.RUNS_DIR / f"C_results{output_suffix}.json", "w") as f:
        json.dump(results, f, indent=2)
    np.save(C.RUNS_DIR / f"C_perm_null{output_suffix}.npy", np.array(perm_counts))
    print(f"Saved results to {C.RUNS_DIR}")

    return results


def _run_associations(A: np.ndarray, idx_sub: pd.DataFrame,
                      C_sub: np.ndarray) -> dict:
    """
    Compute partial Spearman for each valid (atom, param) pair.
    Returns dict (atom_i, param) -> {"rho", "q", "n"}
    """
    n_atoms = A.shape[1]
    all_rho = {}
    all_pval = {}
    n_map = {}

    for param in ALL_PARAMS:
        if param not in idx_sub.columns:
            continue
        p_vec = pd.to_numeric(idx_sub[param], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(p_vec)
        if valid.sum() < 20:
            continue

        rhos = partial_spearman(A[valid], p_vec[valid], C_sub[valid])
        n    = valid.sum()
        pvs  = rho_to_pval(rhos, n)

        for atom_i, (rho, pv) in enumerate(zip(rhos, pvs)):
            all_rho[(atom_i, param)]  = float(rho)
            all_pval[(atom_i, param)] = float(pv)
            n_map[(atom_i, param)]    = int(n)

    if not all_pval:
        return {}

    # BH FDR across all pairs
    keys   = list(all_pval.keys())
    pvals  = np.array([all_pval[k] for k in keys])
    sig    = bh_fdr(pvals, q=C.ASSOC_Q_THR)
    # Compute q-values (ranks / N * q_thresh, corrected)
    n_tot = len(pvals)
    order = np.argsort(pvals)
    qs = np.empty(n_tot)
    qs[order] = (np.arange(1, n_tot+1) / n_tot) * C.ASSOC_Q_THR / (pvals[order] / pvals[order])
    # Simpler: use raw p * n / rank
    ranks = np.empty(n_tot, dtype=int)
    ranks[order] = np.arange(1, n_tot+1)
    q_vals = np.minimum(1.0, pvals * n_tot / ranks)

    return {k: {"rho": all_rho[k], "q": float(q_vals[i]), "n": n_map[k]}
            for i, k in enumerate(keys)}


def _permutation_null(A: np.ndarray, idx_sub: pd.DataFrame,
                      C_sub: np.ndarray, n_perms: int,
                      include_run_id_covariate: bool = False) -> list:
    """Count passing atoms under permutation within folder and model strata."""
    rng = np.random.default_rng(0)
    counts = []
    for perm in range(n_perms):
        idx_perm = idx_sub.copy()
        for param in MECHANISTIC_PARAMS:
            if param not in idx_perm.columns:
                continue
            # Preserve all active covariates within each permutation stratum.
            strata = ["folder", "model_id"]
            if include_run_id_covariate:
                strata.append("run_id")
            for _, grp in idx_perm.groupby(strata):
                shuffled = idx_perm.loc[grp.index, param].values.copy()
                rng.shuffle(shuffled)
                idx_perm.loc[grp.index, param] = shuffled
        conf = _run_associations(A, idx_perm, C_sub)
        n_pass = sum(1 for v in conf.values()
                     if v["q"] < C.ASSOC_Q_THR and abs(v["rho"]) >= C.ASSOC_RHO_THR)
        counts.append(n_pass)
        if (perm+1) % 100 == 0:
            print(f"  Permutation {perm+1}/{n_perms}  mean_pass={np.mean(counts):.1f}", end="\r")
    print()
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-perms", type=int, default=1000)
    ap.add_argument("--include-run-id-covariate", action="store_true")
    args = ap.parse_args()
    run_test_c(args.layer, args.device, args.n_perms, args.include_run_id_covariate)
