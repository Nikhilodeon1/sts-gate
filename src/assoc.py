"""
assoc.py – Test C: Mechanistic parameter and ECG morphology associations.

Steps:
  1. For each MedalCare ECG, compute max token activation per SAE atom (batched),
     apply log1p. Shape: (N_ecgs, d_sae) for both Pretrained and Random-Init Control SAEs.
  2. Compute 6 ECG morphology features from Lead II via fast signal delineation on
     `load_medalcare(ecg_path)` (compatible with both raw CSVs and compact bundle shards).
  3. Build covariate matrix: one-hot subfolder + one-hot anatomical model ID
     (optional sensitivity: + one-hot run_id).
  4. Discovery pass on train split: partial Spearman + BH FDR q < 0.05.
  5. Confirmation pass on held-out (val + test): BH FDR q < 0.05, |rho| >= threshold,
     and consistent correlation sign between discovery and confirmation.
  6. Controls:
     - 1,000 within-(subfolder, model_id) permutations on held-out split (pre-projected fast solver)
     - Random-init control SAE + random-init ECG-FM confirmation pass
  7. Bin confirmed atoms: Physiology vs Morphology-only vs Unmapped, and save figures/tables.

Run:
    python src/assoc.py --layer 8 [--device cuda] [--exclude-run-ids run_S64 run_S67]
"""
import sys, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from data import build_index, load_medalcare


# Mechanistic parameters (column names as they appear in index.parquet after prefix)
MECHANISTIC_PARAMS = [
    # Ventricular action potential duration & 3D dispersion gradients
    "v_APD.min",
    "v_APD.max",
    "v_APD.rho_d",          # Transmural APD gradient
    "v_APD.z_d",            # Apico-basal APD gradient
    "v_APD.v_d",            # Inter-ventricular APD gradient
    # Ventricular endocardial activation / His-Purkinje fascicle sites
    "v_stim[0].z",          # LV septal fascicle height
    "v_stim[0].phi",        # LV septal fascicle angle
    "v_stim[0].thr",        # LV septal fascicle threshold
    "v_stim[1].z",          # LV anterior fascicle height
    "v_stim[1].phi",        # LV anterior fascicle angle
    "v_stim[2].z",          # LV posterior fascicle height
    "v_stim[2].phi",        # LV posterior fascicle angle
    "v_stim[3].z",          # RV septal site height
    "v_stim[3].phi",        # RV septal site angle
    "v_stim[4].time",       # RV moderator-band delay
    "v_stim[4].z",          # RV moderator-band height
    "v_stim[4].phi",        # RV moderator-band angle
    # Ischemic zone geometry & transmurality (MI simulations)
    "v_isch[0].size",       # Ischemic zone size
    "v_isch[0].z",          # Ischemic zone apico-basal center
    "v_isch[0].phi",        # Ischemic zone circumferential angle
    "v_isch[0].rho_eps_max",# Ischemic transmurality (0.3 vs 1.0)
    # Atrial conduction velocities & 3D orientation
    "a_cv_t.BulkTissue",    # Atrial bulk conduction velocity
    "a_cv_t.BachmannsBundle",
    "a_cv_t.CristaTerminalis",
    "a_cv_t.PectinateMuscles",
    "a_cv_t.InferiorIsthmus",
    "a_rot.x",              # Atrial X rotation
    "a_rot.y",              # Atrial Y rotation
    "a_rot.z",              # Atrial Z rotation
    "a_fibperc",            # Fibrosis percentage
]

# Surface ECG morphology features computed from Lead II
MORPHOLOGY_PARAMS = [
    "morph_hr",
    "morph_pr_interval",
    "morph_qrs_duration",
    "morph_qt_interval",
    "morph_r_amplitude",
    "morph_t_amplitude",
]

ALL_PARAMS = MECHANISTIC_PARAMS + MORPHOLOGY_PARAMS


# ─────────────────────────────────────────────────────────────────────────────
# Collinearity-safe Partial Spearman & Monotone BH-FDR
# ─────────────────────────────────────────────────────────────────────────────

def partial_spearman(
    A: np.ndarray,
    p: np.ndarray,
    C_mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute partial Spearman correlation between each atom column in A and parameter p,
    removing shared variance explained by covariates C_mat.
    Guards against zero-variance residuals when a parameter or atom is collinear with C_mat.
    Returns (rhos, valid_atoms_mask).
    """
    n = len(p)
    R = rankdata(A, axis=0).astype(np.float64)   # (n, n_atoms)
    r = rankdata(p).astype(np.float64)           # (n,)

    active_cov = C_mat.std(axis=0) > 1e-8
    if active_cov.any():
        X = np.c_[np.ones(n, dtype=np.float64), C_mat[:, active_cov].astype(np.float64)]
    else:
        X = np.ones((n, 1), dtype=np.float64)

    beta_r, *_ = np.linalg.lstsq(X, r, rcond=None)
    rp = r - X @ beta_r
    rp_std = float(rp.std())
    if rp_std < 1e-6:
        return np.zeros(A.shape[1], dtype=float), np.zeros(A.shape[1], dtype=bool)
    rp = (rp - rp.mean()) / rp_std

    beta_A, *_ = np.linalg.lstsq(X, R, rcond=None)
    Ra = R - X @ beta_A
    Ra_mean = Ra.mean(axis=0, keepdims=True)
    Ra_std  = Ra.std(axis=0, keepdims=True)
    valid_atoms = (Ra_std.squeeze(0) >= 1e-6) & (A.std(axis=0) >= 1e-6)

    Ra_norm = np.zeros_like(Ra)
    Ra_norm[:, valid_atoms] = (Ra[:, valid_atoms] - Ra_mean[:, valid_atoms]) / Ra_std[:, valid_atoms]

    rhos = (Ra_norm.T @ rp) / float(n)
    return np.clip(rhos, -0.999999, 0.999999), valid_atoms


def bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    """Compute monotone Benjamini-Hochberg step-up adjusted q-values."""
    n = len(pvals)
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(pvals)
    ranks = np.arange(1, n + 1, dtype=float)
    raw_q = pvals[order] * (n / ranks)
    mono_q = np.minimum.accumulate(raw_q[::-1])[::-1]
    mono_q = np.clip(mono_q, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = mono_q
    return out


def bh_fdr(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns boolean mask of discoveries."""
    return bh_qvalues(pvals) <= q


def rho_to_pval(rho: np.ndarray, n: int) -> np.ndarray:
    """Approximate two-sided p-value for Spearman rho via t-distribution."""
    from scipy.stats import t as t_dist
    df = max(int(n) - 2, 1)
    t_stat = rho * np.sqrt(df / np.maximum(1.0 - rho ** 2, 1e-12))
    return 2.0 * t_dist.sf(np.abs(t_stat), df=df)


# ─────────────────────────────────────────────────────────────────────────────
# Batched Atom Activation Matrix
# ─────────────────────────────────────────────────────────────────────────────

def compute_atom_activations(
    index: pd.DataFrame,
    sae,
    model,
    layer: int,
    scale: float,
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """
    For each ECG in index: run both 5s windows through model and SAE in batches,
    take log1p(max_token_activation) across both windows. Returns (N, d_sae) float32.
    """
    import torch
    from model import extract_layers

    sae.eval()
    model.eval()

    all_acts = []
    N = len(index)

    for start in range(0, N, batch_size):
        batch = index.iloc[start:start + batch_size]
        win0, win1 = [], []
        for _, row in batch.iterrows():
            try:
                w = load_medalcare(row["ecg_path"])
            except Exception:
                w = np.zeros((2, 12, C.WIN_SAMPLES), dtype=np.float32)
            win0.append(w[0])
            win1.append(w[1])

        x0 = torch.from_numpy(np.stack(win0)).float()
        x1 = torch.from_numpy(np.stack(win1)).float()
        feats0 = extract_layers(model, x0, layers=[layer], device=device)[layer].to(device)
        feats1 = extract_layers(model, x1, layers=[layer], device=device)[layer].to(device)

        B_cur, T_cur, D_cur = feats0.shape
        with torch.no_grad():
            _, z0 = sae(feats0.reshape(-1, D_cur) / scale)
            _, z1 = sae(feats1.reshape(-1, D_cur) / scale)
            m0 = z0.reshape(B_cur, T_cur, -1).max(dim=1).values
            m1 = z1.reshape(B_cur, T_cur, -1).max(dim=1).values
            max_acts = torch.log1p(torch.maximum(m0, m1)).cpu().numpy().astype(np.float32)

        all_acts.append(max_acts)
        print(f"  Atom activations: {min(start + batch_size, N)}/{N}", end="\r", flush=True)

    print()
    return np.concatenate(all_acts, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Fast Bundle-Safe Morphology Feature Extraction (500 Hz Lead II)
# ─────────────────────────────────────────────────────────────────────────────

def compute_morphology(ecg_path: str) -> dict:
    """
    Compute 6 surface ECG morphology features from 10s Lead II (500 Hz)
    using `load_medalcare(ecg_path)` (supports both raw CSVs and bundle shards).
    """
    nan_row = {k: float("nan") for k in MORPHOLOGY_PARAMS}
    try:
        w = load_medalcare(ecg_path)  # (2, 12, 2500)
        lead_ii = np.concatenate([w[0, 1], w[1, 1]]).astype(np.float64)  # (5000,)
        fs = float(C.SAMPLE_RATE)  # 500 Hz

        # Detect R-peaks (minimum 280 ms refractory period = 140 samples)
        prom = max(0.35, 0.45 * float(np.std(lead_ii)))
        peaks, _ = find_peaks(lead_ii, distance=int(0.28 * fs), prominence=prom)
        if len(peaks) < 2:
            peaks, _ = find_peaks(np.abs(lead_ii), distance=int(0.28 * fs), prominence=0.3)
        if len(peaks) == 0:
            return nan_row

        # 1. Heart rate (bpm)
        if len(peaks) >= 2:
            rr_sec = np.diff(peaks) / fs
            valid_rr = rr_sec[(rr_sec >= 0.3) & (rr_sec <= 2.5)]
            hr = float(60.0 / np.median(valid_rr)) if len(valid_rr) > 0 else float(60.0 / np.median(rr_sec))
        else:
            hr = float("nan")

        # 2. R-wave amplitude
        r_amp = float(np.mean(lead_ii[peaks]))

        # Per-beat wave delineation around interior R-peaks
        pr_list, qrs_list, qt_list, t_amp_list = [], [], [], []
        n_samp = len(lead_ii)
        for pk in peaks:
            if pk < int(0.24 * fs) or pk + int(0.45 * fs) >= n_samp:
                continue
            baseline = float(np.median(lead_ii[pk - int(0.24 * fs): pk - int(0.18 * fs)]))

            # Q-onset / nadir in [-60 ms, 0 ms]
            q_win = lead_ii[pk - int(0.06 * fs): pk + 1]
            q_rel = int(np.argmin(q_win))
            q_idx = (pk - int(0.06 * fs)) + q_rel
            # Walk back from Q-nadir up to 25 ms to find onset returning toward baseline
            onset_search = lead_ii[max(0, q_idx - int(0.025 * fs)): q_idx + 1]
            if len(onset_search) > 1:
                q_onset = max(0, q_idx - int(0.025 * fs)) + int(np.argmin(np.abs(onset_search - baseline)))
            else:
                q_onset = q_idx

            # S-nadir / offset in [0 ms, +80 ms]
            s_win = lead_ii[pk: pk + int(0.08 * fs) + 1]
            s_rel = int(np.argmin(s_win))
            s_idx = pk + s_rel
            offset_search = lead_ii[s_idx: min(n_samp, s_idx + int(0.035 * fs))]
            if len(offset_search) > 1:
                s_offset = s_idx + int(np.argmin(np.abs(offset_search - baseline)))
            else:
                s_offset = s_idx
            qrs_ms = float((s_offset - q_onset) * 1000.0 / fs)
            if 20.0 <= qrs_ms <= 220.0:
                qrs_list.append(qrs_ms)

            # P-wave peak & onset in [-220 ms, -50 ms] before R-peak
            p_start = pk - int(0.22 * fs)
            p_end   = pk - int(0.05 * fs)
            p_seg   = lead_ii[p_start:p_end]
            if len(p_seg) > 5:
                p_pk_rel = int(np.argmax(np.abs(p_seg - baseline)))
                p_pre = p_seg[:p_pk_rel + 1]
                thr_p = baseline + 0.15 * (p_seg[p_pk_rel] - baseline)
                below_p = np.where(np.abs(p_pre - baseline) <= np.abs(thr_p - baseline))[0]
                p_on_rel = int(below_p[-1]) if len(below_p) > 0 else 0
                p_onset = p_start + p_on_rel
                pr_ms = float((q_onset - p_onset) * 1000.0 / fs)
                if 60.0 <= pr_ms <= 350.0:
                    pr_list.append(pr_ms)

            # T-wave peak & offset in [+90 ms, +440 ms] after R-peak
            t_start = pk + int(0.09 * fs)
            t_end   = pk + int(0.44 * fs)
            t_seg   = lead_ii[t_start:t_end]
            if len(t_seg) > 10:
                t_amp_list.append(float(np.max(t_seg)))
                t_pk_rel = int(np.argmax(np.abs(t_seg - baseline)))
                t_post = t_seg[t_pk_rel:]
                thr_t = baseline + 0.12 * (t_seg[t_pk_rel] - baseline)
                below_t = np.where(np.abs(t_post - baseline) <= np.abs(thr_t - baseline))[0]
                t_off_rel = t_pk_rel + (int(below_t[0]) if len(below_t) > 0 else len(t_post) - 1)
                t_offset = t_start + t_off_rel
                qt_ms = float((t_offset - q_onset) * 1000.0 / fs)
                if 180.0 <= qt_ms <= 600.0:
                    qt_list.append(qt_ms)

        return {
            "morph_hr":           hr,
            "morph_pr_interval":  float(np.median(pr_list)) if pr_list else float("nan"),
            "morph_qrs_duration": float(np.median(qrs_list)) if qrs_list else float("nan"),
            "morph_qt_interval":  float(np.median(qt_list)) if qt_list else float("nan"),
            "morph_r_amplitude":  r_amp,
            "morph_t_amplitude":  float(np.mean(t_amp_list)) if t_amp_list else float("nan"),
        }
    except Exception:
        return nan_row


# ─────────────────────────────────────────────────────────────────────────────
# Association & Fast Pre-Projected Permutation Engine
# ─────────────────────────────────────────────────────────────────────────────

def _run_associations(
    A: np.ndarray,
    idx_sub: pd.DataFrame,
    C_sub: np.ndarray,
    candidate_keys: set[tuple[int, str]] | None = None,
) -> dict:
    """
    Compute partial Spearman for each valid (atom, param) pair.
    If `candidate_keys` is provided (Stage 2 confirmation of Stage 1 discoveries),
    computes both full-matrix BH-FDR (`q_full`) and candidate-restricted BH-FDR (`q`).
    """
    all_rho = {}
    all_pval = {}
    n_map = {}

    for param in ALL_PARAMS:
        if param not in idx_sub.columns:
            continue
        p_vec = pd.to_numeric(idx_sub[param], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(p_vec)
        n = int(valid.sum())
        if n < 20 or float(np.std(p_vec[valid])) < 1e-8:
            continue

        rhos, valid_atoms = partial_spearman(A[valid], p_vec[valid], C_sub[valid])
        if not valid_atoms.any():
            continue
        pvs = rho_to_pval(rhos, n)

        for atom_i in np.where(valid_atoms)[0]:
            key = (int(atom_i), param)
            all_rho[key]  = float(rhos[atom_i])
            all_pval[key] = float(pvs[atom_i])
            n_map[key]    = n

    if not all_pval:
        return {}

    keys  = list(all_pval.keys())
    pvals = np.array([all_pval[k] for k in keys], dtype=float)
    q_full = bh_qvalues(pvals)

    if candidate_keys is not None and len(candidate_keys) > 0:
        cand_mask = np.array([k in candidate_keys for k in keys], dtype=bool)
        q_cand = np.ones_like(q_full)
        if cand_mask.any():
            q_cand[cand_mask] = bh_qvalues(pvals[cand_mask])
    else:
        q_cand = q_full

    return {
        k: {
            "rho": all_rho[k],
            "pval": all_pval[k],
            "q": float(q_cand[i]),
            "q_full": float(q_full[i]),
            "n": n_map[k],
        }
        for i, k in enumerate(keys)
    }


def _permutation_null(
    A: np.ndarray,
    idx_sub: pd.DataFrame,
    C_sub: np.ndarray,
    n_perms: int,
    include_run_id_covariate: bool = False,
    rho_thr: float = C.ASSOC_RHO_THR,
) -> list[int]:
    """
    Fast, exact permutation null within (folder, model_id[, run_id]) strata.
    Precomputes rank-residualized atom matrix Ra_norm and covariate projector X_pinv
    once per parameter validity mask, reducing 1,000 permutations from hours to seconds.
    """
    rng = np.random.default_rng(0)
    strata_cols = ["folder", "model_id"]
    if include_run_id_covariate:
        strata_cols.append("run_id")

    idx_reset = idx_sub.reset_index(drop=True)
    strata_groups = [
        grp.index.to_numpy(dtype=int)
        for _, grp in idx_reset.groupby(strata_cols)
        if len(grp) > 1
    ]

    # Precompute per-parameter invariant matrices
    precomputed = []
    for param in MECHANISTIC_PARAMS:
        if param not in idx_reset.columns:
            continue
        p_vec = pd.to_numeric(idx_reset[param], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(p_vec)
        n = int(valid.sum())
        if n < 20 or float(np.std(p_vec[valid])) < 1e-8:
            continue

        active_cov = C_sub[valid].std(axis=0) > 1e-8
        if active_cov.any():
            X = np.c_[np.ones(n, dtype=np.float64), C_sub[valid][:, active_cov].astype(np.float64)]
        else:
            X = np.ones((n, 1), dtype=np.float64)
        X_pinv = np.linalg.pinv(X)

        r_orig = rankdata(p_vec[valid]).astype(np.float64)
        rp_orig = r_orig - X @ (X_pinv @ r_orig)
        if float(rp_orig.std()) < 1e-6:
            continue

        R = rankdata(A[valid], axis=0).astype(np.float64)
        Ra = R - X @ (X_pinv @ R)
        Ra_mean = Ra.mean(axis=0, keepdims=True)
        Ra_std  = Ra.std(axis=0, keepdims=True)
        valid_atoms = (Ra_std.squeeze(0) >= 1e-6) & (A[valid].std(axis=0) >= 1e-6)
        if not valid_atoms.any():
            continue
        Ra_norm = (Ra[:, valid_atoms] - Ra_mean[:, valid_atoms]) / Ra_std[:, valid_atoms]

        # Map strata indices restricted to valid subset
        valid_strata = []
        if valid.all():
            valid_strata = strata_groups
        else:
            global_to_local = np.full(len(p_vec), -1, dtype=int)
            global_to_local[np.where(valid)[0]] = np.arange(n, dtype=int)
            for g_idx in strata_groups:
                loc = global_to_local[g_idx]
                loc = loc[loc >= 0]
                if len(loc) > 1:
                    valid_strata.append(loc)

        precomputed.append({
            "param": param,
            "p_valid": p_vec[valid].copy(),
            "n": n,
            "X": X,
            "X_pinv": X_pinv,
            "Ra_norm": Ra_norm,
            "strata": valid_strata,
        })

    if not precomputed:
        return [0] * n_perms

    counts = []
    for perm in range(n_perms):
        pvals_list = []
        rhos_list  = []
        for item in precomputed:
            p_perm = item["p_valid"].copy()
            for loc in item["strata"]:
                p_perm[loc] = p_perm[rng.permutation(loc)]
            r = rankdata(p_perm).astype(np.float64)
            rp = r - item["X"] @ (item["X_pinv"] @ r)
            rp_std = float(rp.std())
            if rp_std < 1e-6:
                rhos = np.zeros(item["Ra_norm"].shape[1], dtype=float)
                pvs  = np.ones(item["Ra_norm"].shape[1], dtype=float)
            else:
                rp = (rp - rp.mean()) / rp_std
                rhos = np.clip((item["Ra_norm"].T @ rp) / float(item["n"]), -0.999999, 0.999999)
                pvs  = rho_to_pval(rhos, item["n"])
            rhos_list.append(rhos)
            pvals_list.append(pvs)

        all_rhos  = np.concatenate(rhos_list)
        all_pvals = np.concatenate(pvals_list)
        all_q     = bh_qvalues(all_pvals)
        n_pass    = int(np.sum((all_q < C.ASSOC_Q_THR) & (np.abs(all_rhos) >= rho_thr)))
        counts.append(n_pass)
        if (perm + 1) % 100 == 0 or (perm + 1) == n_perms:
            print(f"  Permutation {perm+1}/{n_perms}  mean_pass={np.mean(counts):.2f}", end="\r", flush=True)
    print()
    return counts


def _plot_top_associations(confirmed_df: pd.DataFrame, out_path: Path):
    if confirmed_df.empty:
        return
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top = confirmed_df.reindex(
            confirmed_df["rho_conf"].abs().sort_values(ascending=False).index
        ).head(15).iloc[::-1]

        fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.35 * len(top))))
        labels = [f"Atom {int(r.atom)} : {r.param}" for _, r in top.iterrows()]
        y = np.arange(len(top))
        ax.barh(y - 0.16, top["rho_disc"], height=0.30, label="Discovery (train)", color="#4c72b0")
        ax.barh(y + 0.16, top["rho_conf"], height=0.30, label="Confirmation (val+test)", color="#55a868")
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.axvline(C.ASSOC_RHO_THR, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(-C.ASSOC_RHO_THR, color="gray", linestyle="--", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlabel("Partial Spearman rho (controlling for subfolder & model_id)")
        ax.set_title("Test C: Top Confirmed Mechanistic & Morphology SAE Atom Associations")
        ax.legend(loc="lower right", fontsize=8.5)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved association figure to {out_path}")
    except Exception as e:
        print(f"[WARN] Could not plot associations: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Test C pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_test_c(
    layer: int = C.LOCKED_LAYER or 8,
    device: str = "cpu",
    n_permutations: int = 1000,
    include_run_id_covariate: bool = False,
    exclude_run_ids: list[str] | None = None,
    rebuild_acts: bool = False,
    rebuild_morph: bool = False,
) -> dict:
    import torch
    from sae import BatchTopKSAE
    from model import load_ecgfm

    excluded = sorted(dict.fromkeys(exclude_run_ids or []))
    suffix = ""
    if excluded:
        id_str = "_".join(r.removeprefix("run_") for r in excluded)
        suffix = f"_sensitivity_without_{id_str}"
    if include_run_id_covariate:
        suffix += "_with_run_id_covariate"

    results = {
        "layer": int(layer),
        "excluded_run_ids": excluded,
        "run_id_covariate_sensitivity": bool(include_run_id_covariate),
    }

    # ── Load index ────────────────────────────────────────────────────────
    print("\n[Test C] Loading index...")
    index = build_index()
    if excluded:
        n_before = len(index)
        index = index.loc[~index["run_id"].isin(excluded)].reset_index(drop=True)
        print(f"  Excluding {n_before - len(index)} ECGs from run IDs: {excluded}")
    print(f"  Total ECGs: {len(index)}")

    if "model_id" not in index or index["model_id"].isna().any():
        raise ValueError("Dataset index must contain a model_id for every ECG before Test C.")
    split_counts = index.groupby("model_id")["split"].nunique()
    overlapping_models = split_counts[split_counts > 1]
    if not overlapping_models.empty:
        raise RuntimeError(
            f"Test C is blocked: {len(overlapping_models)} model IDs occur in multiple "
            "predefined splits. Resolve the model-level split assignment before association testing."
        )

    base_suffix = f"_sensitivity_without_{'_'.join(r.removeprefix('run_') for r in excluded)}" if excluded else ""

    # ── Pretrained SAE & atom activations ─────────────────────────────────
    act_cache = C.CACHE_DIR / f"atom_acts{base_suffix}_layer{layer}.npy"
    if act_cache.exists() and not rebuild_acts and len(np.load(act_cache, mmap_mode="r")) == len(index):
        print(f"[Test C] Loading cached pretrained atom activations from {act_cache}")
        A_all = np.load(act_cache)
    else:
        print("[Test C] Computing pretrained SAE atom activations...")
        sae_path = C.CACHE_DIR / f"sae_pretrained{base_suffix}_layer{layer}.pt"
        if not sae_path.exists():
            sae_path = C.SAE_WEIGHTS
        scale_path = C.CACHE_DIR / f"scale_pretrained{base_suffix}_layer{layer}.npy"
        if not scale_path.exists():
            scale_path = C.CACHE_DIR / f"scale_pretrained_layer{layer}.npy"

        sae = BatchTopKSAE()
        sae.load_state_dict(torch.load(sae_path, map_location="cpu"))
        sae.to(device)
        model, _, _ = load_ecgfm(random_init=False)
        model.to(device)
        scale = float(np.load(scale_path))
        A_all = compute_atom_activations(index, sae, model, layer, scale, device)
        np.save(act_cache, A_all)
        del model, sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Control SAE & atom activations ────────────────────────────────────
    ctrl_act_cache = C.CACHE_DIR / f"atom_acts_ctrl{base_suffix}_layer{layer}.npy"
    ctrl_sae_path  = C.CACHE_DIR / f"sae_ctrl_seed0{base_suffix}_layer{layer}.pt"
    if not ctrl_sae_path.exists():
        ctrl_sae_path = C.SAE_CTRL_WEIGHTS
    ctrl_scale_path = C.CACHE_DIR / f"scale_ctrl_seed0{base_suffix}_layer{layer}.npy"
    if not ctrl_scale_path.exists():
        ctrl_scale_path = C.CACHE_DIR / f"scale_ctrl_seed0_layer{layer}.npy"

    if ctrl_act_cache.exists() and not rebuild_acts and len(np.load(ctrl_act_cache, mmap_mode="r")) == len(index):
        print(f"[Test C] Loading cached random-init control atom activations from {ctrl_act_cache}")
        A_ctrl = np.load(ctrl_act_cache)
    elif ctrl_sae_path.exists() and ctrl_scale_path.exists():
        print("[Test C] Computing random-init control SAE atom activations...")
        sae_ctrl = BatchTopKSAE()
        sae_ctrl.load_state_dict(torch.load(ctrl_sae_path, map_location="cpu"))
        sae_ctrl.to(device)
        model_ctrl, _, _ = load_ecgfm(random_init=True, seed=0)
        model_ctrl.to(device)
        scale_ctrl = float(np.load(ctrl_scale_path))
        A_ctrl = compute_atom_activations(index, sae_ctrl, model_ctrl, layer, scale_ctrl, device)
        np.save(ctrl_act_cache, A_ctrl)
        del model_ctrl, sae_ctrl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        A_ctrl = None
        print("[WARN] No control SAE weights found; run `python src/sae.py --ctrl` for random-init null.")

    # ── Add morphology features to index ─────────────────────────────────
    morph_cache = C.CACHE_DIR / f"morphology{base_suffix}.parquet"
    use_cached_morph = False
    if morph_cache.exists() and not rebuild_morph:
        morph_df = pd.read_parquet(morph_cache)
        if len(morph_df) == len(index) and morph_df["morph_hr"].notna().mean() > 0.5:
            use_cached_morph = True

    if use_cached_morph:
        print(f"[Test C] Loading cached morphology features from {morph_cache}")
        index = index.reset_index(drop=True)
        for col in MORPHOLOGY_PARAMS:
            if col in morph_df.columns:
                index[col] = morph_df[col].values
    else:
        print("[Test C] Extracting Lead II ECG morphology features...")
        morph_rows = []
        N_m = len(index)
        for i, (_, row) in enumerate(index.iterrows()):
            morph_rows.append(compute_morphology(row["ecg_path"]))
            if (i + 1) % 1000 == 0 or (i + 1) == N_m:
                print(f"  Morphology: {i+1}/{N_m}", end="\r", flush=True)
        print()
        morph_df = pd.DataFrame(morph_rows)
        morph_df.to_parquet(morph_cache, index=False)
        for col in MORPHOLOGY_PARAMS:
            index[col] = morph_df[col].values

    # ── Build covariate matrix ────────────────────────────────────────────
    subfolder_dummies = pd.get_dummies(index["folder"], prefix="f", drop_first=True).astype(float)
    model_dummies     = pd.get_dummies(index["model_id"].astype(str), prefix="model", drop_first=True).astype(float)
    covariates = [subfolder_dummies.values, model_dummies.values]
    if include_run_id_covariate:
        run_dummies = pd.get_dummies(index["run_id"].astype(str), prefix="run", drop_first=True).astype(float)
        covariates.append(run_dummies.values)
    C_mat = np.concatenate(covariates, axis=1)

    # ── Split masks ───────────────────────────────────────────────────────
    tr_mask   = (index["split"] == "train").values
    va_mask   = (index["split"] == "validation").values
    te_mask   = (index["split"] == "test").values
    held_mask = (index["split"].isin(["validation", "test"])).values

    # ── Discovery pass (train) ────────────────────────────────────────────
    print(f"\n[Test C] Discovery pass ({tr_mask.sum()} train ECGs)...")
    discovery = _run_associations(A_all[tr_mask], index[tr_mask], C_mat[tr_mask])
    disc_candidates_25 = {
        k for k, v in discovery.items()
        if v["q"] < C.ASSOC_Q_THR and abs(v["rho"]) >= C.ASSOC_RHO_THR
    }
    disc_candidates_15 = {
        k for k, v in discovery.items()
        if v["q"] < C.ASSOC_Q_THR and abs(v["rho"]) >= 0.15
    }
    print(f"  Discovered candidate (atom, param) pairs on train: "
          f"{len(disc_candidates_25)} (|rho|>={C.ASSOC_RHO_THR}), {len(disc_candidates_15)} (|rho|>=0.15)")

    # ── Confirmation pass (held-out combined & locked test split) ─────────
    print(f"[Test C] Confirmation pass ({held_mask.sum()} held-out ECGs, {te_mask.sum()} locked test ECGs)...")
    confirmation = _run_associations(
        A_all[held_mask], index[held_mask], C_mat[held_mask],
        candidate_keys=disc_candidates_25 if disc_candidates_25 else disc_candidates_15,
    )
    conf_test_locked = _run_associations(
        A_all[te_mask], index[te_mask], C_mat[te_mask],
        candidate_keys=disc_candidates_25 if disc_candidates_25 else disc_candidates_15,
    )

    def _collect_confirmed(conf_dict: dict, rho_thr: float) -> pd.DataFrame:
        rows = []
        for (atom_i, param), disc_row in discovery.items():
            if disc_row["q"] >= C.ASSOC_Q_THR or abs(disc_row["rho"]) < rho_thr:
                continue
            conf_row = conf_dict.get((atom_i, param))
            if conf_row is None:
                continue
            # Require consistent sign, q < 0.05, and |rho| >= rho_thr on confirmation split
            if (conf_row["q"] < C.ASSOC_Q_THR
                    and abs(conf_row["rho"]) >= rho_thr
                    and np.sign(disc_row["rho"]) == np.sign(conf_row["rho"])):
                rows.append({
                    "atom":      int(atom_i),
                    "param":     param,
                    "category":  "physiology" if param in MECHANISTIC_PARAMS else "morphology",
                    "rho_disc":  float(disc_row["rho"]),
                    "q_disc":    float(disc_row["q"]),
                    "rho_conf":  float(conf_row["rho"]),
                    "q_conf":    float(conf_row["q"]),
                    "q_conf_full": float(conf_row["q_full"]),
                    "n_samples": int(conf_row["n"]),
                })
        return pd.DataFrame(rows)

    confirmed_df = _collect_confirmed(confirmation, C.ASSOC_RHO_THR)
    confirmed_df_15 = _collect_confirmed(confirmation, 0.15)
    confirmed_locked_test_df = _collect_confirmed(conf_test_locked, C.ASSOC_RHO_THR)

    n_confirmed = int(confirmed_df["atom"].nunique()) if not confirmed_df.empty else 0
    n_pairs_confirmed = len(confirmed_df)
    n_confirmed_15 = int(confirmed_df_15["atom"].nunique()) if not confirmed_df_15.empty else 0
    print(f"\nConfirmed unique atoms (|rho|>={C.ASSOC_RHO_THR}): {n_confirmed} "
          f"({n_pairs_confirmed} atom-param pairs; threshold: >= {C.ASSOC_MIN_ATOMS})")
    print(f"Confirmed unique atoms (|rho|>=0.15): {n_confirmed_15} ({len(confirmed_df_15)} pairs)")

    # ── Permutation null ──────────────────────────────────────────────────
    print(f"[Test C] Running {n_permutations} stratified permutations...")
    perm_counts = _permutation_null(
        A_all[held_mask], index[held_mask], C_mat[held_mask],
        n_permutations, include_run_id_covariate, rho_thr=C.ASSOC_RHO_THR,
    )
    results["perm_null_mean"] = float(np.mean(perm_counts))
    results["perm_null_p95"]  = float(np.percentile(perm_counts, 95))
    results["perm_null_max"]  = int(np.max(perm_counts))

    # ── Random-init null ──────────────────────────────────────────────────
    if A_ctrl is not None:
        print("[Test C] Running random-init control discovery -> confirmation...")
        ctrl_disc = _run_associations(A_ctrl[tr_mask], index[tr_mask], C_mat[tr_mask])
        ctrl_cands = {
            k for k, v in ctrl_disc.items()
            if v["q"] < C.ASSOC_Q_THR and abs(v["rho"]) >= C.ASSOC_RHO_THR
        }
        ctrl_conf = _run_associations(
            A_ctrl[held_mask], index[held_mask], C_mat[held_mask],
            candidate_keys=ctrl_cands if ctrl_cands else None,
        )
        ctrl_atoms = {
            atom_i for (atom_i, param), v in ctrl_conf.items()
            if (atom_i, param) in ctrl_cands
            and v["q"] < C.ASSOC_Q_THR
            and abs(v["rho"]) >= C.ASSOC_RHO_THR
            and np.sign(ctrl_disc[(atom_i, param)]["rho"]) == np.sign(v["rho"])
        }
        results["ctrl_confirmed_atoms"] = int(len(ctrl_atoms))
        print(f"Random-init confirmed unique atoms: {len(ctrl_atoms)}")
    else:
        results["ctrl_confirmed_atoms"] = None

    # ── Atom binning ──────────────────────────────────────────────────────
    if not confirmed_df.empty:
        phys_atoms  = set(confirmed_df[confirmed_df["param"].isin(MECHANISTIC_PARAMS)]["atom"])
        morph_atoms = set(confirmed_df[~confirmed_df["param"].isin(MECHANISTIC_PARAMS)]["atom"]) - phys_atoms
        per_param_counts = confirmed_df.groupby("param")["atom"].nunique().sort_values(ascending=False).to_dict()
    else:
        phys_atoms = morph_atoms = set()
        per_param_counts = {}

    if not confirmed_df_15.empty:
        phys_atoms_15  = set(confirmed_df_15[confirmed_df_15["param"].isin(MECHANISTIC_PARAMS)]["atom"])
        morph_atoms_15 = set(confirmed_df_15[~confirmed_df_15["param"].isin(MECHANISTIC_PARAMS)]["atom"]) - phys_atoms_15
        per_param_counts_15 = confirmed_df_15.groupby("param")["atom"].nunique().sort_values(ascending=False).to_dict()
    else:
        phys_atoms_15 = morph_atoms_15 = set()
        per_param_counts_15 = {}

    results["n_physiology_atoms"] = int(len(phys_atoms))
    results["n_morphology_atoms"] = int(len(morph_atoms))
    results["n_confirmed_total"]  = int(n_confirmed)
    results["n_confirmed_pairs"]  = int(n_pairs_confirmed)
    results["per_param_confirmed_atoms"] = {k: int(v) for k, v in per_param_counts.items()}

    results["moderate_thr_015"] = {
        "n_physiology_atoms": int(len(phys_atoms_15)),
        "n_morphology_atoms": int(len(morph_atoms_15)),
        "n_confirmed_total":  int(n_confirmed_15),
        "n_confirmed_pairs":  int(len(confirmed_df_15)),
        "per_param_confirmed_atoms": {k: int(v) for k, v in per_param_counts_15.items()},
    }

    if not confirmed_locked_test_df.empty:
        phys_atoms_te = set(confirmed_locked_test_df[confirmed_locked_test_df["param"].isin(MECHANISTIC_PARAMS)]["atom"])
        morph_atoms_te = set(confirmed_locked_test_df[~confirmed_locked_test_df["param"].isin(MECHANISTIC_PARAMS)]["atom"]) - phys_atoms_te
        results["locked_test_confirmation"] = {
            "n_confirmed_total": int(confirmed_locked_test_df["atom"].nunique()),
            "n_confirmed_pairs": int(len(confirmed_locked_test_df)),
            "n_physiology_atoms": int(len(phys_atoms_te)),
            "n_morphology_atoms": int(len(morph_atoms_te)),
        }
    else:
        results["locked_test_confirmation"] = {
            "n_confirmed_total": 0,
            "n_confirmed_pairs": 0,
            "n_physiology_atoms": 0,
            "n_morphology_atoms": 0,
        }

    # ── Pass / fail ───────────────────────────────────────────────────────
    ctrl_count = results.get("ctrl_confirmed_atoms") or 0
    ratio_ok   = (n_confirmed >= 2 * max(ctrl_count, 1))
    if (n_confirmed >= C.ASSOC_MIN_ATOMS
            and ratio_ok
            and results["n_physiology_atoms"] > 0):
        verdict = "PASS"
    elif (n_confirmed_15 >= C.ASSOC_MIN_ATOMS and len(phys_atoms_15) > 0):
        verdict = "PASS_MODERATE_RHO"
    elif (results["n_morphology_atoms"] > 0 and results["n_physiology_atoms"] == 0):
        verdict = "MORPHOLOGY_ONLY"
    elif n_confirmed == 0 and n_confirmed_15 == 0:
        verdict = "NO_ATOMS"
    else:
        verdict = "MARGINAL"
    results["verdict"] = verdict
    print(f"\n[Test C] Verdict: {verdict}")

    # ── Save ──────────────────────────────────────────────────────────────
    C.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = confirmed_df if not confirmed_df.empty else confirmed_df_15
    if not confirmed_df.empty:
        confirmed_df.to_csv(C.RUNS_DIR / f"C_confirmed_atoms{suffix}.csv", index=False)
    if not confirmed_df_15.empty:
        confirmed_df_15.to_csv(C.RUNS_DIR / f"C_confirmed_atoms_rho015{suffix}.csv", index=False)
    with open(C.RUNS_DIR / f"C_results{suffix}.json", "w") as f:
        json.dump(results, f, indent=2)
    np.save(C.RUNS_DIR / f"C_perm_null{suffix}.npy", np.array(perm_counts))
    _plot_top_associations(plot_df, C.RUNS_DIR / f"C_top_associations{suffix}.png")
    print(f"Saved results to {C.RUNS_DIR}")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=C.LOCKED_LAYER or 8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-perms", type=int, default=1000)
    ap.add_argument("--include-run-id-covariate", action="store_true")
    ap.add_argument("--exclude-run-ids", nargs="*", default=None)
    ap.add_argument("--rebuild-acts", action="store_true")
    ap.add_argument("--rebuild-morph", action="store_true")
    args = ap.parse_args()
    run_test_c(
        args.layer, args.device, args.n_perms,
        args.include_run_id_covariate, args.exclude_run_ids,
        args.rebuild_acts, args.rebuild_morph,
    )
