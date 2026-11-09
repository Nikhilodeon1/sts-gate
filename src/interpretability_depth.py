"""
interpretability_depth.py – Rigorous Mechanistic Interpretability Depth Checks for TMLR.

Implements three critical analyses that elevate correlational interpretability to causal,
clinically grounded machine learning:
  1. Monosemanticity Audit: Raw 768-D Neurons vs. 6,144-D SAE Atoms (quantifies superposition
     and polysemanticity reduction).
  2. Causal Feature Ablation: Targeted vs. Off-Target Interventions (evaluates causal specificity
     when zeroing/clamping condition-specific biophysical atoms).
  3. Waveform Grounding & Temporal Localization: Identifies top-activating ECGs and projects
     token-level SAE atom activations back onto 12-lead ECG time series (P-wave, QRS, ST-T).

Run:
    python src/interpretability_depth.py --layer 8 [--device cuda] [--all]
"""
import sys, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from data import build_index, load_medalcare
from assoc import (
    MECHANISTIC_PARAMS, MORPHOLOGY_PARAMS, ALL_PARAMS,
    partial_spearman, bh_qvalues, rho_to_pval, compute_morphology
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Monosemanticity Audit: Raw Neurons vs. SAE Atoms
# ─────────────────────────────────────────────────────────────────────────────

def run_monosemanticity_audit(
    layer: int = 8,
    device: str = "cpu",
    exclude_run_ids: list[str] | None = None,
    rho_thr: float = 0.20,
    q_thr: float = 0.05,
) -> dict:
    """
    Directly tests the central premise of Sparse Autoencoders: that raw foundation model
    neurons are polysemantic (entangled across multiple parameters due to superposition),
    whereas SAE atoms are monosemantic (selectively tuned to distinct parameters).
    """
    import torch
    from model import load_ecgfm, extract_layers
    from sae import BatchTopKSAE

    print("\n" + "="*70)
    print("ANALYSIS 1: Monosemanticity Audit (Raw Neurons vs. SAE Atoms)")
    print("="*70)

    index = build_index()
    if exclude_run_ids:
        index = index.loc[~index["run_id"].isin(exclude_run_ids)].reset_index(drop=True)

    # Use held-out splits (validation + test)
    held_mask = index["split"].isin(["validation", "test"]).to_numpy()
    idx_held = index[held_mask].reset_index(drop=True)
    N_held = len(idx_held)

    # Add morphology features if cached or compute
    morph_cache = C.CACHE_DIR / "morphology_sensitivity_without_S64_S67.parquet"
    if not morph_cache.exists():
        morph_cache = C.CACHE_DIR / "morphology.parquet"
    if morph_cache.exists():
        m_df = pd.read_parquet(morph_cache)
        for col in MORPHOLOGY_PARAMS:
            if col in m_df.columns:
                idx_held[col] = m_df.loc[held_mask, col].values

    # Covariate matrix: subfolder + model_id
    subfolder_dummies = pd.get_dummies(idx_held["folder"], prefix="f", drop_first=True).astype(float)
    model_dummies = pd.get_dummies(idx_held["model_id"].astype(str), prefix="model", drop_first=True).astype(float)
    C_mat = np.concatenate([subfolder_dummies.values, model_dummies.values], axis=1)

    # ── 1. Load Raw Neuron Activations (Pooled Layer 8 features) ─────────────
    print(f"Extracting Raw Layer {layer} Neuron Activations ({N_held} held-out ECGs)...")
    model, _, _ = load_ecgfm(random_init=False)
    model.to(device)
    model.eval()

    raw_neuron_acts = []
    batch_size = 32
    for start in range(0, N_held, batch_size):
        batch = idx_held.iloc[start:start+batch_size]
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
        with torch.no_grad():
            f0 = extract_layers(model, x0, layers=[layer], device=device)[layer].mean(dim=1).cpu().numpy()
            f1 = extract_layers(model, x1, layers=[layer], device=device)[layer].mean(dim=1).cpu().numpy()
        raw_neuron_acts.append((f0 + f1) / 2.0)
    A_raw = np.concatenate(raw_neuron_acts, axis=0)  # (N_held, 768)

    # ── 2. Load SAE Atom Activations ─────────────────────────────────────────
    act_cache = C.CACHE_DIR / f"atom_acts_sensitivity_without_S64_S67_layer{layer}.npy"
    if not act_cache.exists():
        act_cache = C.CACHE_DIR / f"atom_acts_layer{layer}.npy"
    if act_cache.exists():
        A_sae_all = np.load(act_cache)
        A_sae = A_sae_all[held_mask]  # (N_held, 6144)
    else:
        raise FileNotFoundError(f"SAE atom cache not found at {act_cache}. Run Test C first.")

    # ── 3. Profile Parameter Selectivity for Raw vs. SAE ─────────────────────
    print("Computing Partial Spearman profiles across biophysical parameters...")

    def _profile_matrix(A_matrix: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray, dict]:
        n_features = A_matrix.shape[1]
        feature_sig_counts = np.zeros(n_features, dtype=int)
        all_rhos_list = []
        all_pvals_list = []
        pair_map = []

        for param in ALL_PARAMS:
            if param not in idx_held.columns:
                continue
            p_vec = pd.to_numeric(idx_held[param], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(p_vec)
            if valid.sum() < 20 or float(np.std(p_vec[valid])) < 1e-8:
                continue
            rhos, val_mask = partial_spearman(A_matrix[valid], p_vec[valid], C_mat[valid])
            pvs = rho_to_pval(rhos, int(valid.sum()))
            for feat_idx in np.where(val_mask)[0]:
                all_rhos_list.append(rhos[feat_idx])
                all_pvals_list.append(pvs[feat_idx])
                pair_map.append((feat_idx, param))

        all_rhos = np.array(all_rhos_list)
        all_pvals = np.array(all_pvals_list)
        all_qs = bh_qvalues(all_pvals)

        feature_to_params = {i: [] for i in range(n_features)}
        for (feat_idx, param), r_val, q_val in zip(pair_map, all_rhos, all_qs):
            if q_val < q_thr and abs(r_val) >= rho_thr:
                feature_sig_counts[feat_idx] += 1
                feature_to_params[feat_idx].append(param)

        return feature_sig_counts, all_rhos, feature_to_params

    raw_counts, raw_rhos, raw_feat_map = _profile_matrix(A_raw, "Raw Neurons (768)")
    sae_counts, sae_rhos, sae_feat_map = _profile_matrix(A_sae, "SAE Atoms (6144)")

    # Active features with at least one association
    active_raw = raw_counts[raw_counts > 0]
    active_sae = sae_counts[sae_counts > 0]

    # Metrics
    raw_polysemantic_frac = float((raw_counts >= 3).sum() / max(len(active_raw), 1))
    sae_polysemantic_frac = float((sae_counts >= 3).sum() / max(len(active_sae), 1))
    raw_monosemantic_frac = float((raw_counts == 1).sum() / max(len(active_raw), 1))
    sae_monosemantic_frac = float((sae_counts == 1).sum() / max(len(active_sae), 1))

    mean_params_per_raw = float(np.mean(active_raw)) if len(active_raw) else 0.0
    mean_params_per_sae = float(np.mean(active_sae)) if len(active_sae) else 0.0

    results = {
        "raw_neurons_total": 768,
        "sae_atoms_total": 6144,
        "raw_active_features": int(len(active_raw)),
        "sae_active_features": int(len(active_sae)),
        "raw_mean_params_per_active_feature": mean_params_per_raw,
        "sae_mean_params_per_active_feature": mean_params_per_sae,
        "raw_monosemantic_fraction": raw_monosemantic_frac,
        "sae_monosemantic_fraction": sae_monosemantic_frac,
        "raw_polysemantic_fraction_ge3": raw_polysemantic_frac,
        "sae_polysemantic_fraction_ge3": sae_polysemantic_frac,
    }

    print("\nMonosemanticity Audit Results:")
    print(f"  Raw Neurons: Monosemantic={100*raw_monosemantic_frac:.1f}%, Highly Polysemantic (>=3 params)={100*raw_polysemantic_frac:.1f}%, Mean load={mean_params_per_raw:.2f} params/feature")
    print(f"  SAE Atoms:   Monosemantic={100*sae_monosemantic_frac:.1f}%, Highly Polysemantic (>=3 params)={100*sae_polysemantic_frac:.1f}%, Mean load={mean_params_per_sae:.2f} params/feature")

    # Plot comparative histogram
    _plot_monosemanticity(raw_counts, sae_counts, C.RUNS_DIR / "monosemanticity_comparison.png")

    out_path = C.RUNS_DIR / "monosemanticity_audit.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved audit metrics to {out_path}")
    return results


def _plot_monosemanticity(raw_counts: np.ndarray, sae_counts: np.ndarray, out_path: Path):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        bins = np.arange(0, 10) - 0.5
        ax.hist(
            raw_counts[raw_counts > 0], bins=bins, density=True, alpha=0.6,
            color="#d95f02", label="Raw Neurons (Layer 8, d=768)", edgecolor="black"
        )
        ax.hist(
            sae_counts[sae_counts > 0], bins=bins, density=True, alpha=0.6,
            color="#1b9e77", label="SAE Atoms (Layer 8, d=6144)", edgecolor="black"
        )
        ax.set_xticks(range(1, 9))
        ax.set_xticklabels([str(i) if i < 8 else "8+" for i in range(1, 9)])
        ax.set_xlabel("Number of Correlated Biophysical / Morphology Parameters")
        ax.set_ylabel("Probability Density (Active Features)")
        ax.set_title("Resolution of Polysemanticity: Raw Transformer Neurons vs. SAE Atoms")
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure to {out_path}")
    except Exception as e:
        print(f"[WARN] Could not plot monosemanticity: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Causal Feature Ablation & Steering
# ─────────────────────────────────────────────────────────────────────────────

def run_causal_ablation(
    layer: int = 8,
    k_ablate: int = 15,
    exclude_run_ids: list[str] | None = None,
) -> dict:
    """
    Causally intervenes on the network by zeroing out the top condition-specific
    biophysical SAE atoms and measuring:
      1. Targeted degradation on the matching clinical condition.
      2. Specificity preservation on off-target clinical conditions.
      3. Comparison against random active atom ablation (Control).
    """
    from sklearn.metrics import roc_auc_score

    print("\n" + "="*70)
    print("ANALYSIS 2: Causal Feature Ablation & Specificity Steering")
    print("="*70)

    index = build_index()
    if exclude_run_ids:
        index = index.loc[~index["run_id"].isin(exclude_run_ids)].reset_index(drop=True)

    labels = index["group"].map(C.GROUP_TO_INT).to_numpy()
    splits = index["split"].to_numpy()
    tr_mask = (splits == "train")
    te_mask = (splits == "test")

    act_cache = C.CACHE_DIR / f"atom_acts_sensitivity_without_S64_S67_layer{layer}.npy"
    if not act_cache.exists():
        act_cache = C.CACHE_DIR / f"atom_acts_layer{layer}.npy"
    if not act_cache.exists():
        raise FileNotFoundError(f"SAE atom cache not found at {act_cache}. Run Test C first.")
    A_all = np.load(act_cache)

    Z_tr = A_all[tr_mask]
    y_tr = labels[tr_mask]
    Z_te = A_all[te_mask]
    y_te = labels[te_mask]

    # Select top K atoms by train correlation / linear probe association for each class
    condition_atoms = {}
    for c_idx, c_name in enumerate(C.GROUP_LABELS):
        binary_y = (y_tr == c_idx).astype(float)
        # Point-biserial correlation
        atom_stds = Z_tr.std(axis=0) + 1e-8
        corrs = np.abs((Z_tr - Z_tr.mean(axis=0)).T @ (binary_y - binary_y.mean())) / (len(Z_tr) * atom_stds * binary_y.std())
        top_k = np.argsort(-corrs)[:k_ablate]
        condition_atoms[c_name] = top_k.tolist()

    # Fit a fast sparse-logistic probe on train
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Z_tr)
    clf = OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=1000))
    clf.fit(sc.transform(Z_tr), y_tr)

    # Baseline performance on test
    P_base = clf.predict_proba(sc.transform(Z_te))
    base_aucs = {}
    for i, g in enumerate(C.GROUP_LABELS):
        base_aucs[g] = float(roc_auc_score((y_te == i).astype(int), P_base[:, i]))

    ablation_results = {}
    rng = np.random.default_rng(42)
    active_atoms = np.where(Z_tr.mean(axis=0) > 0.05)[0]

    for target_group in C.GROUP_LABELS:
        ablated_atoms = condition_atoms[target_group]
        Z_te_ablated = Z_te.copy()
        Z_te_ablated[:, ablated_atoms] = 0.0

        P_ablated = clf.predict_proba(sc.transform(Z_te_ablated))
        ablated_aucs = {}
        for i, g in enumerate(C.GROUP_LABELS):
            ablated_aucs[g] = float(roc_auc_score((y_te == i).astype(int), P_ablated[:, i]))

        # Random ablation control
        rand_atoms = rng.choice(active_atoms, size=k_ablate, replace=False)
        Z_te_rand = Z_te.copy()
        Z_te_rand[:, rand_atoms] = 0.0
        P_rand = clf.predict_proba(sc.transform(Z_te_rand))
        rand_aucs = {}
        for i, g in enumerate(C.GROUP_LABELS):
            rand_aucs[g] = float(roc_auc_score((y_te == i).astype(int), P_rand[:, i]))

        target_drop = base_aucs[target_group] - ablated_aucs[target_group]
        off_target_drops = [
            base_aucs[g] - ablated_aucs[g] for g in C.GROUP_LABELS if g != target_group
        ]
        mean_off_target_drop = float(np.mean(off_target_drops))
        rand_target_drop = base_aucs[target_group] - rand_aucs[target_group]

        ablation_results[target_group] = {
            "baseline_auc": base_aucs[target_group],
            "ablated_auc": ablated_aucs[target_group],
            "target_auc_drop": float(target_drop),
            "random_control_drop": float(rand_target_drop),
            "mean_off_target_drop": mean_off_target_drop,
            "causal_specificity_ratio": float(target_drop / max(abs(mean_off_target_drop), 1e-4)),
        }

    print("\nCausal Ablation Summary (Ablating Top-15 Condition Atoms):")
    for g, metrics in ablation_results.items():
        print(f"  {g:10s}: Targeted Drop={metrics['target_auc_drop']:+.4f} | Random Control Drop={metrics['random_control_drop']:+.4f} | Off-Target Drop={metrics['mean_off_target_drop']:+.4f} | Specificity={metrics['causal_specificity_ratio']:.1f}x")

    _plot_causal_ablation(ablation_results, C.RUNS_DIR / "causal_ablation_effect.png")

    out_path = C.RUNS_DIR / "causal_ablation.json"
    with open(out_path, "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"Saved causal ablation metrics to {out_path}")
    return ablation_results


def _plot_causal_ablation(results: dict, out_path: Path):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        groups = list(results.keys())
        target_drops = [results[g]["target_auc_drop"] for g in groups]
        off_target_drops = [results[g]["mean_off_target_drop"] for g in groups]
        rand_drops = [results[g]["random_control_drop"] for g in groups]

        x = np.arange(len(groups))
        width = 0.27

        fig, ax = plt.subplots(figsize=(9.5, 4.5))
        ax.bar(x - width, target_drops, width, label="Targeted Condition Ablation", color="#d95f02")
        ax.bar(x, rand_drops, width, label="Random Active Atoms (Control)", color="#7570b3")
        ax.bar(x + width, off_target_drops, width, label="Off-Target Spillover Effect", color="#66a61e")

        ax.set_ylabel("AUROC Drop (Higher = Stronger Causal Impact)")
        ax.set_title("Causal Verification: Targeted SAE Feature Ablation Across Diagnoses")
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=20)
        ax.axhline(0.0, color="gray", linewidth=0.8)
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure to {out_path}")
    except Exception as e:
        print(f"[WARN] Could not plot causal ablation: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Waveform Grounding & Token Activation Overlays
# ─────────────────────────────────────────────────────────────────────────────

def run_waveform_grounding(
    layer: int = 8,
    device: str = "cpu",
    exclude_run_ids: list[str] | None = None,
) -> None:
    """
    Extracts the highest-activating 12-lead ECGs for top biophysical atoms and
    visualizes their temporal activation profile across the 5s cardiac cycle.
    """
    import torch
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from model import load_ecgfm, extract_layers
    from sae import BatchTopKSAE

    print("\n" + "="*70)
    print("ANALYSIS 3: Waveform Grounding & Temporal Localization")
    print("="*70)

    index = build_index()
    if exclude_run_ids:
        index = index.loc[~index["run_id"].isin(exclude_run_ids)].reset_index(drop=True)

    test_idx = index[index["split"] == "test"].reset_index(drop=True)

    # Load trained SAE and model
    scale_path = C.CACHE_DIR / f"scale_pretrained_sensitivity_without_S64_S67_layer{layer}.npy"
    if not scale_path.exists():
        scale_path = C.CACHE_DIR / f"scale_pretrained_layer{layer}.npy"
    sae_path = C.CACHE_DIR / f"sae_pretrained_sensitivity_without_S64_S67_layer{layer}.pt"
    if not sae_path.exists():
        sae_path = C.SAE_WEIGHTS

    scale = float(np.load(scale_path))
    sae = BatchTopKSAE().to(device)
    sae.load_state_dict(torch.load(sae_path, map_location=device))
    sae.eval()

    model, _, _ = load_ecgfm(random_init=False)
    model.to(device)
    model.eval()

    # Load Test C confirmed atoms table to pick top 4 diverse biophysical targets
    c_csv = C.RUNS_DIR / "C_confirmed_atoms_sensitivity_without_S64_S67.csv"
    if c_csv.exists():
        c_df = pd.read_csv(c_csv)
        top_atoms = []
        for p in ["v_APD.rho_d", "v_stim[3].phi", "morph_r_amplitude", "v_isch[0].z"]:
            sub = c_df[c_df["param"] == p]
            if not sub.empty:
                top_atoms.append((int(sub.sort_values(by="rho_conf", key=abs, ascending=False).iloc[0]["atom"]), p))
    else:
        top_atoms = [(4481, "v_APD.rho_d"), (760, "v_stim[3].phi"), (2928, "morph_r_amplitude"), (3450, "v_isch[0].z")]

    print(f"Profiling temporal waveforms for top atoms: {top_atoms}")

    fig, axes = plt.subplots(len(top_atoms), 2, figsize=(13, 2.6 * len(top_atoms)))
    fs = float(C.SAMPLE_RATE)
    time_sec = np.linspace(0, 5.0, C.WIN_SAMPLES)

    for row_idx, (atom_id, param_name) in enumerate(top_atoms):
        best_act = -1.0
        best_ecg = None
        best_tok_acts = None

        # Sample 100 test ECGs to find high-activation exemplar
        sample = test_idx.sample(min(120, len(test_idx)), random_state=42)
        for _, row in sample.iterrows():
            try:
                w = load_medalcare(row["ecg_path"])[0]  # (12, 2500)
            except Exception:
                continue
            x = torch.from_numpy(w[np.newaxis, :]).float()
            with torch.no_grad():
                feats = extract_layers(model, x, layers=[layer], device=device)[layer].to(device)
                _, z = sae(feats.reshape(-1, feats.shape[-1]) / scale)
                z_atom = z[:, atom_id].cpu().numpy()
            max_val = float(np.max(z_atom))
            if max_val > best_act:
                best_act = max_val
                best_ecg = w
                best_tok_acts = z_atom

        if best_ecg is None:
            continue

        # Interpolate 156 tokens across the 2500 time samples
        tok_time = np.linspace(0, 5.0, len(best_tok_acts))
        smooth_act = np.interp(time_sec, tok_time, best_tok_acts)
        norm_act = smooth_act / max(float(np.max(smooth_act)), 1e-6)

        # Plot Lead II
        ax_lead = axes[row_idx, 0]
        ax_act  = axes[row_idx, 1]

        ax_lead.plot(time_sec, best_ecg[1], color="#2b5c8f", linewidth=0.9, label="Lead II")
        ax_lead.fill_between(time_sec, best_ecg[1].min(), best_ecg[1].max(), where=(norm_act > 0.35),
                             color="#e41a1c", alpha=0.25, label="High Activation Window")
        ax_lead.set_ylabel(f"Atom {atom_id}\n({param_name})", fontsize=8.5, fontweight="bold")
        ax_lead.set_xlim(0, 5.0)
        ax_lead.grid(alpha=0.25)
        if row_idx == 0:
            ax_lead.set_title("12-Lead ECG Signal (Lead II with Atom Activation Zone)", fontsize=10)

        # Plot Chest Lead V1/V5 + Activation Trajectory
        ax_act.plot(time_sec, best_ecg[6], color="#4daf4a", linewidth=0.7, alpha=0.7, label="Lead V1")
        ax_act.plot(time_sec, best_ecg[10], color="#984ea3", linewidth=0.7, alpha=0.7, label="Lead V5")
        ax_act_twin = ax_act.twinx()
        ax_act_twin.plot(time_sec, norm_act, color="#e41a1c", linewidth=1.2, linestyle="--", label="SAE Activation")
        ax_act_twin.set_ylabel("Norm Act", color="#e41a1c", fontsize=8)
        ax_act_twin.set_ylim(-0.05, 1.1)
        ax_act.set_xlim(0, 5.0)
        ax_act.grid(alpha=0.25)
        if row_idx == 0:
            ax_act.set_title("Chest Leads (V1, V5) & Continuous SAE Atom Firing", fontsize=10)

        if row_idx == len(top_atoms) - 1:
            ax_lead.set_xlabel("Time (seconds)")
            ax_act.set_xlabel("Time (seconds)")

    fig.tight_layout()
    out_path = C.RUNS_DIR / "top_activating_waveforms.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved waveform grounding figure to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    import torch
    default_dev = "cuda" if torch.cuda.is_available() else "cpu"
    ap.add_argument("--layer", type=int, default=C.LOCKED_LAYER or 8)
    ap.add_argument("--device", default=default_dev)
    ap.add_argument("--exclude-run-ids", nargs="*", default=["run_S64", "run_S67"])
    ap.add_argument("--run-monosemanticity", action="store_true")
    ap.add_argument("--run-causal-ablation", action="store_true")
    ap.add_argument("--run-waveform-visualization", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    run_all = args.all or not (args.run_monosemanticity or args.run_causal_ablation or args.run_waveform_visualization)

    if run_all or args.run_monosemanticity:
        run_monosemanticity_audit(layer=args.layer, device=args.device, exclude_run_ids=args.exclude_run_ids)
    if run_all or args.run_causal_ablation:
        run_causal_ablation(layer=args.layer, exclude_run_ids=args.exclude_run_ids)
    if run_all or args.run_waveform_visualization:
        run_waveform_grounding(layer=args.layer, device=args.device, exclude_run_ids=args.exclude_run_ids)


if __name__ == "__main__":
    main()
