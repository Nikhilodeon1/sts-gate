"""
phase_localization.py – Cardiac Phase & Fiducial Token Localization Audit.

Maps temporal tokens (t=1..156, ~32 ms/token) to cardiac cycle phases
(QRS depolarization, ST-T repolarization, and isoelectric baseline) using
ECG fiducials. Evaluates:
  1. Activation mass concentration of Stimulus site atoms (depolarization)
  2. Activation mass concentration of APD atoms (repolarization)
  3. Comparison against unassociated control dictionary atoms
  4. Within-record temporal phase permutation test (1,000 perms)
  5. Token L0 and patch RMS energy by cardiac phase (settling the L0 mechanism)

Run:
    python src/phase_localization.py [--device cpu] [--layer 8]
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from data import build_index, load_medalcare
from assoc import compute_morphology, MECHANISTIC_PARAMS

SUFFIX = "_sensitivity_without_S64_S67"


def delineate_phases_lead2(ecg_12x5000: np.ndarray, fs: int = 500, n_tokens: int = 156) -> np.ndarray:
    """
    Given a 10s 12-lead ECG, delineates 5s Window 0 (2500 samples) of Lead II (index 1).
    Maps each of the 156 tokens (16 samples = 32 ms per token) to:
      0 = isoelectric baseline (T-P segment / diastolic interval)
      1 = QRS complex (ventricular depolarization)
      2 = ST-T complex (ventricular repolarization)
    Returns: (156,) int array.
    """
    lead2 = ecg_12x5000[1, :2500]  # 5s Window 0
    token_labels = np.zeros(n_tokens, dtype=np.int32)
    step = 2500 / n_tokens  # ~16.025 samples per token

    # Find R peaks
    peaks, props = find_peaks(lead2, distance=fs * 0.4, prominence=0.3 * np.std(lead2))
    if len(peaks) == 0:
        # fallback: use center of beat
        return token_labels

    for r in peaks:
        # QRS window: ~ -50 ms to +60 ms around R peak (-25 to +30 samples)
        qrs_start = max(0, int(r - 0.05 * fs))
        qrs_end = min(2500, int(r + 0.06 * fs))
        
        # ST-T window: ~ +60 ms to +380 ms after R peak (+30 to +190 samples)
        st_t_start = qrs_end
        st_t_end = min(2500, int(r + 0.38 * fs))

        # Map to tokens
        t_qrs_0, t_qrs_1 = int(qrs_start / step), min(n_tokens, int(np.ceil(qrs_end / step)))
        t_st_0, t_st_1 = int(st_t_start / step), min(n_tokens, int(np.ceil(st_t_end / step)))

        token_labels[t_qrs_0:t_qrs_1] = 1
        token_labels[t_st_0:t_st_1] = 2

    return token_labels


def compute_token_activations_sample(index: pd.DataFrame, sae, model, layer: int, scale: float,
                                     n_samples: int = 200, device: str = "cpu"):
    """
    Extracts raw token activations z_t (B, T, D) and Lead II phase labels for n_samples held-out ECGs.
    """
    import torch
    from model import extract_layers

    sae.eval()
    model.eval()

    sample_rows = index.iloc[:n_samples]
    all_z = []
    all_phases = []
    all_energies = []

    for _, row in sample_rows.iterrows():
        try:
            w = load_medalcare(row["ecg_path"])
        except Exception:
            continue
        
        phases = delineate_phases_lead2(np.concatenate([w[0], w[1]], axis=1))
        # patch RMS energy per token
        w0_lead2 = w[0][1] # 2500 samples
        tokens = np.array_split(w0_lead2, 156)
        rms = np.array([np.sqrt(np.mean(tok**2)) for tok in tokens])

        x0 = torch.from_numpy(w[0][None, ...]).float()
        with torch.no_grad():
            feats = extract_layers(model, x0, layers=[layer], device=device)[layer].to(device)
            B, T, D = feats.shape
            _, z = sae(feats.reshape(-1, D) / scale)
            z = z.reshape(B, T, -1).cpu().numpy()[0] # (156, d_sae)

        all_z.append(z)
        all_phases.append(phases)
        all_energies.append(rms)

    return np.array(all_z), np.array(all_phases), np.array(all_energies)


def run_phase_localization(z_tokens: np.ndarray, phases: np.ndarray, energies: np.ndarray,
                           stim_atoms: list[int], apd_atoms: list[int],
                           n_perms: int = 1000, seed: int = 0) -> dict:
    """
    Computes activation mass in QRS (phase 1) vs ST-T (phase 2) vs baseline (phase 0).
    """
    rng = np.random.default_rng(seed)
    N, T, D = z_tokens.shape

    is_qrs = (phases == 1)
    is_st_t = (phases == 2)
    is_base = (phases == 0)

    # L0 and energy by phase
    l0_per_token = (z_tokens > 0).sum(axis=-1)  # (N, T)
    l0_by_phase = {
        "QRS": float(l0_per_token[is_qrs].mean()) if is_qrs.any() else 0.0,
        "ST_T": float(l0_per_token[is_st_t].mean()) if is_st_t.any() else 0.0,
        "Baseline": float(l0_per_token[is_base].mean()) if is_base.any() else 0.0,
    }
    energy_by_phase = {
        "QRS": float(energies[is_qrs].mean()) if is_qrs.any() else 0.0,
        "ST_T": float(energies[is_st_t].mean()) if is_st_t.any() else 0.0,
        "Baseline": float(energies[is_base].mean()) if is_base.any() else 0.0,
    }

    # Activation mass fraction
    total_z = np.maximum(z_tokens.sum(axis=1), 1e-12)  # (N, D)

    # Stimulus site atoms -> expected QRS mass
    z_stim = z_tokens[:, :, stim_atoms]
    qrs_mass_stim = (z_stim * is_qrs[..., None]).sum(axis=1) / np.maximum(z_stim.sum(axis=1), 1e-12)
    mean_qrs_mass_stim = float(qrs_mass_stim.mean())

    # APD atoms -> expected ST-T mass
    z_apd = z_tokens[:, :, apd_atoms]
    st_mass_apd = (z_apd * is_st_t[..., None]).sum(axis=1) / np.maximum(z_apd.sum(axis=1), 1e-12)
    mean_st_mass_apd = float(st_mass_apd.mean())

    # Background atoms (all non-confirmed)
    all_conf = set(stim_atoms) | set(apd_atoms)
    bg_atoms = [i for i in range(D) if i not in all_conf][:500]
    z_bg = z_tokens[:, :, bg_atoms]
    qrs_mass_bg = (z_bg * is_qrs[..., None]).sum(axis=1) / np.maximum(z_bg.sum(axis=1), 1e-12)
    st_mass_bg = (z_bg * is_st_t[..., None]).sum(axis=1) / np.maximum(z_bg.sum(axis=1), 1e-12)

    # Within-record permutation null (shuffle phase sequence within each record)
    null_stim_qrs = []
    null_apd_st = []
    for _ in range(n_perms):
        shuf_phases = np.array([rng.permutation(phases[i]) for i in range(N)])
        shuf_qrs = (shuf_phases == 1)
        shuf_st = (shuf_phases == 2)
        q_m = (z_stim * shuf_qrs[..., None]).sum(axis=1) / np.maximum(z_stim.sum(axis=1), 1e-12)
        s_m = (z_apd * shuf_st[..., None]).sum(axis=1) / np.maximum(z_apd.sum(axis=1), 1e-12)
        null_stim_qrs.append(float(q_m.mean()))
        null_apd_st.append(float(s_m.mean()))

    pval_stim = float((np.array(null_stim_qrs) >= mean_qrs_mass_stim).mean())
    pval_apd = float((np.array(null_apd_st) >= mean_st_mass_apd).mean())

    return {
        "l0_by_cardiac_phase": l0_by_phase,
        "energy_by_cardiac_phase": energy_by_phase,
        "stimulus_atoms": {
            "n_atoms": len(stim_atoms),
            "mean_activation_mass_in_QRS": mean_qrs_mass_stim,
            "background_mass_in_QRS": float(qrs_mass_bg.mean()),
            "perm_null_mean_in_QRS": float(np.mean(null_stim_qrs)),
            "perm_pval": pval_stim
        },
        "apd_atoms": {
            "n_atoms": len(apd_atoms),
            "mean_activation_mass_in_ST_T": mean_st_mass_apd,
            "background_mass_in_ST_T": float(st_mass_bg.mean()),
            "perm_null_mean_in_ST_T": float(np.mean(null_apd_st)),
            "perm_pval": pval_apd
        }
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--n-perms", type=int, default=1000)
    args = ap.parse_args()

    csv_path = C.RUNS_DIR / f"C_confirmed_atoms{SUFFIX}.csv"
    if not csv_path.exists():
        print(f"[ERROR] {csv_path} missing.")
        return

    df = pd.read_csv(csv_path)
    stim_atoms = sorted(df[df["param"].str.startswith("v_stim")]["atom"].unique())
    apd_atoms = sorted(df[df["param"].str.startswith("v_APD")]["atom"].unique())
    print(f"Loaded {len(stim_atoms)} stimulus atoms and {len(apd_atoms)} APD atoms.")

    # Check for cached token activations or run extraction
    tok_cache = C.CACHE_DIR / "sample_phase_tokens.npz"
    if tok_cache.exists():
        data = np.load(tok_cache)
        z_tokens, phases, energies = data["z_tokens"], data["phases"], data["energies"]
        print(f"Loaded cached token activations {z_tokens.shape}")
    else:
        print("[WARN] Local token cache not found. Please run on GPU/Colab with data bundle.")
        return

    res = run_phase_localization(z_tokens, phases, energies, stim_atoms, apd_atoms, n_perms=args.n_perms)
    out_path = C.RUNS_DIR / "R4_phase_localization.json"
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n[DONE] Saved phase localization audit to {out_path}")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
