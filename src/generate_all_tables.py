"""
generate_all_tables.py – Single Source of Truth for all Dossier & Manuscript Tables.
Reads raw data files directly and prints exact verified numbers.
"""
import json
import pandas as pd
import numpy as np

def main():
    print("=" * 80)
    print("1. PROVENANCE & SPLITS")
    print("=" * 80)
    with open('runs/provenance_report.json') as f:
        prov = json.load(f)
    sens = prov['sensitivity_index']
    print(f"Clean Cohort Total: N = {sens['n_ecgs']} ECGs, {sens['n_meshes']} meshes")
    print(f"  Train: {sens['ecgs_per_split']['train']} ECGs ({sens['meshes_per_split']['train']} meshes)")
    print(f"  Val:   {sens['ecgs_per_split']['validation']} ECGs ({sens['meshes_per_split']['validation']} meshes)")
    print(f"  Test:  {sens['ecgs_per_split']['test']} ECGs ({sens['meshes_per_split']['test']} meshes)")
    print(f"Test C Confirmation Pool (Val+Test): {sens['ecgs_per_split']['validation'] + sens['ecgs_per_split']['test']} ECGs")
    
    print("\n" + "=" * 80)
    print("2. TABLE 1: TEST A PROBING")
    print("=" * 80)
    with open('runs/A_results_sensitivity_without_S64_S67.json') as f:
        a_res = json.load(f)
    for l, d in a_res['layer_results'].items():
        v = d['val']
        t = d['test']
        print(f"Layer {l:2s}: Val AUROC={v['macro_auc']:.6f}, Test AUROC={t['macro_auc']:.6f}, Test Acc={t['accuracy']*100:.2f}%, best_C={t['best_C']}")
    print(f"Control Val AUROC (3 seeds): mean={a_res['ctrl_mean_auc']:.4f}, raw={a_res['ctrl_val_aucs']}")

    print("\n" + "=" * 80)
    print("3. TABLE 2: TEST B SAE FIDELITY")
    print("=" * 80)
    with open('runs/B_results_sensitivity_without_S64_S67.json') as f:
        b_pre = json.load(f)['checks']
    with open('runs/B_results_ctrl_sensitivity_without_S64_S67.json') as f:
        b_ctrl = json.load(f)['checks']
    print(f"Pretrained : Held Token EV={b_pre['ev_held_per_token']*100:.2f}%, Global EV={b_pre['ev_held_global']*100:.2f}%, Dead={b_pre['dead_fraction']*100:.2f}% ({int(b_pre['dead_fraction']*6144)}/6144)")
    print(f"             Raw Test AUROC={b_pre['raw_test_auc']:.4f}, Recon Test AUROC={b_pre['recon_test_auc']:.4f}, Sparse Test AUROC={b_pre['sparse_z_test_auc']:.4f}")
    print(f"Random-Init: Held Token EV={b_ctrl['ev_held_per_token']*100:.2f}%, Global EV={b_ctrl['ev_held_global']*100:.2f}%, Dead={b_ctrl['dead_fraction']*100:.2f}% ({int(b_ctrl['dead_fraction']*6144)}/6144)")
    print(f"             Raw Test AUROC={b_ctrl['raw_test_auc']:.4f}, Recon Test AUROC={b_ctrl['recon_test_auc']:.4f}, Sparse Test AUROC={b_ctrl['sparse_z_test_auc']:.4f}")

    print("\n" + "=" * 80)
    print("4. TABLE 3: TEST C BIOPHYSICAL ATOMS (EXACT RAW COUNTS)")
    print("=" * 80)
    df = pd.read_csv('runs/C_confirmed_atoms_sensitivity_without_S64_S67.csv')
    cats = {
        'APD': df['param'].str.startswith('v_APD'),
        'Stim': df['param'].str.startswith('v_stim'),
        'Ischemia': df['param'].str.startswith('v_isch'),
        'Morphology': df['param'].str.startswith('morph_'),
    }
    for name, mask in cats.items():
        sub = df[mask]
        rhos = sub['rho_conf'].abs()
        print(f"{name:11s}: N_pairs={len(sub):3d}, unique_atoms={sub['atom'].nunique():3d}, mean={rhos.mean():.4f}, median={rhos.median():.4f}, std={rhos.std():.4f}, max={rhos.max():.4f}")
    
    headline = df[cats['APD'] | cats['Stim']]
    rhos_h = headline['rho_conf'].abs()
    print(f"Headline (APD|Stim) : N_pairs={len(headline):3d}, unique_atoms={headline['atom'].nunique():3d}, mean={rhos_h.mean():.4f}, median={rhos_h.median():.4f}, std={rhos_h.std():.4f}, max={rhos_h.max():.4f}")
    
    multi = df[cats['APD'] | cats['Stim'] | cats['Morphology']]
    rhos_m = multi['rho_conf'].abs()
    print(f"Multi (APD|Stim|Morph): N_pairs={len(multi):3d}, unique_atoms={multi['atom'].nunique():3d}, mean={rhos_m.mean():.4f}, median={rhos_m.median():.4f}, std={rhos_m.std():.4f}, max={rhos_m.max():.4f}")
    
    rhos_all = df['rho_conf'].abs()
    print(f"Total (All 4 domains) : N_pairs={len(df):3d}, unique_atoms={df['atom'].nunique():3d}, mean={rhos_all.mean():.4f}, median={rhos_all.median():.4f}, std={rhos_all.std():.4f}, max={rhos_all.max():.4f}")

    print("\n" + "=" * 80)
    print("5. SELECTIVITY (ATOMS & PARAMETER LOAD)")
    print("=" * 80)
    counts = df.groupby('atom')['param'].nunique()
    print(f"Total Associated Atoms (at |rho| >= 0.25): {len(counts)} ({len(counts)/6144*100:.2f}% of dictionary)")
    print(f"Unassociated Atoms: {6144 - len(counts)} ({(6144 - len(counts))/6144*100:.2f}% of dictionary)")
    for k, v in counts.value_counts().sort_index().items():
        print(f"  {k} params: {v:3d} atoms ({v/len(counts)*100:.2f}% of associated, {v/6144*100:.2f}% of dictionary)")
    print(f"Mean parameter load among associated: {counts.mean():.4f}")

if __name__ == "__main__":
    main()
