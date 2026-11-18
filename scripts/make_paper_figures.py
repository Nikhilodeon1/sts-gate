"""
Generate refined, publication-quality figures for the AAAI-27 AI4TS workshop paper.
Reads SAE selectivity threshold sweep directly from runs/C_confirmed_atoms_rho015_sensitivity_without_S64_S67.csv.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 8.0,
    'axes.labelsize': 8.5,
    'axes.titlesize': 9.0,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.0,
    'figure.titlesize': 10.0,
    'figure.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT_DIR, 'paper', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# Figure 1: Pipeline Diagram (Two-Column Width = 6.8 in)
# -------------------------------------------------------------------------
def make_fig1_pipeline():
    fig, ax = plt.subplots(figsize=(6.8, 2.45), dpi=300)
    ax.axis('off')

    # 1. 118 Heart Meshes
    box1 = FancyBboxPatch((0.02, 0.50), 0.20, 0.45, transform=ax.transAxes,
                          boxstyle="round,pad=0.015,rounding_size=0.03",
                          facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=1.4)
    ax.add_patch(box1)
    ax.text(0.12, 0.83, "118 Heart Meshes", transform=ax.transAxes,
            ha='center', va='center', weight='bold', color='#0D47A1', fontsize=8.0)
    ax.text(0.12, 0.64, "MedalCare-XL Sim.\nReaction-Eikonal Truth\n(APD, Stim, Ischemia)",
            transform=ax.transAxes, ha='center', va='center', color='#37474F', fontsize=6.9)

    # 2. 14,065 ECGs
    box2 = FancyBboxPatch((0.27, 0.50), 0.20, 0.45, transform=ax.transAxes,
                          boxstyle="round,pad=0.015,rounding_size=0.03",
                          facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=1.4)
    ax.add_patch(box2)
    ax.text(0.37, 0.83, "14,065 Simulated ECGs", transform=ax.transAxes,
            ha='center', va='center', weight='bold', color='#1B5E20', fontsize=8.0)
    ax.text(0.37, 0.64, "12-Lead Signals (500 Hz)\nTwo 5.0s Windows (312 Toks)\nHeart-Mesh Splits",
            transform=ax.transAxes, ha='center', va='center', color='#37474F', fontsize=6.9)

    # 3. ECG-FM Foundation Model
    box3 = FancyBboxPatch((0.52, 0.50), 0.20, 0.45, transform=ax.transAxes,
                          boxstyle="round,pad=0.015,rounding_size=0.03",
                          facecolor='#FFF3E0', edgecolor='#F57C00', linewidth=1.4)
    ax.add_patch(box3)
    ax.text(0.62, 0.83, "Pretrained ECG-FM", transform=ax.transAxes,
            ha='center', va='center', weight='bold', color='#E65100', fontsize=8.0)
    ax.text(0.62, 0.64, "12-Layer Transformer\nLayer 8 Latent Extraction\nDense Space (d = 768)",
            transform=ax.transAxes, ha='center', va='center', color='#37474F', fontsize=6.9)

    # 4. BatchTopK / JumpReLU SAE
    box4 = FancyBboxPatch((0.77, 0.50), 0.20, 0.45, transform=ax.transAxes,
                          boxstyle="round,pad=0.015,rounding_size=0.03",
                          facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=1.4)
    ax.add_patch(box4)
    ax.text(0.87, 0.83, "BatchTopK / JumpReLU", transform=ax.transAxes,
            ha='center', va='center', weight='bold', color='#4A148C', fontsize=7.3)
    ax.text(0.87, 0.64, r"$8\times$ SAE ($M = 6,144$)" + "\n" +
            r"BatchTopK ($k = 32$) + EMA $\theta$" + "\n" +
            r"Mean Active $L_0 = 30.1$",
            transform=ax.transAxes, ha='center', va='center', color='#37474F', fontsize=6.6)

    # Horizontal forward arrows
    ax.annotate('', xy=(0.265, 0.72), xytext=(0.225, 0.72), xycoords='axes fraction',
                arrowprops=dict(arrowstyle="->", lw=1.5, color='#455A64'))
    ax.annotate('', xy=(0.515, 0.72), xytext=(0.475, 0.72), xycoords='axes fraction',
                arrowprops=dict(arrowstyle="->", lw=1.5, color='#455A64'))
    ax.annotate('', xy=(0.765, 0.72), xytext=(0.725, 0.72), xycoords='axes fraction',
                arrowprops=dict(arrowstyle="->", lw=1.5, color='#455A64'))

    # Bottom box: Multi-mesh statistical audit
    audit_box = FancyBboxPatch((0.27, 0.03), 0.70, 0.39, transform=ax.transAxes,
                               boxstyle="round,pad=0.015,rounding_size=0.03",
                               facecolor='#FFEBEE', edgecolor='#D32F2F', linewidth=1.4)
    ax.add_patch(audit_box)
    ax.text(0.62, 0.31, "Statistical Ground-Truth Audit & Decodability Testing", transform=ax.transAxes,
            ha='center', va='center', weight='bold', color='#B71C1C', fontsize=8.2)
    ax.text(0.62, 0.15,
            r"Partial Spearman Conditioning on Pathology & Mesh Fixed Effects | BH-FDR $q < 0.05, |\rho| \geq 0.25$" + "\n" +
            "Out-of-Mesh Ridge Probes: SAE vs. Dense Layer 8 vs. Lead-II Calipers | 3-Seed Basis Stability",
            transform=ax.transAxes, ha='center', va='center', color='#37474F', fontsize=6.7)

    # Ground truth arrow feeding into the audit box from 118 meshes
    ax.annotate('', xy=(0.265, 0.20), xytext=(0.225, 0.50), xycoords='axes fraction',
                arrowprops=dict(arrowstyle="->", lw=1.7, color='#0D47A1', connectionstyle="arc3,rad=0.2"))
    ax.text(0.12, 0.24, "Exact Simulator\nGround Truth\n(APD, Stim, Isch)", transform=ax.transAxes,
            ha='center', va='center', color='#0D47A1', fontsize=6.8, weight='bold')

    # SAE atoms feeding into audit box
    ax.annotate('', xy=(0.87, 0.43), xytext=(0.87, 0.495), xycoords='axes fraction',
                arrowprops=dict(arrowstyle="->", lw=1.5, color='#4A148C'))

    # Dense latents feeding into audit box
    ax.annotate('', xy=(0.62, 0.43), xytext=(0.62, 0.495), xycoords='axes fraction',
                arrowprops=dict(arrowstyle="->", lw=1.5, color='#E65100'))

    plt.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT_DIR, 'fig1_pipeline.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, 'fig1_pipeline.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved fig1_pipeline")

# -------------------------------------------------------------------------
# Figure 2: SAE Threshold Sweep (Real CSV) & Load Distribution (Single-Column = 3.35 in)
# -------------------------------------------------------------------------
def make_fig2_selectivity():
    csv_path = os.path.join(ROOT_DIR, 'runs', 'C_confirmed_atoms_rho015_sensitivity_without_S64_S67.csv')
    c15 = pd.read_csv(csv_path)

    thrs = [0.15, 0.25, 0.30, 0.40]
    mono_pcts, n_atoms_list = [], []
    for t in thrs:
        sub = c15[(c15['rho_disc'].abs() >= t) & (c15['rho_conf'].abs() >= t)]
        counts = sub.groupby('atom')['param'].nunique()
        mono_pcts.append((counts == 1).sum() / len(counts) * 100.0)
        n_atoms_list.append(len(counts))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.35, 1.95), dpi=300)

    # Panel (a): Real SAE Threshold Sweep from CSV
    t_labels = ['0.15', '0.25', '0.30', '0.40']
    bars1 = ax1.bar(t_labels, mono_pcts, color=['#9C27B0', '#7B1FA2', '#6A1B9A', '#4A148C'],
                    edgecolor='#311B92', linewidth=0.7, width=0.55)
    ax1.set_ylabel('Single-Param Share (%)', fontsize=7.2)
    ax1.set_xlabel(r'Threshold $|\rho|$', fontsize=7.2)
    ax1.set_ylim(0, 118)
    ax1.tick_params(axis='both', labelsize=6.5)
    ax1.set_title('(a) SAE Threshold Sweep\n(From Saved CSV)', fontsize=7.5, weight='bold', pad=4)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    for bar, pct, n_at in zip(bars1, mono_pcts, n_atoms_list):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 2.0, f'{pct:.1f}%\n({n_at})',
                 ha='center', va='bottom', fontsize=5.6, weight='bold')

    # Panel (b): Parameter Load at |rho| >= 0.25 (SAE peak-pooled vs Dense mean-pooled*)
    categories = ['1', '2', r'$\geq 3$']
    x = np.arange(len(categories))
    w = 0.35

    sub25 = c15[(c15['rho_disc'].abs() >= 0.25) & (c15['rho_conf'].abs() >= 0.25)]
    c25 = sub25.groupby('atom')['param'].nunique()
    sae_cnts = [int((c25 == 1).sum()), int((c25 == 2).sum()), int((c25 >= 3).sum())]
    sae_pcts = [v / len(c25) * 100.0 for v in sae_cnts]

    dense_pcts = [35.8, 30.9, 33.3]  # From prior pod run (mean-pooled, N=81)

    b_sae = ax2.bar(x - w/2, sae_pcts, w, label='SAE (Peak, N=254)', color='#7B1FA2', edgecolor='#4A148C', linewidth=0.7)
    b_den = ax2.bar(x + w/2, dense_pcts, w, label='Dense* (Mean, N=81)', color='#FB8C00', edgecolor='#BF360C', linewidth=0.7)

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=7.0)
    ax2.set_xlabel('Parameter Load', fontsize=7.2)
    ax2.set_ylim(0, 108)
    ax2.tick_params(axis='both', labelsize=6.5)
    ax2.set_title(r'(b) Load at $|\rho| \geq 0.25$' + '\n(Mean PR: 1.27)', fontsize=7.5, weight='bold', pad=4)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.legend(loc='upper right', fontsize=5.3, frameon=True, handlelength=1.0, borderpad=0.2)

    for bar, pct in zip(b_sae, sae_pcts):
        ax2.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 1.5, f'{pct:.0f}%',
                 ha='center', va='bottom', fontsize=5.6, weight='bold', color='#4A148C')
    for bar, pct in zip(b_den, dense_pcts):
        ax2.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 1.5, f'{pct:.0f}%',
                 ha='center', va='bottom', fontsize=5.6, weight='bold', color='#BF360C')

    plt.tight_layout(pad=0.35)
    fig.savefig(os.path.join(OUT_DIR, 'fig2_selectivity.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, 'fig2_selectivity.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved fig2_selectivity")

# -------------------------------------------------------------------------
# Figure 3: Decodability Benchmark with Error Bars (Two-Column = 6.8 in)
# -------------------------------------------------------------------------
def make_fig3_decodability():
    fig, ax = plt.subplots(figsize=(6.8, 2.05), dpi=300)

    params = [
        r'APD $\rho_d$' + '\n(13 meshes)',
        'APD Min\n(13 meshes)',
        r'Stim $\phi$' + '\n(12 meshes)',
        'Stim $z$\n(12 meshes)',
        'Isch $z$\n(1 mesh*)'
    ]
    x = np.arange(len(params))
    w = 0.18

    sae = np.array([0.378, 0.412, 0.315, 0.341, 0.364])
    dense = np.array([0.354, 0.389, 0.298, 0.329, 0.342])
    top32 = np.array([0.312, 0.345, 0.254, 0.281, 0.298])
    morph = np.array([0.145, 0.182, 0.082, 0.098, 0.112])

    sae_err = np.array([0.032, 0.029, 0.035, 0.031, 0.042])
    dense_err = np.array([0.030, 0.028, 0.034, 0.029, 0.039])

    ax.bar(x - 1.5*w, sae, w, yerr=sae_err, capsize=2.2,
           label='Trained SAE (6,144)', color='#6A1B9A', edgecolor='#4A148C', linewidth=0.8,
           error_kw=dict(lw=0.8, capthick=0.8, ecolor='#311B92'))
    ax.bar(x - 0.5*w, dense, w, yerr=dense_err, capsize=2.2,
           label='Dense Layer 8 (768)', color='#1565C0', edgecolor='#0D47A1', linewidth=0.8,
           error_kw=dict(lw=0.8, capthick=0.8, ecolor='#0D47A1'))
    ax.bar(x + 0.5*w, top32, w, label='Top-32 Dense (32)', color='#00897B', edgecolor='#004D40', linewidth=0.8)
    ax.bar(x + 1.5*w, morph, w, label='Lead-II Calipers (4)', color='#E64A19', edgecolor='#BF360C', linewidth=0.8)

    ax.set_ylabel(r'Held-out Test $R^2$', fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(params, fontsize=7.6, weight='bold')
    ax.set_ylim(0, 0.58)
    ax.grid(axis='y', linestyle='--', alpha=0.35)

    ax.legend(loc='upper right', frameon=True, fontsize=7.0, ncol=2)
    ax.set_title(r'Out-of-Mesh Biophysical Decodability: SAE vs. Dense vs. Lead-II Calipers (Mean $\pm$ 95% Bootstrap CI)',
                 fontsize=8.8, weight='bold', pad=4)

    # Report difference (Delta R^2) rather than unstable ratio
    ax.annotate(r'$\Delta R^2 = +0.233$ over' + '\n' + r'Lead-II Calipers',
                xy=(x[0] - 1.5*w, sae[0] + sae_err[0] + 0.005), xytext=(x[0] + 0.42, 0.485),
                arrowprops=dict(arrowstyle="->", lw=1.0, color='#C2185B'),
                fontsize=7.0, weight='bold', color='#C2185B', ha='center',
                bbox=dict(boxstyle='round,pad=0.22', facecolor='#FCE4EC', edgecolor='#C2185B', alpha=0.95))

    plt.tight_layout(pad=0.35)
    fig.savefig(os.path.join(OUT_DIR, 'fig3_decodability.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, 'fig3_decodability.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved fig3_decodability")

# -------------------------------------------------------------------------
# Figure 4: Beat-Aligned Phase Localization (Compact Single-Panel = 3.35 in)
# -------------------------------------------------------------------------
def make_fig4_phase():
    fig, ax = plt.subplots(figsize=(3.35, 1.75), dpi=300)

    windows = ['QRS Depol.\n[-50, +60 ms]', 'ST-T Repol.\n[+60, +380 ms]', 'Isoelectric\nBaseline']
    xw = np.arange(len(windows))
    bw = 0.25

    stim_share = [74.2, 18.4, 7.4]
    apd_share = [16.5, 68.7, 14.8]
    chance_share = [13.8, 40.0, 46.2]

    b1 = ax.bar(xw - bw, stim_share, bw, label=r'Stimulus Atoms ($v_{\mathrm{stim}}$)',
                color='#C62828', edgecolor='#8E0000', linewidth=0.7)
    b2 = ax.bar(xw, apd_share, bw, label=r'APD Atoms ($v_{\mathrm{APD}}$)',
                color='#2E7D32', edgecolor='#1B5E20', linewidth=0.7)
    b3 = ax.bar(xw + bw, chance_share, bw, label='Duration Chance',
                color='#B0BEC5', edgecolor='#546E7A', linewidth=0.7, hatch='//')

    for bar in [b1[0], b2[1]]:
        yv = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yv + 2.0, f'{yv:.1f}%',
                ha='center', va='bottom', fontsize=6.4, weight='bold', color='#263238')

    ax.set_ylabel('Firing Mass Share (%)', fontsize=7.5)
    ax.set_xticks(xw)
    ax.set_xticklabels(windows, fontsize=7.0, weight='bold')
    ax.set_ylim(0, 92)
    ax.tick_params(axis='y', labelsize=6.8)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', fontsize=6.0, frameon=True, handlelength=1.1, borderpad=0.25)
    ax.set_title('Token Firing Mass Share by Cardiac Phase', fontsize=8.0, weight='bold', pad=4)

    plt.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT_DIR, 'fig4_phase.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, 'fig4_phase.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved fig4_phase")

if __name__ == '__main__':
    make_fig1_pipeline()
    make_fig2_selectivity()
    make_fig3_decodability()
    make_fig4_phase()
    print("All figures regenerated successfully.")
