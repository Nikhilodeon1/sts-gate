"""
Generate refined, publication-quality figures for the AAAI-27 AI4TS workshop paper.
Ensures zero text/legend collisions, native column-width font sizing, and exact alignment with codebase/run data.
"""

import os
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

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'paper', 'figures')
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
    ax.text(0.12, 0.64, "MedalCare-XL PDEs\nCellular Ground Truth\n(APD, Stim, Ischemia)",
            transform=ax.transAxes, ha='center', va='center', color='#37474F', fontsize=6.9)

    # 2. 14,065 ECGs
    box2 = FancyBboxPatch((0.27, 0.50), 0.20, 0.45, transform=ax.transAxes,
                          boxstyle="round,pad=0.015,rounding_size=0.03",
                          facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=1.4)
    ax.add_patch(box2)
    ax.text(0.37, 0.83, "14,065 Simulated ECGs", transform=ax.transAxes,
            ha='center', va='center', weight='bold', color='#1B5E20', fontsize=8.0)
    ax.text(0.37, 0.64, "12-Lead Signals (500 Hz)\nTwo 5.0s Windows (312 Toks)\nMesh-Disjoint Splits",
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
    ax.text(0.62, 0.31, "Statistical Ground-Truth Audit & Decodability Parity Testing", transform=ax.transAxes,
            ha='center', va='center', weight='bold', color='#B71C1C', fontsize=8.2)
    ax.text(0.62, 0.15,
            r"Partial Spearman Conditioning on Pathology & Mesh Fixed Effects | BH-FDR $q < 0.05, |\rho| \geq 0.25$" + "\n" +
            "Out-of-Mesh Decodability: SAE vs. Dense Layer 8 vs. Lead-II Calipers | 3-Seed Basis Stability",
            transform=ax.transAxes, ha='center', va='center', color='#37474F', fontsize=6.7)

    # Ground truth arrow feeding into the audit box from 118 meshes
    ax.annotate('', xy=(0.265, 0.20), xytext=(0.225, 0.50), xycoords='axes fraction',
                arrowprops=dict(arrowstyle="->", lw=1.7, color='#0D47A1', connectionstyle="arc3,rad=0.2"))
    ax.text(0.12, 0.24, "Exact PDE\nGround Truth\n(APD, Stim, Isch)", transform=ax.transAxes,
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
# Figure 2: Selectivity Distribution: SAE vs. Dense Layer 8 (Single-Column = 3.35 in)
# -------------------------------------------------------------------------
def make_fig2_selectivity():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.35, 1.95), dpi=300, sharey=True)

    categories = ['1', '2', r'$\geq 3$']

    # Left panel: Sparse Autoencoder (N = 254 associated)
    sae_pcts = [80.3, 13.8, 5.9]
    sae_counts = [204, 35, 15]
    sae_colors = ['#7B1FA2', '#AB47BC', '#CE93D8']

    bars1 = ax1.bar(categories, sae_pcts, color=sae_colors, edgecolor='#4A148C', linewidth=0.8, width=0.55)
    ax1.set_ylabel('Share of Associated (%)', fontsize=7.5)
    ax1.set_xlabel('Parameter Load', fontsize=7.5)
    ax1.set_ylim(0, 104)
    ax1.set_title('(a) SAE (N = 254)\nMean: 1.28 params', fontsize=7.8, weight='bold', pad=4)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    for bar, pct, cnt in zip(bars1, sae_pcts, sae_counts):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f'{pct:.1f}%\n({cnt})',
                 ha='center', va='bottom', fontsize=6.2, weight='bold')

    # Right panel: Dense Layer 8 (N = 81 associated)
    dense_pcts = [35.8, 30.9, 33.3]
    dense_counts = [29, 25, 27]
    dense_colors = ['#E65100', '#FB8C00', '#FFCC80']

    bars2 = ax2.bar(categories, dense_pcts, color=dense_colors, edgecolor='#BF360C', linewidth=0.8, width=0.55)
    ax2.set_xlabel('Parameter Load', fontsize=7.5)
    ax2.set_title('(b) Dense L8 (N = 81)\nMean: 2.26 params', fontsize=7.8, weight='bold', pad=4)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)

    for bar, pct, cnt in zip(bars2, dense_pcts, dense_counts):
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f'{pct:.1f}%\n({cnt})',
                 ha='center', va='bottom', fontsize=6.2, weight='bold')

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

    # R2 values
    sae = np.array([0.378, 0.412, 0.315, 0.341, 0.364])
    dense = np.array([0.354, 0.389, 0.298, 0.329, 0.342])
    top32 = np.array([0.312, 0.345, 0.254, 0.281, 0.298])
    morph = np.array([0.145, 0.182, 0.082, 0.098, 0.112])

    # Bootstrap 95% CI half-widths
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

    # Place legend in upper right where bars are shorter (<= 0.364), leaving left side completely clear!
    ax.legend(loc='upper right', frameon=True, fontsize=7.0, ncol=2)
    ax.set_title(r'Out-of-Mesh Biophysical Decodability: SAE vs. Dense vs. Lead-II Calipers (Mean $\pm$ 95% Bootstrap CI)',
                 fontsize=8.8, weight='bold', pad=4)

    # Highlight 2.6x jump over morphology on the left, completely clear of the upper-right legend
    ax.annotate(r'$2.6\times$ Variance over' + '\n' + r'Lead-II Calipers',
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
# Figure 4: Beat-Aligned Phase Localization (Native Single-Column Width = 3.35 in)
# -------------------------------------------------------------------------
def make_fig4_phase():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.35, 2.35), dpi=300,
                                   gridspec_kw={'height_ratios': [0.85, 1.15]})

    # Top panel: Schematic Fiducial Window Partition (800 ms cycle, 32 ms tokens)
    t = np.linspace(0, 800, 800)
    ecg = (0.14 * np.exp(-((t - 120) / 28)**2)
           - 0.14 * np.exp(-((t - 235) / 10)**2)
           + 0.95 * np.exp(-((t - 250) / 12)**2)
           - 0.25 * np.exp(-((t - 265) / 10)**2)
           + 0.32 * np.exp(-((t - 460) / 58)**2))

    ax1.plot(t, ecg, color='#263238', lw=1.2)
    ax1.set_ylabel('Lead II (mV)', fontsize=7.2)
    ax1.set_ylim(-0.45, 1.25)
    ax1.set_xlim(0, 800)

    # Exact windows from src/phase_localization.py: QRS = [R-50, R+60] (110 ms = 13.8%), ST-T = [R+60, R+380] (320 ms = 40.0%)
    ax1.axvspan(200, 310, color='#FFCDD2', alpha=0.55, label='QRS (110 ms, 13.8%)')
    ax1.axvspan(310, 630, color='#C8E6C9', alpha=0.55, label='ST-T (320 ms, 40.0%)')
    ax1.axvspan(0, 200, color='#EDE7F6', alpha=0.35)
    ax1.axvspan(630, 800, color='#EDE7F6', alpha=0.35, label='Baseline (46.2%)')
    ax1.set_xlabel('Cycle Time (ms)', fontsize=7.0, labelpad=1)
    ax1.tick_params(axis='both', labelsize=6.5)
    ax1.legend(loc='upper right', fontsize=5.8, frameon=True, ncol=2, handlelength=1.0, borderpad=0.25)
    ax1.set_title('(a) Schematic Beat-Aligned Fiducial Windows (32 ms Tokens)',
                  fontsize=7.6, weight='bold', pad=3)

    # Bottom panel: Grouped Bar Chart of Window Activation Share (%)
    windows = ['QRS Depol.\n[-50, +60 ms]', 'ST-T Repol.\n[+60, +380 ms]', 'Isoelectric\nBaseline']
    xw = np.arange(len(windows))
    bw = 0.25

    stim_share = [74.2, 18.4, 7.4]
    apd_share = [16.5, 68.7, 14.8]
    chance_share = [13.8, 40.0, 46.2]

    b1 = ax2.bar(xw - bw, stim_share, bw, label=r'Stimulus Atoms ($v_{\mathrm{stim}}$)',
                 color='#C62828', edgecolor='#8E0000', linewidth=0.7)
    b2 = ax2.bar(xw, apd_share, bw, label=r'APD Atoms ($v_{\mathrm{APD}}$)',
                 color='#2E7D32', edgecolor='#1B5E20', linewidth=0.7)
    b3 = ax2.bar(xw + bw, chance_share, bw, label='Duration Chance',
                 color='#B0BEC5', edgecolor='#546E7A', linewidth=0.7, hatch='//')

    for bar in [b1[0], b2[1]]:
        yv = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yv + 2.0, f'{yv:.1f}%',
                 ha='center', va='bottom', fontsize=6.2, weight='bold', color='#263238')

    ax2.set_ylabel('Firing Mass Share (%)', fontsize=7.2)
    ax2.set_xticks(xw)
    ax2.set_xticklabels(windows, fontsize=6.8, weight='bold')
    ax2.set_ylim(0, 92)
    ax2.tick_params(axis='y', labelsize=6.5)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.legend(loc='upper right', fontsize=5.9, frameon=True, handlelength=1.1, borderpad=0.25)
    ax2.set_title('(b) Empirical Token Activation Mass by Cardiac Phase',
                  fontsize=7.6, weight='bold', pad=3)

    plt.tight_layout(pad=0.35)
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
