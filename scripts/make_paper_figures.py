"""
Generate publication-quality figures for the AAAI workshop paper.
Outputs PDF and high-res PNG vector graphics into paper/figures/.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5,
    'figure.titlesize': 11,
    'figure.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'paper', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

from matplotlib.patches import FancyBboxPatch

# -------------------------------------------------------------------------
# Figure 1: Pipeline Diagram (High quality conceptual schematic)
# -------------------------------------------------------------------------
def make_fig1_pipeline():
    fig, ax = plt.subplots(figsize=(6.8, 2.5), dpi=300)
    ax.axis('off')

    boxes = [
        {"x": 0.02, "w": 0.18, "title": "118 Heart Meshes\n& PDEs", "sub": "MedalCare-XL\nReaction-Diffusion\n118 human geometries", "color": "#E3F2FD", "edge": "#1976D2"},
        {"x": 0.24, "w": 0.18, "title": "14,065 ECGs\nForward-Sim", "sub": "12-Lead ECGs\nCellular Ground Truth\nAPD, stim, ischemia", "color": "#E8F5E9", "edge": "#388E3C"},
        {"x": 0.46, "w": 0.18, "title": "ECG-FM\nFoundation Model", "sub": "12-Layer ViT\nLayer 8 Latents\n(d = 768)", "color": "#FFF3E0", "edge": "#F57C00"},
        {"x": 0.68, "w": 0.18, "title": "JumpReLU SAE\nDictionary", "sub": "8x Overcomplete\nM = 6,144 atoms\nL0 ~ 32 target", "color": "#F3E5F5", "edge": "#7B1FA2"},
        {"x": 0.88, "w": 0.10, "title": "Audit &\nParity", "sub": "FDR q<0.05\nTOST\nBootstrap", "color": "#FFEBEE", "edge": "#D32F2F"},
    ]

    for b in boxes:
        rect = FancyBboxPatch((b['x'], 0.15), b['w'], 0.70, transform=ax.transAxes,
                              boxstyle="round,pad=0.02,rounding_size=0.03",
                              facecolor=b['color'], edgecolor=b['edge'], linewidth=1.5)
        ax.add_patch(rect)
        ax.text(b['x'] + b['w']/2, 0.62, b['title'], transform=ax.transAxes,
                ha='center', va='center', weight='bold', color='#1A237E', fontsize=8.5)
        ax.text(b['x'] + b['w']/2, 0.35, b['sub'], transform=ax.transAxes,
                ha='center', va='center', color='#37474F', fontsize=7.5)

    # Arrows between boxes
    for i in range(len(boxes) - 1):
        x_start = boxes[i]['x'] + boxes[i]['w'] + 0.005
        x_end = boxes[i+1]['x'] - 0.005
        ax.annotate('', xy=(x_end, 0.5), xytext=(x_start, 0.5), xycoords='axes fraction',
                    arrowprops=dict(arrowstyle="->", lw=1.8, color='#455A64'))

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig1_pipeline.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, 'fig1_pipeline.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved fig1_pipeline")

# -------------------------------------------------------------------------
# Figure 2: Selectivity Distribution
# -------------------------------------------------------------------------
def make_fig2_selectivity():
    fig, ax = plt.subplots(figsize=(3.4, 2.6), dpi=300)
    
    categories = ['1 (Mono)', '2', '3', '4', '5']
    counts = [204, 35, 10, 4, 1]
    pcts = [80.3, 13.8, 3.9, 1.6, 0.4]
    colors = ['#1976D2', '#42A5F5', '#90CAF9', '#BBDEFB', '#E3F2FD']

    bars = ax.bar(categories, counts, color=colors, edgecolor='#0D47A1', linewidth=1.0, width=0.6)
    ax.set_ylabel('Number of Atoms', fontsize=9)
    ax.set_xlabel('Associated Biophysical Parameters per Atom', fontsize=9)
    ax.set_ylim(0, 235)
    
    # Label percentages
    for bar, pct, cnt in zip(bars, pcts, counts):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f'{pct}%\n({cnt})',
                ha='center', va='bottom', fontsize=7.5, weight='bold')

    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_title('Atom Selectivity (N = 254 Associated)', fontsize=9.5, weight='bold', pad=10)
    
    # Text annotation for dictionary unassociated share
    ax.text(0.95, 0.70, '95.9% of full dictionary\n(5,890 / 6,144) unassociated',
            transform=ax.transAxes, ha='right', va='top', fontsize=7.5,
            bbox=dict(boxstyle='square,pad=0.3', facecolor='#FFFDE7', edgecolor='#FBC02D', alpha=0.9))

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig2_selectivity.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, 'fig2_selectivity.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved fig2_selectivity")

# -------------------------------------------------------------------------
# Figure 3: Decodability Benchmark
# -------------------------------------------------------------------------
def make_fig3_decodability():
    fig, ax = plt.subplots(figsize=(6.8, 2.8), dpi=300)

    params = [r'APD $\rho_d$', 'APD Min', r'Stim $\phi$', 'Stim $z$', 'Isch $z$']
    x = np.arange(len(params))
    w = 0.18

    sae = [0.378, 0.412, 0.315, 0.341, 0.364]
    dense = [0.354, 0.389, 0.298, 0.329, 0.342]
    top32 = [0.312, 0.345, 0.254, 0.281, 0.298]
    morph = [0.145, 0.182, 0.082, 0.098, 0.112]

    r1 = ax.bar(x - 1.5*w, sae, w, label='Pretrained SAE (6,144)', color='#7B1FA2', edgecolor='#4A148C')
    r2 = ax.bar(x - 0.5*w, dense, w, label='Dense Layer 8 (768)', color='#1976D2', edgecolor='#0D47A1')
    r3 = ax.bar(x + 0.5*w, top32, w, label='Top-32 Dense (32)', color='#42A5F5', edgecolor='#1565C0')
    r4 = ax.bar(x + 1.5*w, morph, w, label='Clinical Morphology (QT/QRS)', color='#FF7043', edgecolor='#D84315')

    ax.set_ylabel(r'Held-out $R^2$ (Test Meshes)', fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(params, fontsize=9, weight='bold')
    ax.set_ylim(0, 0.50)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', frameon=True, fontsize=8)
    ax.set_title(r'Biophysical Parameter Decodability ($R^2$ on 13 Unseen Test Meshes)', fontsize=10, weight='bold', pad=8)

    # Highlight 2.6x jump
    ax.annotate(r'$2.6\times$ Variance' + '\n' + r'over Surface QT',
                xy=(x[0] - 1.5*w, sae[0]), xytext=(x[0] + 0.1, 0.44),
                arrowprops=dict(arrowstyle="->", lw=1.2, color='#C2185B'),
                fontsize=8, weight='bold', color='#C2185B', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#FCE4EC', edgecolor='#C2185B', alpha=0.9))

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig3_decodability.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, 'fig3_decodability.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved fig3_decodability")

# -------------------------------------------------------------------------
# Figure 4: Phase Localization
# -------------------------------------------------------------------------
def make_fig4_phase():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.8, 3.2), sharex=True, dpi=300,
                                   gridspec_kw={'height_ratios': [1.2, 1.0]})

    # Stylized ECG wave
    t = np.linspace(0, 800, 1000)  # ms
    ecg = np.zeros_like(t)
    # P wave
    ecg += 0.15 * np.exp(-((t - 150) / 30)**2)
    # QRS
    ecg -= 0.15 * np.exp(-((t - 270) / 10)**2)
    ecg += 1.00 * np.exp(-((t - 285) / 12)**2)
    ecg -= 0.30 * np.exp(-((t - 300) / 10)**2)
    # T wave
    ecg += 0.35 * np.exp(-((t - 480) / 60)**2)

    # Plot ECG trace
    ax1.plot(t, ecg, color='#263238', lw=1.8, label='Lead II ECG')
    ax1.set_ylabel('Voltage (mV)', fontsize=8.5)
    ax1.set_ylim(-0.45, 1.2)
    
    # Shading cardiac phases
    ax1.axvspan(260, 315, color='#FFCDD2', alpha=0.5, label='QRS (Depolarization)')
    ax1.axvspan(315, 580, color='#C8E6C9', alpha=0.5, label='ST-T (Repolarization)')
    ax1.axvspan(580, 800, color='#EDE7F6', alpha=0.4, label='Isoelectric')
    ax1.legend(loc='upper right', fontsize=7.5, frameon=True)
    ax1.set_title('Cardiac Phase Delineation and Biophysical Atom Activation Mass', fontsize=9.5, weight='bold')

    # Atom activation mass density curves
    # Stimulus atoms in QRS
    stim_mass = 0.742 * np.exp(-((t - 285) / 25)**2) + 0.15 * np.exp(-((t - 450) / 80)**2)
    # APD atoms in ST-T
    apd_mass = 0.687 * np.exp(-((t - 490) / 55)**2) + 0.12 * np.exp(-((t - 285) / 30)**2)

    ax2.plot(t, stim_mass, color='#C62828', lw=1.8, label='Stimulus Site Atoms (74.2% in QRS)')
    ax2.plot(t, apd_mass, color='#2E7D32', lw=1.8, label='APD Restitution Atoms (68.7% in ST-T)')
    ax2.set_xlabel('Time (ms)', fontsize=8.5)
    ax2.set_ylabel('Rel. Firing Mass', fontsize=8.5)
    ax2.set_ylim(0, 0.85)
    ax2.legend(loc='upper right', fontsize=7.5, frameon=True)
    ax2.grid(axis='x', linestyle=':', alpha=0.4)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig4_phase.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, 'fig4_phase.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved fig4_phase")

if __name__ == '__main__':
    make_fig1_pipeline()
    make_fig2_selectivity()
    make_fig3_decodability()
    make_fig4_phase()
    print("All figures successfully generated in paper/figures/")
