# Research Update & Reviewer Briefing Dossier
**Project:** `sts-gate` / ECG Ground-Truth Mechanistic Interpretability  
**Paper Title:** *Do Sparse Autoencoders Recover Biophysical Features in Pretrained Cardiac Foundation Models? An in silico Ground-Truth Case Study*  
**Primary Target Venue:** AAAI-27 Workshop on Foundation Models for Multisensory Intelligence (*Making Sense*) / AI4TS (Deadline: Nov 20, 2026; 7 pages + unlimited appendix)  
**Accompanying Manuscript PDF:** [`paper/main.pdf`](file:///c:/Users/nikhi/Codes/TEST/sts-gate/paper/main.pdf) (8 pages: 7 pages main text + references, 1 page appendix)

---

## 1. Executive Summary for the Mock Reviewer

Pretrained cardiac foundation models (such as ECG-FM) demonstrate state-of-the-art diagnostic capabilities across multi-lead electrocardiography. However, understanding their latent representations using mechanistic interpretability (e.g., Sparse Autoencoders / SAEs) faces a fundamental verification bottleneck: **in living human patient cohorts, true cellular-level cardiac states cannot be recorded invasively.** Prior benchmarks (e.g., CADENCE, ECG-InterpBench) relied exclusively on surface-interval calipers (e.g., QT intervals) or clinical ICD codes, which cannot distinguish genuine cellular biophysics from torso geometry or electrode shortcuts.

### Our Solution
We conduct the first **in silico ground-truth case study** utilizing the MedalCare-XL simulation framework:
- **118 Anatomically Distinct Human Geometries:** Patient-specific ventricular and torso meshes derived from clinical MRI/CT scans.
- **Multiscale Biophysical Forward Modeling:** Microscopic monodomain reaction-diffusion PDEs coupled to human cellular ionic models (ten Tusscher–Panfilov) forward-projected to standard 12-lead body-surface potentials ($N = 14,065$ clean primary cohort).
- **Exact Intracellular Ground Truth:** Every record logs mathematical ground truth for local action potential duration (APD) restitution ($v_{\text{APD}}$), 3D myocardial activation/stimulus coordinates ($v_{\text{stim}}$), and transmural ischemic lesions ($v_{\text{isch}}$).
- **Representation & Dictionary Learning:** Intermediate representations extracted from Layer 8 of ECG-FM ($d = 768$) and an $8\times$ overcomplete JumpReLU Sparse Autoencoder ($M = 6,144$ dictionary atoms).

---

## 2. Core Empirical Findings & Evidence Guide

| Claim | Finding & Evidence | Figure / Table in Paper |
| :--- | :--- | :--- |
| **1. Diagnostic Depth** | Layer 8 captures peak diagnostic information: Test AUROC $= 0.9925 \pm 0.003$ and Accuracy $= 90.84\%$ (vs. $0.8107$ for 3-seed random-weight control, $p < 10^{-15}$). | **Table 1** |
| **2. SAE Information Preservation** | Pretrained JumpReLU SAE achieves $88.18\%$ token explained variance with only 21 dead atoms ($0.34\%$). Linear probe on sparse codes achieves $0.9911$ AUROC. | **Table 2** |
| **3. Ground-Truth Biophysical Recovery** | Multi-mesh permutation testing with Benjamini-Hochberg FDR control ($q < 0.05, \|\rho\| \ge 0.25$) confirms **174 pure biophysical atoms** (120 APD, 58 stimulus site, 4 overlap). | **Table 3** & **Figure 1** |
| **4. High Parameter Selectivity** | **80.31% of associated atoms (204 / 254) are strictly monosemantic**, correlating with exactly one biophysical parameter (mean load $1.28 \pm 0.61$). | **Figure 2** |
| **5. Exceeding Clinical Calipers** | Latents decode APD restitution ($R^2 = 0.378$) with **$2.6\times$ higher variance explained** than standard surface QT intervals ($R^2 = 0.145$), ruling out surface-shortcut heuristics. Torso conductivities yield $R^2 \approx 0.000$. | **Table 4** & **Figure 3** |
| **6. The SAE-Dense Parity Result** | The SAE does **not** beat the dense representation in decodability ($\Delta R^2 \le +0.024$, 95% bootstrap CI $[-0.008, +0.056]$ crosses zero; TOST equivalent within $\pm 0.05$). Dense neurons have a higher hit rate ($10.5\%$ vs. $4.1\%$). | **Table 4** & **Figure 3** |
| **7. Dictionary Stability Paradox** | Dictionary atom vectors are unstable across random training seeds ($\sim 8\%$ vector cosine match $\ge 0.70$), yet **functional association replicates at $68.39\%$** ($119 / 174$), demonstrating that electrophysiological subspaces are robust but basis decomposition is non-unique. | **Table 5** |
| **8. Beat-Aligned Phase Localization** | Stimulus site atoms concentrate $74.2\%$ of activation mass within QRS depolarization ($p < 0.001$), while APD atoms concentrate $68.7\%$ within ST-T repolarization ($p < 0.001$). Token $L_0$ is inversely correlated with cardiac energy ($\rho = -0.345$). | **Figure 4** |

---

## 3. Visual Presentation Overview

The paper includes four publication-grade vector figures generated via matplotlib (`paper/figures/`):
- **Figure 1 (System Pipeline, Page 2):** High-level workflow from 118 ventricular meshes $\to$ PDEs $\to$ 14,065 ECGs $\to$ ECG-FM Layer 8 $\to$ JumpReLU SAE $\to$ FDR-controlled in-silico ground-truth verification.
- **Figure 2 (Atom Selectivity Distribution, Page 4):** Clear histogram showing the 80.3% monosemantic isolation rate across associated dictionary atoms.
- **Figure 3 (Biophysical Decodability Benchmark, Page 6):** Grouped bar chart comparing $R^2$ across SAE, Dense Layer 8, Top-32 Dense, and Clinical Morphology, highlighting the $2.6\times$ advantage over surface calipers and visual parity between SAE and Dense.
- **Figure 4 (Beat-Aligned Cardiac Phase Delineation, Page 6):** Multi-panel Lead II ECG with shaded cardiac intervals (QRS, ST-T, isoelectric) plotted against the empirical activation mass densities of stimulus and APD atoms.

---

## 4. Key Questions for the Mock Reviewer to Analyze

When reviewing [`paper/main.pdf`](file:///c:/Users/nikhi/Codes/TEST/sts-gate/paper/main.pdf), please evaluate:
1. **Calibration & Tone:** Does the framing strike the right balance between highlighting a genuine scientific discovery (pretrained representations encode intracellular kinetics far beyond clinical calipers) and sober restraint (SAEs provide coordinate transformation rather than superior decodability or causal control)?
2. **Visual Flow & Layout:** How do Figures 1–4 and Tables 1–5 read across the two-column layout? Does the visual hierarchy guide the reader naturally?
3. **Pacing within the 7-Page Limit:** The main body and references occupy pages 1–7, with supplementary methodology on page 8. Is the narrative concise yet mathematically and biologically self-contained?
4. **Differentiation from Prior Work:** Is the distinction from clinical-only benchmarks (CADENCE, ECG-InterpBench) sufficiently clear and compelling?

---

## 5. Single Source of Truth Verification
All reported metrics can be independently reproduced in 4 seconds by running:
```bash
python src/generate_all_tables.py
```
Outputs match the underlying raw JSON/CSV artifacts in `runs/` with 100% precision.
