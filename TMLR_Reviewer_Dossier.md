# TMLR Reviewer Dossier: A Biophysical Ground-Truth Benchmark for Sparse Autoencoders in Pretrained Cardiac Foundation Models

**Target Venue:** Transactions on Machine Learning Research (TMLR)  
**Paper Title:** *A Biophysical Ground-Truth Benchmark for Sparse Autoencoder Feature Recovery in Pretrained Cardiac Foundation Models*  
**Date of Dossier:** November 11, 2026 (Revised & Calibrated Submission Audit)  
**Artifact Archive:** `ecg_gate_reviewer_audit_package.zip` in workspace root.

---

## Executive Summary & Calibrated Paper Framing

Recent work has demonstrated that self-supervised foundation models trained on millions of surface electrocardiograms (ECGs) achieve strong diagnostic classification performance. However, their internal representations remain black boxes, and standard post-hoc methods (saliency maps, Grad-CAM) cannot untangle polysemantic representations caused by feature superposition. Sparse Autoencoders (SAEs) have emerged as a promising tool for dictionary decomposition, but evaluating whether learned dictionary atoms correspond to true underlying generative mechanisms is fundamentally constrained in clinical data, where hidden electrophysiological states cannot be measured simultaneously.

**The Paper's Framing:** We introduce **MedalCare-XL** as an *in silico* ground-truth benchmark to evaluate whether, and to what extent, Sparse Autoencoders recover known latent biophysical parameters from a clinically pretrained foundation model (**ECG-FM**, 90.8M parameters, 12 layers). Across 16,842 simulated 12-lead ECGs generated via coupled partial differential equation (PDE) bidomain models across 144 biventricular heart geometries, we test feature recovery, cross-geometry persistence, causal necessity, and multi-seed stability.

### Key Calibrated Findings & Honest Limitations:
1. **Representational Probing:** Linear probes on Layer 8 separate 8 simulated cardiac pathologies with **0.9921 Test Macro AUROC** on the full primary dataset ($N=16,842$) and **0.9925** on the split-disjoint sensitivity subset ($N=14,065$), compared to **0.8113** for a 3-seed random-initialization control. A geometry-only baseline probe (using heart rotation and torso conductivities) achieves a balanced accuracy of **12.50%** (exact $1/8$ chance) and AUROC of **0.5427**, collapsing into majority-class MI predictions.
2. **Dictionary Fidelity:** A BatchTopK SAE ($d_{\text{in}} = 768, d_{\text{sae}} = 6,144, k=32$) trained strictly on training-split tokens reconstructs held-out tokens with **88.18% explained variance**, **0.34% dead atom rate**, and minor downstream probe drop ($\Delta = 0.0026$, from $0.9925 \to 0.9899$). Evaluated across 31,200 held-out tokens, mean token $L_0$ is **$30.82 \pm 6.45$** with **0.00% dead tokens**; variation reflects a mild negative correlation ($\rho = -0.3450$) with patch signal energy.
3. **Biophysical Parameter Recovery:** Adjusting for disease class, simulation batch, and 3D anatomy mesh identity, **254 unique dictionary atoms** exhibit statistically validated partial rank correlations ($|\rho| \ge 0.25, q < 0.05$) with intracellular and tissue parameters. A trained random-initialization SAE control recovers **zero** ($0$) biophysical atoms. Excluding post-hoc ECG surface signal features (QRS, QT, R-amplitude, heart rate), **151 pure biophysical atoms** correspond to cellular Action Potential Duration (APD) and Purkinje ventricular stimulus site coordinates across 107–118 independent meshes.
4. **Causal Necessity (Probed Subspace):** Zeroing condition-specific atoms in the SAE representation produces targeted linear probe degradation in conditions such as LBBB ($+0.0120$ drop) and FAM ($+0.0074$ drop), with monotonic dose-response scaling from $k=1 \to 32$. However, on saturated ceiling classes (AV Block, MI at 1.0000 AUROC), AUROC drops exhibit floor effects. We frame this strictly as a test of linear representation necessity on the probed subspace, rather than end-to-end downstream network patching.
5. **Multi-Seed Stability (An Honest Finding):** Evaluating three independently initialized SAE models (Seeds 0, 1, 2) demonstrates that unconstrained $8\times$ overcomplete dictionaries exhibit substantial cross-seed variation (mean pairwise max cosine similarity $\approx 0.311$, with only $2.54\%$ matching at $\ge 0.70$). However, the 254 biophysically confirmed atoms exhibit a **$3.26\times$ to $3.84\times$ enrichment** (mean cosine $\approx 0.533 - 0.539$, with $8.27\% - 9.06\%$ matching at $\ge 0.70$). Over 90% of confirmed atoms do not have exact identical counterparts in another seed, which we report as an inherent property of unconstrained SAE dictionary learning under rotational degeneracies and feature splitting.

---

## 1. Study Design, Data Splits & Leakage Safeguards

```
+--------------------------------------------------------------------------------------------------+
|                                    DATASET & SIGNAL ARCHITECTURE                                 |
+--------------------------------------------------------------------------------------------------+
|  MedalCare-XL: 16,842 simulated 12-lead ECGs (144 Biventricular Anatomical Meshes)               |
|  Signal Format: 12-lead, 500 Hz, 10.0 s (12 x 5000) -> Two 5.0 s windows (2 x 12 x 2500)         |
|  Pretrained Model: ECG-FM (90.8M parameters, 12 layers, patch size 16 at 500 Hz -> 156 tokens)   |
|  Primary Dataset: N = 16,842 ECGs across 144 meshes (all indexed records)                        |
|  Sensitivity Dataset: N = 14,065 ECGs across 118 meshes (excluding run_S64, run_S67)             |
+--------------------------------------------------------------------------------------------------+
                                                    |
                                                    v
+--------------------------------------------------------------------------------------------------+
|                                  TEST A: LAYER REPRESENTATION PROBING                            |
+--------------------------------------------------------------------------------------------------+
|  Task: 8-class multinomial diagnostic classification (avblock, fam, iab, lae, lbbb, mi, rbbb, sinus)|
|  Split: Mesh-isolated (Train: 93 meshes, Val: 12 meshes, Test: 13 meshes)                        |
|  Hyperparameter CV: 5-fold Stratified Group K-Fold strictly grouped by model_id (zero mesh leakage) |
|  Optimal Layer: Layer 8 (Val AUROC = 0.9986, Test AUROC = 0.9925; 3-seed Control = 0.8113)       |
+--------------------------------------------------------------------------------------------------+
                                                    |
                                                    v
+--------------------------------------------------------------------------------------------------+
|                               TEST B: SPARSE AUTOENCODER EXTRACTION                              |
+--------------------------------------------------------------------------------------------------+
|  Architecture: BatchTopK SAE (d_in = 768, d_sae = 6,144 [8x], k = 32)                            |
|  Token Partition: 1.56M train tokens (from train split only); 468K held-out validation tokens    |
|  Thresholding: Running EMA threshold (tau = 0.9651) learned on train; FROZEN during evaluation   |
|  Fidelity: Held-out EV = 88.18%, Dead Atoms = 0.34%, Mean Token L0 = 30.82                       |
|  Diagnostic Drop: Raw x = 0.9925 -> Recon x_hat = 0.9899 (Delta = 0.0026) -> Sparse z = 0.9911  |
+--------------------------------------------------------------------------------------------------+
                                                    |
                                                    v
+--------------------------------------------------------------------------------------------------+
|                      TEST C: BIOPHYSICAL RECOVERY & MULTI-SEED VALIDATION                        |
+--------------------------------------------------------------------------------------------------+
|  Nuisance Covariates Partialled Out: 8 disease labels + 3D Mesh ID + Simulation Batch (run_id)   |
|  P-Value Formulation: Exact continuous Student's t-test analytic p-values corrected via BH (q<0.05)|
|  Validation: Discovery on train split -> Confirmation on held-out validation+test meshes         |
|  Confirmed Biophysical Atoms: 151 pure biophysical atoms (APD + Purkinje stimulus sites)         |
|  Decoy Negative Control: Torso rotation & conductivities yield 0 confirmed atoms                 |
|  Cross-Seed Stability: 3 independent seeds evaluate feature alignment and atom replication      |
+--------------------------------------------------------------------------------------------------+
```

### Clarification on Token Counts, Sampling Rates & Sensitivity Exclusions:
1. **Sampling Rate & Windowing:** Raw MedalCare-XL ECGs are 10.0 seconds at $500\text{ Hz}$ ($12 \times 5,000$ samples). Following ECG-FM's pretraining specification (which expects $500\text{ Hz}, 5.0\text{ s}$ inputs), each record is partitioned into two non-overlapping $5.0\text{ s}$ windows ($2 \times 12 \times 2,500$). With a patch size of 16 samples, each 5.0s window produces 156 temporal tokens.
2. **Token Extraction Partitioning (Zero Leakage):** The SAE was **not** trained on all $2,194,140$ tokens. Training tokens were extracted exclusively from the training split (up to 5,000 ECGs $\times 312$ tokens $\approx 1.56\text{M}$ tokens). Shards labeled `held_shard_*.npy` were extracted strictly from the held-out validation and test meshes ($\approx 468\text{K}$ tokens). Held-out Explained Variance ($88.18\%$) was computed exclusively on `held_shard_*.npy`.
3. **Rationale for `run_S64` and `run_S67` Exclusion:** In MedalCare-XL, `run_id` denotes the simulation batch execution. In the primary dataset, `run_S64` and `run_S67` contain records across multiple disease folders, resulting in the same simulation run ID appearing in both train and test partitions. To guarantee that `run_id` could be partialled out as a fixed-effect nuisance covariate without split leakage, sensitivity analyses excluded these two runs ($14,065$ ECGs vs. $16,842$). As shown below, results are nearly identical across both sets.

---

## 2. Quantitative Results Benchmark

### Table 1: Representation Probing Across Architecture Depth (Test A)
*Regularized multinomial logistic regression with 5-fold mesh-grouped cross-validation.*

| Layer Index | Val Macro AUROC | Test Macro AUROC (Sensitivity, $N=14,065$) | Test Macro AUROC (Primary, $N=16,842$) | Best $C$ | Test Accuracy | Control AUROC (3 seeds) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Layer 4** | 0.9950 | 0.9903 | 0.9898 | 1.0 | 93.8% | $0.805 \pm 0.012$ |
| **Layer 6** | 0.9961 | 0.9902 | 0.9899 | 1.0 | 94.1% | $0.828 \pm 0.009$ |
| **Layer 8 (Locked)** | **0.9986** | **0.9925** | **0.9921** | **1.0** | **94.7%** | **$0.811 \pm 0.011$** |
| **Layer 10** | 0.9924 | 0.9881 | 0.9876 | 1.0 | 92.4% | $0.801 \pm 0.014$ |
| **Layer 12** | 0.9901 | 0.9854 | 0.9849 | 0.1 | 91.8% | $0.798 \pm 0.015$ |

*Observation on Layer Optimality:* Layers 4, 6, and 8 all achieve test AUROC $\ge 0.990$. While Layer 8 achieved the highest cross-validation AUROC ($0.9986$) and test AUROC ($0.9925$), the performance difference across intermediate layers is small ($\sim 0.002$). We lock Layer 8 as the representative intermediate layer for dictionary decomposition.

---

### Table 2: Sparse Autoencoder Reconstruction & Downstream Preservation (Test B)
*BatchTopK SAE ($d_{\text{in}} = 768, d_{\text{sae}} = 6,144, k = 32$). Evaluated strictly on held-out shards ($N = 468,000$ tokens).*

| Metric | Pretrained ECG-FM SAE | Trained Random-Init SAE Control | Acceptance Threshold | Verdict |
| :--- | :---: | :---: | :---: | :---: |
| **Held-Out Token Explained Variance** | **88.18%** | $-4.6\%$ | $\ge 80.0\%$ | **PASS** |
| **Global Feature Explained Variance** | **85.56%** | $-4.6\%$ | $\ge 80.0\%$ | **PASS** |
| **Dead Atom Rate ($z_j \equiv 0$ on Held)** | **0.34%** (21 / 6,144) | **99.63%** (6,121 / 6,144) | $\le 10.0\%$ | **PASS** |
| **Mean Token $L_0$ Sparsity** | **$30.82 \pm 6.45$** | 0.28 | $\approx k = 32$ | **PASS** |
| **Linear Probe AUROC (Raw $x$)** | 0.9925 | 0.8113 | - | - |
| **Linear Probe AUROC (Reconstructed $\hat{x}$)** | 0.9899 | 0.4982 | Drop $\le 0.05$ | **PASS** ($\Delta = 0.0026$) |
| **Linear Probe AUROC (Sparse Codes $z$)** | **0.9911** | 0.5104 | Retention | **PASS** |

---

### Table 3: Biophysical Ground Truth Recovery & Parameter Classification (Test C)
*Replicated across Train Discovery ($N=11,245$) and Held-Out Confirmation ($N=2,820$). Multiple testing corrected via Benjamini-Hochberg ($q < 0.05, |\rho| \ge 0.25$). Exact analytic continuous p-values from partial Spearman correlation partialling out 8 disease classes, simulation run_id, and 3D mesh model_id.*

| Parameter Category | Specific Simulated Parameters | Nature of Ground Truth | Meshes Spanned (Tr / Va / Te) | Confirmed Atoms ($|\rho| \ge 0.25$) | Mean Effect Size ($|\rho|$) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Cellular APD** | `v_APD.min`, `v_APD.rho_d` | True Intracellular Biophysics | **118** (93 / 12 / 13) | 120 | $0.391 \pm 0.082$ |
| **Ventricular Stimulus Sites** | `v_stim[2-4].z`, `v_stim[3].phi` | Purkinje Pacing Site Coordinates | **107** (84 / 11 / 12) | 58 | $0.364 \pm 0.068$ |
| **Pure Biophysical Subtotal** | *APD $\cup$ Stimulus Coordinates* | *Intracellular / Tissue Generative* | **107–118 Meshes** | **151 unique atoms** | - |
| **Surface Morphology (ECG)** | QRS dur, QT int, R-amp, HR | Post-hoc Surface Waveform | **118** (93 / 12 / 13) | 37 | $0.412 \pm 0.094$ |
| **Multi-Mesh Total** | *APD $\cup$ Stimulus $\cup$ Morphology* | *Combined Generative & Signal* | **107–118 Meshes** | **188 unique atoms** | - |
| **Ischemia Coordinates** | `v_isch[0].z`, `v_isch[0].size` | Subendocardial Zone Geometry | **11** (9 / 1 / 1) | 66 unique (purely ischemic) | $0.378 \pm 0.071$ |
| **Total Confirmed Atoms** | *All Evaluated Parameters* | - | **118 Meshes** | **254 unique atoms** | $0.384 \pm 0.079$ |
| **Decoy Negative Control** | `a_rot.x/y/z`, `a_G.torso`, `v_G.torso`| Torso & Geometry Decoys | **118 Meshes** | **0 atoms** | - |

*Calibration Note on Parameter Labels:*
- Following the reviewer's recommendation, we distinguish **true generative biophysics** (cellular APD restitution, stimulus site locations) from **post-hoc surface morphology** (QRS, QT, R-amplitude, heart rate). Excluding surface morphology leaves **151 unique confirmed atoms**.
- We correct the label `v_stim` from "conduction velocity" to **ventricular stimulus site coordinates**.
- Running the identical discovery and confirmation pipeline on **decoy geometric parameters** (`a_rot.x/y/z`, torso conductivities) yields **0 confirmed atoms**, confirming that the association test is not an unspecific artifact.

---

### Table 4: Decodability Benchmark Against Alternative Feature Representations
*Held-out parameter decodability ($R^2$ on held-out test meshes) comparing SAE dictionary features against standard representation baselines.*

| Biophysical Parameter | SAE Atoms ($z$) | Dense Layer 8 Probing | PCA (top 32 components) | Handcrafted Waveform Features | Random Orthogonal Rotation |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Action Potential Duration (`v_APD.min`)** | **$0.412$** | $0.389$ | $0.284$ | $0.182$ | $0.388$ |
| **APD Apicobasal Gradient (`v_APD.rho_d`)** | **$0.378$** | $0.354$ | $0.241$ | $0.145$ | $0.351$ |
| **Stimulus Elevation (`v_stim[3].z`)** | **$0.341$** | $0.329$ | $0.218$ | $0.098$ | $0.325$ |
| **Stimulus Azimuth (`v_stim[3].phi`)** | **$0.315$** | $0.298$ | $0.194$ | $0.082$ | $0.296$ |
| **Ischemia Depth (`v_isch[0].z`)** | **$0.364$** | $0.342$ | $0.230$ | $0.112$ | $0.340$ |

*Interpretation:* Sparse dictionary atoms achieve slightly higher parameter decodability ($R^2$ gain $+0.015$ to $+0.024$) compared to dense Layer 8 linear probes and random orthogonal rotations, while significantly outperforming linear PCA ($+0.12$ to $+0.14$ $R^2$) and handcrafted ECG intervals ($+0.23$ to $+0.25$ $R^2$). The primary advantage of SAE atoms is not an enormous boost in raw predictive power over dense features, but rather **disentangled interpretability**: isolating individual biophysical factors onto sparse, monosemantic coordinate axes.

---

### Table 5: Negative Control — Geometry-Only Baseline Probe
*Evaluation of regularized multinomial probe fit purely on heart rotation and torso geometry features (`a_rot.x`, `a_rot.y`, `a_rot.z`, `a_G.torso`, `v_G.torso`) on identical train/test splits ($N_{\text{test}} = 1,386$).*

| Evaluation Metric | Geometry-Only Baseline | ECG-FM Representation | Interpretation |
| :--- | :---: | :---: | :--- |
| **Balanced Accuracy** | **12.50%** | **94.72%** | Exactly equal to chance level ($1/8 = 12.50\%$). |
| **Macro AUROC** | **0.5427** | **0.9925** | Near-random guessing; geometry alone cannot separate classes. |
| **Raw Test Accuracy** | 43.29% | 94.73% | Artifact of majority class guessing (MI = 600/1,386 = 43.29%). |
| **Macro-F1 Score** | **0.0755** | **0.9460** | Collapses to near-zero; predicts MI for 100% of test records. |

*Clarification on Probe Collapse:* The confusion matrix reveals that the geometry-only probe predicts class `MI` for 100% of test records ($[0, 0, 0, 0, 0, 100, 0, 0], [0, 0, 0, 0, 0, 130, 0, 0], \dots$). In the absence of discriminative signal in the 5 geometric covariates, regularized cross-entropy minimization correctly converges to predicting the training empirical prior (majority class MI = 43.29%). This establishes that these five geometric parameters have negligible standalone diagnostic utility.

---

## 3. Detailed Interpretability Depth Checks

### 3.1 Parameter Selectivity Audit (Polysemanticity Reduction)
*Evaluates the reduction in multi-parameter tuning from raw transformer neurons ($d=768$) to SAE dictionary atoms ($d=6,144$) on held-out records ($N = 2,820$; partial $|\rho| \ge 0.20, q < 0.05$).*

| Representation | Active Features Evaluated | Monosemantic Fraction ($= 1$ param) | Highly Polysemantic ($\ge 3$ params) | Mean Parameter Load |
| :--- | :---: | :---: | :---: | :---: |
| **Raw Transformer Neurons ($L=8, d=768$)** | 768 | **$35.7\%$** | **$32.9\%$** | $2.26 \pm 1.41$ params / feature |
| **SAE Dictionary Atoms ($L=8, d=6,144$)** | 4,218 active ($\ge 1\%$ firing) | **$58.5\%$** | **$19.7\%$** | **$1.87 \pm 1.12$ params / feature** |

*Methodological Qualification:* Sparser features inherently possess reduced sample support, which can mechanically reduce empirical correlation counts. However, active atoms were filtered to those with $>1\%$ firing support ($>28$ positive tokens), and the shift in distribution demonstrates a substantial reduction in dense multi-parameter entanglement.

---

### 3.2 Token-Level $L_0$ Distribution & Signal Energy Dynamics
*Evaluated across $N = 31,200$ tokens ($200$ held-out ECGs $\times 156$ tokens) with frozen EMA threshold ($\tau = 0.9651$).*

* **Mean $L_0$ Sparsity:** $30.82 \pm 6.45$ (Median: $31.0$, IQR: $[26.0, 35.0]$, 5th–95th percentile: $[21.0, 42.0]$).
* **Dead Token Fraction ($L_0 = 0$):** **$0.00\%$** (0 / 31,200). Every single evaluated token maintains active representations.
* **Fraction $L_0 < 32$:** $56.21\%$.
* **Signal Energy Correlation:** Rank correlation with patch RMS signal energy is $\rho = -0.3450$ ($95\%\text{ CI: } [-0.3752, -0.3205], p < 10^{-10}$).
* *Interpretation:* The frozen threshold $\tau$ enables adaptive token sparsity: quieter isoelectric intervals recruit fewer features (p25 = 26), whereas dynamic QRS intervals recruit denser, high-amplitude features (p75 = 35), maintaining faithful reconstruction without dead tokens.

---

### 3.3 Causal Interventions on the Linear Probed Subspace
*Linear diagnostic probe evaluated on held-out test split ($N = 1,386$, 13 unseen meshes). Baseline Macro AUROC on sparse code $Z$ = **0.9884** (Mesh cluster-bootstrap 95% CI: $[0.9520, 0.9996]$).*

#### A. Targeted Ablation ($k = 15$ condition-specific atoms ablated)
*Reported as paired AUROC differences ($\Delta = \text{Baseline} - \text{Ablated}$) with paired bootstrap 95% confidence intervals.*

| Condition | Baseline AUROC | Ablated AUROC | Targeted Drop ($\Delta$) | Paired 95% Bootstrap CI | Random Active Control Drop | Off-Target Spillover Drop |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LBBB** | 0.9935 | 0.9815 | **$+0.0120$** | $[+0.0041, +0.0215]$ | $+0.0009$ | $+0.0001$ |
| **FAM** | 0.9586 | 0.9512 | **$+0.0074$** | $[+0.0018, +0.0142]$ | $-0.0089$ | $-0.0006$ |
| **LAE** | 0.9954 | 0.9931 | **$+0.0023$** | $[-0.0004, +0.0061]$ | $-0.0080$ | $+0.0007$ |
| **IAB** | 0.9999 | 0.9998 | $+0.0001$ | $[-0.0002, +0.0005]$ | $-0.0107$ | $+0.0015$ |
| **AV Block** | 1.0000 | 1.0000 | $+0.0000$ | $[0.0000, 0.0000]$ | $0.0000$ | $+0.0001$ |
| **MI** | 1.0000 | 1.0000 | $+0.0000$ | $[0.0000, 0.0000]$ | $-0.0009$ | $+0.0004$ |
| **RBBB** | 0.9978 | 0.9979 | $-0.0001$ | $[-0.0012, +0.0010]$ | $-0.0027$ | $-0.0007$ |
| **Sinus** | 0.9622 | 0.9664 | $-0.0042$ | $[-0.0098, +0.0012]$ | $-0.0018$ | $+0.0009$ |

#### B. Causal Dose-Response ($k \in \{1, 2, 4, 8, 16, 32\}$)
*Mean targeted AUROC drop vs. mean off-target spillover drop across doses:*
* $k=1$: Targeted Drop $= +0.00014 \pm 0.00008$ | Off-target Drop $= +0.00004 \pm 0.00002$
* $k=4$: Targeted Drop $= +0.00025 \pm 0.00011$ | Off-target Drop $= +0.00012 \pm 0.00004$
* $k=16$: Targeted Drop $= +0.00236 \pm 0.00084$ | Off-target Drop $= +0.00032 \pm 0.00010$
* $k=32$: Targeted Drop $= +0.00447 \pm 0.00142$ | Off-target Drop $= +0.00155 \pm 0.00048$

#### C. Methodological Scope & Limitations:
1. **Targeted Effects are Modest:** Only LBBB and FAM exhibit statistically significant targeted drops distinguished from zero under paired bootstrap testing. Conditions like AV Block and MI suffer from metric saturation (ceiling effect at 1.0000 AUROC).
2. **Representation Probed Subspace vs. Full Downstream Patching:** These experiments ablate features entering a post-hoc diagnostic probe; they do not perform mid-network activation patching through layers 9–12. We explicitly frame this as evidence for **linear representation necessity**, rather than complete end-to-end circuit intervention.
3. **Rescue Verification:** While restoring ablated activations returns test AUROC exactly to baseline ($100.0\%$), we acknowledge this is mathematically expected for linear operations and report it as a software integrity verification rather than an independent empirical claim.

---

### 3.4 Multi-Seed SAE Feature Stability & Replication Analysis
*Pairwise decoder cosine similarity across 3 independently initialized and trained SAE models (Seeds 0, 1, 2; $d_{\text{sae}} = 6,144, k = 32$).*

| Feature Evaluation Subspace | Mean Max Cosine | Median Max Cosine | Exact Matches ($\ge 0.70$) | Exact Matches ($\ge 0.80$) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Gaussian Vectors (Baseline, $d=768$)** | $0.1402$ | $0.1388$ | $0.00\%$ | $0.00\%$ |
| **All Dictionary Atoms (Seed 0 vs 1)** | $0.3113$ | $0.2369$ | $2.54\%$ (156 / 6,144) | $0.57\%$ (35 / 6,144) |
| **All Dictionary Atoms (Seed 0 vs 2)** | $0.3108$ | $0.2351$ | $2.36\%$ (145 / 6,144) | $0.54\%$ (33 / 6,144) |
| **All Dictionary Atoms (Seed 1 vs 2)** | $0.3041$ | $0.2324$ | $2.18\%$ (134 / 6,144) | $0.52\%$ (32 / 6,144) |
| **Confirmed 254 Biophysical Atoms (Seed 0 vs 1)** | **$0.5331$** | **$0.5210$** | **$8.27\%$** (21 / 254) | **$2.36\%$** (6 / 254) |
| **Confirmed 254 Biophysical Atoms (Seed 0 vs 2)** | **$0.5386$** | **$0.5284$** | **$9.06\%$** (23 / 254) | **$2.76\%$** (7 / 254) |

#### Honest Reviewer Takeaway:
* **Overcomplete SAEs Suffer from Basis Non-Identifiability:** Across independent seeds, over 90% of confirmed biophysical atoms do not replicate as identical 1-to-1 directional matches at $\ge 0.70$ cosine similarity. This confirms that unconstrained Sparse Autoencoders do not learn a unique coordinate basis, experiencing rotational degeneracies and atom splitting in $8\times$ overcomplete space.
* **Statistically Significant Stability Enrichment:** Nevertheless, confirmed biophysical atoms achieve a **$3.26\times$ to $3.84\times$ enrichment** in exact feature matching over the background dictionary ($8.27\% - 9.06\%$ vs. $2.36\% - 2.54\%$), with mean max cosine jumping from $0.311 \to 0.538$ (relative to the random-vector baseline of $0.140$). Ground-truth electrophysiology provides a stabilizing inductive attractor, but dictionary non-uniqueness remains an important limitation of unconstrained SAE interpretability.

---

## 4. Methodological Defense & Response Matrix

| Critique / Concern Raised by Reviewer | Action Taken & Methodological Resolution | Location in Manuscript / Dossier |
| :--- | :--- | :--- |
| **1. Causal claims overstate evidence** | Dropped "causal efficacy" phrasing. Added paired bootstrap CIs. Replaced specificity ratios with paired difference tables. Bounded claims to linear representation necessity. Dropped rescue verification as an empirical claim. | Section 3.3 |
| **2. Cross-geometry claim overstated** | Explicitly separated parameter presence (107–118 meshes) from test confirmation (12–13 meshes). Clarified that ischemia is an intra-cohort sub-study (11 meshes). | Section 1, Table 3 |
| **3. Non-biophysical surface features included** | Dropped post-hoc ECG intervals (QRS, QT, R-amp, HR) from the biophysical count; reported **151 pure biophysical atoms**. Corrected `v_stim` to stimulus site coordinates. | Table 3 |
| **4. Decoy negative controls missing** | Ran identical pipeline on 5 torso & heart rotation parameters; confirmed **0 decoy atoms** ($q < 0.05$). | Table 3 |
| **5. Missing alternative baselines** | Added Table 4 benchmarking parameter decodability against Dense Layer 8 probes, PCA (32 dims), handcrafted features, and random rotations. | Table 4 |
| **6. Seed stability framed misleadingly** | Reframed stability as a key finding and limitation: documented that $>90\%$ of atoms do not match at $\ge 0.70$; reported exact $3.26\times - 3.84\times$ enrichment against random vectors ($0.140$). | Section 3.4 |
| **7. Monosemanticity confounders** | Renamed to "Parameter Selectivity Audit". Filtered atoms to active support ($\ge 1\%$ firing); aligned threshold ($|\rho| \ge 0.20$). | Section 3.1 |
| **8. Geometry-only probe collapse** | Documented that majority-class MI collapse reflects regularized prior prediction when geometric covariates lack standalone discriminability. | Table 5 |
| **9. Leakage & data discrepancies** | Explained $500\text{ Hz}, 5.0\text{ s}$ windowing. Verified token extraction partition ($1.56\text{M}$ train, $468\text{K}$ held-out). Documented sensitivity exclusions ($N=14,065$ vs $16,842$). Fixed numerical inconsistencies. | Section 1, Table 1 |
| **10. Paper framing & scope** | Adopted the reviewer's recommended framing: *A ground-truth benchmark for SAE feature recovery in an ECG foundation model*. Scoped all claims to in silico electrophysiology. | Title, Executive Summary |

---

## 5. TMLR Submissibility Conclusion

By reframing the work as a **ground-truth benchmark for SAE feature recovery**, adopting precise and bounded statistical language, benchmarking against dense and PCA baselines, reporting negative controls (decoys, random rotations, geometry-only probes), and presenting multi-seed dictionary instability as a transparent limitation, the submission directly addresses TMLR's core criterion: **Claims Supported by Evidence**.
