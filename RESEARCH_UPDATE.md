# Research Update & Blind Peer Review Resolution Dossier
**Project:** `sts-gate` / Cardiac Foundation Model Biophysical Ground-Truth Evaluation  
**Paper Title:** *Do Sparse Autoencoders Recover Biophysical Features in Pretrained Cardiac Foundation Models? An in silico Ground-Truth Case Study*  
**Target Venue:** The 4th Workshop on Artificial Intelligence for Time Series Analysis (AI4TS) at AAAI-27 (Deadline: Nov 20, 2026)  
**Compiled Manuscript PDF:** [`paper/main.pdf`](file:///c:/Users/nikhi/Codes/TEST/sts-gate/paper/main.pdf) (8 pages total: Pages 1–7 = Main Text + 14 Verified References; Page 8 = Appendix A + Table 6)

---

## 1. Resolution of All Blind Peer Review Findings

Following an exhaustive independent blind review of the rendered PDF pages, figures, LaTeX logs, bibliography, and codebase (`src/`, `runs/`, `cache/`), every visual defect, code-to-paper discrepancy, numerical contradiction, and citation issue has been resolved:

### A. Visual & Layout Fixes (`0 Overfull \hbox`, `0 Underfull \vbox`, `0 Collisions`)
1. **Page 1 Subtitle Line-Wrap:** Wrapped the workshop subtitle across two centered lines in `\author{...}`, eliminating the `66.8 pt` right-margin clipping.
2. **Page 6 Table 5 Gutter Collision Eliminated:** Resized all single-column tables (Tables 1, 2, 5, 6) to `\columnwidth` and two-column tables (Tables 3, 4) to `0.96\textwidth`, completely eliminating the collision between Table 5 and Section 6 as well as margin protrusions on Pages 3, 6, and 8.
3. **Figure 3 Legend vs. Callout Separation:** Placed the legend in the upper right and the `2.6× Variance over Lead-II Calipers` callout on the left, eliminating all overlap. Added explicit per-parameter test mesh counts (`13 meshes`, `12 meshes`, `1 mesh*`) on the x-axis.
4. **Figure 4 Native Single-Column Sizing & Grouped Bar Chart:** Redesigned [`paper/figures/fig4_phase.pdf`](file:///c:/Users/nikhi/Codes/TEST/sts-gate/paper/figures/fig4_phase.pdf) at native single-column width (`figsize=(3.35, 2.35)`) so fonts render at 100% scale (7–8 pt instead of shrunk to 3.5 pt), and replaced synthetic Gaussian curves with a grouped bar chart of empirical token firing mass share across the three fiducial windows (`QRS [-50, +60 ms]`, `ST-T [+60, +380 ms]`, `Baseline`).
5. **Typography & Spacing:** Added `\raggedbottom` to eliminate `badness 10000` vertical column gaps, fixed `(~91M` to `($\sim$91M`, removed `\enlargethispage`, and shortened section headers (`5 Biophysical Decodability & Parity`, `A Supplementary Audit`) to eliminate stretched headings.

### B. Codebase & Ground-Truth Alignment
1. **Exact SAE Formulation (`BatchTopKSAE` + Inference JumpReLU Gate):** Updated Section 2.3 (Eqs. 3–4) and Figure 1 to match [`src/sae.py`](file:///c:/Users/nikhi/Codes/TEST/sts-gate/src/sae.py): an $8\times$ `BatchTopKSAE` ($M = 6{,}144, k = 32$) trained for 3 epochs with Adam ($lr = 3 \times 10^{-4}$) under pure reconstruction MSE loss while tracking an exponential moving average (EMA) scalar threshold $\theta$ that acts as a global JumpReLU gate at inference time (mean active $L_0 = 30.13$).
2. **Exact Waveform Windowing & Pooling Operators:** Updated Section 2.1 and 2.3 to match [`src/config.py`](file:///c:/Users/nikhi/Codes/TEST/sts-gate/src/config.py) and [`src/assoc.py`](file:///c:/Users/nikhi/Codes/TEST/sts-gate/src/assoc.py): 10.0-second 12-lead records at 500 Hz split into two 5.0-second windows (2,500 samples per lead, 156 tokens per window, $T = 312$ tokens per record, $\approx 32.05\text{ ms}$ per token), with mean pooling for dense linear probes and peak log-activation $\tilde{f}_j = \log(1 + \max_t f_{j,t})$ for record-level SAE atom features.
3. **True Split-Disjointness Exclusion Reason (`run_S64`, `run_S67`):** Documented that `run_S64` and `run_S67` (2,777 records across 26 meshes from the 16,842-record full index) were excluded because their batch identifiers crossed predefined train/val/test split boundaries.
4. **Exact Per-Domain Mesh Counts (`Table 3`, `Table 4`, `Section 2.2`):** Reported exact held-out mesh counts from [`runs/mesh_generalization_audit.json`](file:///c:/Users/nikhi/Codes/TEST/sts-gate/runs/mesh_generalization_audit.json): APD parameters span 25 confirmation meshes (12 val, 13 test), Purkinje stimulus coordinates span 23 confirmation meshes (11 val, 12 test), and the MI ischemia sub-cohort (6,600 records across 11 meshes) spans 2 confirmation meshes (1 val mesh with 650 ECGs, 1 test mesh with 600 ECGs).
5. **Exact Test C Covariates, P-Values, and Unrecovered Parameter List:** Documented that Test C conditions on one-hot pathology subfolder (`folder`) and anatomical mesh ID (`model_id`) fixed effects, computes Student-$t$ partial Spearman $p$-values with Benjamini–Hochberg FDR $q < 0.05$ and binding effect-size filter $|\rho| \ge 0.25$, and accurately listed the 14 unrecovered multi-mesh parameters (`v_APD.max, z_d, v_d`, secondary `v_stim` coordinates, and atrial rotations `a_rot.x, y, z`).
6. **Table 2 Dual EV Reporting & Table 5 Subspace Coverage:** Table 2 now reports both `Tok EV` (`88.18%` vs. `100.0%`) and `Glob EV` (`85.56%` vs. `-4.60%`), explaining the constant-bias collapse on random tokens (`99.63%` dead atoms). Section 5.2 and Table 5 clearly distinguish atom-level cosine alignment ($8.05\%\text{--}8.62\%$) from parameter-subspace coverage ($68.39\%$, reflecting consistent recovery of $v_{\text{APD.}\rho_d}$).

### C. Bibliography Verification ([`paper/references.bib`](file:///c:/Users/nikhi/Codes/TEST/sts-gate/paper/references.bib))
- Fixed author first names in `heap2025saes` (**Thomas Heap, Tim Lawson, Lucy Farnik, and Laurence Aitchison**).
- Added co-author **Wei Qiu** to `duan2026ecginterpbench`.
- Added volume, issue, and article number (`8(5):ooaf122`) to `mckeen2025ecgfm`.
- Added and cited **Bussmann, Leask, and Nanda (2024)** (`bussmann2024batchtopk`, *arXiv:2412.06410*) and **Benjamini and Hochberg (1995)** (`benjamini1995controlling`).
