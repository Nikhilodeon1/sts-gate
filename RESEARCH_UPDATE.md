# Research Update & Mock Review Response Dossier
**Project:** `sts-gate` / Cardiac Foundation Model Biophysical Ground-Truth Evaluation  
**Paper Title:** *Do Sparse Autoencoders Recover Biophysical Features in Pretrained Cardiac Foundation Models? An in silico Ground-Truth Case Study*  
**Target Venue:** AAAI-27 Workshop on Foundation Models for Multisensory Intelligence (*Making Sense*) / AI4TS (Deadline: Nov 20, 2026)  
**Compiled Manuscript PDF:** [`paper/main.pdf`](file:///c:/Users/nikhi/Codes/TEST/sts-gate/paper/main.pdf) (8 pages total: 7 pages main text + references, 1 page appendix)

---

## 1. Summary of Changes in Response to Mock Review

Following the rigorous mock review, we performed a comprehensive pass over the bibliography, equations, figures, and claims to eliminate vulnerabilities:

### A. Reference Integrity (100% Rebuilt & Verified)
- **MedalCare-XL:** Corrected to Gillette, Gsell, Nagel, Prassl, Doessel, Plank, and Loewe, *Scientific Data* vol. 10, article 531, 2023 (16,900-subject dataset).
- **ECG-FM:** Corrected to McKeen et al., *JAMIA Open* 2025. Removed inaccurate "vision-transformer-style" and "masked autoregressive" phrasing; accurately described as a 12-layer transformer trained with masked reconstruction and contrastive objectives.
- **Paulo & Belrose:** Corrected title and citation to *"Sparse Autoencoders Trained on the Same Data Learn Different Features"* (arXiv 2501.16615, 2025). Contextualized our 8% vector overlap with their observation of only ~30% feature overlap across seeds and heightened seed sensitivity in TopK/thresholded architectures.
- **CADENCE & ECG-InterpBench:** Corrected citations to Duan et al. (2026) and Duan (Rice, 2026). Directly noted CADENCE's finding that sparse probes match/beat dense representations on clinical targets, establishing our unique differentiator: microscopic PDE ground truth.
- **Added Heap et al. (2025):** *"SAEs Can Interpret Randomly Initialized Transformers"*, explaining why evaluating random controls is essential.

### B. Mathematical Formulations & Physics Nomenclature
- **Monodomain Equation (Eq. 1):** Added the membrane surface-to-volume ratio $\chi$:
  $$\nabla \cdot (\bm{\sigma} \nabla V_m) = \chi \left( C_m \frac{\partial V_m}{\partial t} + I_{\text{ion}}(V_m, \bm{w}) \right)$$
- **Thoracic Potential Field (Eq. 2):** Updated to the boundary element transfer formulation ($\bm{\Phi}_{\text{body}} = \bm{T} \bm{J}_i$) rather than an unbounded homogeneous free-space integral.
- **Nomenclature:** Replaced the misnomer "APD restitution" with exact terms: baseline action potential duration ($v_{\text{APD.min}}$) and apical-basal repolarization gradient ($v_{\text{APD.}\rho_d}$).

### C. Calibrated Scientific Claims & Rigor
- **Decodability Advantage:** Abstract and text now clearly distinguish Dense ($2.4\times$ variance over surface calipers: $0.354$ vs. $0.145$) and SAE ($2.6\times$ variance: $0.378$ vs. $0.145$).
- **Parity Result Reframing:** Replaced the flawed TOST $p < 0.01$ claim with a mathematically defensible confidence interval interpretation: the 95% cluster-bootstrap CI $[-0.008, +0.056]$ crosses zero and excludes decodability advantages above $+0.06$.
- **Term Usage:** Replaced "coordinate rotation" with "sparse overcomplete feature transformation" (since JumpReLU is non-linear and $8\times$ overcomplete).
- **Nuisance Control:** Frankly stated that torso conductivities and electrode coordinates have zero within-mesh variance by design in MedalCare-XL, explaining why $R^2 \approx 0.000$ and avoiding overclaiming shortcut immunity.
- **Ablation & Phase Localization:** Demoted the ablation study (acknowledging it falls within random perturbation noise floor) and compressed Section 6 to one crisp paragraph acknowledging biophysical coupling and time autocorrelation.

### D. Redrawn Publication Figures (`paper/figures/`)
- **Figure 1 (Pipeline):** Redrawn with clear forward flow, larger text/arrows, non-overlapping boxes, and a direct arrow from MedalCare-XL exact PDE ground truth feeding into the multi-mesh statistical audit box.
- **Figure 2 (Selectivity):** Compact single-column layout clearly showing that 80.3% of associated atoms correlate with 1 parameter, while 95.9% of the full dictionary is unassociated.
- **Figure 3 (Decodability):** Moved legend to upper left (preventing collision with Isch $z$), added error bars showing 95% cluster-bootstrap CIs, and assigned distinct color palettes for Dense vs. Top-32.
- **Figure 4 (Phase):** Added uniform temporal chance baseline (dotted line), clear window bounds, and noted that it illustrates fiducial gating.

---

## 2. Evidence & Table Reconciliation

All numbers in the paper match [`src/generate_all_tables.py`](file:///c:/Users/nikhi/Codes/TEST/sts-gate/src/generate_all_tables.py) with 100% precision:
- **Table 1:** Diagnostic probing (Layer 8 test AUROC $0.9925$, random control $0.8107$). Clarified that Table 1 uses regularized logistic regression on mean-pooled record latents, while Table 2 uses token-level probes.
- **Table 2:** SAE fidelity ($88.18\%$ held token EV, $0.34\%$ dead atoms, $0.9911$ sparse AUROC).
- **Table 3:** Confirmation of 174 headline biophysical atoms (120 APD, 58 stimulus site, 4 overlap) under multi-mesh FDR $q < 0.05, |\rho| \ge 0.25$.
- **Table 4:** Decodability benchmark across 7 representations on 13 unseen test meshes.
- **Table 5:** Stability analysis on 174 atoms ($\sim 8\%$ vector overlap vs. $68.4\%$ functional replication).
- **Table 6 (Appendix):** Parameter recall audit across all 29 non-zero-variance parameters (20 biophysics, 4 ischemia, 5 morphology; 12 confirmed = $41.4\%$).

---

## 3. Reviewer Verification Links
- Compiled PDF: [`paper/main.pdf`](file:///c:/Users/nikhi/Codes/TEST/sts-gate/paper/main.pdf)
- LaTeX Source: [`paper/main.tex`](file:///c:/Users/nikhi/Codes/TEST/sts-gate/paper/main.tex)
- BibTeX Database: [`paper/references.bib`](file:///c:/Users/nikhi/Codes/TEST/sts-gate/paper/references.bib)
- Table Generator: [`src/generate_all_tables.py`](file:///c:/Users/nikhi/Codes/TEST/sts-gate/src/generate_all_tables.py)
