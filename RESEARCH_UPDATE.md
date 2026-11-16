# Research Update & Mock Review Response Dossier
**Project:** `sts-gate` / Cardiac Foundation Model Biophysical Ground-Truth Evaluation  
**Paper Title:** *Do Sparse Autoencoders Recover Biophysical Features in Pretrained Cardiac Foundation Models? An in silico Ground-Truth Case Study*  
**Target Venue:** The 4th Workshop on Artificial Intelligence for Time Series Analysis (AI4TS) at AAAI-27 (Deadline: Nov 20, 2026)  
**Compiled Manuscript PDF:** [`paper/main.pdf`](file:///c:/Users/nikhi/Codes/TEST/sts-gate/paper/main.pdf) (8 pages total: 7 pages main text + references, 1 page appendix)

---

## 1. Summary of Comprehensive Revisions in Response to Reviewers 1 & 2

Following the detailed feedback from both reviewers, we conducted a systematic pass over the manuscript, figures, methodology, and empirical claims:

### A. Clarified Framing & Scope (Addressing Reviewer 1 & 2 Core Concerns)
1. **Target Venue Alignment:** Subtitle and framing set specifically for **The 4th Workshop on Artificial Intelligence for Time Series Analysis (AI4TS) at AAAI-27**. AI4TS explicitly emphasizes foundation models, health applications, and interpretability for time-series data.
2. **Abstract Streamlining:** Rewritten to a concise, focused ~170 words. Framed as a rigorous, single-model, single-layer exploratory case study on ECG-FM Layer 8. Typo ("cannot be recorded non-invasively in living patients") eliminated.
3. **De-Hyped Plain Technical Language:** Replaced buzzwords ("decodability engines", "verification bottleneck", "information-expanding operator", "definitively confirms", "calibrated verdict") with direct empirical descriptions.
4. **Ischemia Sub-Cohort Clarification:** Corrected the erroneous "single-mesh" text in Section 4.1. Ischemia is simulated across a sub-cohort of **13 distinct meshes** (6,100 myocardial infarction ECGs spanning LAD, RCA, and LCX vascular territories).
5. **Initial Candidate Filter Rate Reported:** Clarified that of 712 initial candidate feature-parameter pairs on discovery meshes, **325 pairs (45.6%)** replicate under within-mesh permutation FDR control on the 25 held-out confirmation meshes.
6. **Sign Invariance Reporting:** Replaced the unclustered binomial test formulation with the exact empirical count: across all 325 confirmed pairs, **325 of 325 (100.0%)** preserve identical correlation sign between discovery and confirmation meshes.
7. **Paired Pretrained vs. Random-Init Metric:** Replaced unclustered $p < 10^{-12}$ with the paired difference on test meshes: $\Delta R^2 = +0.312$, 95% cluster-bootstrap CI $[+0.268, +0.356]$.
8. **Functional Replication Formally Defined:** Explicitly defined the replication criterion: an atom in Seed 0 is replicated if an atom in Seed 1 or 2 correlates with the same parameter with matching sign at $|\rho| \ge 0.25, q < 0.05$.

### B. Visual & Layout Polishing (Exact 8-Page Snap)
1. **Figure 1 (Pipeline):** Curved the ground-truth arrow (`arc3,rad=0.2`), repositioned the PDE label to `(0.12, 0.24)`, and enlarged/padded the bottom audit box to prevent any text clipping.
2. **Figure 2 (Selectivity):** Updated the dense layer parameter load percentages to exact values matching counts: 29/81 (**35.8%**), 25/81 (**30.9%**), 27/81 (**33.3%**).
3. **Figure 4 (Phase Gating):** Height adjusted to 2.15 inches (`figsize=(6.8, 2.15)`), eliminating the 4-line overflow on Page 7.
4. **Page Budget Adherence:** The paper snaps cleanly to **exactly 8 pages** (Pages 1–7: Main Text + References [1]–[12]; Page 8: Appendix A + Table 6), strictly honoring the AI4TS 4–7 page limit (+ appendix).

---

## 2. Evidence & Table Reconciliation

All numbers in the paper match [`src/generate_all_tables.py`](file:///c:/Users/nikhi/Codes/TEST/sts-gate/src/generate_all_tables.py) with 100% precision:
- **Table 1:** Diagnostic probing (Layer 8 test AUROC $0.9925$, random control $0.8107$).
- **Table 2:** SAE fidelity ($88.18\%$ held token EV, $0.34\%$ dead atoms, $0.9911$ sparse AUROC).
- **Table 3:** Confirmation of 174 headline biophysical atoms (120 APD, 58 stimulus site, 4 overlap) under multi-mesh FDR $q < 0.05, |\rho| \ge 0.25$. Across all domains, 254 atoms and 325 pairs confirmed; 80.3% (204/254) are selectively correlated with exactly one parameter.
- **Table 4:** Decodability benchmark across 8 representations on 13 unseen test meshes (including `RandInit` at $0.042$, demonstrating pretraining emergence).
- **Table 5:** Stability analysis on 174 atoms ($\sim 8\%$ vector overlap vs. $68.4\%$ functional replication).
- **Table 6 (Appendix):** Parameter recall audit across all 29 non-zero-variance parameters (12 confirmed = $41.4\%$).

---

## 3. Reviewer Verification Links
- Compiled PDF: [`paper/main.pdf`](file:///c:/Users/nikhi/Codes/TEST/sts-gate/paper/main.pdf)
- LaTeX Source: [`paper/main.tex`](file:///c:/Users/nikhi/Codes/TEST/sts-gate/paper/main.tex)
- BibTeX Database: [`paper/references.bib`](file:///c:/Users/nikhi/Codes/TEST/sts-gate/paper/references.bib)
- Table Generator: [`src/generate_all_tables.py`](file:///c:/Users/nikhi/Codes/TEST/sts-gate/src/generate_all_tables.py)
