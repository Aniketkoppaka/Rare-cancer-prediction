# Final Manuscript Number & Evidence Manifest

**Benchmark Version:** `RARE_DISEASE_ZS_BENCHMARK_v1.0`  
**Status:** AUTHORITATIVE SINGLE-SOURCE-OF-TRUTH FOR MANUSCRIPT DRAFTING (v1.1)  
**Last Verified:** 2026-08-23  

---

## 1. BENCHMARK COHORT & SEARCH SPACE

| Dimension | Canonical Value | Source Artifact | Description |
| :--- | :---: | :--- | :--- |
| **Total Evaluable Patient Cases** | **10,601** | `clean_phenopackets.json` | Clinically annotated patient phenopackets. |
| **Total Rare Disease Search Universe** | **12,956** | `phenotype.hpoa` | Complete knowledge base candidate diseases. |
| **Total Benchmark Rare Diseases** | **803** | `clean_phenopackets.json` | Diseases with at least 1 evaluable patient case. |
| **Training Split Diseases** | **562** | Partition (Seed 42) | 7,721 patient cases (70.0% of cohort). |
| **Validation Split Diseases** | **120** | Partition (Seed 42) | 1,000 patient cases (15.0% of cohort). |
| **Zero-Shot Test Split Diseases** | **121** | Partition (Seed 42) | 1,880 patient cases (15.0% of cohort). |
| **Mean Phenotypes per Patient** | **5.14 ± 2.8** | `clean_phenopackets.json` | Clinical presentation length. |

---

## 2. STATIC RETRIEVAL BASELINE LADDER (LOCKED TEST SET, N=1,880)

*Primary Metric Convention: Strict Inequality ($S(d) > S(T)$)*

| Model / Baseline Ladder | Hits@1 (%) | Hits@5 (%) | Hits@10 (%) | MRR | Source Artifact |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Unanchored Dense Two-Tower (InfoNCE)** | **0.00%** | **0.16%** | **0.43%** | **0.0084** | `phase5_dense_failure_audit.md` |
| **Cosine-Normalized IC Baseline** | **39.68%** | **60.69%** | **67.82%** | **0.4942** | `run_phase10_9_cosine.py` |
| **Raw Information Content ($A0$, Köhler et al.)**| **66.81%** | **80.27%** | **84.47%** | **0.7314** | `phase6_7_reproducibility.md` |
| **Deterministic Semantic Residual ($A2$)** | **68.51%** | **80.85%** | **83.51%** | **0.7380** | `phase6_8_symbolic_residual.md` |
| **Learned Symbolic Residual ($A3$)** | **68.56%** | **80.90%** | **83.51%** | **0.7387** | `phase6_8_symbolic_residual.md` |
| **Power-Normalized IC ($A1$, $\gamma=0.10$ — Primary)** | **68.62%** | **80.90%** | **83.51%** | **0.7399** | `kaggle_phase10_7_runner.py` |

---

## 3. ACTIVE ACQUISITION & CONTROL BENCHMARK (LOCKED TEST SET, N=1,880)

| Query Budget ($Q$) | Policy A (Info-Theoretic) | Random Pool (mean ± std) | Highest-IC | Policy A Advantage | Net Errors Resolved |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$Q = 0$ (Static Baseline)** | **68.62%** (MRR: 0.7399) | 68.62% ± 0.00% | 68.62% | **+0.00 pp** | 0 / 590 (0.0%) |
| **$Q = 1$** | **74.84%** (MRR: 0.7843) | 71.31% ± 0.27% | 70.11% | **+3.53 pp** | 117 / 590 (19.8%) |
| **$Q = 2$** | **77.77%** (MRR: 0.8088) | 73.91% ± 0.15% | 72.02% | **+3.85 pp** | 172 / 590 (29.2%) |
| **$Q = 3$** | **79.79%** (MRR: 0.8295) | 76.11% ± 0.21% | 74.47% | **+3.68 pp** | 210 / 590 (35.6%) |
| **$Q = 5$ (Canonical Target)** | **85.00%** (MRR: 0.8684) | **80.43% ± 0.38%** | **77.18%** | **+4.63 pp** | **308 / 590 (52.2%)** |
| **$Q = 10$** | **89.95%** (MRR: 0.9049) | — | — | — | 401 / 590 (68.0%) |
| **Oracle Upper Bound ($Q=1$)** | **90.59%** (MRR: 0.9253) | — | — | — | 413 / 590 (70.0%) |

### Primary Statistical Significance at $Q=5$ (Policy A vs. Random Acquisition):
- **Paired Advantage:** $\mathbf{+4.63\text{ pp}}$ (95% Bootstrap CI: $[+4.09\text{ pp}, +6.65\text{ pp}]$)
- **Discordant Pairs:** Policy A wins on **123 cases** | Random wins on **23 cases**
- **McNemar Significance:** $\chi^2 = \mathbf{67.1} \quad (p = 2.6 \times 10^{-16})$
- **Demoted Cases (Zero-Worsening):** **0 / 1,290 cases worsened (0.00%)** for Policy A; **0 / 1,290 (0.00%)** across all 5 Random seeds.

---

## 4. CANDIDATE UNIVERSE SCALING RELATIONSHIP (5 SEEDS PER SCALE)

| Candidate Universe ($N$) | Static Hits@1 (mean ± std) | 1-Step Hits@1 (mean ± std) | 5-Step Hits@1 (mean ± std) | 5-Step Active Gain |
| :---: | :---: | :---: | :---: | :---: |
| **$N = 100$** | 90.73% ± 0.33% | 93.73% ± 0.25% | 96.20% ± 0.16% | **+5.47 pp ± 0.16 pp** |
| **$N = 500$** | 84.60% ± 0.21% | 88.46% ± 0.31% | 93.13% ± 0.27% | **+8.53 pp ± 0.27 pp** |
| **$N = 1,000$** | 81.95% ± 0.19% | 86.48% ± 0.22% | 92.21% ± 0.24% | **+10.26 pp ± 0.24 pp** |
| **$N = 5,000$** | 73.88% ± 0.28% | 79.36% ± 0.26% | 88.02% ± 0.33% | **+14.14 pp ± 0.33 pp** |
| **$N = 12,956$ (Full KB)** | **68.62% ± 0.00%** | **74.84% ± 0.00%** | **85.00% ± 0.00%** | **+16.38 pp ± 0.00 pp** |

---

## 5. ENTROPY MECHANISM & QUANTILE SUBGROUPS

| Entropy Subgroup | Entropy Range | Cohort Size | Static Hits@1 | 1-Step ($Q=1$) | 5-Step ($Q=5$) | 5-Step Marginal Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Quartile 1 (Lowest Ambiguity)** | $H < 0.001\text{ bits}$ | $N = 470$ | **99.57%** | 99.57% (+0.00 pp) | **99.57%** | **+0.00 pp** |
| **Quartile 2 (Low-Mid Ambiguity)**| $0.001 \le H < 0.409\text{ bits}$ | $N = 470$ | **94.68%** | 95.11% (+0.43 pp) | **95.53%** | **+0.85 pp** |
| **Quartile 3 (Mid-High Ambiguity)**| $0.409 \le H < 2.196\text{ bits}$ | $N = 470$ | **61.06%** | 77.02% (+15.96 pp) | **82.13%** | **+21.06 pp** |
| **Quartile 4 (Highest Ambiguity)**| $H \ge 2.196\text{ bits}$ | $N = 470$ | **19.15%** | 27.66% (+8.51 pp) | **62.77%** | **+43.62 pp** |
