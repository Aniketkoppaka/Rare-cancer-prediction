# Active Phenotype Acquisition for Zero-Shot Rare Disease Prioritization

Official benchmark repository for:  
**"Active Phenotype Acquisition Resolves the Observation Bottleneck in Zero-Shot Rare Disease Prioritization over a 12,956-Disease Candidate Universe"**

---

## 📌 Benchmark Overview (`RARE_DISEASE_ZS_BENCHMARK_v1.0`)

This benchmark formalizes the evaluation of interactive, sequential phenotype elicitation for rare disease diagnosis across a large-scale candidate universe.

- **Total Evaluable Cases:** 10,601 clinically annotated patient phenopackets across 803 rare diseases.
- **Candidate Retrieval Universe:** 12,956 rare diseases from the Human Phenotype Ontology Annotation (`phenotype.hpoa`).
- **Disease-Disjoint Splits (Seed 42):**
  - **Training:** 562 diseases (7,721 patient cases)
  - **Validation:** 120 diseases (1,000 patient cases)
  - **Held-Out Zero-Shot Test:** 121 diseases (1,880 patient cases)

---

## 🚀 Quickstart & Reproducibility

### 1. Requirements
The benchmark evaluation requires standard Python 3.9+ with `numpy`:
```bash
pip install numpy
```

### 2. Run Primary Benchmark & Acquisition Policy Control
To evaluate the static baseline ($A1$), Information-Theoretic Active Acquisition (Policy A), 5-seed Random Candidate-Pool Acquisition, Highest-IC control, and statistical significance tests:

```bash
python benchmark_primary_control.py
```

### 3. Run Enhanced & Optimized Policy (Fast Inverted-Index Acquisition)
To run the fast vectorized acquisition engine with candidate elimination margin weighting and dynamic softmax temperature scaling ($\tau$):

```bash
python benchmark_adaptive_margin.py
```

### 4. Continuous & Extended Query Trajectory Benchmarks ($Q=0 \dots 25$)
To evaluate step-by-step diagnostic resolution and theoretical scaling limits:

```bash
# Extended query budget (Q = 0 to 15) on unseen validation & test cohorts
python benchmark_extended_q.py

# Continuous integer trajectory (Q = 0 to 25)
python benchmark_continuous_trajectory.py

# 5-Fold cross-validation & clinical noise / EHR error stress testing
python audit_crossval_and_noise.py

# High-speed data integrity & null value audit
python audit_data_integrity.py
```

---

## 📊 Summary of Primary Results (Locked Zero-Shot Test Set, N=1,880)

### Baseline & Policy Control ($Q = 5$)
| Method / Acquisition Strategy | Query Budget ($Q$) | Hits@1 (%) | MRR | Net Error Reduction |
| :--- | :---: | :---: | :---: | :---: |
| **Static Power-Normalized IC Baseline ($A1$)** | $Q = 0$ | 68.62% | 0.7399 | — |
| **Random Candidate-Pool Acquisition (5 Seeds)** | $Q = 5$ | 80.43% ± 0.38% | — | +11.81 pp over static |
| **Greedy Specificity Baseline (Highest-IC)** | $Q = 5$ | 77.18% | — | +8.56 pp over static |
| **Information-Theoretic Policy A** | $Q = 5$ | 85.00% | 0.8684 | +16.38 pp over static (52.20% errors resolved) |
| **Enhanced Adaptive Policy (Margin Boost)** | $Q = 5$ | **84.52%** | **0.8659** | **+15.90 pp over static** |

### Extended Query Inquiry Trajectory ($Q = 0 \dots 15$)
| Query Budget ($Q$) | Hits@1 (%) | Hits@5 (%) | Hits@10 (%) | MRR | Net Error Resolved |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$Q = 0$ (Static)** | 68.62% | 80.90% | 83.51% | 0.7399 | — |
| **$Q = 1$** | 74.73% | 82.71% | 84.73% | 0.7819 | +19.49% |
| **$Q = 3$** | 80.85% | 86.76% | 88.14% | 0.8338 | +38.98% |
| **$Q = 5$** | 84.52% | 88.94% | 90.37% | 0.8659 | +50.68% |
| **$Q = 10$** | 89.95% | 91.12% | 91.97% | 0.9057 | +68.00% |
| **$Q = 15$** | **91.81%** | **92.77%** | **93.09%** | **0.9226** | **+73.90%** |

### Robustness, Noise Resilience & Generalization:
- **5-Fold Cross-Validation:** Stable diagnostic gain of **$+18.80\text{ pp} \pm 1.20\text{ pp}$** across 5 independent random unseen disease partitions.
- **Clinical Noise Tolerance:** Preserves **$84.41\%$ Hits@1** at $Q=10$ under $10\%$ clinician misreporting / EHR error rates.
- **Controlled Statistical Significance ($Q=5$):** Policy advantage over random acquisition: $\mathbf{+4.63\text{ pp}}$ ($p = 2.6 \times 10^{-16}$, McNemar $\chi^2 = 67.1$).
- **Zero Demotion Rate:** $0 / 1,290$ initially correct cases demoted ($0.00\%$).
- **Inference Speed:** **~18.09 ms/case** over 12,956 disease search space using inverted indices.

---

## 📂 Repository Structure

```
├── data/
│   └── benchmark/
│       ├── clean_phenopackets.json       # 10,601 patient phenopackets (803 diseases)
│       ├── failure_cases.json            # Error-case diagnostics & logs
│       └── phenotype.hpoa                # Knowledge base profiles for 12,956 diseases
├── paper/
│   ├── paper_draft_v1.0.md               # Original manuscript draft
│   └── paper_draft_v1.1.md               # Updated comprehensive academic manuscript
├── research/
│   ├── final_manuscript_number_manifest.md  # Authoritative numerical manifest
│   ├── phase14_claim_evidence_matrix.md     # Master claim-evidence traceability
│   ├── phase16_prior_art_matrix.md          # Comprehensive literature matrix
│   └── phase18_external_submission_audit.md # Final pre-submission audit report
├── audit_crossval_and_noise.py        # 5-fold cross-validation & clinical noise stress test
├── audit_data_integrity.py            # High-speed data cleanliness & null audit
├── benchmark_adaptive_margin.py       # Vectorized fast acquisition engine & margin policy
├── benchmark_continuous_trajectory.py # Full step-by-step resolution trajectory (Q=0..25)
├── benchmark_extended_q.py            # Extended query scaling benchmark (Q=0..15)
├── benchmark_primary_control.py       # Standalone benchmark & control runner
├── generate_graphs.py                 # Trajectory & performance matrix visualizer
├── SUMMARY.md                         # Comprehensive architecture & results documentation
└── trajectory_and_performance_matrix.png # High-res visual benchmark summary
```

---

## 📜 Citation

```bibtex
@article{active_phenotype_acquisition_2026,
  title={Active Phenotype Acquisition Resolves the Observation Bottleneck in Zero-Shot Rare Disease Prioritization over a 12,956-Disease Candidate Universe},
  author={Anonymous},
  journal={Under Review},
  year={2026}
}
```
