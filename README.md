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
python run_phase13_control.py
```

**Expected Runtime:** $< 2\text{ minutes}$ on standard CPU.

---

## 📊 Summary of Primary Results (Locked Test Set, N=1,880)

| Method / Acquisition Strategy | Query Budget ($Q$) | Hits@1 (%) | MRR | Net Error Reduction |
| :--- | :---: | :---: | :---: | :---: |
| **Static Power-Normalized IC Baseline ($A1$)** | $Q = 0$ | 68.62% | 0.7399 | — |
| **Random Candidate-Pool Acquisition (5 Seeds)** | $Q = 5$ | 80.43% ± 0.38% | — | +11.81 pp over static |
| **Greedy Specificity Baseline (Highest-IC)** | $Q = 5$ | 77.18% | — | +8.56 pp over static |
| **Information-Theoretic Active Acquisition (Policy A)** | $Q = 5$ | **85.00%** | **0.8684** | **+16.38 pp over static (52.20% errors resolved)** |

### Controlled Statistical Significance ($Q=5$):
- **Policy A Advantage over Random Acquisition:** $\mathbf{+4.63\text{ pp}}$ (95% Bootstrap CI: $[+4.09\text{ pp}, +6.65\text{ pp}]$)
- **Paired McNemar Test:** $\chi^2 = 67.1 \quad (p = 2.6 \times 10^{-16})$
- **Zero-Worsening Stability:** $0 / 1,290$ initially correct cases demoted ($0.00\%$).

---

## 📂 Repository Structure

```
├── data/
│   └── benchmark/
│       ├── clean_phenopackets.json  # 10,601 patient phenopackets (803 diseases)
│       └── phenotype.hpoa           # Knowledge base profiles for 12,956 diseases
├── paper/
│   └── paper_draft_v1.1.md          # Complete academic manuscript draft
├── research/
│   ├── final_manuscript_number_manifest.md  # Authoritative numerical manifest
│   ├── phase14_claim_evidence_matrix.md     # Master claim-evidence traceability
│   ├── phase16_prior_art_matrix.md          # Comprehensive literature matrix
│   └── phase18_external_submission_audit.md # Final pre-submission audit report
├── run_phase13_control.py           # Standalone benchmark & control runner
└── trace_q3.py                      # Verification script for Q=3 case count
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
