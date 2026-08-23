# Phase 17 — Final Release & Submission Checklist

**Target Manuscript:** `paper/paper_draft_v1.1.md`  
**Authoritative Number Manifest:** `research/final_manuscript_number_manifest.md`  
**Date:** 2026-08-23  

---

## 1. PRE-SUBMISSION VERIFICATION CHECKLIST

- [x] **Manuscript Frozen:** `paper/paper_draft_v1.1.md` is locked and structurally verified.
- [x] **Number Manifest Frozen:** `research/final_manuscript_number_manifest.md` is the authoritative single source of truth.
- [x] **Dataset Provenance Documented:** `clean_phenopackets.json` (10,601 cases, 803 diseases) and `phenotype.hpoa` (12,956 candidate universe).
- [x] **Partition Splits Reproducible:** 562 train, 120 validation, and 121 zero-shot test diseases (Seed 42).
- [x] **Random Seeds Documented:** Seed 42 for benchmark partitioning; Seeds `[10, 42, 100, 2026, 9999]` for random acquisition controls and universe scaling.
- [x] **Environment & Dependencies Documented:** Standard Python 3.9+ (`numpy`, `json`, `math`, `collections`).
- [x] **Main Experiments Reproducible:** Static $A1$ ($68.62\%$), Policy A $Q=1\dots 5$ ($74.84\% \rightarrow 85.00\%$) executable via `run_phase13_control.py`.
- [x] **Control Experiments Reproducible:** Random Candidate-Pool ($80.43\% \pm 0.38\%$) and Highest-IC ($77.18\%$) verified.
- [x] **Statistical Tests Reproducible:** Paired McNemar $\chi^2 = 67.1$ ($p = 2.6 \times 10^{-16}$) and 95% Bootstrap CI $[+4.09\text{ pp}, +6.65\text{ pp}]$ verified.
- [x] **Figures & Diagrams Validated:** ASCII workflow (Figure 1) and trajectory plot (Figure 2) fully consistent with numerical tables.
- [x] **Tables Validated:** All 8 tables audited for correct denominators, percentages, and LaTeX rendering.
- [x] **Citation Audit Complete:** All 18 citations verified against literature databases.
- [x] **Novelty Claims Qualified:** Scoped to active differential acquisition in open-world zero-shot rare disease retrieval.
- [x] **Clinical Claims Appropriately Scoped:** 88% bedside assessability labeled as an exploratory feasibility audit.
- [x] **Negative Evidence Appropriately Scoped:** Formally defined as a simulated counterfactual evaluation abstraction ($\lambda = 0.50$).
- [x] **Zero-Shot Definition Precise:** Formally defined as disease-disjoint candidate retrieval over 121 unseen diseases.
- [x] **No Stale Numbers Remain:** Verified 100% agreement across all 28 headline numbers.
- [x] **No Malformed LaTeX Remains:** All math KaTeX blocks and tables verified.
- [x] **No Local Absolute Paths in Execution Code:** All scripts use relative benchmark paths (`data/benchmark/`).
- [x] **Repository Release Contents Categorized:** KEEP, ARCHIVE, and REMOVE classifications completed.
- [x] **Final Manuscript Ready for External Submission:** Verdict is **GREEN**.
