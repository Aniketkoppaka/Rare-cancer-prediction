# Phase 16 — Strategic Manuscript Novelty & Related Work Positioning

**Target Manuscript:** `paper/paper_draft_v1.1.md` (and future revisions)  
**Date:** 2026-08-23  

---

## 1. STRATEGIC POSITIONING BLUEPRINT

To ensure maximal reviewer acceptance at top-tier venues (NeurIPS, ICLR, JAMIA, Nature Digital Medicine), the manuscript must adopt the following calibrated positioning:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   CALIBRATED POSITIONING MATRIX                                  │
│                                                                                                  │
│  DO NOT CLAIM:                                    CLAIM INSTEAD:                                 │
│  ──────────────────────────────────────────────   ─────────────────────────────────────────────  │
│  "We invented active feature acquisition."         "We adapt information-theoretic feature       │
│                                                    acquisition to open-world zero-shot rare      │
│                                                    disease candidate retrieval."                 │
│                                                                                                  │
│  "Entropy x IC is a universal law."               "Policy A is a computationally tractable       │
│                                                    differential-partitioning objective that      │
│                                                    empirically outperforms greedy specificity."  │
│                                                                                                  │
│  "Active questioning helps because more data      "Active questioning improves accuracy because │
│   always helps."                                  of BOTH raw feature accumulation (+11.81 pp)  │
│                                                   AND intelligent entropy balancing (+4.63 pp,  │
│                                                   p = 2.6e-16)."                                 │
│                                                                                                  │
│  "We evaluate 12,956 test diseases."              "We evaluate zero-shot prioritization over     │
│                                                   121 unseen diseases against a 12,956-disease   │
│                                                   candidate retrieval universe."                 │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. CANONICAL CONTRIBUTION STATEMENT (FOR INTRODUCTION & ABSTRACT)

> **Our Primary Contributions are Threefold:**
> 1. **Conceptual Formulation & Benchmark:** We formalize `RARE_DISEASE_ZS_BENCHMARK_v1.0`, establishing the first disease-disjoint zero-shot benchmark for sequential phenotype acquisition across 10,601 clinically annotated patient phenopackets evaluated over a massive 12,956-disease candidate universe.
> 2. **Controlled Empirical Disambiguation:** Through a 5-seed random-acquisition control and greedy specificity baseline, we demonstrate that information-theoretic query selection (Policy A) provides a statistically decisive **$+4.63\text{ pp}$ advantage over random feature acquisition ($p = 2.6 \times 10^{-16}$, McNemar $\chi^2 = 67.1$)**, establishing that intelligent feature selection provides substantial value beyond arbitrary feature accumulation.
> 3. **Candidate Universe Scaling Relationship:** We discover that the marginal diagnostic benefit of active questioning triples (from $+5.47\text{ pp}$ at $N=100$ to $+16.38\text{ pp}$ at $N=12,956$) as the candidate search universe scales to open-world dimensions, proving that active acquisition directly overcomes the observation bottleneck caused by differential distractor crowding.

---

## 3. UPDATED RELATED WORK SUMMARY SECTION

The Related Work in Section 2 of `paper/paper_draft_v1.1.md` already captures the key taxonomy (Static Phenotype Prioritization vs. Active Feature Acquisition). All newly identified prior works (BOQA, SHEPHERD, Phenomizer, LIRICAL, Inquiry, Context-Aware Symptom Checking) are properly contextualized and cited.

No structural modifications to `paper/paper_draft_v1.1.md` are required; Draft v1.1 is already fully aligned with this strategic novelty positioning.
