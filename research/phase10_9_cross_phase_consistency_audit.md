# Phase 10.9 — Final Cross-Phase Consistency Audit Before Manuscript

**Date:** 2026-08-23  
**Benchmark Version:** `RARE_DISEASE_ZS_BENCHMARK_v1.0`  
**Evaluation Cohort:** 10,601 evaluable cases | 803 benchmark diseases | 121 zero-shot test diseases (1,880 cases)  
**Search Space:** 12,956 rare candidate diseases  
**Auditor Mode:** Adversarial Cross-Phase Forensic Verification  

---

## 1. EXECUTIVE AUDIT VERDICT

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             PHASE 10.9 CROSS-PHASE CONSISTENCY VERDICT                           │
│                                                                                                  │
│  FINAL GATE STATUS: [ GREEN — FULLY RECONCILED ]                                                 │
│                                                                                                  │
│  1. TABLE 2 RAW IC (A0) RECONCILIATION:                                                          │
│     - In Phase 10.8 Table 2, Hits@5 (80.85%) and Hits@10 (83.51%) were typographical duplicates │
│       from row A2.                                                                               │
│     - True canonical Raw IC (A0) values are: Hits@1 = 66.81%, Hits@5 = 80.27%,                   │
│       Hits@10 = 84.47%, MRR = 0.7314. Corrected and locked in the manuscript manifest.          │
│                                                                                                  │
│  2. DENSE BASELINE FORMAL SPECIFICATION:                                                         │
│     - Formally defined as "Unanchored Dense Two-Tower Transformer (256-d, InfoNCE loss on        │
│       562 training diseases)", achieving Hits@1 = 0.00%, Hits@5 = 0.16%, MRR = 0.0084.          │
│                                                                                                  │
│  3. ERROR COUNT RECONCILIATION (308 vs 309):                                                     │
│     - Canonical Point Estimate at Q=5: 1,598 correct out of 1,880.                              │
│     - 1,598 - 1,290 = EXACTLY 308 net errors resolved (52.20% of 590 errors).                    │
│     - The 309 value in earlier draft logs was traced to an exploratory K=25 run (1,599 correct). │
│                                                                                                  │
│  4. COMPLETE CROSS-PHASE NUMBER TRACEABILITY:                                                    │
│     - All numbers across all 10 project phases are traced, explained, and indexed in             │
│       `research/final_manuscript_number_manifest.md`.                                            │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. DETAILED AUDIT & RECONCILIATION FINDINGS

### 2.1 Table 2 Baseline Ladder Audit (Raw IC $A0$)
- **Historical Record (Phase 6.7 & 6.8):**  
  Raw IC ($A0$): Hits@1 = **66.81%** (1,256 / 1,880) | Hits@5 = **80.27%** | Hits@10 = **84.47%** | MRR = **0.7314**.
- **Phase 10.8 Anomaly:** Phase 10.8 Table 2 accidentally reported Hits@5 as $80.85\%$ and Hits@10 as $83.51\%$ because those columns were copied from the row below ($A2$).
- **Reconciliation:** We re-ran the exact evaluation script (`run_phase10_9_cosine.py` and `run_phase10_9_audit.py`). The true canonical numbers for $A0$ are confirmed as **66.81% / 80.27% / 84.47% / 0.7314**. Corrected in the manifest.

---

### 2.2 Reconciling $A1$ Static Baseline ($68.56\%$ vs $68.62\%$)
- **Strict Inequality ($S(d) > S(T)$):** Hits@1 = **68.62%** (1,290 / 1,880), Hits@5 = **80.90%**, Hits@10 = **83.51%**, MRR = **0.7399**.
- **Pessimistic Equal ($S(d) \ge S(T)$):** Hits@1 = **68.56%** (1,289 / 1,880), Hits@5 = **80.85%**, Hits@10 = **83.51%**, MRR = **0.7385**.
- **Trace:** The discrepancy is caused by **exactly 1 patient case** where a distractor disease tied the true disease's floating-point score. The manuscript primary tables use strict inequality consistently.

---

### 2.3 Reconciling Table 3 Error Counts ($308$ vs $309$)
- **Canonical $Q=5$ Point Estimate ($K=10$, Strict Inequality):**
  $$\text{Hits@1} = 85.00\% = \frac{1,598}{1,880}$$
  $$\text{Net Errors Resolved} = 1,598 - 1,290 = \mathbf{308 \text{ cases}} \quad \left(\frac{308}{590} = 52.20\%\right)$$
- **Trace of Historical 309:** In Phase 10.5 exploratory sensitivity testing, $K=25$ achieved $85.05\%$ ($1,599$ correct), yielding $1,599 - 1,290 = 309$.
- **Resolution:** The canonical manuscript number for $K=10$ is **308 errors resolved**.

---

### 2.4 Unanchored Dense Baseline Description
- **Architecture:** Two-Tower Dense Bi-Encoder (256-dimensional embedding space).
- **Training Objective:** InfoNCE Contrastive Loss with in-batch negatives.
- **Training Data:** 562 training diseases (7,721 patient cases).
- **Evaluation:** Zero-shot evaluation across all 12,956 candidate disease embeddings.
- **Result:** Hits@1 = **0.00%**, Hits@5 = **0.16%**, Hits@10 = **0.43%**, MRR = **0.0084**.
- **Explanation:** In disease-disjoint zero-shot prioritization across 12,956 candidates, unanchored neural embeddings collapse because disease representations are mapped to generic semantic clusters rather than exact discriminative symptom overlap.

---

## 3. MASTER CROSS-PHASE NUMBER RECONCILIATION TABLE

| Metric / Quantity | Phase 6/7 | Phase 8/9 | Phase 10 | Phase 10.5 | Phase 10.6 | Phase 10.7 | Phase 10.8 | Final Manifest | Trace / Explanation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Raw IC Hits@1** | 66.81% | 66.81% | 66.81% | 66.81% | 66.81% | 66.81% | 66.81% | **66.81%** | Identical across all phases. |
| **Raw IC Hits@5** | 80.27% | 80.27% | 80.27% | 80.27% | 80.27% | 80.27% | 80.85% | **80.27%** | Phase 10.8 typo corrected. |
| **Raw IC Hits@10** | 84.47% | 84.47% | 84.47% | 84.47% | 84.47% | 84.47% | 83.51% | **84.47%** | Phase 10.8 typo corrected. |
| **Raw IC MRR** | 0.7314 | 0.7314 | 0.7314 | 0.7314 | 0.7314 | 0.7314 | 0.7314 | **0.7314** | Identical across all phases. |
| **A1 Hits@1** | 68.62% | 68.62% | 68.62% | 68.62% | 68.62% | 68.62% | 68.62% | **68.62%** | Canonical strict point estimate. |
| **A1 MRR** | 0.7399 | 0.7399 | 0.7399 | 0.7399 | 0.7399 | 0.7399 | 0.7399 | **0.7399** | Canonical strict MRR. |
| **Active Q=1 Hits@1** | — | — | 74.84% | 74.84% | 74.84% | 74.84% | 74.84% | **74.84%** | Canonical 1-step point estimate. |
| **Active Q=5 Hits@1** | — | — | — | 85.05% | 85.00% | 85.00% | 85.00% | **85.00%** | Canonical 5-step point estimate. |
| **Active Q=10 Hits@1**| — | — | — | 89.95% | 89.95% | 89.95% | 89.95% | **89.95%** | Canonical 10-step point estimate. |
| **Resolved Errors Q=1**| — | — | 117 | 117 | 117 | 117 | 117 | **117 / 590** | 19.83% of residual errors. |
| **Resolved Errors Q=5**| — | — | — | 309 | 308 | 308 | 308 | **308 / 590** | 52.20% of residual errors ($K=10$). |
| **Worsened Cases** | — | — | 0 | 0 | 0 | 0 | 0 | **0 / 1,290** | 0.00% demoted cases. |
| **N=12,956 Q=5 Gain** | — | — | — | +16.44 pp | +16.38 pp | +16.38 pp | +16.38 pp | **+16.38 pp** | Strict point estimate gain. |

---

## 4. FINAL PUBLICATION GATE STATUS: [ GREEN — READY FOR MANUSCRIPT DRAFTING ]

- **100% of manuscript numbers are traced and reconciled.**
- **All historical discrepancies are documented and explained.**
- **`research/final_manuscript_number_manifest.md` is frozen as the authoritative single source of truth.**
