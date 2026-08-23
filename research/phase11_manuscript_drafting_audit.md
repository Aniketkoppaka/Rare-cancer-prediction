# Phase 11 — Manuscript Drafting & Evidence Audit Report

**Date:** 2026-08-23  
**Manuscript Target:** `paper/paper_draft_v1.0.md`  
**Single Source of Truth:** `research/final_manuscript_number_manifest.md`  
**Benchmark Version:** `RARE_DISEASE_ZS_BENCHMARK_v1.0`  
**Auditor Mode:** Adversarial Manuscript Verification Gatekeeper  

---

## 1. EXECUTIVE AUDIT VERDICT

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               PHASE 11 MANUSCRIPT AUDIT VERDICT                                  │
│                                                                                                  │
│  FINAL STATUS: [ GREEN — MANUSCRIPT FULLY VERIFIED & FROZEN ]                                    │
│                                                                                                  │
│  1. NUMBER MANIFEST COMPLIANCE:                                                                  │
│     - 100% of numerical quantities in `paper/paper_draft_v1.0.md` match                          │
│       `research/final_manuscript_number_manifest.md` exactly.                                    │
│     - Zero numbers were retrieved from older exploratory draft logs.                             │
│                                                                                                  │
│  2. SCIENTIFIC CLAIM CALIBRATION:                                                                │
│     - "Observation ceiling": Strictly defined as an empirical observation limitation on sparse   │
│       HPO records, not an impossible mathematical boundary.                                      │
│     - "Zero-worsening": Framed as an empirical stability property on clinical phenotype profiles │
│       rather than an unconstrained universal mathematical theorem.                              │
│     - "Universe scaling": Framed as an empirical monotonic relationship rather than a universal  │
│       scaling law.                                                                               │
│     - "Dense baseline": Accurately specified as an Unanchored Dense Two-Tower Transformer        │
│       (256-d, InfoNCE) rather than generalizing to all possible neural architectures.            │
│     - "Clinical feasibility": Labeled explicitly as an exploratory feasibility audit.            │
│                                                                                                  │
│  3. STATISTICAL REPORTING:                                                                       │
│     - Point estimates (85.00% at Q=5) are clearly distinguished from bootstrap means (84.97%).   │
│     - Paired 95% Bootstrap CIs and McNemar chi2 test statistics are fully documented.            │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. COMPREHENSIVE NUMBER-BY-NUMBER MANIFEST AUDIT

| Manuscript Section / Table | Quantity / Metric | Manifest Value | Manuscript Draft Value | Audit Status |
| :--- | :--- | :---: | :---: | :---: |
| **Abstract & Benchmark** | Total evaluable cases | 10,601 | 10,601 | **MATCH** |
| **Abstract & Benchmark** | Candidate disease universe | 12,956 | 12,956 | **MATCH** |
| **Abstract & Benchmark** | Benchmark diseases | 803 | 803 | **MATCH** |
| **Benchmark Table 1** | Train diseases / cases | 562 / 7,721 | 562 / 7,721 | **MATCH** |
| **Benchmark Table 1** | Val diseases / cases | 120 / 1,000 | 120 / 1,000 | **MATCH** |
| **Benchmark Table 1** | Test diseases / cases | 121 / 1,880 | 121 / 1,880 | **MATCH** |
| **Baseline Table 2** | Dense Two-Tower Hits@1 | 0.00% | 0.00% | **MATCH** |
| **Baseline Table 2** | Dense Two-Tower MRR | 0.0084 | 0.0084 | **MATCH** |
| **Baseline Table 2** | Cosine IC Hits@1 | 39.68% | 39.68% | **MATCH** |
| **Baseline Table 2** | Raw IC ($A0$) Hits@1 | 66.81% | 66.81% | **MATCH** |
| **Baseline Table 2** | Raw IC ($A0$) Hits@5 | 80.27% | 80.27% | **MATCH** |
| **Baseline Table 2** | Raw IC ($A0$) Hits@10 | 84.47% | 84.47% | **MATCH** |
| **Baseline Table 2** | Raw IC ($A0$) MRR | 0.7314 | 0.7314 | **MATCH** |
| **Baseline Table 2** | Power-Norm ($A1$) Hits@1 | 68.62% | 68.62% | **MATCH** |
| **Baseline Table 2** | Power-Norm ($A1$) MRR | 0.7399 | 0.7399 | **MATCH** |
| **Trajectory Table 3** | 1-Step ($Q=1$) Hits@1 | 74.84% | 74.84% | **MATCH** |
| **Trajectory Table 3** | 1-Step ($Q=1$) MRR | 0.7843 | 0.7843 | **MATCH** |
| **Trajectory Table 3** | 1-Step Errors Resolved | 117 / 590 (19.8%) | 117 / 590 (19.8%) | **MATCH** |
| **Trajectory Table 3** | 5-Step ($Q=5$) Hits@1 | 85.00% | 85.00% | **MATCH** |
| **Trajectory Table 3** | 5-Step ($Q=5$) MRR | 0.8684 | 0.8684 | **MATCH** |
| **Trajectory Table 3** | 5-Step Errors Resolved | 308 / 590 (52.2%) | 308 / 590 (52.2%) | **MATCH** |
| **Trajectory Table 3** | 10-Step ($Q=10$) Hits@1 | 89.95% | 89.95% | **MATCH** |
| **Trajectory Table 3** | 10-Step ($Q=10$) MRR | 0.9049 | 0.9049 | **MATCH** |
| **Trajectory Table 3** | Oracle Upper Bound Hits@1| 90.59% | 90.59% | **MATCH** |
| **Scaling Table 4** | $N=100$ Static / Gain | 90.73% / +5.47 pp | 90.73% / +5.47 pp | **MATCH** |
| **Scaling Table 4** | $N=500$ Static / Gain | 84.60% / +8.53 pp | 84.60% / +8.53 pp | **MATCH** |
| **Scaling Table 4** | $N=1,000$ Static / Gain | 81.95% / +10.26 pp | 81.95% / +10.26 pp| **MATCH** |
| **Scaling Table 4** | $N=5,000$ Static / Gain | 73.88% / +14.14 pp | 73.88% / +14.14 pp| **MATCH** |
| **Scaling Table 4** | $N=12,956$ Static / Gain | 68.62% / +16.38 pp | 68.62% / +16.38 pp| **MATCH** |
| **Entropy Table 5** | Q1 ($H < 0.001$) Gain | +0.00 pp | +0.00 pp | **MATCH** |
| **Entropy Table 5** | Q2 ($0.001 \le H < 0.409$) Gain | +0.85 pp | +0.85 pp | **MATCH** |
| **Entropy Table 5** | Q3 ($0.409 \le H < 2.196$) Gain | +21.06 pp | +21.06 pp | **MATCH** |
| **Entropy Table 5** | Q4 ($H \ge 2.196$) Gain | +43.62 pp | +43.62 pp | **MATCH** |
| **Robustness Table 6** | Uniform 50% Hits@1 | 78.40% (+9.78 pp) | 78.40% (+9.78 pp) | **MATCH** |
| **Robustness Table 6** | IC-Weighted 50% Hits@1 | 78.99% (+10.37 pp) | 78.99% (+10.37 pp)| **MATCH** |
| **Robustness Table 6** | IC-Weighted 75% Hits@1 | 76.01% (+7.39 pp) | 76.01% (+7.39 pp) | **MATCH** |
| **Feasibility Table 7** | Bedside Assessable (A,B,C)| 88% | 88% | **MATCH** |

---

## 3. FINAL GATE DECISION: [ GREEN — MANUSCRIPT FULLY READY ]

`paper/paper_draft_v1.0.md` is complete, internally consistent, mathematically calibrated, and fully backed by the frozen reproducibility package.
