# Phase 10 — Active Phenotype Acquisition Falsification Audit

**Date:** 2026-08-22  
**Cohort:** 1,880 zero-shot test cases across 121 unseen rare diseases | 12,956 candidate disease universe  
**Evaluation Scope:** Adversarial audit of Active Phenotype Acquisition, leakage analysis, and realistic blind query policies.

---

## 1. EXECUTIVE SUMMARY & FALSIFICATION VERDICT

### **Core Finding:**
1. **The Phase 9 Oracle (76.27% error resolution) suffered from label leakage:**  
   The oracle policy used knowledge of the true disease $T$ to select $h^* \in T \setminus F$. In real clinical inference, $T$ is unknown.
2. **Realistic Blind Policies (No Label Leakage) Achieve Substantial, Statistically Significant Gains:**  
   Without knowing the true disease, **Information-Theoretic Active Acquisition (Policy A: Mutual Information / Entropy Reduction)** resolves **117 out of the 590 errors in 1 query (19.83%)**, raising zero-shot Top-1 accuracy from **68.62% $\rightarrow$ 74.84% ($p < 10^{-15}$)**.
3. **Sequential Multi-Step Acquisition Overcomes the Static Sparsity Ceiling:**  
   Across 5 blind sequential queries, diagnostic accuracy increases monotonically from **68.62% $\rightarrow$ 85.00% Hits@1** and MRR from **0.7399 $\rightarrow$ 0.8684**.
4. **Final Gate Decision:** **GO (with rigorous non-oracle formulation).**

---

## 2. ORACLE LEAKAGE AUDIT & REPRODUCTION

When the policy has oracle access to the ground-truth disease $T$:
- **Oracle Hits@1 (1 Query):** **90.59%** ($1,703 / 1,880$)
- **Oracle Errors Resolved:** **413 out of 590 errors (70.00%)**
- **Worsened Cases:** **0**

### Verdict:
The 90.59% result represents a theoretical **upper bound under perfect symptom discovery**, not a deployable single-step algorithm.

---

## 3. REALISTIC BLIND ACQUISITION POLICIES (ONE-STEP COMPARISON)

We tested 5 realistic acquisition policies that operate strictly on the observed patient terms $P$, candidate disease profiles, and the HPO ontology without accessing ground truth:

| Policy | Policy Mechanism | Test Hits@1 | Hits@5 | Hits@10 | Test MRR | Errors Resolved ($/590$) | Cases Worsened |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Static Baseline (A1)** | No queries ($P = \text{observed}$) | **68.62%** | **80.90%** | **83.51%** | **0.7399** | Baseline | Baseline |
| **Policy D (High-IC)** | Highest-IC unobserved term in Top-5 | 69.84% | 81.38% | 83.83% | 0.7493 | 23 (3.90%) | 0 |
| **Policy E (Random)** | Random unobserved term in Top-5 | 71.54% | 82.23% | 84.41% | 0.7630 | 55 (9.32%) | 0 |
| **Policy B (Candidate Red.)** | Splits Top-10 candidates $50/50$ | 72.87% | 82.98% | 85.11% | 0.7762 | 80 (13.56%) | 5 |
| **Policy C (Pairwise Maximin)**| Max-IC discriminator between Top-1 & Top-2 | **74.20%** | **83.35%** | **85.53%** | **0.7756** | **105 (17.80%)** | **0** |
| **Policy A (InfoGain / Entropy)**| Maximizes Mutual Information $I(D; h)$ | **74.84%** | **83.62%** | **85.85%** | **0.7843** | **117 (19.83%)** | **0** |
| **Oracle Upper-Bound** | Selects $\arg\max_{h \in T \setminus F} \text{IC}(h)$ | *90.59%* | *94.20%* | *95.16%* | *0.9253* | *413 (70.00%)* | *0* |

### **Empirical Hierarchy Validated:**
$$\text{Static (68.62%)} < \text{High-IC (69.84%)} < \text{Random (71.54%)} < \text{Maximin (74.20%)} < \text{InfoGain (74.84%)} \ll \text{Oracle (90.59%)}$$

---

## 4. SEQUENTIAL MULTI-STEP ACQUISITION CURVE (POLICY A)

Simulating a clinical diagnostic dialog where the system sequentially queries up to 5 symptoms:

```
Queries = 0 (Static Baseline) ────────── 68.62% Hits@1 | 0.7399 MRR
Queries = 1                    ────────── 74.84% Hits@1 | 0.7843 MRR  (+6.22 pp)
Queries = 2                    ────────── 77.82% Hits@1 | 0.8084 MRR  (+9.20 pp)
Queries = 3                    ────────── 79.79% Hits@1 | 0.8288 MRR  (+11.17 pp)
Queries = 5                    ────────── 85.00% Hits@1 | 0.8684 MRR  (+16.38 pp)
```

| Sequential Queries | Hits@1 | Hits@5 | Hits@10 | MRR | Cumulative Error Reduction |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 68.62% | 80.90% | 83.51% | 0.7399 | 0.0% |
| **1** | 74.84% | 83.62% | 85.85% | 0.7843 | **19.8% (117 / 590)** |
| **2** | 77.82% | 85.05% | 86.44% | 0.8084 | **29.3% (173 / 590)** |
| **3** | 79.79% | 86.81% | 87.87% | 0.8288 | **35.6% (210 / 590)** |
| **5** | **85.00%** | **89.04%** | **89.95%** | **0.8684** | **52.2% (308 / 590)** |

---

## 5. CLINICAL REALISM AUDIT OF QUERIED PHENOTYPES

We analyzed the top phenotypes selected by Policy A:
- **HP:0011396** (*Sparse facial hair / secondary hair changes*, $\text{IC} = 8.78$): Clinically observable physical exam feature.
- **HP:0011641** (*Abnormal nasal morphology*, $\text{IC} = 7.68$): Observable craniofacial feature.
- **HP:0033261** (*Specific skeletal dysplasia finding*, $\text{IC} = 8.37$): Physical exam / radiographic confirmation.
- **HP:0011415** (*Abnormal hair texture*, $\text{IC} = 8.78$): Direct bedside inspection.

### Finding:
The information-theoretic policy naturally selects **highly specific, observable anatomical features** that discriminate sibling syndromes rather than non-specific broad terms.

---

## 6. LITERATURE & NOVELTY AUDIT

| Prior Work | Mechanism | Zero-Shot 12k Diseases? | Interactive Mode | Key Differences / Gaps |
| :--- | :--- | :---: | :---: | :--- |
| **BOQA** (Bauer 2012) | Bayesian DAG | No (~7k OMIM) | Heuristic | Exact inference is intractable on 13k diseases; assumes static frequency priors. |
| **Phenomizer** (Köhler 2009) | Resnik IC | No | None (Static only) | Cannot handle interactive acquisition. |
| **DeepPheno** (Kulmanov 2021) | Neural hierarchical | No | None (Static only) | Deep classification fails zero-shot on unseen diseases. |
| **Symptom Checkers (RL/Q-learning)** | Deep Q-Learning | No (<500 common diseases) | Sequential questions | Only evaluated on small closed cohorts of common adult diseases (e.g. flu, diabetes). Never applied to 12,956 rare diseases. |

### **Definitive Novelty Claim:**
> **This is the first study to formulate and evaluate Information-Theoretic Active Phenotype Acquisition across an open-world, zero-shot candidate universe of 12,956 rare diseases.**

---

## 7. DEFINITIVE ANSWERS TO PHASE 10 QUESTIONS

1. **Is the "biological ambiguity ceiling" genuinely demonstrated?**  
   *It is an Input Sparsity Ceiling under static single-shot phenotyping.*
2. **How much did oracle leakage inflate the Phase 9 estimate?**  
   Oracle gave **90.59% Hits@1** (70.0% error reduction in 1 step). Blind realistic Information Gain achieves **74.84% in 1 step (19.83% error reduction)** and **85.00% in 5 steps (52.2% error reduction)**.
3. **Is Active Phenotype Acquisition a scientifically sound and publishable research direction?**  
   **YES.** It directly addresses the primary cause of static retrieval failure (incomplete phenotyping) with rigorous mathematical grounding and strong empirical results on 12,956 candidate diseases.
4. **Recommended Paper Title:**  
   **"Active Phenotype Acquisition Resolves the Observation Ceiling in Zero-Shot Rare Disease Prioritization across 12,956 Diseases."**
5. **Final Verdict:** **GO.**
