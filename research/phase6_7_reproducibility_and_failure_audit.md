# Phase 6.7 — Reproducibility Audit, Failure Mechanism Analysis, and Symbolic-Grounded Reassessment

**Date:** 2026-08-22  
**Dataset:** 10,658 raw cases → 10,601 evaluable cases | 803 diseases | 562 train / 120 val / 121 zero-shot test  
**Test Cohort:** 1,880 cases | Candidate Universe: 12,956 candidate diseases  

---

## 1. BENCHMARK RECONCILIATION: WHY DID RAW IC (A0) REPORT 66.81% vs 63.62%?

### 1.1 Direct Comparison
A forensic audit comparing the Phase 6/6.5 code against the Phase 6.6 runner was executed:

| Attribute | Phase 6 & 6.5 Pipeline | Phase 6.6 Runner Pipeline | Status |
| :--- | :--- | :--- | :---: |
| **Cases** | 1,880 test cases | 1,880 test cases | **Identical** |
| **Diseases** | 121 test diseases | 121 test diseases | **Identical** |
| **Candidate Universe** | 12,956 candidate diseases | 12,956 candidate diseases | **Identical** |
| **IC Corpus & Values** | Background corpus HPOA | Background corpus HPOA | **Identical** |
| **HPO Preprocessing** | Exact term lookup | Exact term lookup | **Identical** |
| **Ranking Implementation** | `scores.sort(key=lambda x: x[0], reverse=True)` | `scores.sort(reverse=True)` | **DISCREPANCY** |
| **Tie-Breaking Mechanism** | Stable sort (preserves candidate order) | Tuple sort (breaks ties by Disease ID string reverse-alphanumeric) | **ROOT CAUSE** |
| **Hits@1** | **66.81%** (1,256 / 1,880) | **63.62%** (1,196 / 1,880) | Explained |
| **Hits@5** | **80.27%** | **78.46%** | Explained |
| **Hits@10** | **84.47%** | **82.39%** | Explained |
| **MRR** | **0.7314** | **0.7049** | Explained |

### 1.2 The Root Cause
For unnormalized Raw IC ($A0$), many candidate diseases produce tied scores (especially $0.0$ overlap).
- When using Python stable sorting `scores.sort(key=lambda x: x[0], reverse=True)`, equal-score ties preserve natural insertion order.
- When sorting raw tuples `(score, did)` with `scores.sort(reverse=True)`, Python compares the second tuple element `did` (e.g. `ORPHA:9...` vs `ORPHA:1...` or `OMIM:6...` vs `OMIM:1...`).
- When $\gamma > 0$ ($A1$), profile lengths differentiate non-zero ties, which is why $A1$ achieved **69.15%** in both scripts.

**Standardized Benchmark Baseline:**
The canonical, reproducible Raw IC baseline is **Hits@1 = 66.81%, MRR = 0.7314** (stable candidate ordering).

---

## 2. VERIFYING A1 ON VALIDATION ONLY ($\gamma$ SELECTION)

To prevent data snooping, $\gamma$ was swept across validation cases ($N=1,000$, 120 disjoint validation diseases) over $\gamma \in [0.0, 0.50]$:

| $\gamma$ Value | Validation Hits@1 ($N=1,000$) | Validation MRR |
| :---: | :---: | :---: |
| $\gamma = 0.000$ (A0) | 71.00% | 0.7648 |
| $\gamma = 0.010$ | 71.40% | 0.7633 |
| $\gamma = 0.025$ | 71.60% | 0.7651 |
| $\gamma = 0.050$ | 71.40% | 0.7644 |
| $\gamma = 0.075$ | 72.00% | 0.7674 |
| $\mathbf{\gamma = 0.100}$ | $\mathbf{72.50\%}$ | $\mathbf{0.7701}$ |
| $\gamma = 0.150$ | 71.90% | 0.7654 |
| $\gamma = 0.250$ | 70.30% | 0.7505 |
| $\gamma = 0.500$ | 52.50% | 0.6058 |

**Validation Decision:** $\gamma = 0.10$ was selected as the optimal hyperparameter on validation data.

### 2.1 Single Evaluation of Pre-Selected $\gamma = 0.10$ on Locked Test Set
- **Test A0 (Raw IC):** Hits@1 = **66.81%** | Hits@5 = **80.27%** | Hits@10 = **84.47%** | MRR = **0.7314**
- **Test A1 ($\gamma = 0.10$, Pre-Selected):** Hits@1 = **68.56%** | Hits@5 = **80.85%** | Hits@10 = **83.51%** | MRR = **0.7387**
- **Gain:** $+1.75\text{ pp}$ Hits@1, $+0.0073$ MRR ($p = 0.0097$, McNemar test).

---

## 3. AUDIT OF MODEL B COLLAPSE (39.63% vs 7.07%)

- **Phase 6 Model B (39.63%):** Computed exact symbolic IC intersection divided by symmetric denominator $\sqrt{|P| \cdot |D|}$:
  $$S(P, D) = \frac{\sum_{h \in P \cap D} \text{IC}(h)}{\sqrt{|P| \cdot |D|}}$$
- **Phase 6.6 Model B (7.07%):** Replaced symbolic IC overlap with dense vector mean-pooling and cosine similarity:
  $$S(P, D) = \text{cosine}(\text{mean}(e(P)), \text{mean}(e(D)))$$
- **Finding:** Compressing discrete phenotype sets into unanchored mean-pooled dense vectors destroys the sparse, pathognomonic specificity of individual HPO terms, dropping retrieval performance by $32.56\text{ pp}$.

---

## 4. AUDIT OF C0 & C1 COLLAPSE (0.00% Hits@1)

### 4.1 Diagnostic Inspection
1. **Vanishing Gradient Bottleneck:** In Phase 6.6, frozen embeddings with $63/64$ random noise dimensions combined with unconstrained MLP scoring without dot-product alignment caused vanishing gradients.
2. **Loss Stagnation:** Training loss stagnated at $\ln(10) \approx 2.302$, meaning the model output near-uniform scores across all candidates.
3. **Severe Candidate Discrepancy:** Training on 31 batch negatives failed to constrain the unanchored MLP across 12,956 candidate diseases.

---

## 5. SMALL-SCALE OVERFIT & SEEN VS UNSEEN GENERALIZATION TESTS

### 5.1 Controlled Overfit Test (10 Diseases, 100 Patient Cases)
When re-implemented with proper dot-product scaling and trainable set projections:
- **Epoch 1:** Loss = $2.3032$ | Hits@1 = $1.00\%$
- **Epoch 10:** Loss = $2.0312$ | Hits@1 = $55.00\%$
- **Epoch 20:** Loss = $1.1820$ | Hits@1 = $59.00\%$
- **Epoch 30:** Loss = $0.9396$ | Hits@1 = $\mathbf{66.00\%}$

### 5.2 Seen vs Unseen Disease Test
- **Seen 10-Disease Accuracy:** $\mathbf{66.00\%}$ (High training retrieval)
- **Unseen 10-Disease Accuracy:** $\mathbf{14.00\%}$ (Random baseline = $10.00\%$)
- **Scientific Conclusion (CASE D):** The cross-attention neural model is capable of fitting seen disease combinations, but **fails to generalize zero-shot to unseen candidate diseases** when relying purely on unanchored dense representations.

---

## 6. WHY DOES $\gamma \approx 0.05 - 0.10$ WORK? (MECHANISM ANALYSIS)

Paired case analysis on the locked test set ($N=1,880$):
- **A1 Rescued ($A1=1, A0>1$):** **93 cases**
  - Average Patient HPO count: **6.2 terms**
  - Average True Disease Profile Size: **47.1 terms** (Medium/Large profiles)
- **A0 Lost ($A0=1, A1>1$):** **60 cases**
  - Average Patient HPO count: **7.0 terms**
  - Average True Disease Profile Size: **94.2 terms** (Massive outlier profiles)

### Mechanism Discovered:
1. Candidate profile size $|D|$ and Total IC mass $\sum_{h \in D} \text{IC}(h)$ are collinear ($r = 0.9811$).
2. Raw IC ($A0$) suffers from **incidental overlap accumulation**: massive disease profiles ($>100$ terms) accumulate accidental overlap on common phenotypes, pushing correct medium-sized diseases out of Rank 1.
3. Mild length normalization ($\gamma = 0.05 - 0.10$) suppresses accidental overlap from giant distractor diseases, rescuing 93 cases.
4. However, static length normalization is a blunt instrument: it penalizes true massive diseases (avg size 94.2), losing 60 cases.

---

## 7. SYMBOLIC-GROUNDED RESIDUAL MODEL PROOF-OF-CONCEPT

Instead of replacing symbolic IC with dense embeddings, we tested the **Symbolic-Grounded Residual Formulation**:
$$S_{\text{residual}}(P, D) = \sum_{h \in P \cap D} \text{IC}(h) + \lambda \cdot \mathcal{R}\big(\text{OverlapCount}, \text{SumIC}, \text{MaxIC}, \text{MeanIC}, \log |D|, \log |P|\big)$$

### Empirical Results:
- The model successfully learned $\lambda \approx 0.398$ and converged smoothly with margin loss ($0.204 \to 0.117$).
- On validation evaluation ($N=200$), the residual model maintained full symbolic retrieval fidelity (**80.50%** vs **81.00%** for Raw IC) without any catastrophic collapse.

---

## 8. FINAL SCIENTIFIC VERDICT & DECISION TREE (CASE E / D)

| Question | Finding |
| :--- | :--- |
| **A0 Discrepancy Resolved?** | **YES** (Stable sort = 66.81% vs tuple-string tie sort = 63.62%) |
| **A1 Validated Out-of-Sample?** | **YES** ($\gamma=0.10$ pre-selected on Val $\to$ 68.56% on Test, $+1.75\text{ pp}$, $p < 0.01$) |
| **Why Did Dense C1 Fail?** | **Dense compression destroys pathognomonic IC spikes + Zero-shot generalization gap** |
| **Can Cross-Attn Overfit Seen Data?** | **YES** ($1\% \to 66\%$ on 10-disease overfit test) |
| **Is Symbolic Residual Feasible?** | **YES** ($S = S_{\text{IC}} + \lambda R$ preserves exact match while enabling learned calibration) |

### **FINAL SCIENTIFIC PIVOT:**
> The hypothesis that **"unanchored neural embeddings can replace symbolic ontology matching"** is **FALSIFIED**.  
> The valid, high-impact research direction is:  
> **"Symbolic-Grounded Ontology Calibration: Can query-conditioned graph neural residuals calibrate exact phenotypic overlap without destroying pathognomonic evidence in zero-shot rare disease diagnosis?"**

This establishes the clean foundation for integrating Orphadata and ontology hierarchies in the next phase!
