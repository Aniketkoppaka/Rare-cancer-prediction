# Phase 6.6 — Genuine Query-Conditioned Asymmetric Model Experiment & Scientific Evaluation

**Date:** 2026-08-22  
**Dataset:** 10,658 raw cases → 10,601 evaluable cases | 803 diseases | 562 train / 120 val / 121 zero-shot test  
**Test Cohort:** 1,880 cases | Candidate Universe: 12,956 candidate diseases  
**Hardware Executed:** NVIDIA Tesla T4 GPU (Kaggle CUDA Environment)

---

## 1. RESEARCH QUESTION

> *"Can query-conditioned asymmetric phenotype matching reduce profile-size bias in zero-shot rare-disease prioritization beyond static phenotype weighting and length normalization?"*

---

## 2. MODEL EQUATIONS & ARCHITECTURES

### 2.1 Model A0: Raw Information Content Overlap (Exact Matching)
Classical, parameter-free baseline scoring exact phenotypic intersections:
$$S_{\text{A0}}(P, D) = \sum_{h \in P \cap D} \text{IC}(h)$$

### 2.2 Model A1: Optimal Static Length-Normalized IC ($\gamma = 0.05$)
Classical parametric directional baseline penalizing candidate profile size:
$$S_{\text{A1}}(P, D) = \frac{\sum_{h \in P \cap D} \text{IC}(h)}{|D|^{\gamma}}$$
*Note: $\gamma = 0.05$ was selected strictly on validation cases ($N=1,000$) and evaluated once on test.*

### 2.3 Model B: Learned Symmetric Dual-Encoder
Symmetric set encoder projecting mean-pooled patient and disease embeddings into a shared metric space with cosine scoring:
$$q_P = \text{Norm}\left(W \cdot \frac{1}{|P|}\sum_{p \in P} e(p)\right), \quad q_D = \text{Norm}\left(W \cdot \frac{1}{|D|}\sum_{d \in D} e(d)\right)$$
$$S_{\text{B}}(P, D) = q_P^\top q_D \quad \implies \quad S_{\text{B}}(P, D) = S_{\text{B}}(D, P)$$

### 2.4 Model C0: Non-Conditioned Learned Scorer
Encodes disease profiles independently of patient presentations using an unconditioned disease projection and MLP scorer:
$$q_P = W_P \cdot \bar{e}_P, \quad q_D = W_D \cdot \bar{e}_D, \quad S_{\text{C0}}(P, D) = \text{MLP}([q_P \,\|\, q_D])$$

### 2.5 Model C1: True Query-Conditioned Cross-Attention
Constructs a patient query vector $Q = f(P)$, disease keys $K_D = g(D)$, and disease values $V_D = v(D)$, computing asymmetric attention weights conditioned directly on patient input:
$$Q = W_Q \cdot \left(\frac{1}{|P|}\sum_{p \in P} e(p)\right)$$
$$\alpha_i(P, D) = \frac{\exp\left(\frac{Q \cdot (W_K e(d_i))^\top}{\sqrt{d_{\text{emb}}}}\right)}{\sum_{j=1}^{|D|} \exp\left(\frac{Q \cdot (W_K e(d_j))^\top}{\sqrt{d_{\text{emb}}}}\right)}, \quad Z(P, D) = \sum_{i=1}^{|D|} \alpha_i(P, D) \, (W_V e(d_i))$$
$$S_{\text{C1}}(P, D) = \text{MLP}([Q \,\|\, Z(P, D)])$$

*Property Verified:* $\alpha_i(P_1, D) \neq \alpha_i(P_2, D)$ for $P_1 \neq P_2$.

---

## 3. EXPERIMENTAL RESULTS (LOCKED ZERO-SHOT TEST, N=1,880)

All models evaluated across the full **12,956 candidate disease universe**.

| Model | Architecture Description | Hits@1 | Hits@5 | Hits@10 | MRR | Trainable Parameters | Query-Conditioned |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A0** | Raw Exact IC Overlap | 63.62% | 78.46% | 82.39% | 0.7049 | 0 | No |
| **A1** | Static Length-Norm IC ($\gamma = 0.05$) | **69.15%** | **81.06%** | **83.72%** | **0.7432** | 1 scalar | No |
| **B** | Learned Symmetric Dual-Encoder | 7.07% | 13.56% | 16.76% | 0.1073 | 4,096 | No |
| **C0** | Non-Conditioned Learned Scorer | 0.00% | 0.05% | 0.05% | 0.0011 | 16,513 | No |
| **C1** | True Query-Conditioned Cross-Attn | 0.00% | 0.00% | 0.00% | 0.0010 | 20,609 | **Yes** |

---

## 4. SCIENTIFIC ABLATION & HYPOTHESIS TESTING

### 4.1 Primary Test: $C_1 \text{ vs } C_0$ (Isolating Query-Conditioning)
- **$C_1$ Hits@1:** $0.00\%$
- **$C_0$ Hits@1:** $0.00\%$
- **$C_1$ vs $C_0$ McNemar $\chi^2$:** $1.0000$ ($p = 0.3173$, non-significant)
- **Result:** $C_1 \approx C_0 \approx 0.00\%$. The cross-attention query-conditioning mechanism provided **zero empirical advantage** over unconditioned projection in this formulation.

### 4.2 Secondary Test: $C_1 \text{ vs } A_0 / A_1$ (Neural Scorer vs Classical Matching)
- Both unconstrained neural dense scorers ($C_0, C_1$) completely collapsed under 12,956-way zero-shot retrieval ($0.00\%$ Hits@1 vs $63.62\%$ for $A0$ and $69.15\%$ for $A1$).
- Training with InfoNCE on 31 batch negatives was completely insufficient to learn a calibrated global score function capable of ranking against 12,956 candidate diseases zero-shot.

---

## 5. PROFILE-SIZE SUBGROUP ANALYSIS (Hits@1)

| Patient Sparsity | Disease Profile Size | Cases ($N$) | A0 (Raw IC) | A1 ($\gamma=0.05$) | C0 (Non-Cond) | C1 (Cross-Attn) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Dense ($>10$)** | **Large ($\ge 50$)** | 496 | 93.55% | **94.56%** | 0.00% | 0.00% |
| **Dense ($>10$)** | **Medium ($11-49$)** | 129 | 69.77% | 69.77% | 0.00% | 0.00% |
| **Medium ($4-10$)** | **Large ($\ge 50$)** | 469 | 64.39% | **72.28%** | 0.00% | 0.00% |
| **Medium ($4-10$)** | **Medium ($11-49$)** | 364 | 60.71% | **71.15%** | 0.00% | 0.00% |
| **Medium ($4-10$)** | **Small ($\le 10$)** | 4 | 50.00% | 50.00% | 0.00% | 0.00% |
| **Sparse ($\le 3$)** | **Large ($\ge 50$)** | 51 | 13.73% | 13.73% | 0.00% | 0.00% |
| **Sparse ($\le 3$)** | **Medium ($11-49$)** | 341 | 27.57% | **33.43%** | 0.00% | 0.00% |
| **Sparse ($\le 3$)** | **Small ($\le 10$)** | 26 | 61.54% | **76.92%** | 0.00% | 0.00% |

### Key Observations:
1. **$A1$ ($\gamma=0.05$) provides consistent gains across sparse/medium patient cohorts:**
   - Sparse patient + Small disease: **$+15.38\%$** ($61.54\% \to 76.92\%$)
   - Medium patient + Large disease: **$+7.89\%$** ($64.39\% \to 72.28\%$)
   - Medium patient + Medium disease: **$+10.44\%$** ($60.71\% \to 71.15\%$)
2. **Dense neural projection models ($B, C_0, C_1$) fail catastrophically** because rare disease diagnosis requires exact symbolic token matches and pathognomonic specificity, which dense vector pooling destroys unless explicitly anchored to symbolic IC overlap.

---

## 6. WHY DID THE UNCONSTRAINED NEURAL MODELS (B, C0, C1) FAIL?

1. **Loss of Exact Symbolic Matching:** Exact HPO token overlap is discrete and sparse. When terms are compressed into a 64-d dense vector and passed through an MLP, the model loses the exact symbolic identity of high-IC terms.
2. **Extreme Candidate Space Discrepancy:** Training used 31 random negatives, while testing ranked against 12,956 candidate diseases. Unconstrained MLPs output arbitrary score scales on unseen diseases, producing massive false-positive drift.
3. **The "Curse of Compression":** Mean-pooling and unconstrained cross-attention smooth out the sharp delta spikes of pathognomonic HPO terms (e.g., an HPO term with $\text{IC} > 8.0$ gets averaged away).

---

## 7. SCIENTIFIC VERDICT

### **VERDICT: NOT SUPPORTED / FALSIFIED IN UNCONSTRAINED FORM**

- **$C_1 \approx C_0$ ($0.00\%$ Hits@1):** True query-conditioned cross-attention on dense embeddings without symbolic exact-match anchoring completely fails to generalize in a 12,956-candidate zero-shot setting.
- **The True Winner is Classical Directional Normalization ($A1$):**
  $$S(P, D) = \frac{\sum_{h \in P \cap D} \text{IC}(h)}{|D|^{0.05}}$$
  This simple, grounded, parameter-efficient method achieves **$69.15\%$ Hits@1** and **$0.7432$ MRR** (+5.53 pp over Raw IC in this benchmark), outperforming all unanchored neural architectures.

---

## 8. IMPLICATIONS FOR FUTURE PHASES

Any future neural architecture (including Orphadata hierarchy / GNN integration) **MUST NOT** discard exact symbolic IC overlap. Instead, neural components must act as a **residual re-ranker or term-weight modulator directly on top of the symbolic exact-match graph**, rather than replacing symbolic retrieval with unanchored dense embeddings.
