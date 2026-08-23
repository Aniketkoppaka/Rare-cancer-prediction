# Phase 12 — Required Manuscript Revisions Plan (v1.0 ➔ v1.1)

**Target:** `paper/paper_draft_v1.0.md`  
**Purpose:** Concrete, line-by-line remediation of all 5 fatal reviewer vulnerabilities identified in the Phase 12 Adversarial Review.  

---

## 1. MUST FIX BEFORE SUBMISSION (CRITICAL REVISIONS)

### 🔴 MUST FIX 1: Retitle to Eliminate Search Universe vs. Test Disease Scope Ambiguity
- **Current Issue:** The title *"Active Phenotype Acquisition Resolves the Observation Ceiling in Open-World Zero-Shot Rare Disease Prioritization across 12,956 Diseases"* implies that the benchmark evaluated 12,956 distinct test diseases.
- **Exact Revision:** Change Title to:  
  **"Active Phenotype Acquisition Resolves the Observation Bottleneck in Zero-Shot Rare Disease Prioritization over a 12,956-Disease Candidate Universe"**
- **Scientific Rationale:** Accurately states that the evaluation tests zero-shot prioritization over 121 unseen disease targets against a massive 12,956-disease retrieval candidate universe.

---

### 🔴 MUST FIX 2: Incorporate the Random Acquisition Baselines into Table 3
- **Current Issue:** Table 3 compares Policy A only to Static Baseline and Oracle, leaving the paper vulnerable to the reviewer objection: *"Any additional feature from the top candidates improves retrieval; what is the mathematical contribution of Information Gain?"*
- **Exact Revision:** Add two rows to Table 3:
  1. `Uniform Random HPO Query (Q=1)`: $72.76\% \pm 0.08\%$
  2. `Candidate-Pool Random Query (Q=1)`: $74.30\% \pm 0.36\%$
  3. `Information-Theoretic Policy A (Q=1)`: $\mathbf{74.84\%}$ ($p < 10^{-25}$)
  4. `Information-Theoretic Policy A (Q=5)`: $\mathbf{85.00\%}$ ($p < 10^{-65}$)
- **Scientific Rationale:** Proves that Information-Theoretic acquisition outperforms unguided and candidate-pool random querying, establishing that active feature selection provides mathematical value beyond arbitrary feature accumulation.

---

### 🔴 MUST FIX 3: Formalize Counterfactual Negative Evidence Assumptions ($\lambda = 0.50$)
- **Current Issue:** Reviewer B correctly notes that unrecorded symptoms in real clinical charts are not guaranteed negative findings.
- **Exact Revision:** In Section 5.1 (Symmetric Negative Evidence Scoring Update) and Section 11 (Limitations), explicitly state:  
  > *"Under the benchmark simulation protocol, unannotated phenotypes for target disease $T$ are modeled as counterfactual absent findings. In prospective clinical deployments, negative evidence must be elicited through active physician confirmation to distinguish genuine clinical absence from unexamined features."*
- **Scientific Rationale:** Transparently scopes negative evidence updates and prevents clinical geneticist reviewers from rejecting the paper on semantic documentation grounds.

---

### 🔴 MUST FIX 4: Refine Clinical Feasibility Categorization as "Exploratory"
- **Current Issue:** Claiming that $88\%$ of queried terms are "clinically feasible" without formal double-blind inter-rater annotation is an easy target for rejection.
- **Exact Revision:** In Abstract, Section 8, and Table 7:
  - Rename Table 7 title to: *"Exploratory clinical acquisition mode classification of 100 top queried HPO terms."*
  - Rephrase prose to: *"In an exploratory feasibility audit of the 100 most frequently queried HPO terms, $88\%$ were classified as potentially assessable at the bedside through patient history, visual inspection, or standard physical examination without immediate specialist imaging or genetic assays."*
- **Scientific Rationale:** Appropriately frames the analysis as an exploratory feasibility assessment rather than a validated clinical trial.

---

### 🔴 MUST FIX 5: Precisely Scope the Unanchored Dense Baseline
- **Current Issue:** Table 2 lists *"Unanchored Dense Embeddings (BioBERT / GCN)"*, which overgeneralizes the failure of one specific model to all neural methods.
- **Exact Revision:** Update Table 2 and Section 6.1 to read:  
  *"Unanchored Dense Two-Tower Transformer (256-d InfoNCE)"*. Add a clarifying sentence in Section 6.1:  
  > *"While this specific unanchored contrastive bi-encoder collapsed under open-world zero-shot evaluation, future hybrid architectures combining symbolic anchoring with deep graph representations represent an important direction for further exploration."*
- **Scientific Rationale:** Prevents ML reviewers from defending neural representations against overgeneralized claims.

---

## 2. SHOULD FIX (METHODOLOGICAL & TERMINOLOGICAL POLISH)

### 🟡 SHOULD FIX 1: Clarify "Active Feature Acquisition" vs. "Active Learning"
- **Revision:** Throughout the text, use the exact term **"Active Phenotype Acquisition"** or **"Sequential Feature Acquisition (SFA)"** and explicitly clarify that model weights ($\gamma, \lambda$) are fixed at inference time, while the patient's feature representation is dynamically augmented.

### 🟡 SHOULD FIX 2: Replace "Observation Ceiling" with "Observation Bottleneck"
- **Revision:** Use **"Observation Bottleneck"** or **"Input-Sparsity Ceiling"** in prose to avoid implying an insurmountable mathematical boundary.

### 🟡 SHOULD FIX 3: Quantify $K=10$ Computational Tradeoff
- **Revision:** In Section 7.3, state explicitly that while $K=25$ achieves $86.40\%$ on validation ($+0.80\text{ pp}$ over $K=10$), $K=10$ was selected as the canonical operating point because it reduces differential candidate computation by $60\%$ with negligible performance difference.

---

## 3. OPTIONAL POLISH

### 🟢 OPTIONAL POLISH 1: Highlight the Tripling of Marginal Active Value
- Emphasize the finding in the Abstract and Discussion that expanding the candidate search space from $N=100 \rightarrow 12,956$ **triples the marginal diagnostic value of active acquisition ($+5.47\text{ pp} \rightarrow +16.38\text{ pp}$)** due to distractor crowding.

### 🟢 OPTIONAL POLISH 2: Clarify Test Set Disease-Disjoint Status
- In Section 4.1, explicitly remind the reader that the 121 test diseases were completely excluded from training and validation, ensuring strict zero-shot generalization.

---

## 4. ACTION SUMMARY
All required revisions are purely textual, terminological, and contextual, requiring **zero changes to the frozen numbers or the locked benchmark**. We are ready to apply these concrete revisions to produce `paper/paper_draft_v1.1.md`.
