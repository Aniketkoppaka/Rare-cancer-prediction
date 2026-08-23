# Phase 6 Implementation & Pipeline Audit

## 1. Existing Relevant Modules & Preprocessing
* **`data/benchmark/clean_phenopackets.json`**: Contains 10,658 extracted patient profiles. 10,601 cases cleanly map to candidate rare diseases present in `phenotype.hpoa`.
* **`data/benchmark/phenotype.hpoa`**: Curated knowledge base containing 12,956 rare diseases and 285,599 phenotype annotations.
* **`data/benchmark/hp.obo`**: Complete HPO directed acyclic graph (20,413 terms, 24,378 edges).

## 2. Strict Zero-Shot Partitioning Logic
* **Random Seed:** Locked at `Seed = 42`.
* **Disease-Disjoint Split Strategy:**
  * Total Evaluable Diseases: **803 diseases** (10,601 cases).
  * **Train Diseases (70%):** 562 diseases (7,721 patient cases).
  * **Validation Diseases (15%):** 120 diseases (1,000 patient cases).
  * **Zero-Shot Test Diseases (15%):** 121 diseases (1,880 patient cases).
* **Test Candidate Universe:** Full $|\mathcal{D}| = 12,956$ rare disease profiles in `phenotype.hpoa`.

## 3. Leakage Audit & Zero-Shot Verification Checklist
* [x] **No Disease-ID Lookup Table:** Neither Model B nor Model C will contain an `nn.Embedding(num_diseases, dim)` table.
* [x] **Ontological Projection Only:** A disease representation $z_D$ or memory $\{K_D, V_D\}$ is generated strictly on-the-fly from its HPO terms.
* [x] **Disjoint Disease Constraint:** No patient with a test disease is seen during model training.
* [x] **Locked Test Set:** Architecture exploration and hyperparameter decisions will be tuned solely on the 1,000 validation cases.

## 4. Model Architecture Matrix (Minimum Viable Prototypes)
* **Model A (Raw IC Baseline):** Exact unnormalized lexical overlap $\sum_{h \in P \cap D} \text{IC}(h)$.
* **Model B (Learned Symmetric Dual-Encoder):**
  * Patient Encoder: $z_P = \text{MeanPool}(\text{Linear}(E_P))$.
  * Disease Encoder: $z_D = \text{MeanPool}(\text{Linear}(E_D))$.
  * Similarity: $S(P, D) = \text{Cosine}(z_P, z_D) = S(D, P)$.
* **Model C (Learned Query-Conditioned Asymmetric Encoder):**
  * Patient Query: $Q_P = \text{MeanPool}(\text{Linear}(E_P)) \in \mathbb{R}^{1 \times d}$.
  * Disease Memory: $K_D, V_D = \text{Linear}(E_D) \in \mathbb{R}^{|D| \times d}$.
  * Asymmetric Matching: $S(P, D) = \text{ScoreHead}(\text{CrossAttention}(Q_P, K_D, V_D)) \neq S(D, P)$.
