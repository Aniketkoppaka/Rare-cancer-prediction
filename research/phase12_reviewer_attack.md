# Phase 12 — Hostile Adversarial Reviewer Attack on Manuscript v1.0

**Date:** 2026-08-23  
**Target Manuscript:** `paper/paper_draft_v1.0.md`  
**Review Venue:** Top-Tier ML / Biomedical Informatics Conference (NeurIPS / ICLR / JAMIA / Nature Digital Medicine)  
**Auditor Mode:** Hostile Adversarial Peer Reviewers (Simulated Rejection Committee)  

---

## 1. COMPREHENSIVE REVIEWER REPORTS

### 🧑‍⚖️ Reviewer A (Machine Learning & Representation Learning Specialist)
**Score:** `6 / 10` (Borderline Reject / Major Revision)  
**Confidence:** `4 / 5` (High)

#### Summary of Contribution:
The paper presents an interactive phenotype elicitation framework for rare disease retrieval across a large knowledge base of 12,956 diseases. Using a power-normalized Information Content scoring baseline and an entropy-weighted Information Content acquisition policy, the authors show that asking 5 targeted questions improves Top-1 retrieval from $68.62\%$ to $85.00\%$ on 1,880 zero-shot patient cases.

#### Critical ML Weaknesses & Methodological Concerns:
1. **Terminology Inflation ("Active Learning" vs. "Active Feature Acquisition"):**
   - The paper frequently blends active learning with sequential feature acquisition. In traditional active learning, an algorithm selects *unlabeled data instances to annotate for model parameter updating*. Here, the model parameters ($\gamma, \lambda$) are completely fixed; the system is performing *test-time sequential feature acquisition (SFA)* or *cost-sensitive feature elicitation*. Calling this "active acquisition" without clearly defining it as sequential test-time feature selection will mislead ML reviewers.
2. **Missing Random Acquisition Baseline in Main Manuscript Table:**
   - The paper claims that the Information-Theoretic policy ($H_b(p_h) \cdot \text{IC}(h)$) is essential. However, Table 3 only compares Policy A to static baseline and an oracle. While Phase 10.5 repository logs show an unguided random HPO baseline ($72.76\%$) and candidate-pool random baseline ($74.30\%$), **neither is presented in Table 3 of the manuscript**. Without explicitly showing the Random Candidate-Pool baseline in the primary trajectory table, a reviewer will argue: *"Any additional feature from the top candidates improves retrieval; the entropy formula adds negligible value."*
3. **Overgeneralization of the Dense Baseline Failure:**
   - Table 2 reports an unanchored dense bi-encoder achieving $0.00\%$ Hits@1. While it is true that an unconditioned 256-d InfoNCE bi-encoder struggles with zero-shot set-retrieval, modern graph neural networks (e.g., SHEPHERD, GIMME-HPO) or hybrid dense-symbolic retrievers with hard negative mining might perform better. The manuscript should state that *this specific unanchored two-tower architecture* collapsed, not imply that dense neural retrieval is fundamentally incapable.
4. **Finite Known Set vs. "Open-World" Claim:**
   - The candidate universe consists of 12,956 known HPOA disease entries. This is an open-world setting in terms of *disease-disjoint unseen test categories*, but the candidate retrieval vocabulary is closed and finite. The phrasing *"over a 12,956-disease candidate universe"* is more mathematically precise than *"open-world zero-shot prioritization across 12,956 diseases"*.

---

### 🧑‍⚕️ Reviewer B (Medical Informatics & Clinical AI Specialist)
**Score:** `5 / 10` (Reject / Major Revision)  
**Confidence:** `5 / 5` (Very High)

#### Summary of Clinical Contribution:
The authors propose an interactive questioning mechanism to disambiguate rare genetic disease differential diagnoses by querying unobserved Human Phenotype Ontology terms.

#### Critical Clinical & Semantic Vulnerabilities:
1. **The Dangerous Fallacy of Counterfactual Negative Evidence ($\lambda = 0.50$):**
   - In the benchmark simulation, if a queried phenotype $h^*$ is not in the ground-truth disease profile $\Phi(T)$, the simulator returns $y = \text{Absent}$, and every competitor disease possessing $h^*$ is penalized by $-\lambda \text{IC}(h^*)$.
   - **Fatal Clinical Reality:** In clinical genetics, the absence of an HPO annotation in a patient's chart does NOT mean the patient definitely lacks the symptom. It often means the clinician *did not examine or document it*. Treating non-documented phenotypes as confirmed negative findings heavily inflates the simulated discriminative power of negative evidence.
   - **Required Fix:** The manuscript must explicitly state that negative answers are *counterfactual simulated negative observations* under the HPOA disease representation, and discuss the sensitivity of $\lambda$.
2. **Exploratory Nature of the 100-Term Feasibility Audit:**
   - Table 7 claims that $88\%$ of queried terms are "bedside assessable." Who performed this annotation? Was there double-blind clinical annotation with Cohen's $\kappa$ inter-rater agreement? If this was classified post-hoc by the authors without independent board-certified clinical geneticists, it must be explicitly labeled an **exploratory feasibility audit**, not presented as proven clinical feasibility.
3. **Ambiguity Between Patient History vs. Physical Findings:**
   - Dysmorphic terms like "HP:6000029 Specific facial dysmorphism" or "HP:0011641 Abnormal nasal morphology" cannot simply be asked to a patient as a questionnaire item. They require specialized dysmorphology training by a medical geneticist.

---

### 🧐 Reviewer C (Skeptical General Reviewer)
**Score:** `6 / 10` (Borderline / Weak Accept)  
**Confidence:** `4 / 5` (High)

#### Summary & Skeptical Challenge:
The paper shows that if you ask a patient 5 clarifying questions, you can diagnose their rare disease with higher accuracy ($68.62\% \rightarrow 85.00\%$). 

#### Fundamental Conceptual Attack:
1. **"Is this Result Trivial? Asking for More Data Improves Retrieval":**
   - *"If I have a search engine and the user types 5 more keywords, the search results improve. Why is this a major scientific discovery?"*
   - **How the manuscript must defend itself:** The scientific contribution is NOT that more data helps. The contribution is:
     - (a) Demonstrating that static ranking failures are overwhelmingly caused by an *input-sparsity bottleneck* (where 98.6% of errors share identical observed feature subsets) rather than ranking capacity;
     - (b) Proving that blind information-theoretic feature acquisition (without label leakage) captures $>70\%$ of an omniscient oracle's potential in 5 steps;
     - (c) Discovering that the marginal benefit of active acquisition scales with candidate universe size (tripling from $+5.47\text{ pp}$ to $+16.38\text{ pp}$).
2. **Title Overclaim: "Across 12,956 Diseases":**
   - The paper evaluates **121 unseen diseases** (1,880 cases) against a candidate retrieval set of 12,956 diseases. A reader could easily misunderstand the title to mean that the system was tested on patients representing all 12,956 diseases.
   - **Recommended Title Change:** *"Active Phenotype Acquisition Resolves the Observation Bottleneck in Zero-Shot Rare Disease Prioritization over a 12,956-Disease Candidate Universe"*.

---

## 2. TOP FIVE FATAL REJECTION RISKS & REMEDIATION PLAN

| Risk # | Rejection Vulnerability | Reviewer Target | Severity | Remediation Strategy |
| :---: | :--- | :---: | :---: | :--- |
| **Risk 1** | **Triviality Objection:** "Asking more questions obviously improves accuracy; what is novel?" | Reviewers A & C | **CRITICAL** | Include the **Random Candidate-Pool baseline ($74.30\%$)** in Table 3 to prove Policy A ($85.00\%$) provides mathematically superior disambiguation, and frame the paper around the input-sparsity bottleneck. |
| **Risk 2** | **Title / Scope Misunderstanding:** Title implies 12,956 evaluated test diseases rather than 121 test diseases across a 12,956 candidate universe. | Reviewer C | **CRITICAL** | Retitle to: *"over a 12,956-Disease Candidate Universe"*. Clarify test cohort ($N=1,880, 121$ unseen diseases) in Abstract & Intro. |
| **Risk 3** | **Negative Evidence Simulation Leakage:** Assuming unrecorded symptoms are true negatives ($\lambda=0.50$). | Reviewer B | **MAJOR** | Formally define negative answers as *simulated counterfactual observations under benchmark representations* and discuss missingness bounds. |
| **Risk 4** | **Unvalidated Clinical Feasibility:** Claiming 88% bedside feasibility without clinical inter-rater reliability. | Reviewer B | **MAJOR** | Explicitly label Table 7 as an *exploratory clinical feasibility audit* and add extensive clinical limitations. |
| **Risk 5** | **Overgeneralized Dense Failure Claim:** Claiming "dense representations fail" based on one 256-d bi-encoder. | Reviewer A | **MAJOR** | Precisely specify the exact architecture: *"Unanchored Dense Two-Tower Transformer (256-d InfoNCE)"* and constrain the claim. |

---

## 3. DETAILED CLAIM-EVIDENCE AUDIT MATRIX

| # | Manuscript Claim in Draft v1.0 | Evidence & Provenance | Directly Supported? | Reviewer Risk Level | Required Modification |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **1** | *"Active phenotype acquisition resolves the observation ceiling across 12,956 diseases"* | 1,880 test cases, 121 unseen diseases, 12,956 candidate universe | **YELLOW** (Scope Ambiguity) | **HIGH** | Change "across" to "over a 12,956-disease candidate universe". Change "ceiling" to "observation bottleneck". |
| **2** | *"Unanchored Dense Embeddings fail zero-shot (0.00% Hits@1)"* | 256-d Two-Tower InfoNCE model | **YELLOW** (Overgeneralization) | **MEDIUM** | Specifically name the exact architecture and scope the conclusion. |
| **3** | *"Zero cases were worsened after active acquisition"* | 1,290 / 1,290 preserved on locked test | **GREEN** (Empirically True) | **MEDIUM** | Emphasize this as an empirical stability observation on clinical profiles, not a universal theorem. |
| **4** | *"Active acquisition exhibits a monotonic scaling relationship with universe size"* | 5 seeds across $N \in [100, 12956]$ | **GREEN** (Directly Supported) | **LOW** | Use "empirical scaling relationship" rather than "universal scaling law". |
| **5** | *"88% of queried terms are assessable at the bedside"* | 100-term exploratory categorization | **YELLOW** (Unvalidated) | **HIGH** | Explicitly label as an exploratory feasibility audit. |
| **6** | *"Policy A resolves 52.20% of residual errors in 5 queries"* | $308 / 590$ resolved on locked test set | **GREEN** (Directly Supported) | **LOW** | Retain canonical numbers ($85.00\%$, 308 resolved). |

---

## 4. SIMULATED REVIEW COMMITTEE DECISION: [ YELLOW — MAJOR REVISION REQUIRED ]

The paper contains strong, reproducible empirical results ($+16.38\text{ pp}$ gain, $p < 10^{-65}$, multi-seed universe scaling), but **requires concrete manuscript revisions to prevent fatal reviewer pushback regarding terminology, scope, random baseline inclusion, and clinical realism**.
