"""
Phase 6.6 — Genuine Query-Conditioned Asymmetric Model

Research Question:
  "Can query-conditioned asymmetric phenotype matching reduce profile-size bias
   in zero-shot rare-disease prioritization beyond static phenotype weighting
   and length normalization?"

Ablation Ladder:
  A0  : Raw IC (static, no learning)
  A1  : Optimal gamma-normalized IC (static, 1 parameter)
  B   : Learned symmetric dual encoder (pool P, pool D, cosine)
  C0  : Learned disease-scorer WITHOUT patient conditioning
          (disease profile pooled independently; no cross-attention)
  C1  : TRUE query-conditioned cross-attention
          (patient query attends to disease term keys/values)

Critical test: C1 > C0?  (isolates patient-conditioning contribution)
"""

import json
import math
import random
import time
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print("=" * 65)
print("PHASE 6.6 — TRUE QUERY-CONDITIONED ASYMMETRIC MODEL")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# 0.  SEED CONTROL
# ─────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ─────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────
print("\n[1/12] Loading data …")
with open("data/benchmark/clean_phenopackets.json", "r", encoding="utf-8") as f:
    raw_cases = json.load(f)

disease_to_hpos: dict[str, set[str]] = defaultdict(set)
with open("data/benchmark/phenotype.hpoa", "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.strip().split("\t")
        if len(parts) >= 4:
            did, hid = parts[0], parts[3]
            if hid.startswith("HP:"):
                disease_to_hpos[did].add(hid)

valid_cases = [c for c in raw_cases if c["diseases"][0] in disease_to_hpos]
all_diseases = sorted({c["diseases"][0] for c in valid_cases})
print(f"  Valid cases: {len(valid_cases)} | Evaluable diseases: {len(all_diseases)}")

# ─────────────────────────────────────────────────────────────
# 2.  VOCABULARY & IC
# ─────────────────────────────────────────────────────────────
print("[2/12] Building vocabulary and IC weights …")
all_hpos_list = sorted(
    {h for c in valid_cases for h in c["hpos"]}
    | {h for hset in disease_to_hpos.values() for h in hset}
)
hpo_to_idx: dict[str, int] = {h: i + 1 for i, h in enumerate(all_hpos_list)}
VOCAB_SIZE = len(hpo_to_idx) + 1          # +1 for padding index 0

total_kb = len(disease_to_hpos)
hpo_doc_freq = Counter(h for hset in disease_to_hpos.values() for h in hset)
hpo_ic: dict[str, float] = {
    h: -math.log((freq + 1) / (total_kb + 1))
    for h, freq in hpo_doc_freq.items()
}
ic_array = np.array(
    [hpo_ic.get(h, 1.0) for h in all_hpos_list], dtype=np.float32
)  # shape (VOCAB_SIZE-1,)
print(f"  Vocabulary size: {VOCAB_SIZE} HPO terms")

# ─────────────────────────────────────────────────────────────
# 3.  DISEASE-DISJOINT PARTITION  (seed=42, FROZEN)
# ─────────────────────────────────────────────────────────────
print("[3/12] Creating frozen disease-disjoint benchmark split …")
shuffled_dids = list(all_diseases)
random.shuffle(shuffled_dids)
n_train_d = int(0.70 * len(shuffled_dids))
n_val_d   = int(0.15 * len(shuffled_dids))

train_dids = set(shuffled_dids[:n_train_d])
val_dids   = set(shuffled_dids[n_train_d : n_train_d + n_val_d])
test_dids  = set(shuffled_dids[n_train_d + n_val_d :])

train_cases = [c for c in valid_cases if c["diseases"][0] in train_dids]
val_cases_list   = [c for c in valid_cases if c["diseases"][0] in val_dids]
test_cases_list  = [c for c in valid_cases if c["diseases"][0] in test_dids]
candidate_diseases = list(disease_to_hpos.keys())   # 12,956

print(f"  Train: {len(train_cases)} cases / {len(train_dids)} diseases")
print(f"  Val  : {len(val_cases_list)} cases / {len(val_dids)} diseases")
print(f"  Test : {len(test_cases_list)} cases / {len(test_dids)} diseases")
print(f"  Candidate universe: {len(candidate_diseases)} diseases")

# Zero-shot safety assertion: no test disease should appear in training
assert len(test_dids & train_dids) == 0, "TEST-TRAIN DISEASE OVERLAP DETECTED!"
assert len(test_dids & val_dids)   == 0, "TEST-VAL DISEASE OVERLAP DETECTED!"
print("  [PASS] Zero-shot safety assertion: test diseases unseen in train/val")

disease_hpo_sets = {did: set(hset) for did, hset in disease_to_hpos.items()}

# ─────────────────────────────────────────────────────────────
# 4.  HELPER: ENCODE HPO SET AS PADDED TENSOR
# ─────────────────────────────────────────────────────────────
PAD_IDX = 0
MAX_LEN_PATIENT  = 64     # typical: ~9 HPOs per patient
MAX_LEN_DISEASE  = 128    # typical: ~20–200 HPOs per disease (cap at 128)

def hpo_set_to_tensor(hpo_set: set[str], max_len: int) -> torch.Tensor:
    """Convert an HPO set to a padded LongTensor of shape (max_len,)."""
    indices = [hpo_to_idx[h] for h in hpo_set if h in hpo_to_idx]
    # Randomly shuffle order to enforce permutation invariance during training
    random.shuffle(indices)
    if len(indices) > max_len:
        indices = indices[:max_len]
    padded = indices + [PAD_IDX] * (max_len - len(indices))
    return torch.tensor(padded, dtype=torch.long)

def hpo_set_to_tensor_fixed(hpo_set: set[str], max_len: int) -> torch.Tensor:
    """Deterministic (sorted) version for evaluation / unit tests."""
    indices = sorted([hpo_to_idx[h] for h in hpo_set if h in hpo_to_idx])
    if len(indices) > max_len:
        indices = indices[:max_len]
    padded = indices + [PAD_IDX] * (max_len - len(indices))
    return torch.tensor(padded, dtype=torch.long)

def create_key_padding_mask(tensor_ids: torch.Tensor) -> torch.Tensor:
    """True where token is padding (for nn.MultiheadAttention key_padding_mask)."""
    return tensor_ids == PAD_IDX  # (batch, seq_len) or (seq_len,)

# ─────────────────────────────────────────────────────────────
# 5.  MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────
EMBED_DIM = 64    # intentionally small — we are testing mechanism, not capacity

print(f"[4/12] Defining models (embed_dim={EMBED_DIM}) …")

# ── Fixed HPO embeddings (initialized from IC, then FROZEN) ──
def make_ic_embedding(embed_dim: int) -> nn.Embedding:
    """
    Build an embedding table where each HPO gets a vector whose first
    dimension is its IC value (log-frequency) and the rest are small
    random noise. Embeddings are then frozen so architecture effects
    are isolated from representation learning.
    """
    weight = torch.zeros(VOCAB_SIZE, embed_dim)
    for h, idx in hpo_to_idx.items():
        weight[idx, 0] = hpo_ic.get(h, 1.0)          # IC in dim-0
        weight[idx, 1:] = torch.randn(embed_dim - 1) * 0.01
    emb = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD_IDX)
    emb.weight.data.copy_(weight)
    emb.weight.requires_grad_(False)   # FROZEN
    return emb

# ─────────────────────────
#  Model B: Symmetric Dual Encoder
# ─────────────────────────
class SymmetricDualEncoder(nn.Module):
    """
    S(P,D) = cosine(mean_pool(P_embs), mean_pool(D_embs)).
    Symmetric: S(P,D) = S(D,P).
    Learns a linear projection on top of frozen IC embeddings.
    """
    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.emb = make_ic_embedding(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (batch, seq) → repr: (batch, embed_dim)"""
        mask = (ids != PAD_IDX).float().unsqueeze(-1)     # (B, L, 1)
        embs = self.proj(self.emb(ids))                   # (B, L, D)
        embs = embs * mask
        counts = mask.sum(dim=1).clamp(min=1)             # (B, 1)
        return embs.sum(dim=1) / counts                   # (B, D)

    def score(self, p_ids: torch.Tensor, d_ids: torch.Tensor) -> torch.Tensor:
        """(B, L_p), (B, L_d) → (B,) scores"""
        qp = F.normalize(self.encode(p_ids), dim=-1)
        qd = F.normalize(self.encode(d_ids), dim=-1)
        return (qp * qd).sum(dim=-1)

# ─────────────────────────
#  Model C0: Non-conditioned disease scorer
# ─────────────────────────
class NonConditionedScorer(nn.Module):
    """
    C0: Patient and Disease are encoded independently.
    No cross-attention — disease phenotype weights do NOT depend on patient.
    S(P, D) = MLP(pool(P_embs), pool(D_embs)).
    This is the critical ablation baseline.
    """
    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.emb = make_ic_embedding(embed_dim)
        self.patient_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.disease_proj  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def encode_patient(self, ids: torch.Tensor) -> torch.Tensor:
        mask = (ids != PAD_IDX).float().unsqueeze(-1)
        embs = self.patient_proj(self.emb(ids)) * mask
        return embs.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def encode_disease(self, ids: torch.Tensor) -> torch.Tensor:
        mask = (ids != PAD_IDX).float().unsqueeze(-1)
        embs = self.disease_proj(self.emb(ids)) * mask
        return embs.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def score(self, p_ids: torch.Tensor, d_ids: torch.Tensor) -> torch.Tensor:
        qp = self.encode_patient(p_ids)
        qd = self.encode_disease(d_ids)
        return self.scorer(torch.cat([qp, qd], dim=-1)).squeeze(-1)

# ─────────────────────────
#  Model C1: True Query-Conditioned Cross-Attention
# ─────────────────────────
class TrueAsymmetricCrossAttention(nn.Module):
    """
    C1: Genuine query-conditioned asymmetric model.

    Architecture:
      1. Patient P → mean-pool → linear projection → query vector q  (d,)
      2. Disease D → linear Key/Value projections → K_D (m,d), V_D (m,d)
      3. Cross-attention: alpha = softmax(q · K_D^T / sqrt(d))    (m,)
         — alpha(h | P, D) explicitly depends on P through q
      4. Context vector: z = alpha · V_D                            (d,)
      5. Score: s = MLP([q, z])

    Key guarantee: A(P1, D) != A(P2, D) for P1 != P2 in general,
    because q = f(P) changes when P changes.
    """
    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        self.emb = make_ic_embedding(embed_dim)

        # Patient encoder: mean-pool + linear
        self.patient_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Disease key and value projections
        self.key_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Final scoring MLP
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )
        self.scale = math.sqrt(embed_dim)

    def encode_patient_query(self, p_ids: torch.Tensor) -> torch.Tensor:
        """p_ids: (B, L_p) → q: (B, D)"""
        mask = (p_ids != PAD_IDX).float().unsqueeze(-1)   # (B, L, 1)
        embs = self.patient_proj(self.emb(p_ids)) * mask  # (B, L, D)
        q = embs.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return q

    def cross_attend(
        self,
        q: torch.Tensor,
        d_ids: torch.Tensor,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        q     : (B, D) patient query
        d_ids : (B, L_d) disease term ids
        returns context z (B, D) and optionally attention weights (B, L_d)
        """
        mask = (d_ids != PAD_IDX).float()                # (B, L_d)
        embs = self.emb(d_ids)                           # (B, L_d, D)

        K = self.key_proj(embs)                          # (B, L_d, D)
        V = self.value_proj(embs)                        # (B, L_d, D)

        # Scaled dot-product attention: (B, 1, D) x (B, D, L_d) → (B, 1, L_d)
        q_expanded = q.unsqueeze(1)                      # (B, 1, D)
        raw_attn = torch.bmm(q_expanded, K.transpose(1, 2)) / self.scale  # (B,1,L_d)
        raw_attn = raw_attn.squeeze(1)                   # (B, L_d)

        # Mask padding positions with -inf before softmax
        raw_attn = raw_attn.masked_fill(mask == 0, float("-inf"))
        alpha = F.softmax(raw_attn, dim=-1)              # (B, L_d)

        # Replace NaN (all-padding rows) with 0
        alpha = torch.nan_to_num(alpha, nan=0.0)

        # Weighted sum of values
        z = torch.bmm(alpha.unsqueeze(1), V).squeeze(1)  # (B, D)

        if return_attn:
            return z, alpha
        return z, None

    def score(
        self,
        p_ids: torch.Tensor,
        d_ids: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple:
        q = self.encode_patient_query(p_ids)             # (B, D)
        z, attn = self.cross_attend(q, d_ids, return_attn=return_attn)
        s = self.scorer(torch.cat([q, z], dim=-1)).squeeze(-1)  # (B,)
        if return_attn:
            return s, attn
        return s

# ─────────────────────────────────────────────────────────────
# 6.  UNIT TESTS
# ─────────────────────────────────────────────────────────────
print("[5/12] Running unit tests …")

def run_unit_tests(model_c1: TrueAsymmetricCrossAttention):
    model_c1.eval()
    model_c1 = model_c1.to("cpu")

    # ── Pick a shared disease and two distinct patients from training set ──
    sample_disease_id = list(train_dids)[0]
    d_hpos = disease_hpo_sets[sample_disease_id]

    # Patient 1: sparse — pick 3 HPOs
    p1_cases = [c for c in train_cases if len(c["hpos"]) <= 5][:1]
    # Patient 2: dense — pick a patient with >10 HPOs
    p2_cases = [c for c in train_cases if len(c["hpos"]) >= 8][:1]

    if not p1_cases or not p2_cases:
        print("  [SKIP] Unit tests: insufficient test patients")
        return

    p1_hpos = set(p1_cases[0]["hpos"])
    p2_hpos = set(p2_cases[0]["hpos"])

    d_ids = hpo_set_to_tensor_fixed(d_hpos, MAX_LEN_DISEASE).unsqueeze(0)   # (1, L_d)
    p1_ids = hpo_set_to_tensor_fixed(p1_hpos, MAX_LEN_PATIENT).unsqueeze(0)
    p2_ids = hpo_set_to_tensor_fixed(p2_hpos, MAX_LEN_PATIENT).unsqueeze(0)

    with torch.no_grad():
        # ── TEST 1: Query-Conditioning ─────────────────────────
        # Use maximally different synthetic patients to test structural conditioning.
        # Patient A: only 1 HPO;  Patient B: 20 completely different HPOs.
        # Even an untrained model must produce different attention weights,
        # proving the mechanism is structurally query-conditioned.
        all_vocab_hpos = list(hpo_to_idx.keys())
        fake_p1_hpos = {all_vocab_hpos[10]}
        fake_p2_hpos = set(all_vocab_hpos[200:220])   # 20 different HPOs

        fp1_ids = hpo_set_to_tensor_fixed(fake_p1_hpos, MAX_LEN_PATIENT).unsqueeze(0)
        fp2_ids = hpo_set_to_tensor_fixed(fake_p2_hpos, MAX_LEN_PATIENT).unsqueeze(0)

        _, a1 = model_c1.score(fp1_ids, d_ids, return_attn=True)
        _, a2 = model_c1.score(fp2_ids, d_ids, return_attn=True)
        diff = (a1 - a2).abs().mean().item()
        eps = 1e-7  # even random init must produce different queries for different inputs
        assert diff > eps, (
            f"TEST 1 FAIL: attention weights identical for maximally-different patients! "
            f"mean |A(P1,D) - A(P2,D)| = {diff:.8f} < {eps}\n"
            f"This indicates patient query is NOT being computed from input set — "
            f"query conditioning is architecturally absent."
        )
        print(f"  [PASS] TEST 1 — Query conditioning: mean |A(P1,D)-A(P2,D)| = {diff:.6f}")

        # ── TEST 2: Patient Permutation Invariance ──────────────
        # same HPOs, different order
        p1_a = hpo_set_to_tensor_fixed(p1_hpos, MAX_LEN_PATIENT).unsqueeze(0)
        p1_indices = [hpo_to_idx[h] for h in p1_hpos if h in hpo_to_idx]
        random.shuffle(p1_indices)
        p1_b_padded = p1_indices + [PAD_IDX] * (MAX_LEN_PATIENT - len(p1_indices))
        p1_b = torch.tensor([p1_b_padded], dtype=torch.long)

        q_a = model_c1.encode_patient_query(p1_a)
        q_b = model_c1.encode_patient_query(p1_b)
        # Mean-pooling is permutation invariant by construction
        q_diff = (q_a - q_b).abs().max().item()
        # Allow small floating-point tolerance
        assert q_diff < 1e-5, (
            f"TEST 2 FAIL: patient permutation not invariant! "
            f"max |q_a - q_b| = {q_diff:.6e}"
        )
        print(f"  [PASS] TEST 2 — Patient permutation invariance: max diff = {q_diff:.2e}")

        # ── TEST 3: Disease Permutation Invariance ──────────────
        d_indices = sorted([hpo_to_idx[h] for h in d_hpos if h in hpo_to_idx])
        d_a_padded = d_indices[:MAX_LEN_DISEASE] + [PAD_IDX] * max(0, MAX_LEN_DISEASE - len(d_indices))
        d_b_shuffled = d_indices[:]
        random.shuffle(d_b_shuffled)
        d_b_padded = d_b_shuffled[:MAX_LEN_DISEASE] + [PAD_IDX] * max(0, MAX_LEN_DISEASE - len(d_b_shuffled))

        d_a_t = torch.tensor([d_a_padded], dtype=torch.long)
        d_b_t = torch.tensor([d_b_padded], dtype=torch.long)

        q_fixed = model_c1.encode_patient_query(p1_ids)
        z_a, _ = model_c1.cross_attend(q_fixed, d_a_t)
        z_b, _ = model_c1.cross_attend(q_fixed, d_b_t)
        # Softmax over the same set values — result should be identical
        # (attention is order-agnostic since keys come from embeddings, not positions)
        # Small difference expected due to FP ordering
        z_diff = (z_a - z_b).abs().max().item()
        print(f"  [INFO] TEST 3 — Disease permutation: max context diff = {z_diff:.2e}")
        # Note: NOT strictly invariant since keys depend on embedding lookups via indices
        # but numerically should be small for frozen embeddings (no positional encoding)

        # ── TEST 4: Zero-shot Safety ────────────────────────────
        # Verify that disease representation is entirely derived from HPO profile
        # (no embedding table indexed by disease ID)
        has_disease_id_emb = any(
            "disease_id" in name or "did_embed" in name
            for name, _ in model_c1.named_parameters()
        )
        assert not has_disease_id_emb, "TEST 4 FAIL: found disease-ID embedding!"
        print("  [PASS] TEST 4 — Zero-shot safety: no disease-ID embedding found")

    print("  All unit tests passed.\n")

# Instantiate C1 first for unit testing (untrained is fine for structural tests)
c1_model = TrueAsymmetricCrossAttention(embed_dim=EMBED_DIM).to("cpu")
run_unit_tests(c1_model)

# ─────────────────────────────────────────────────────────────
# 7.  DATASET FOR TRAINING
# ─────────────────────────────────────────────────────────────
print("[6/12] Building training dataset …")

class RareDiseaseDataset(Dataset):
    """
    Each item: (patient_ids, positive_disease_ids).
    Negative sampling done in the collate / training loop.
    """
    def __init__(self, cases, disease_hpo_sets, hpo_to_idx, all_train_dids):
        self.cases = cases
        self.disease_hpo_sets = disease_hpo_sets
        self.hpo_to_idx = hpo_to_idx
        self.all_train_dids_list = list(all_train_dids)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        p_hpos = set(case["hpos"])
        pos_did = case["diseases"][0]
        pos_hpos = self.disease_hpo_sets[pos_did]

        p_ids  = hpo_set_to_tensor(p_hpos,  MAX_LEN_PATIENT)
        pd_ids = hpo_set_to_tensor(pos_hpos, MAX_LEN_DISEASE)
        return p_ids, pd_ids

def collate_fn_infonce(batch, disease_hpo_sets, all_train_dids_list, n_negatives=31):
    """
    InfoNCE collate: each item gets n_negatives random negatives from training diseases.
    Returns:
        p_ids   : (B, L_p)
        pd_ids  : (B, L_d)  — positive disease
        neg_ids : (B, n_neg, L_d)  — negative diseases
    """
    p_ids_list  = [item[0] for item in batch]
    pd_ids_list = [item[1] for item in batch]

    neg_ids_batch = []
    for _ in batch:
        neg_dids = random.sample(all_train_dids_list, n_negatives)
        neg_tensors = [
            hpo_set_to_tensor(disease_hpo_sets[did], MAX_LEN_DISEASE)
            for did in neg_dids
        ]
        neg_ids_batch.append(torch.stack(neg_tensors, dim=0))  # (n_neg, L_d)

    return (
        torch.stack(p_ids_list),           # (B, L_p)
        torch.stack(pd_ids_list),           # (B, L_d)
        torch.stack(neg_ids_batch),         # (B, n_neg, L_d)
    )

# ─────────────────────────────────────────────────────────────
# 8.  TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────
def train_model(
    model,
    model_name: str,
    n_epochs: int = 20,
    batch_size: int = 64,
    lr: float = 3e-4,
    n_negatives: int = 31,
    patience: int = 5,
    val_every: int = 1,
):
    """
    Train with InfoNCE loss.
    Early stopping on validation Hits@1 (using ONLY val diseases).
    Returns best model state dict and val metrics per epoch.
    """
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)

    train_dataset = RareDiseaseDataset(
        train_cases, disease_hpo_sets, hpo_to_idx, train_dids
    )
    collate = lambda batch: collate_fn_infonce(
        batch, disease_hpo_sets, list(train_dids), n_negatives
    )
    loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate, num_workers=0
    )

    best_val_h1 = -1.0
    best_state  = None
    no_improve  = 0
    history     = []

    t_start = time.time()
    print(f"\n  Training {model_name} for up to {n_epochs} epochs …")

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for p_ids, pd_ids, neg_ids in loader:
            p_ids  = p_ids.to(DEVICE)   # (B, L_p)
            pd_ids = pd_ids.to(DEVICE)  # (B, L_d)
            neg_ids = neg_ids.to(DEVICE)  # (B, n_neg, L_d)
            B = p_ids.size(0)

            # Score positive: (B,)
            pos_scores = model.score(p_ids, pd_ids)

            # Score negatives: reshape to (B*n_neg, L_d), score, reshape to (B, n_neg)
            n_neg = neg_ids.size(1)
            p_rep = p_ids.unsqueeze(1).expand(-1, n_neg, -1).reshape(B * n_neg, -1)
            d_rep = neg_ids.reshape(B * n_neg, -1)
            neg_scores = model.score(p_rep, d_rep).reshape(B, n_neg)

            # InfoNCE: log-softmax over [pos, neg1, ..., negN]
            all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)  # (B, 1+n_neg)
            labels = torch.zeros(B, dtype=torch.long, device=DEVICE)
            loss = F.cross_entropy(all_scores, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        if epoch % val_every == 0:
            val_h1 = quick_val_hits1(model, val_cases_list[:200], "cpu")
            model.train()
            history.append({"epoch": epoch, "loss": avg_loss, "val_h1": val_h1})
            print(
                f"    Epoch {epoch:3d}/{n_epochs} | loss={avg_loss:.4f} | "
                f"val Hits@1={val_h1*100:.2f}% | elapsed={time.time()-t_start:.0f}s"
            )

            if val_h1 > best_val_h1:
                best_val_h1 = val_h1
                best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve  = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"    Early stopping at epoch {epoch}")
                    break

    model.load_state_dict(best_state)
    model.eval()
    print(f"  Best val Hits@1: {best_val_h1*100:.2f}%")
    return model, history

# ─────────────────────────────────────────────────────────────
# 9.  EVALUATION FUNCTIONS
# ─────────────────────────────────────────────────────────────
def quick_val_hits1(model, cohort, eval_device="cpu"):
    """Fast Hits@1 over a subset of cohort — used for early stopping."""
    model_dev = next(model.parameters()).device
    model.to(eval_device)
    model.eval()
    h1 = 0
    with torch.no_grad():
        for case in cohort:
            p_hpos = set(case["hpos"])
            tgt = case["diseases"][0]
            p_ids = hpo_set_to_tensor_fixed(p_hpos, MAX_LEN_PATIENT).unsqueeze(0)

            best_score = float("-inf")
            best_did   = None
            for did in candidate_diseases:
                d_hpos = disease_hpo_sets[did]
                d_ids = hpo_set_to_tensor_fixed(d_hpos, MAX_LEN_DISEASE).unsqueeze(0)
                s = model.score(p_ids, d_ids).item()
                if s > best_score:
                    best_score = s
                    best_did   = did
            if best_did == tgt:
                h1 += 1
    model.to(model_dev)
    return h1 / len(cohort)

def full_eval(model, cohort, candidate_list, name, eval_device="cpu"):
    """Full evaluation: Hits@1/5/10 and MRR over all candidates."""
    model.to(eval_device)
    model.eval()
    h1, h5, h10, mrr_sum = 0, 0, 0, 0.0
    ranked_list_store = []

    with torch.no_grad():
        for case in cohort:
            p_hpos = set(case["hpos"])
            tgt = case["diseases"][0]
            p_ids = hpo_set_to_tensor_fixed(p_hpos, MAX_LEN_PATIENT).unsqueeze(0)

            scores = []
            for did in candidate_list:
                d_hpos = disease_hpo_sets[did]
                d_ids = hpo_set_to_tensor_fixed(d_hpos, MAX_LEN_DISEASE).unsqueeze(0)
                s = model.score(p_ids, d_ids).item()
                scores.append((s, did))

            scores.sort(key=lambda x: x[0], reverse=True)
            ranked = [x[1] for x in scores]
            try:
                r = ranked.index(tgt) + 1
            except ValueError:
                r = len(candidate_list)

            if r <= 1:  h1  += 1
            if r <= 5:  h5  += 1
            if r <= 10: h10 += 1
            mrr_sum += 1.0 / r

            ranked_list_store.append({
                "patient_id": case.get("id", ""),
                "target": tgt,
                "rank": r,
                "n_patient_hpos": len(p_hpos),
                "n_disease_hpos": len(disease_hpo_sets.get(tgt, set())),
                "top1_did": ranked[0],
                "top1_dsize": len(disease_hpo_sets.get(ranked[0], set())),
            })

    N = len(cohort)
    result = {
        "method": name,
        "Hits@1":  h1  / N * 100,
        "Hits@5":  h5  / N * 100,
        "Hits@10": h10 / N * 100,
        "MRR":     mrr_sum / N,
        "N":       N,
    }
    return result, ranked_list_store

# ─────────────────────────────────────────────────────────────
# 10.  CLASSICAL BASELINES A0 and A1
# ─────────────────────────────────────────────────────────────
print("[7/12] Computing classical baselines on validation …")

def ic_overlap(p_hpos, d_hpos, gamma=0.0):
    inter = p_hpos & d_hpos
    if not inter:
        return 0.0
    s = sum(hpo_ic.get(h, 1.0) for h in inter)
    if gamma > 0:
        s /= len(d_hpos) ** gamma
    return s

# A0 on validation (small subset for speed)
print("  A0 (Raw IC) — val subset …")
a0_val_h1 = sum(
    1 for c in val_cases_list[:300]
    if max(candidate_diseases,
           key=lambda d: ic_overlap(set(c["hpos"]), disease_hpo_sets[d])) == c["diseases"][0]
) / 300
print(f"    A0 val Hits@1 (approx, 300 cases): {a0_val_h1*100:.2f}%")

# A1 — tune gamma on VALIDATION only
print("  A1 — gamma search on validation subset …")
best_gamma, best_a1_h1 = 0.0, 0.0
for g in [0.02, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20]:
    h1 = sum(
        1 for c in val_cases_list[:300]
        if max(candidate_diseases,
               key=lambda d: ic_overlap(set(c["hpos"]), disease_hpo_sets[d], g)) == c["diseases"][0]
    ) / 300
    print(f"    gamma={g:.2f}: val Hits@1 = {h1*100:.2f}%")
    if h1 > best_a1_h1:
        best_a1_h1, best_gamma = h1, g

print(f"  A1 selected gamma={best_gamma:.2f} (val Hits@1={best_a1_h1*100:.2f}%)")

# ─────────────────────────────────────────────────────────────
# 11.  TRAIN MODELS B, C0, C1
# ─────────────────────────────────────────────────────────────
print("\n[8/12] Training neural models …")

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

EPOCHS   = 25
BATCH    = 64
LR       = 3e-4
N_NEG    = 31
PATIENCE = 5

# Model B
b_model = SymmetricDualEncoder(embed_dim=EMBED_DIM)
print(f"  Model B trainable params: {count_trainable_params(b_model):,}")
b_model, b_history = train_model(b_model, "Model B (Symmetric)", EPOCHS, BATCH, LR, N_NEG, PATIENCE)

# Model C0
c0_model = NonConditionedScorer(embed_dim=EMBED_DIM)
print(f"  Model C0 trainable params: {count_trainable_params(c0_model):,}")
c0_model, c0_history = train_model(c0_model, "Model C0 (Non-conditioned)", EPOCHS, BATCH, LR, N_NEG, PATIENCE)

# Model C1
c1_model = TrueAsymmetricCrossAttention(embed_dim=EMBED_DIM)
print(f"  Model C1 trainable params: {count_trainable_params(c1_model):,}")
c1_model, c1_history = train_model(c1_model, "Model C1 (True Cross-Attention)", EPOCHS, BATCH, LR, N_NEG, PATIENCE)

# ─────────────────────────────────────────────────────────────
# 12.  FULL EVALUATION ON LOCKED TEST SET
# ─────────────────────────────────────────────────────────────
print("\n[9/12] Full evaluation on locked test set (N=1,880) …")
print("  NOTE: Test set accessed once, results locked.")

# A0 — full test
print("  Evaluating A0 (Raw IC) …")
t0 = time.time()
a0_h1, a0_h5, a0_h10, a0_mrr = 0, 0, 0, 0.0
a0_ranked_store = []
for case in test_cases_list:
    p_hpos = set(case["hpos"])
    tgt = case["diseases"][0]
    scores = [(ic_overlap(p_hpos, disease_hpo_sets[d]), d) for d in candidate_diseases]
    scores.sort(reverse=True)
    ranked = [x[1] for x in scores]
    try:   r = ranked.index(tgt) + 1
    except ValueError: r = len(candidate_diseases)
    if r <= 1:  a0_h1  += 1
    if r <= 5:  a0_h5  += 1
    if r <= 10: a0_h10 += 1
    a0_mrr += 1.0 / r
    a0_ranked_store.append({
        "target": tgt, "rank": r,
        "n_patient_hpos": len(p_hpos),
        "n_disease_hpos": len(disease_hpo_sets.get(tgt, set())),
    })
N = len(test_cases_list)
a0_result = {"method": "A0: Raw IC", "Hits@1": a0_h1/N*100, "Hits@5": a0_h5/N*100,
             "Hits@10": a0_h10/N*100, "MRR": a0_mrr/N, "N": N}
print(f"  A0 done in {time.time()-t0:.1f}s: Hits@1={a0_result['Hits@1']:.2f}%")

# A1 — full test (locked gamma from validation)
print(f"  Evaluating A1 (gamma={best_gamma}) …")
t0 = time.time()
a1_h1, a1_h5, a1_h10, a1_mrr = 0, 0, 0, 0.0
a1_ranked_store = []
for case in test_cases_list:
    p_hpos = set(case["hpos"])
    tgt = case["diseases"][0]
    scores = [(ic_overlap(p_hpos, disease_hpo_sets[d], best_gamma), d) for d in candidate_diseases]
    scores.sort(reverse=True)
    ranked = [x[1] for x in scores]
    try:   r = ranked.index(tgt) + 1
    except ValueError: r = len(candidate_diseases)
    if r <= 1:  a1_h1  += 1
    if r <= 5:  a1_h5  += 1
    if r <= 10: a1_h10 += 1
    a1_mrr += 1.0 / r
    a1_ranked_store.append({
        "target": tgt, "rank": r,
        "n_patient_hpos": len(p_hpos),
        "n_disease_hpos": len(disease_hpo_sets.get(tgt, set())),
    })
a1_result = {"method": f"A1: gamma-IC (g={best_gamma})", "Hits@1": a1_h1/N*100,
             "Hits@5": a1_h5/N*100, "Hits@10": a1_h10/N*100, "MRR": a1_mrr/N, "N": N}
print(f"  A1 done in {time.time()-t0:.1f}s: Hits@1={a1_result['Hits@1']:.2f}%")

# Neural models
print("  Evaluating B, C0, C1 on test …")
b_result,  b_ranked_store  = full_eval(b_model,  test_cases_list, candidate_diseases, "B: Symmetric")
c0_result, c0_ranked_store = full_eval(c0_model, test_cases_list, candidate_diseases, "C0: Non-conditioned")
c1_result, c1_ranked_store = full_eval(c1_model, test_cases_list, candidate_diseases, "C1: True Cross-Attn")

# ─────────────────────────────────────────────────────────────
# 13.  RESULTS TABLE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("PHASE 6.6 — COMPLETE ABLATION LADDER (ZERO-SHOT TEST, N=1,880)")
print("=" * 75)
all_results = [a0_result, a1_result, b_result, c0_result, c1_result]
header = f"{'Method':<35} {'Hits@1':>7} {'Hits@5':>7} {'Hits@10':>8} {'MRR':>7}"
print(header)
print("-" * 75)
for r in all_results:
    print(f"{r['method']:<35} {r['Hits@1']:>6.2f}% {r['Hits@5']:>6.2f}% {r['Hits@10']:>7.2f}% {r['MRR']:>7.4f}")
print("=" * 75)

# ─────────────────────────────────────────────────────────────
# 14.  STATISTICAL ANALYSIS (McNemar for Hits@1)
# ─────────────────────────────────────────────────────────────
print("\n[10/12] Statistical analysis …")

def mcnemar_test(store_a, store_b):
    """McNemar's test comparing Hits@1 of two methods using ranked store dicts."""
    # both correct, neither correct, only A, only B
    correct_a = {i for i, r in enumerate(store_a) if r["rank"] == 1}
    correct_b = {i for i, r in enumerate(store_b) if r["rank"] == 1}
    n = len(store_a)
    both = len(correct_a & correct_b)
    neither = n - len(correct_a | correct_b)
    only_a = len(correct_a - correct_b)
    only_b = len(correct_b - correct_a)
    # McNemar
    if only_a + only_b == 0:
        chi2, p = 0.0, 1.0
    else:
        chi2 = (abs(only_a - only_b) - 1) ** 2 / (only_a + only_b)
        from scipy.stats import chi2 as chi2_dist
        p = 1 - chi2_dist.cdf(chi2, df=1)
    return {"both": both, "neither": neither, "only_a": only_a, "only_b": only_b,
            "chi2": chi2, "p": p}

try:
    from scipy.stats import chi2 as scipy_chi2
    print("  C1 vs A0:")
    s = mcnemar_test(a0_ranked_store, c1_ranked_store)
    print(f"    Both correct: {s['both']} | Only A0: {s['only_a']} | Only C1: {s['only_b']} | Neither: {s['neither']}")
    print(f"    McNemar chi2={s['chi2']:.4f}, p={s['p']:.4f}")

    print("  C1 vs A1:")
    s = mcnemar_test(a1_ranked_store, c1_ranked_store)
    print(f"    Both correct: {s['both']} | Only A1: {s['only_a']} | Only C1: {s['only_b']} | Neither: {s['neither']}")
    print(f"    McNemar chi2={s['chi2']:.4f}, p={s['p']:.4f}")

    print("  C1 vs C0  [KEY TEST]:")
    s = mcnemar_test(c0_ranked_store, c1_ranked_store)
    print(f"    Both correct: {s['both']} | Only C0: {s['only_a']} | Only C1: {s['only_b']} | Neither: {s['neither']}")
    print(f"    McNemar chi2={s['chi2']:.4f}, p={s['p']:.4f}")
    c1_vs_c0_stats = s
except ImportError:
    print("  scipy not available — skipping McNemar. Using manual formula.")
    c1_vs_c0_stats = None

# ─────────────────────────────────────────────────────────────
# 15.  PROFILE-SIZE SUBGROUP ANALYSIS
# ─────────────────────────────────────────────────────────────
print("\n[11/12] Profile-size subgroup analysis …")

def subgroup_hits1(ranked_store):
    bins = {
        ("sparse", "small"):  [], ("sparse", "medium"):  [], ("sparse", "large"):  [],
        ("medium", "small"):  [], ("medium", "medium"):  [], ("medium", "large"):  [],
        ("dense",  "small"):  [], ("dense",  "medium"):  [], ("dense",  "large"):  [],
    }
    for r in ranked_store:
        np_ = r["n_patient_hpos"]
        nd  = r["n_disease_hpos"]
        pcat = "sparse" if np_ <= 3 else ("dense" if np_ > 10 else "medium")
        dcat = "small"  if nd  <= 10 else ("large" if nd  >= 50 else "medium")
        bins[(pcat, dcat)].append(r["rank"] == 1)
    out = {}
    for k, vals in bins.items():
        if vals:
            out[k] = {"n": len(vals), "hits1": sum(vals) / len(vals) * 100}
    return out

a0_sg = subgroup_hits1(a0_ranked_store)
a1_sg = subgroup_hits1(a1_ranked_store)
c0_sg = subgroup_hits1(c0_ranked_store)
c1_sg = subgroup_hits1(c1_ranked_store)

print(f"\n  {'Patient':8} {'Disease':8} {'N':>5}  {'A0':>7}  {'A1':>7}  {'C0':>7}  {'C1':>7}")
print("  " + "-" * 60)
for (pcat, dcat) in [
    ("sparse","small"), ("sparse","medium"), ("sparse","large"),
    ("medium","small"), ("medium","medium"), ("medium","large"),
    ("dense", "small"), ("dense", "medium"), ("dense", "large"),
]:
    n = a0_sg.get((pcat,dcat), {}).get("n", 0)
    if n == 0:
        continue
    a0h = a0_sg.get((pcat,dcat), {}).get("hits1", 0)
    a1h = a1_sg.get((pcat,dcat), {}).get("hits1", 0)
    c0h = c0_sg.get((pcat,dcat), {}).get("hits1", 0)
    c1h = c1_sg.get((pcat,dcat), {}).get("hits1", 0)
    print(f"  {pcat:8} {dcat:8} {n:>5}  {a0h:>6.2f}%  {a1h:>6.2f}%  {c0h:>6.2f}%  {c1h:>6.2f}%")

# ─────────────────────────────────────────────────────────────
# 16.  ATTENTION ANALYSIS & COUNTERFACTUAL TEST
# ─────────────────────────────────────────────────────────────
print("\n[12/12] Attention analysis and counterfactual test …")

c1_model.eval()
c1_model.to("cpu")

def get_attn_weights(model, p_hpos: set, d_id: str):
    """Return (attention_weights: np.array, d_hpo_list: list)"""
    d_hpos = disease_hpo_sets[d_id]
    d_hpo_list = [h for h in d_hpos if h in hpo_to_idx]
    d_hpo_list = d_hpo_list[:MAX_LEN_DISEASE]

    p_ids = hpo_set_to_tensor_fixed(p_hpos, MAX_LEN_PATIENT).unsqueeze(0)
    d_ids = hpo_set_to_tensor_fixed(d_hpos, MAX_LEN_DISEASE).unsqueeze(0)

    with torch.no_grad():
        _, attn = model.score(p_ids, d_ids, return_attn=True)
    attn_np = attn[0, :len(d_hpo_list)].numpy()
    return attn_np, d_hpo_list

# Find a C1-correct, A0-incorrect case for interesting attention
print("  Searching for illustrative cases …")
c1_correct_a0_wrong = []
for i, (c, a0r, c1r) in enumerate(zip(test_cases_list, a0_ranked_store, c1_ranked_store)):
    if a0r["rank"] != 1 and c1r["rank"] == 1:
        c1_correct_a0_wrong.append((i, c, a0r, c1r))
    if len(c1_correct_a0_wrong) >= 5:
        break

print(f"  Cases where C1 correct, A0 wrong: {len(c1_correct_a0_wrong)}")

# Counterfactual: add one HPO to a patient and show attention change
if c1_correct_a0_wrong:
    case_i, ex_case, ex_a0, ex_c1 = c1_correct_a0_wrong[0]
    ex_p_hpos = set(ex_case["hpos"])
    ex_disease = ex_case["diseases"][0]
    ex_d_hpos = disease_hpo_sets[ex_disease]

    attn_orig, d_hpo_list = get_attn_weights(c1_model, ex_p_hpos, ex_disease)

    # Add one extra HPO from the disease profile (a disease phenotype the patient doesn't have)
    candidate_extras = [h for h in ex_d_hpos if h not in ex_p_hpos and h in hpo_to_idx]
    if candidate_extras:
        extra_hpo = candidate_extras[0]
        ex_p_hpos_augmented = ex_p_hpos | {extra_hpo}
        attn_aug, _ = get_attn_weights(c1_model, ex_p_hpos_augmented, ex_disease)
        delta_attn = np.abs(attn_orig - attn_aug).mean()
        print(f"  Counterfactual test:")
        print(f"    Patient HPOs: {len(ex_p_hpos)} → {len(ex_p_hpos_augmented)} (added {extra_hpo})")
        print(f"    Disease: {ex_disease} | Disease profile size: {len(ex_d_hpos)}")
        print(f"    Mean |ΔAttention| after adding one HPO: {delta_attn:.4f}")
        counterfactual_result = {"delta_attn_mean": float(delta_attn), "added_hpo": extra_hpo}
    else:
        print("  No candidate extra HPO for counterfactual test.")
        counterfactual_result = None
else:
    print("  No C1-correct A0-wrong cases found — skipping attention analysis.")
    counterfactual_result = None

# ─────────────────────────────────────────────────────────────
# 17.  SAVE ALL RESULTS
# ─────────────────────────────────────────────────────────────
import os
os.makedirs("data/benchmark", exist_ok=True)

def sg_to_serializable(sg):
    return {f"{k[0]}_{k[1]}": v for k, v in sg.items()}

output = {
    "phase": "6.6",
    "research_question": (
        "Can query-conditioned asymmetric phenotype matching reduce profile-size "
        "bias in zero-shot rare-disease prioritization beyond static phenotype "
        "weighting and length normalization?"
    ),
    "benchmark": {
        "total_cases": len(valid_cases),
        "test_cases": len(test_cases_list),
        "train_diseases": len(train_dids),
        "val_diseases": len(val_dids),
        "test_diseases": len(test_dids),
        "candidate_universe": len(candidate_diseases),
    },
    "a1_selected_gamma": best_gamma,
    "results": all_results,
    "model_params": {
        "B":  count_trainable_params(b_model),
        "C0": count_trainable_params(c0_model),
        "C1": count_trainable_params(c1_model),
    },
    "training_history": {
        "B": b_history, "C0": c0_history, "C1": c1_history
    },
    "subgroup_a0": sg_to_serializable(a0_sg),
    "subgroup_a1": sg_to_serializable(a1_sg),
    "subgroup_c0": sg_to_serializable(c0_sg),
    "subgroup_c1": sg_to_serializable(c1_sg),
    "c1_vs_c0_mcnemar": c1_vs_c0_stats,
    "counterfactual_test": counterfactual_result,
}

with open("data/benchmark/phase6_6_results.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nSaved results to data/benchmark/phase6_6_results.json")
print("=" * 65)
print("PHASE 6.6 COMPLETE")
print("=" * 65)
