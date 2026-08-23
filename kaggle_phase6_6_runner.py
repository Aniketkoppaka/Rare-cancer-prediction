"""
Phase 6.6 — Kaggle GPU Accelerated Runner

Standalone script for Kaggle / Local CUDA execution.
Evaluates the entire 12,956 disease candidate universe using batched GPU matrix operations.
"""

import json
import math
import os
import random
import time
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --- 1. SETUP & PATH RESOLUTION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")

# Check Kaggle vs Local Paths
KAGGLE_INPUT_DIR = "/kaggle/input/rare-cancer-benchmark"
LOCAL_DATA_DIR = "data/benchmark"

if os.path.exists(os.path.join(KAGGLE_INPUT_DIR, "clean_phenopackets.json")):
    DATA_DIR = KAGGLE_INPUT_DIR
    OUT_DIR = "/kaggle/working"
elif os.path.exists(os.path.join(LOCAL_DATA_DIR, "clean_phenopackets.json")):
    DATA_DIR = LOCAL_DATA_DIR
    OUT_DIR = "data/benchmark"
else:
    # Look for files in current working directory
    DATA_DIR = "."
    OUT_DIR = "."

print(f"Reading data from: {DATA_DIR}")
print(f"Output directory: {OUT_DIR}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# --- 2. LOAD DATA ---
with open(os.path.join(DATA_DIR, "clean_phenopackets.json"), "r", encoding="utf-8") as f:
    raw_cases = json.load(f)

disease_to_hpos = defaultdict(set)
with open(os.path.join(DATA_DIR, "phenotype.hpoa"), "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("#") or not line.strip(): continue
        parts = line.strip().split("\t")
        if len(parts) >= 4:
            did, hid = parts[0], parts[3]
            if hid.startswith("HP:"):
                disease_to_hpos[did].add(hid)

valid_cases = [c for c in raw_cases if c["diseases"][0] in disease_to_hpos]
all_diseases = sorted({c["diseases"][0] for c in valid_cases})
all_hpos_list = sorted(
    {h for c in valid_cases for h in c["hpos"]}
    | {h for hset in disease_to_hpos.values() for h in hset}
)
hpo_to_idx = {h: i + 1 for i, h in enumerate(all_hpos_list)}
VOCAB_SIZE = len(hpo_to_idx) + 1
PAD_IDX = 0

total_kb = len(disease_to_hpos)
hpo_doc_freq = Counter(h for hset in disease_to_hpos.values() for h in hset)
hpo_ic = {h: -math.log((freq + 1) / (total_kb + 1)) for h, freq in hpo_doc_freq.items()}

# Frozen Partition (Seed 42)
shuffled_dids = list(all_diseases)
random.shuffle(shuffled_dids)
n_train_d = int(0.70 * len(shuffled_dids))
n_val_d = int(0.15 * len(shuffled_dids))

train_dids = set(shuffled_dids[:n_train_d])
val_dids = set(shuffled_dids[n_train_d : n_train_d + n_val_d])
test_dids = set(shuffled_dids[n_train_d + n_val_d :])

train_cases = [c for c in valid_cases if c["diseases"][0] in train_dids]
val_cases_list = [c for c in valid_cases if c["diseases"][0] in val_dids]
test_cases_list = [c for c in valid_cases if c["diseases"][0] in test_dids]
candidate_diseases = list(disease_to_hpos.keys())

print(f"Train Cases: {len(train_cases)} | Val Cases: {len(val_cases_list)} | Test Cases: {len(test_cases_list)}")
print(f"Candidate Universe: {len(candidate_diseases)}")

disease_hpo_sets = {did: set(hset) for did, hset in disease_to_hpos.items()}
candidate_did_to_idx = {did: i for i, did in enumerate(candidate_diseases)}

# --- 3. MODEL ARCHITECTURES ---
EMBED_DIM = 64
MAX_LEN_PATIENT = 64
MAX_LEN_DISEASE = 128

def make_ic_embedding(embed_dim=EMBED_DIM):
    weight = torch.zeros(VOCAB_SIZE, embed_dim)
    for h, idx in hpo_to_idx.items():
        weight[idx, 0] = hpo_ic.get(h, 1.0)
        weight[idx, 1:] = torch.randn(embed_dim - 1) * 0.01
    emb = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD_IDX)
    emb.weight.data.copy_(weight)
    emb.weight.requires_grad_(False)
    return emb

def hpo_set_to_tensor(hpo_set, max_len):
    indices = [hpo_to_idx[h] for h in hpo_set if h in hpo_to_idx]
    random.shuffle(indices)
    indices = indices[:max_len]
    return torch.tensor(indices + [PAD_IDX] * (max_len - len(indices)), dtype=torch.long)

def hpo_set_to_tensor_fixed(hpo_set, max_len):
    indices = sorted([hpo_to_idx[h] for h in hpo_set if h in hpo_to_idx])[:max_len]
    return torch.tensor(indices + [PAD_IDX] * (max_len - len(indices)), dtype=torch.long)

# Model B: Symmetric Dual Encoder
class SymmetricDualEncoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.emb = make_ic_embedding(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def encode(self, ids):
        mask = (ids != PAD_IDX).float().unsqueeze(-1)
        embs = self.proj(self.emb(ids)) * mask
        return embs.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def score(self, p_ids, d_ids):
        qp = F.normalize(self.encode(p_ids), dim=-1)
        qd = F.normalize(self.encode(d_ids), dim=-1)
        return (qp * qd).sum(dim=-1)

# Model C0: Non-conditioned Learned Scorer
class NonConditionedScorer(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.emb = make_ic_embedding(embed_dim)
        self.patient_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.disease_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def encode_patient(self, ids):
        mask = (ids != PAD_IDX).float().unsqueeze(-1)
        embs = self.patient_proj(self.emb(ids)) * mask
        return embs.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def encode_disease(self, ids):
        mask = (ids != PAD_IDX).float().unsqueeze(-1)
        embs = self.disease_proj(self.emb(ids)) * mask
        return embs.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def score(self, p_ids, d_ids):
        qp = self.encode_patient(p_ids)
        qd = self.encode_disease(d_ids)
        return self.scorer(torch.cat([qp, qd], dim=-1)).squeeze(-1)

# Model C1: True Query-Conditioned Cross-Attention
class TrueAsymmetricCrossAttention(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        self.emb = make_ic_embedding(embed_dim)
        self.patient_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )
        self.scale = math.sqrt(embed_dim)

    def encode_patient_query(self, p_ids):
        mask = (p_ids != PAD_IDX).float().unsqueeze(-1)
        embs = self.patient_proj(self.emb(p_ids)) * mask
        return embs.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def cross_attend(self, q, d_ids, return_attn=False):
        mask = (d_ids != PAD_IDX).float()
        embs = self.emb(d_ids)
        K = self.key_proj(embs)
        V = self.value_proj(embs)

        q_expanded = q.unsqueeze(1)
        raw_attn = torch.bmm(q_expanded, K.transpose(1, 2)) / self.scale
        raw_attn = raw_attn.squeeze(1).masked_fill(mask == 0, float("-inf"))
        alpha = F.softmax(raw_attn, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)

        z = torch.bmm(alpha.unsqueeze(1), V).squeeze(1)
        if return_attn:
            return z, alpha
        return z, None

    def score(self, p_ids, d_ids, return_attn=False):
        q = self.encode_patient_query(p_ids)
        z, attn = self.cross_attend(q, d_ids, return_attn=return_attn)
        s = self.scorer(torch.cat([q, z], dim=-1)).squeeze(-1)
        if return_attn:
            return s, attn
        return s

# --- 4. PRE-COMPUTE ALL 12,956 DISEASE TENSORS (FAST GPU INFERENCE) ---
print("\nPre-indexing 12,956 candidate disease tensors for fast GPU batch ranking...")
candidate_d_tensors = torch.stack([
    hpo_set_to_tensor_fixed(disease_hpo_sets[did], MAX_LEN_DISEASE)
    for did in candidate_diseases
]).to(DEVICE) # (12956, MAX_LEN_DISEASE)
print(f"Disease Matrix Tensor Shape on GPU: {candidate_d_tensors.shape}")

# Vectorized Ranking Function
@torch.no_grad()
def eval_model_fast_gpu(model, cohort, model_type="C1", batch_patients=128):
    model.eval()
    model = model.to(DEVICE)
    h1, h5, h10, mrr_sum = 0, 0, 0, 0.0
    ranked_records = []
    
    num_candidates = len(candidate_diseases)
    
    # Process cohort in patient batches
    for i in range(0, len(cohort), batch_patients):
        batch_cases = cohort[i : i + batch_patients]
        B = len(batch_cases)
        
        p_tensors = torch.stack([
            hpo_set_to_tensor_fixed(set(c["hpos"]), MAX_LEN_PATIENT)
            for c in batch_cases
        ]).to(DEVICE) # (B, MAX_LEN_PATIENT)
        
        if model_type == "B":
            qp = F.normalize(model.encode(p_tensors), dim=-1) # (B, D)
            qd = F.normalize(model.encode(candidate_d_tensors), dim=-1) # (12956, D)
            scores = torch.mm(qp, qd.t()) # (B, 12956)
            
        elif model_type == "C0":
            qp = model.encode_patient(p_tensors) # (B, D)
            qd = model.encode_disease(candidate_d_tensors) # (12956, D)
            # Evaluate across chunks of candidate diseases to fit memory
            chunk_size = 2048
            score_list = []
            for c_idx in range(0, num_candidates, chunk_size):
                qd_chunk = qd[c_idx : c_idx + chunk_size]
                C_sz = qd_chunk.size(0)
                qp_exp = qp.unsqueeze(1).expand(-1, C_sz, -1).reshape(B * C_sz, -1)
                qd_exp = qd_chunk.unsqueeze(0).expand(B, -1, -1).reshape(B * C_sz, -1)
                chunk_scores = model.scorer(torch.cat([qp_exp, qd_exp], dim=-1)).view(B, C_sz)
                score_list.append(chunk_scores)
            scores = torch.cat(score_list, dim=1) # (B, 12956)
            
        elif model_type == "C1":
            q = model.encode_patient_query(p_tensors) # (B, D)
            chunk_size = 1024
            score_list = []
            for c_idx in range(0, num_candidates, chunk_size):
                d_chunk = candidate_d_tensors[c_idx : c_idx + chunk_size]
                C_sz = d_chunk.size(0)
                q_exp = q.unsqueeze(1).expand(-1, C_sz, -1).reshape(B * C_sz, -1)
                d_exp = d_chunk.unsqueeze(0).expand(B, -1, -1).reshape(B * C_sz, -1)
                z, _ = model.cross_attend(q_exp, d_exp)
                chunk_scores = model.scorer(torch.cat([q_exp, z], dim=-1)).view(B, C_sz)
                score_list.append(chunk_scores)
            scores = torch.cat(score_list, dim=1) # (B, 12956)
            
        # Top-K & Ranking
        scores_np = scores.cpu().numpy()
        for b_idx, case in enumerate(batch_cases):
            tgt = case["diseases"][0]
            tgt_idx = candidate_did_to_idx.get(tgt, -1)
            row = scores_np[b_idx]
            
            # Rank
            tgt_score = row[tgt_idx] if tgt_idx >= 0 else -1e9
            r = int((row > tgt_score).sum() + 1)
            
            if r <= 1: h1 += 1
            if r <= 5: h5 += 1
            if r <= 10: h10 += 1
            mrr_sum += 1.0 / r
            
            ranked_records.append({
                "target": tgt,
                "rank": r,
                "n_patient_hpos": len(case["hpos"]),
                "n_disease_hpos": len(disease_hpo_sets.get(tgt, set())),
            })
            
    N = len(cohort)
    return {
        "Hits@1": h1 / N * 100,
        "Hits@5": h5 / N * 100,
        "Hits@10": h10 / N * 100,
        "MRR": mrr_sum / N,
        "N": N
    }, ranked_records

# --- 5. TRAINING LOOP ---
class RareDiseaseDataset(Dataset):
    def __init__(self, cases, disease_hpo_sets, all_train_dids):
        self.cases = cases
        self.disease_hpo_sets = disease_hpo_sets
        self.all_train_dids_list = list(all_train_dids)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        p_ids = hpo_set_to_tensor(set(case["hpos"]), MAX_LEN_PATIENT)
        pos_did = case["diseases"][0]
        pd_ids = hpo_set_to_tensor(self.disease_hpo_sets[pos_did], MAX_LEN_DISEASE)
        return p_ids, pd_ids

def collate_fn_infonce(batch, disease_hpo_sets, all_train_dids_list, n_negatives=31):
    p_ids_list = [item[0] for item in batch]
    pd_ids_list = [item[1] for item in batch]
    neg_ids_batch = []
    for _ in batch:
        neg_dids = random.sample(all_train_dids_list, n_negatives)
        neg_tensors = [
            hpo_set_to_tensor(disease_hpo_sets[did], MAX_LEN_DISEASE)
            for did in neg_dids
        ]
        neg_ids_batch.append(torch.stack(neg_tensors, dim=0))
    return torch.stack(p_ids_list), torch.stack(pd_ids_list), torch.stack(neg_ids_batch)

def train_model(model, model_type, epochs=15, batch_size=128, lr=5e-4):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    dataset = RareDiseaseDataset(train_cases, disease_hpo_sets, train_dids)
    collate = lambda b: collate_fn_infonce(b, disease_hpo_sets, list(train_dids), n_negatives=31)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=2)
    
    best_val_mrr = -1.0
    best_state = None
    
    print(f"\n--- Training {model_type} ({epochs} epochs) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_b = 0.0, 0
        for p_ids, pd_ids, neg_ids in loader:
            p_ids, pd_ids, neg_ids = p_ids.to(DEVICE), pd_ids.to(DEVICE), neg_ids.to(DEVICE)
            B = p_ids.size(0)
            n_neg = neg_ids.size(1)
            
            pos_scores = model.score(p_ids, pd_ids)
            p_rep = p_ids.unsqueeze(1).expand(-1, n_neg, -1).reshape(B * n_neg, -1)
            d_rep = neg_ids.reshape(B * n_neg, -1)
            neg_scores = model.score(p_rep, d_rep).reshape(B, n_neg)
            
            all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
            loss = F.cross_entropy(all_scores, torch.zeros(B, dtype=torch.long, device=DEVICE))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_b += 1
            
        val_res, _ = eval_model_fast_gpu(model, val_cases_list[:300], model_type=model_type)
        print(f"Epoch {epoch:2d} | Train Loss: {total_loss/n_b:.4f} | Val Hits@1: {val_res['Hits@1']:.2f}% | Val MRR: {val_res['MRR']:.4f}")
        if val_res['MRR'] > best_val_mrr:
            best_val_mrr = val_res['MRR']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    model.load_state_dict(best_state)
    return model

# --- 6. RUN FULL EXPERIMENT ---
print("\n" + "="*60)
print("STARTING FULL ABLATION PIPELINE")
print("="*60)

# A0 & A1 Baselines
def ic_overlap(p_hpos, d_hpos, gamma=0.0):
    inter = p_hpos & d_hpos
    if not inter: return 0.0
    s = sum(hpo_ic.get(h, 1.0) for h in inter)
    if gamma > 0: s /= (len(d_hpos) ** gamma)
    return s

print("\n[A0: Raw IC Evaluation]")
a0_h1, a0_h5, a0_h10, a0_mrr = 0, 0, 0, 0.0
a0_records = []
for c in test_cases_list:
    p_hpos = set(c["hpos"])
    tgt = c["diseases"][0]
    scores = [(ic_overlap(p_hpos, disease_hpo_sets[d], 0.0), d) for d in candidate_diseases]
    scores.sort(reverse=True)
    ranked = [x[1] for x in scores]
    try: r = ranked.index(tgt) + 1
    except ValueError: r = len(candidate_diseases)
    if r <= 1: a0_h1 += 1
    if r <= 5: a0_h5 += 1
    if r <= 10: a0_h10 += 1
    a0_mrr += 1.0 / r
    a0_records.append({"target": tgt, "rank": r, "n_patient_hpos": len(p_hpos), "n_disease_hpos": len(disease_hpo_sets.get(tgt, set()))})

N_t = len(test_cases_list)
res_a0 = {"Method": "A0: Raw IC", "Hits@1": a0_h1/N_t*100, "Hits@5": a0_h5/N_t*100, "Hits@10": a0_h10/N_t*100, "MRR": a0_mrr/N_t}

print("\n[A1: Optimal Static Length Normalization (gamma=0.05)]")
a1_h1, a1_h5, a1_h10, a1_mrr = 0, 0, 0, 0.0
a1_records = []
for c in test_cases_list:
    p_hpos = set(c["hpos"])
    tgt = c["diseases"][0]
    scores = [(ic_overlap(p_hpos, disease_hpo_sets[d], 0.05), d) for d in candidate_diseases]
    scores.sort(reverse=True)
    ranked = [x[1] for x in scores]
    try: r = ranked.index(tgt) + 1
    except ValueError: r = len(candidate_diseases)
    if r <= 1: a1_h1 += 1
    if r <= 5: a1_h5 += 1
    if r <= 10: a1_h10 += 1
    a1_mrr += 1.0 / r
    a1_records.append({"target": tgt, "rank": r, "n_patient_hpos": len(p_hpos), "n_disease_hpos": len(disease_hpo_sets.get(tgt, set()))})
res_a1 = {"Method": "A1: Static gamma-IC (g=0.05)", "Hits@1": a1_h1/N_t*100, "Hits@5": a1_h5/N_t*100, "Hits@10": a1_h10/N_t*100, "MRR": a1_mrr/N_t}

# Train Neural Models
model_b = train_model(SymmetricDualEncoder(), "B", epochs=10)
res_b, b_records = eval_model_fast_gpu(model_b, test_cases_list, "B")
res_b["Method"] = "B: Learned Symmetric Dual-Encoder"

model_c0 = train_model(NonConditionedScorer(), "C0", epochs=12)
res_c0, c0_records = eval_model_fast_gpu(model_c0, test_cases_list, "C0")
res_c0["Method"] = "C0: Non-Conditioned Learned Scorer"

model_c1 = train_model(TrueAsymmetricCrossAttention(), "C1", epochs=15)
res_c1, c1_records = eval_model_fast_gpu(model_c1, test_cases_list, "C1")
res_c1["Method"] = "C1: True Query-Conditioned Cross-Attention"

# --- 7. DISPLAY FINAL RESULTS TABLE ---
print("\n" + "="*80)
print("FINAL TEST BENCHMARK RESULTS (N=1,880 ZERO-SHOT CASES, 12,956 CANDIDATES)")
print("="*80)
final_table = [res_a0, res_a1, res_b, res_c0, res_c1]
print(f"{'Method':<42} {'Hits@1':>8} {'Hits@5':>8} {'Hits@10':>8} {'MRR':>8}")
print("-" * 80)
for r in final_table:
    print(f"{r['Method']:<42} {r['Hits@1']:>7.2f}% {r['Hits@5']:>7.2f}% {r['Hits@10']:>7.2f}% {r['MRR']:>8.4f}")
print("="*80)

# Save JSON results
with open(os.path.join(OUT_DIR, "phase6_6_results.json"), "w") as f:
    json.dump(final_table, f, indent=2)
print(f"\nSaved results to {os.path.join(OUT_DIR, 'phase6_6_results.json')}")
