"""
Phase 6.7: Comprehensive Audit Script

Covers:
1. A0/A1 Benchmark reconciliation & Validation curve for gamma
2. Model B / C0 / C1 diagnostic inspection (training loss, logits, score distribution)
3. Small-scale overfit test (10 diseases, 100 cases)
4. Seen-disease vs Unseen zero-shot test
5. A1 ranking-change case analysis (Why gamma=0.05 works)
6. Score distribution analysis (|D|, sum IC, mean IC, density)
7. Symbolic-grounded residual proof-of-concept: S(P,D) = S_IC(P,D) + lambda * R(P,D)
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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Executing Phase 6.7 Audit on: {DEVICE}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# --- 1. LOAD DATA ---
with open('data/benchmark/clean_phenopackets.json', 'r', encoding='utf-8') as f:
    raw_cases = json.load(f)

disease_to_hpos = defaultdict(set)
with open('data/benchmark/phenotype.hpoa', 'r', encoding='utf-8') as f:
    for line in f:
        if line.startswith('#') or not line.strip(): continue
        parts = line.strip().split('\t')
        if len(parts) >= 4 and parts[3].startswith('HP:'):
            disease_to_hpos[parts[0]].add(parts[3])

valid_cases = [c for c in raw_cases if c['diseases'][0] in disease_to_hpos]
all_diseases = sorted(list(set([c['diseases'][0] for c in valid_cases])))
all_hpos_list = sorted(list(set([h for c in valid_cases for h in c['hpos']] + [h for hset in disease_to_hpos.values() for h in hset])))
hpo_to_idx = {h: i + 1 for i, h in enumerate(all_hpos_list)}
VOCAB_SIZE = len(hpo_to_idx) + 1
PAD_IDX = 0

total_kb = len(disease_to_hpos)
hpo_doc_freq = Counter(h for hset in disease_to_hpos.values() for h in hset)
hpo_ic = {h: -math.log((freq + 1)/(total_kb + 1)) for h, freq in hpo_doc_freq.items()}

# Frozen Split
shuffled_dids = list(all_diseases)
random.shuffle(shuffled_dids)
n_train_d = int(0.70 * len(shuffled_dids))
n_val_d = int(0.15 * len(shuffled_dids))

train_dids = set(shuffled_dids[:n_train_d])
val_dids = set(shuffled_dids[n_train_d:n_train_d+n_val_d])
test_dids = set(shuffled_dids[n_train_d+n_val_d:])

train_cases = [c for c in valid_cases if c['diseases'][0] in train_dids]
val_cases = [c for c in valid_cases if c['diseases'][0] in val_dids]
test_cases = [c for c in valid_cases if c['diseases'][0] in test_dids]
candidate_diseases = list(disease_to_hpos.keys())
disease_hpo_sets = {did: set(hset) for did, hset in disease_to_hpos.items()}

print(f"Data ready: Train={len(train_cases)}, Val={len(val_cases)}, Test={len(test_cases)}, Candidates={len(candidate_diseases)}")

# --- SECTION 2: VALIDATION CURVE FOR GAMMA (A1) ---
print("\n=== SECTION 2: VALIDATION GAMMA SELECTION ===")
gamma_candidates = [0.0, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.25, 0.50]
val_gamma_results = []

def score_gamma(p_hpos, d_hpos, gamma):
    inter = p_hpos & d_hpos
    if not inter: return 0.0
    s = sum(hpo_ic.get(h, 1.0) for h in inter)
    if gamma > 0:
        s /= (len(d_hpos) ** gamma)
    return s

for g in gamma_candidates:
    h1, mrr = 0, 0.0
    for c in val_cases:
        p_hpos = set(c['hpos'])
        tgt = c['diseases'][0]
        scores = [(score_gamma(p_hpos, disease_hpo_sets[d], g), d) for d in candidate_diseases]
        scores.sort(key=lambda x: x[0], reverse=True)
        r = [x[1] for x in scores].index(tgt) + 1
        if r <= 1: h1 += 1
        mrr += 1.0 / r
    h1_pct = h1 / len(val_cases) * 100
    mrr_val = mrr / len(val_cases)
    val_gamma_results.append({'gamma': g, 'Val_Hits@1': h1_pct, 'Val_MRR': mrr_val})
    print(f"  Val Gamma={g:5.3f} | Hits@1: {h1_pct:5.2f}% | MRR: {mrr_val:.4f}")

best_val_gamma = max(val_gamma_results, key=lambda x: x['Val_Hits@1'])['gamma']
print(f"--> Pre-selected Gamma on Validation: {best_val_gamma}")

# Evaluate pre-selected gamma on locked test set ONCE
h1_test, h5_test, h10_test, mrr_test = 0, 0, 0, 0.0
a1_test_ranks = []
a0_test_ranks = []
for c in test_cases:
    p_hpos = set(c['hpos'])
    tgt = c['diseases'][0]
    # A0
    s0 = [(score_gamma(p_hpos, disease_hpo_sets[d], 0.0), d) for d in candidate_diseases]
    s0.sort(key=lambda x: x[0], reverse=True)
    r0 = [x[1] for x in s0].index(tgt) + 1
    a0_test_ranks.append(r0)
    
    # A1 (best_val_gamma)
    s1 = [(score_gamma(p_hpos, disease_hpo_sets[d], best_val_gamma), d) for d in candidate_diseases]
    s1.sort(key=lambda x: x[0], reverse=True)
    r1 = [x[1] for x in s1].index(tgt) + 1
    a1_test_ranks.append(r1)
    if r1 <= 1: h1_test += 1
    if r1 <= 5: h5_test += 1
    if r1 <= 10: h10_test += 1
    mrr_test += 1.0 / r1

N_t = len(test_cases)
a0_hits1 = sum(1 for r in a0_test_ranks if r <= 1) / N_t * 100
a0_mrr = sum(1.0/r for r in a0_test_ranks) / N_t
a1_hits1 = h1_test / N_t * 100
a1_mrr = mrr_test / N_t
print(f"Test A0 (Raw IC):      Hits@1 = {a0_hits1:.2f}% | MRR = {a0_mrr:.4f}")
print(f"Test A1 (Gamma={best_val_gamma}): Hits@1 = {a1_hits1:.2f}% | MRR = {a1_mrr:.4f}")

# --- SECTION 3: TINY-DATA OVERFIT TEST (10 diseases, 100 cases) ---
print("\n=== SECTION 3: TINY-DATA OVERFIT TEST (10 diseases, 100 cases) ===")
overfit_dids = list(train_dids)[:10]
overfit_cases = [c for c in train_cases if c['diseases'][0] in overfit_dids][:100]
print(f"Tiny Overfit Dataset: {len(overfit_cases)} cases across {len(overfit_dids)} diseases")

EMBED_DIM = 64
MAX_LEN_PATIENT = 64
MAX_LEN_DISEASE = 128

def make_ic_embedding():
    weight = torch.zeros(VOCAB_SIZE, EMBED_DIM)
    for h, idx in hpo_to_idx.items():
        weight[idx, 0] = hpo_ic.get(h, 1.0)
        weight[idx, 1:] = torch.randn(EMBED_DIM - 1) * 0.01
    emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=PAD_IDX)
    emb.weight.data.copy_(weight)
    emb.weight.requires_grad_(False)
    return emb

def hpo_set_to_tensor_fixed(hpo_set, max_len):
    indices = sorted([hpo_to_idx[h] for h in hpo_set if h in hpo_to_idx])[:max_len]
    return torch.tensor(indices + [PAD_IDX] * (max_len - len(indices)), dtype=torch.long)

class TrueAsymmetricCrossAttention(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.emb = make_ic_embedding()
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

    def cross_attend(self, q, d_ids):
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
        return z

    def score(self, p_ids, d_ids):
        q = self.encode_patient_query(p_ids)
        z = self.cross_attend(q, d_ids)
        return self.scorer(torch.cat([q, z], dim=-1)).squeeze(-1)

# Train C1 aggressively on tiny dataset (classification loss among the 10 diseases)
c1_overfit = TrueAsymmetricCrossAttention().to(DEVICE)
opt_of = torch.optim.Adam([p for p in c1_overfit.parameters() if p.requires_grad], lr=1e-3)

overfit_d_tensors = torch.stack([
    hpo_set_to_tensor_fixed(disease_hpo_sets[did], MAX_LEN_DISEASE)
    for did in overfit_dids
]).to(DEVICE) # (10, MAX_LEN_DISEASE)

overfit_p_tensors = torch.stack([
    hpo_set_to_tensor_fixed(set(c['hpos']), MAX_LEN_PATIENT)
    for c in overfit_cases
]).to(DEVICE) # (100, MAX_LEN_PATIENT)

overfit_labels = torch.tensor([overfit_dids.index(c['diseases'][0]) for c in overfit_cases], dtype=torch.long).to(DEVICE)

print("Training C1 on tiny dataset for 30 epochs...")
for epoch in range(1, 31):
    c1_overfit.train()
    # Compute scores for all 100 cases vs 10 diseases
    B = len(overfit_cases)
    C_sz = len(overfit_dids)
    
    q = c1_overfit.encode_patient_query(overfit_p_tensors) # (B, D)
    q_exp = q.unsqueeze(1).expand(-1, C_sz, -1).reshape(B * C_sz, -1)
    d_exp = overfit_d_tensors.unsqueeze(0).expand(B, -1, -1).reshape(B * C_sz, -1)
    z = c1_overfit.cross_attend(q_exp, d_exp)
    all_scores = c1_overfit.scorer(torch.cat([q_exp, z], dim=-1)).view(B, C_sz)
    
    loss = F.cross_entropy(all_scores, overfit_labels)
    opt_of.zero_grad()
    loss.backward()
    opt_of.step()
    
    if epoch % 5 == 0 or epoch == 1:
        preds = all_scores.argmax(dim=1)
        acc = (preds == overfit_labels).float().mean().item() * 100
        print(f"  Epoch {epoch:2d}/30 | Loss: {loss.item():.4f} | Training Hits@1 on 10 diseases: {acc:5.2f}%")

# --- SECTION 4: SEEN-DISEASE VS ZERO-SHOT GENERALIZATION ---
print("\n=== SECTION 4: SEEN-DISEASE VS ZERO-SHOT TEST ===")
# Test C1 trained on overfit set against 10 seen diseases vs 10 unseen diseases
unseen_10_dids = list(test_dids)[:10]
unseen_10_cases = [c for c in test_cases if c['diseases'][0] in unseen_10_dids][:100]

unseen_d_tensors = torch.stack([
    hpo_set_to_tensor_fixed(disease_hpo_sets[did], MAX_LEN_DISEASE)
    for did in unseen_10_dids
]).to(DEVICE)

unseen_p_tensors = torch.stack([
    hpo_set_to_tensor_fixed(set(c['hpos']), MAX_LEN_PATIENT)
    for c in unseen_10_cases
]).to(DEVICE)
unseen_labels = torch.tensor([unseen_10_dids.index(c['diseases'][0]) for c in unseen_10_cases], dtype=torch.long).to(DEVICE)

c1_overfit.eval()
with torch.no_grad():
    B_u = len(unseen_10_cases)
    C_u = len(unseen_10_dids)
    q_u = c1_overfit.encode_patient_query(unseen_p_tensors)
    q_exp_u = q_u.unsqueeze(1).expand(-1, C_u, -1).reshape(B_u * C_u, -1)
    d_exp_u = unseen_d_tensors.unsqueeze(0).expand(B_u, -1, -1).reshape(B_u * C_u, -1)
    z_u = c1_overfit.cross_attend(q_exp_u, d_exp_u)
    unseen_scores = c1_overfit.scorer(torch.cat([q_exp_u, z_u], dim=-1)).view(B_u, C_u)
    unseen_preds = unseen_scores.argmax(dim=1)
    unseen_acc = (unseen_preds == unseen_labels).float().mean().item() * 100

print(f"Seen 10-Disease Accuracy:   {acc:5.2f}%")
print(f"Unseen 10-Disease Accuracy: {unseen_acc:5.2f}% (Random baseline = 10.0%)")

# --- SECTION 5: A1 RANKING-CHANGE AND MECHANISM ANALYSIS ---
print("\n=== SECTION 5: A1 RANKING-CHANGE MECHANISM ANALYSIS ===")
# Analyze cases where A1 is correct (Hits@1) and A0 is incorrect
a1_better = [] # A1 rank 1, A0 rank > 1
a0_better = [] # A0 rank 1, A1 rank > 1
both_correct = []

for i, c in enumerate(test_cases):
    r0 = a0_test_ranks[i]
    r1 = a1_test_ranks[i]
    tgt = c['diseases'][0]
    d_len = len(disease_hpo_sets[tgt])
    p_len = len(c['hpos'])
    ic_sum = sum(hpo_ic.get(h, 1.0) for h in (set(c['hpos']) & disease_hpo_sets[tgt]))
    
    record = {
        'case_id': c.get('id', str(i)),
        'target': tgt,
        'r0': r0,
        'r1': r1,
        'patient_hpos': p_len,
        'disease_hpos': d_len,
        'ic_overlap': ic_sum,
    }
    if r1 == 1 and r0 > 1:
        a1_better.append(record)
    elif r0 == 1 and r1 > 1:
        a0_better.append(record)
    elif r0 == 1 and r1 == 1:
        both_correct.append(record)

print(f"A1 Correct, A0 Wrong: {len(a1_better)} cases (Net gained by gamma=0.05)")
print(f"A0 Correct, A1 Wrong: {len(a0_better)} cases (Lost by gamma=0.05)")
print(f"Both Correct:         {len(both_correct)} cases")

avg_p_a1_better = np.mean([x['patient_hpos'] for x in a1_better])
avg_d_a1_better = np.mean([x['disease_hpos'] for x in a1_better])
avg_p_a0_better = np.mean([x['patient_hpos'] for x in a0_better])
avg_d_a0_better = np.mean([x['disease_hpos'] for x in a0_better])

print(f"A1-rescued cases: Avg Patient HPOs = {avg_p_a1_better:.1f}, Avg Disease Profile Size = {avg_d_a1_better:.1f}")
print(f"A0-lost cases:    Avg Patient HPOs = {avg_p_a0_better:.1f}, Avg Disease Profile Size = {avg_d_a0_better:.1f}")

# --- SECTION 6: SCORE DISTRIBUTION ANALYSIS ---
print("\n=== SECTION 6: DISEASE SCORE DISTRIBUTION & PROPERTIES ===")
candidate_stats = []
for did in candidate_diseases:
    hset = disease_hpo_sets[did]
    ics = [hpo_ic.get(h, 1.0) for h in hset]
    candidate_stats.append({
        'did': did,
        'profile_len': len(hset),
        'sum_ic': sum(ics),
        'mean_ic': np.mean(ics) if ics else 0.0,
        'max_ic': max(ics) if ics else 0.0
    })

lens = [x['profile_len'] for x in candidate_stats]
sum_ics = [x['sum_ic'] for x in candidate_stats]
mean_ics = [x['mean_ic'] for x in candidate_stats]

print(f"Candidate Universe (12,956 diseases):")
print(f"  Profile length |D|: Mean={np.mean(lens):.1f}, Median={np.median(lens):.1f}, Max={np.max(lens)}, Min={np.min(lens)}")
print(f"  Sum IC:             Mean={np.mean(sum_ics):.1f}, Median={np.median(sum_ics):.1f}, Max={np.max(sum_ics):.1f}")
print(f"  Mean IC per term:   Mean={np.mean(mean_ics):.2f}, Median={np.median(mean_ics):.2f}")

# Correlation between |D| and sum_ic
corr_len_sum = np.corrcoef(lens, sum_ics)[0, 1]
print(f"  Correlation between |D| and Total IC: r = {corr_len_sum:.4f}")

# --- SECTION 7: SYMBOLIC-GROUNDED RESIDUAL PROOF-OF-CONCEPT ---
print("\n=== SECTION 7: SYMBOLIC-GROUNDED RESIDUAL MODEL PROOF-OF-CONCEPT ===")
# Model: S_residual(P, D) = S_IC(P, D) + lambda * R(P, D)
# where R(P, D) is a simple learned calibration function of symbolic features:
# features: [overlap_count, sum_ic, max_ic, mean_ic, log(|D|), log(|P|)]

class SymbolicResidualScorer(nn.Module):
    def __init__(self):
        super().__init__()
        # Input features: 6 symbolic features
        self.mlp = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.scale_lambda = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def extract_features_batch(self, p_hpos_list, d_hpos_list):
        feats = []
        for p_hpos, d_hpos in zip(p_hpos_list, d_hpos_list):
            inter = p_hpos & d_hpos
            if not inter:
                feats.append([0.0, 0.0, 0.0, 0.0, math.log(max(len(d_hpos), 1)), math.log(max(len(p_hpos), 1))])
            else:
                ics = [hpo_ic.get(h, 1.0) for h in inter]
                feats.append([
                    len(inter),
                    sum(ics),
                    max(ics),
                    np.mean(ics),
                    math.log(max(len(d_hpos), 1)),
                    math.log(max(len(p_hpos), 1))
                ])
        return torch.tensor(feats, dtype=torch.float32)

    def score_single(self, p_hpos, d_hpos):
        inter = p_hpos & d_hpos
        raw_ic = sum(hpo_ic.get(h, 1.0) for h in inter) if inter else 0.0
        if raw_ic == 0.0:
            return 0.0
        ics = [hpo_ic.get(h, 1.0) for h in inter]
        f = torch.tensor([[
            len(inter),
            sum(ics),
            max(ics),
            float(np.mean(ics)),
            math.log(max(len(d_hpos), 1)),
            math.log(max(len(p_hpos), 1))
        ]], dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            delta = self.mlp(f).item()
        return raw_ic + self.scale_lambda.item() * delta

# Train Residual Scorer on Validation Cases
residual_model = SymbolicResidualScorer().to(DEVICE)
opt_res = torch.optim.Adam(residual_model.parameters(), lr=1e-3)

print("Training Symbolic-Grounded Residual Scorer...")
# Fast train on subset of train cases with pairwise ranking loss
for epoch in range(1, 11):
    residual_model.train()
    total_loss = 0.0
    for case in train_cases[:1000]:
        p_hpos = set(case['hpos'])
        pos_did = case['diseases'][0]
        neg_did = random.choice(list(train_dids - {pos_did}))
        
        f_pos = residual_model.extract_features_batch([p_hpos], [disease_hpo_sets[pos_did]]).to(DEVICE)
        f_neg = residual_model.extract_features_batch([p_hpos], [disease_hpo_sets[neg_did]]).to(DEVICE)
        
        ic_pos = sum(hpo_ic.get(h, 1.0) for h in (p_hpos & disease_hpo_sets[pos_did]))
        ic_neg = sum(hpo_ic.get(h, 1.0) for h in (p_hpos & disease_hpo_sets[neg_did]))
        
        score_pos = ic_pos + residual_model.scale_lambda * residual_model.mlp(f_pos).squeeze()
        score_neg = ic_neg + residual_model.scale_lambda * residual_model.mlp(f_neg).squeeze()
        
        # Margin ranking loss
        loss = F.relu(1.0 - (score_pos - score_neg))
        if loss.item() > 0:
            opt_res.zero_grad()
            loss.backward()
            opt_res.step()
            total_loss += loss.item()
    print(f"  Epoch {epoch:2d}/10 | Residual Loss: {total_loss/1000:.4f} | Learned Lambda: {residual_model.scale_lambda.item():.4f}")

# Quick Val Evaluation of Residual Model on 200 Val Cases
residual_model.eval()
res_val_h1 = 0
for c in val_cases[:200]:
    p_hpos = set(c['hpos'])
    tgt = c['diseases'][0]
    scores = []
    for did in candidate_diseases:
        inter = p_hpos & disease_hpo_sets[did]
        if not inter: s = 0.0
        else: s = residual_model.score_single(p_hpos, disease_hpo_sets[did])
        scores.append((s, did))
    scores.sort(key=lambda x: x[0], reverse=True)
    if [x[1] for x in scores].index(tgt) + 1 <= 1:
        res_val_h1 += 1

print(f"\nResidual Model Val Hits@1 (200 cases): {res_val_h1/200*100:.2f}% (vs Raw IC Val: {sum(1 for c in val_cases[:200] if [x[1] for x in sorted([(sum(hpo_ic.get(h, 1.0) for h in (set(c['hpos']) & disease_hpo_sets[d])), d) for d in candidate_diseases], key=lambda x: x[0], reverse=True)].index(c['diseases'][0])+1 <= 1)/200*100:.2f}%)")

print("\n=== PHASE 6.7 AUDIT SCRIPT COMPLETE ===")
