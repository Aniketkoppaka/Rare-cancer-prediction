import json
import math
import random
import os
from collections import defaultdict, Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

print("=== PHASE 6: MINIMUM VIABLE ASYMMETRIC PROTOTYPE (MVM) ===")

# 1. Device and Repro Seeds
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 2. Load Clean Benchmark Data
with open('data/benchmark/clean_phenopackets.json', 'r', encoding='utf-8') as f:
    raw_cases = json.load(f)

disease_to_hpos = defaultdict(set)
with open('data/benchmark/phenotype.hpoa', 'r', encoding='utf-8') as f:
    for line in f:
        if line.startswith('#') or not line.strip(): continue
        parts = line.strip().split('\t')
        if len(parts) >= 4:
            did, hid = parts[0], parts[3]
            if hid.startswith('HP:'):
                disease_to_hpos[did].add(hid)

# Filter valid cases
valid_cases = [c for c in raw_cases if c['diseases'][0] in disease_to_hpos]
all_diseases = sorted(list(set([c['diseases'][0] for c in valid_cases])))

# Build Global Vocabulary of HPO terms
all_hpos = set([h for c in valid_cases for h in c['hpos']])
for hset in disease_to_hpos.values():
    all_hpos.update(hset)
hpo_vocab = {h: i + 1 for i, h in enumerate(sorted(list(all_hpos)))} # 0 is padding
pad_idx = 0
vocab_size = len(hpo_vocab) + 1
print(f"Global HPO Vocabulary Size: {vocab_size}")

# Compute IC
total_kb = len(disease_to_hpos)
hpo_doc_freq = Counter([h for hset in disease_to_hpos.values() for h in hset])
hpo_ic = {h: -math.log((freq + 1)/(total_kb + 1)) for h, freq in hpo_doc_freq.items()}
hpo_ic_weights = np.zeros(vocab_size, dtype=np.float32)
for h, idx in hpo_vocab.items():
    hpo_ic_weights[idx] = hpo_ic.get(h, 1.0)
hpo_ic_tensor = torch.tensor(hpo_ic_weights, device=device).unsqueeze(1)

# Strict Zero-Shot Disease-Disjoint Partition
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

print(f"Dataset Split Contract: Train Cases={len(train_cases)} (from {len(train_dids)} diseases), "
      f"Val Cases={len(val_cases)} (from {len(val_dids)} diseases), "
      f"Test Cases={len(test_cases)} (from {len(test_dids)} diseases)")

# Automated Zero-Shot Safety Assertions
assert len(train_dids & test_dids) == 0, "FATAL: Test disease leaked into train split!"
assert len(val_dids & test_dids) == 0, "FATAL: Test disease leaked into val split!"
print("Zero-Shot Integrity Verified: 100% Disease-Disjoint.")

# Pre-vectorize candidate diseases
disease_hpo_indices = {}
for did in candidate_diseases:
    disease_hpo_indices[did] = [hpo_vocab[h] for h in disease_to_hpos[did] if h in hpo_vocab]

# Dataset Definition
class PhenotypePairDataset(Dataset):
    def __init__(self, cases_list, train_did_pool):
        self.cases = cases_list
        self.did_pool = list(train_did_pool)
    def __len__(self):
        return len(self.cases)
    def __getitem__(self, idx):
        c = self.cases[idx]
        p_indices = [hpo_vocab[h] for h in c['hpos'] if h in hpo_vocab]
        pos_did = c['diseases'][0]
        pos_indices = disease_hpo_indices[pos_did]
        # Sample negative disease from train pool
        neg_did = random.choice(self.did_pool)
        while neg_did == pos_did:
            neg_did = random.choice(self.did_pool)
        neg_indices = disease_hpo_indices[neg_did]
        return p_indices, pos_indices, neg_indices

def collate_fn(batch):
    p_lens = [len(x[0]) for x in batch]
    pos_lens = [len(x[1]) for x in batch]
    neg_lens = [len(x[2]) for x in batch]
    
    max_p = max(max(p_lens), 1)
    max_pos = max(max(pos_lens), 1)
    max_neg = max(max(neg_lens), 1)
    
    B = len(batch)
    p_mat = torch.zeros((B, max_p), dtype=torch.long)
    pos_mat = torch.zeros((B, max_pos), dtype=torch.long)
    neg_mat = torch.zeros((B, max_neg), dtype=torch.long)
    
    p_mask = torch.zeros((B, max_p), dtype=torch.bool)
    pos_mask = torch.zeros((B, max_pos), dtype=torch.bool)
    neg_mask = torch.zeros((B, max_neg), dtype=torch.bool)
    
    for i, (p_idx, pos_idx, neg_idx) in enumerate(batch):
        if p_idx:
            p_mat[i, :len(p_idx)] = torch.tensor(p_idx, dtype=torch.long)
            p_mask[i, :len(p_idx)] = True
        if pos_idx:
            pos_mat[i, :len(pos_idx)] = torch.tensor(pos_idx, dtype=torch.long)
            pos_mask[i, :len(pos_idx)] = True
        if neg_idx:
            neg_mat[i, :len(neg_idx)] = torch.tensor(neg_idx, dtype=torch.long)
            neg_mask[i, :len(neg_idx)] = True
            
    return p_mat, p_mask, pos_mat, pos_mask, neg_mat, neg_mask

# 3. Model B: Learned Symmetric Dual-Encoder
class ModelBSymmetric(nn.Module):
    def __init__(self, vocab_size, emb_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
    def encode_set(self, x, mask):
        # x: (B, L), mask: (B, L)
        emb = self.embedding(x) # (B, L, D)
        proj = self.proj(emb)   # (B, L, D)
        mask_f = mask.unsqueeze(-1).float() # (B, L, 1)
        sum_vec = (proj * mask_f).sum(dim=1) # (B, D)
        counts = mask_f.sum(dim=1).clamp(min=1.0)
        vec = sum_vec / counts
        return nn.functional.normalize(vec, p=2, dim=-1)
    def forward(self, p_mat, p_mask, pos_mat, pos_mask, neg_mat, neg_mask):
        zp = self.encode_set(p_mat, p_mask)
        zpos = self.encode_set(pos_mat, pos_mask)
        zneg = self.encode_set(neg_mat, neg_mask)
        pos_sim = (zp * zpos).sum(dim=-1) # Cosine sim
        neg_sim = (zp * zneg).sum(dim=-1)
        return pos_sim, neg_sim

# 4. Model C: Learned Query-Conditioned Asymmetric Encoder
class ModelCAsymmetric(nn.Module):
    def __init__(self, vocab_size, emb_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.score_head = nn.Linear(emb_dim, 1)
    def encode_patient_query(self, p_mat, p_mask):
        emb = self.embedding(p_mat) # (B, Lp, D)
        proj = self.q_proj(emb)
        mask_f = p_mask.unsqueeze(-1).float()
        q = (proj * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0) # (B, D)
        return q.unsqueeze(1) # (B, 1, D)
    def match(self, q, d_mat, d_mask):
        # q: (B, 1, D), d_mat: (B, Ld), d_mask: (B, Ld)
        d_emb = self.embedding(d_mat) # (B, Ld, D)
        k = self.k_proj(d_emb)        # (B, Ld, D)
        v = self.v_proj(d_emb)        # (B, Ld, D)
        
        # Attention scores
        attn_logits = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(k.size(-1)) # (B, 1, Ld)
        attn_logits = attn_logits.masked_fill(~d_mask.unsqueeze(1), -1e9)
        attn_weights = torch.softmax(attn_logits, dim=-1) # (B, 1, Ld)
        
        # Context vector
        ctx = torch.bmm(attn_weights, v).squeeze(1) # (B, D)
        score = self.score_head(ctx).squeeze(-1)    # (B)
        return score, attn_weights
    def forward(self, p_mat, p_mask, pos_mat, pos_mask, neg_mat, neg_mask):
        q = self.encode_patient_query(p_mat, p_mask)
        pos_score, _ = self.match(q, pos_mat, pos_mask)
        neg_score, _ = self.match(q, neg_mat, neg_mask)
        return pos_score, neg_score

# 5. Training and Validation Loop
def train_and_eval(model_type='B', epochs=10, lr=1e-3, seed=42):
    set_seed(seed)
    train_dataset = PhenotypePairDataset(train_cases, train_dids)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    
    if model_type == 'B':
        model = ModelBSymmetric(vocab_size, emb_dim=64).to(device)
    else:
        model = ModelCAsymmetric(vocab_size, emb_dim=64).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    margin_loss = nn.MarginRankingLoss(margin=0.2)
    
    print(f"\nTraining Model {model_type} (Seed={seed}, Epochs={epochs}, LR={lr})...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for p_mat, p_mask, pos_mat, pos_mask, neg_mat, neg_mask in train_loader:
            p_mat, p_mask = p_mat.to(device), p_mask.to(device)
            pos_mat, pos_mask = pos_mat.to(device), pos_mask.to(device)
            neg_mat, neg_mask = neg_mat.to(device), neg_mask.to(device)
            
            optimizer.zero_grad()
            pos_s, neg_s = model(p_mat, p_mask, pos_mat, pos_mask, neg_mat, neg_mask)
            y = torch.ones_like(pos_s)
            loss = margin_loss(pos_s, neg_s, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")
            
    return model

# 6. Candidate Universe Evaluation Engine (Evaluates against all 12,956 diseases)
def evaluate_neural_model(model, cohort, model_type='B', batch_d=512):
    model.eval()
    hits1, hits5, hits10 = 0, 0, 0
    mrr_total = 0.0
    
    # Pre-encode all candidate diseases if Model B (Static representations)
    if model_type == 'B':
        print("Pre-encoding 12,956 candidate diseases for Model B...")
        d_vecs = []
        with torch.no_grad():
            for i in range(0, len(candidate_diseases), batch_d):
                batch_dids = candidate_diseases[i:i+batch_d]
                mat_list = [disease_hpo_indices[did] for did in batch_dids]
                max_l = max(max([len(x) for x in mat_list]), 1)
                mat = torch.zeros((len(batch_dids), max_l), dtype=torch.long, device=device)
                mask = torch.zeros((len(batch_dids), max_l), dtype=torch.bool, device=device)
                for r, indices in enumerate(mat_list):
                    if indices:
                        mat[r, :len(indices)] = torch.tensor(indices, dtype=torch.long, device=device)
                        mask[r, :len(indices)] = True
                zv = model.encode_set(mat, mask)
                d_vecs.append(zv)
        all_d_vecs = torch.cat(d_vecs, dim=0) # (12956, D)
        
    print(f"Evaluating {len(cohort)} cases against all {len(candidate_diseases)} candidates...")
    with torch.no_grad():
        for c in cohort:
            p_indices = [hpo_vocab[h] for h in c['hpos'] if h in hpo_vocab]
            tgt = c['diseases'][0]
            if not p_indices: continue
            
            p_mat = torch.tensor([p_indices], dtype=torch.long, device=device)
            p_mask = torch.ones((1, len(p_indices)), dtype=torch.bool, device=device)
            
            if model_type == 'B':
                zp = model.encode_set(p_mat, p_mask) # (1, D)
                scores = torch.mm(zp, all_d_vecs.t()).squeeze(0).cpu().numpy() # (12956)
            else:
                # Model C: Query-Conditioned Matching in batches of diseases
                q = model.encode_patient_query(p_mat, p_mask) # (1, 1, D)
                score_chunks = []
                for i in range(0, len(candidate_diseases), batch_d):
                    batch_dids = candidate_diseases[i:i+batch_d]
                    mat_list = [disease_hpo_indices[did] for did in batch_dids]
                    max_l = max(max([len(x) for x in mat_list]), 1)
                    d_mat = torch.zeros((len(batch_dids), max_l), dtype=torch.long, device=device)
                    d_mask = torch.zeros((len(batch_dids), max_l), dtype=torch.bool, device=device)
                    for r, indices in enumerate(mat_list):
                        if indices:
                            d_mat[r, :len(indices)] = torch.tensor(indices, dtype=torch.long, device=device)
                            d_mask[r, :len(indices)] = True
                    q_rep = q.repeat(len(batch_dids), 1, 1)
                    s, _ = model.match(q_rep, d_mat, d_mask)
                    score_chunks.append(s.cpu().numpy())
                scores = np.concatenate(score_chunks)
                
            ranked_indices = np.argsort(-scores)
            ranked_dids = [candidate_diseases[idx] for idx in ranked_indices]
            try:
                r = ranked_dids.index(tgt) + 1
            except ValueError:
                r = len(candidate_diseases)
                
            if r <= 1: hits1 += 1
            if r <= 5: hits5 += 1
            if r <= 10: hits10 += 1
            mrr_total += 1.0 / r
            
    N = len(cohort)
    return {
        'Hits@1': hits1/N*100,
        'Hits@5': hits5/N*100,
        'Hits@10': hits10/N*100,
        'MRR': mrr_total/N
    }

# 7. Execute Validation Tuning
print("\n--- TUNING MODEL B & MODEL C ON VALIDATION COHORT (N=1000) ---")
model_b = train_and_eval('B', epochs=10, lr=1e-3, seed=42)
res_val_b = evaluate_neural_model(model_b, val_cases, 'B')
print(f"Validation Model B (Symmetric):  Hits@1={res_val_b['Hits@1']:5.2f}%, Hits@5={res_val_b['Hits@5']:5.2f}%, Hits@10={res_val_b['Hits@10']:5.2f}%, MRR={res_val_b['MRR']:.4f}")

model_c = train_and_eval('C', epochs=10, lr=1e-3, seed=42)
res_val_c = evaluate_neural_model(model_c, val_cases, 'C')
print(f"Validation Model C (Asymmetric): Hits@1={res_val_c['Hits@1']:5.2f}%, Hits@5={res_val_c['Hits@5']:5.2f}%, Hits@10={res_val_c['Hits@10']:5.2f}%, MRR={res_val_c['MRR']:.4f}")

# 8. Locked Holdout Evaluation on Test Cohort (N=1880)
print("\n=== FINAL TEST EVALUATION ON ZERO-SHOT HOLDOUT (N=1880) ===")
res_test_b = evaluate_neural_model(model_b, test_cases, 'B')
res_test_c = evaluate_neural_model(model_c, test_cases, 'C')

test_mvm_results = {
    'Model_A_Raw_IC': {'Hits@1': 66.91, 'Hits@5': 80.43, 'Hits@10': 84.57, 'MRR': 0.7324},
    'Model_B_Symmetric': res_test_b,
    'Model_C_Asymmetric': res_test_c
}

print(f"\nFinal Locked Zero-Shot Test Summary across 12,956 Candidates:")
for m_name, metrics in test_mvm_results.items():
    print(f"  {m_name:<25} | Hits@1: {metrics['Hits@1']:5.2f}% | Hits@5: {metrics['Hits@5']:5.2f}% | Hits@10: {metrics['Hits@10']:5.2f}% | MRR: {metrics['MRR']:.4f}")

with open('data/benchmark/phase6_mvm_test_results.json', 'w') as out:
    json.dump(test_mvm_results, out, indent=2)
print("Saved final results to data/benchmark/phase6_mvm_test_results.json")
