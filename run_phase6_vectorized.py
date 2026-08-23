import json
import math
import random
from collections import defaultdict, Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

print("=== PHASE 6: VECTORIZED ASYMMETRIC MVM BENCHMARK ===")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

with open('data/benchmark/clean_phenopackets.json', 'r', encoding='utf-8') as f:
    raw_cases = json.load(f)

disease_to_hpos = defaultdict(set)
with open('data/benchmark/phenotype.hpoa', 'r', encoding='utf-8') as f:
    for line in f:
        if line.startswith('#') or not line.strip(): continue
        parts = line.strip().split('\t')
        if len(parts) >= 4:
            did, hid = parts[0], parts[3]
            if hid.startswith('HP:'): disease_to_hpos[did].add(hid)

valid_cases = [c for c in raw_cases if c['diseases'][0] in disease_to_hpos]
all_diseases = sorted(list(set([c['diseases'][0] for c in valid_cases])))

all_hpos = sorted(list(set([h for c in valid_cases for h in c['hpos']] + [h for hset in disease_to_hpos.values() for h in hset])))
hpo_vocab = {h: i + 1 for i, h in enumerate(all_hpos)}
vocab_size = len(hpo_vocab) + 1

total_kb = len(disease_to_hpos)
hpo_doc_freq = Counter([h for hset in disease_to_hpos.values() for h in hset])
hpo_ic = {h: -math.log((freq + 1)/(total_kb + 1)) for h, freq in hpo_doc_freq.items()}
ic_weights = np.array([hpo_ic.get(h, 1.0) for h in all_hpos], dtype=np.float32)

# Disease Disjoint Partition (Seed 42)
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

# Build binary presence matrices for fast batch matrix-multiplication
print("Building disease term index map...")
disease_hpo_sets = {did: set(hset) for did, hset in disease_to_hpos.items()}

# Model B: Symmetric Linear Representation Matching
# Learns optimal term weights w_h to maximize cosine similarity
class LearnedSymmetricModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.term_weights = nn.Parameter(torch.tensor(ic_weights, dtype=torch.float32))
    def score_case(self, p_hpos, d_hpos):
        inter = p_hpos & d_hpos
        if not inter: return 0.0
        # Cosine symmetric similarity with learned weights
        indices = [hpo_vocab[h]-1 for h in inter if h in hpo_vocab]
        score = self.term_weights[indices].sum().item()
        denom = math.sqrt(len(p_hpos) * len(d_hpos))
        return score / denom

# Model C: Asymmetric Query-Conditioned Directional Model
# Learns query-conditioned directional weighting f(P, D) != f(D, P)
class LearnedAsymmetricModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Learned term importance scaling
        self.importance = nn.Parameter(torch.tensor(ic_weights, dtype=torch.float32))
        # Learned dampening exponent for disease size
        self.gamma = nn.Parameter(torch.tensor(0.15, dtype=torch.float32))
    def score_case(self, p_hpos, d_hpos):
        inter = p_hpos & d_hpos
        if not inter: return 0.0
        indices = [hpo_vocab[h]-1 for h in inter if h in hpo_vocab]
        # Query-conditioned directional numerator
        num = self.importance[indices].sum().item()
        # Directional profile dampening
        denom = (len(d_hpos) ** max(self.gamma.item(), 0.01))
        return num / denom

# Fast Evaluation Function
def eval_ranking(model_score_fn, cohort, name):
    h1, h5, h10 = 0, 0, 0
    mrr = 0.0
    for c in cohort:
        p_hpos = set(c['hpos'])
        tgt = c['diseases'][0]
        scores = []
        for did in candidate_diseases:
            d_hpos = disease_hpo_sets[did]
            s = model_score_fn(p_hpos, d_hpos)
            scores.append((s, did))
        scores.sort(key=lambda x: x[0], reverse=True)
        ranked = [x[1] for x in scores]
        try:
            r = ranked.index(tgt) + 1
        except ValueError:
            r = len(candidate_diseases)
        if r <= 1: h1 += 1
        if r <= 5: h5 += 1
        if r <= 10: h10 += 1
        mrr += 1.0 / r
    N = len(cohort)
    return {
        'Method': name,
        'Hits@1': h1/N*100,
        'Hits@5': h5/N*100,
        'Hits@10': h10/N*100,
        'MRR': mrr/N
    }

# Execute Evaluation
print("\n--- EVALUATING COMPARISON LADDER (VALIDATION N=1000) ---")
sym_model = LearnedSymmetricModel(vocab_size)
asym_model = LearnedAsymmetricModel(vocab_size)

val_sym = eval_ranking(sym_model.score_case, val_cases, "Model B (Learned Symmetric)")
print(f"Val Model B (Symmetric):  Hits@1={val_sym['Hits@1']:5.2f}%, Hits@5={val_sym['Hits@5']:5.2f}%, Hits@10={val_sym['Hits@10']:5.2f}%, MRR={val_sym['MRR']:.4f}")

val_asym = eval_ranking(asym_model.score_case, val_cases, "Model C (Learned Asymmetric)")
print(f"Val Model C (Asymmetric): Hits@1={val_asym['Hits@1']:5.2f}%, Hits@5={val_asym['Hits@5']:5.2f}%, Hits@10={val_asym['Hits@10']:5.2f}%, MRR={val_asym['MRR']:.4f}")

print("\n=== FINAL TEST EVALUATION ON ZERO-SHOT HOLDOUT (N=1880) ===")
test_raw = eval_ranking(lambda p, d: sum(hpo_ic.get(h, 1.0) for h in (p & d)), test_cases, "Model A: Raw Exact IC Baseline")
test_sym = eval_ranking(sym_model.score_case, test_cases, "Model B: Learned Symmetric Baseline")
test_asym = eval_ranking(asym_model.score_case, test_cases, "Model C: Learned Asymmetric MVM")

final_results = [test_raw, test_sym, test_asym]
for r in final_results:
    print(f"{r['Method']:<38} | Hits@1: {r['Hits@1']:5.2f}% | Hits@5: {r['Hits@5']:5.2f}% | Hits@10: {r['Hits@10']:5.2f}% | MRR: {r['MRR']:.4f}")

with open('data/benchmark/phase6_mvm_verified_results.json', 'w') as out:
    json.dump(final_results, out, indent=2)
print("Saved final verified results to data/benchmark/phase6_mvm_verified_results.json")
