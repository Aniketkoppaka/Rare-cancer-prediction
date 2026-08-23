"""
Phase 10.9 — Baseline Ladder & Discrepancy Verification Script
Computes exact Hits@1, Hits@5, Hits@10, MRR for A0, A1, A2, A3 on locked test set.
"""

import json
import math
from collections import defaultdict, Counter
import numpy as np

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
disease_hpo_sets = {did: set(hset) for did, hset in disease_to_hpos.items()}
candidate_diseases = list(disease_to_hpos.keys())
all_candidate_set = set(candidate_diseases)

total_kb = len(disease_to_hpos)
hpo_doc_freq = Counter(h for hset in disease_to_hpos.values() for h in hset)
hpo_ic = {h: -math.log((freq + 1)/(total_kb + 1)) for h, freq in hpo_doc_freq.items()}

# Inverted index
hpo_to_dids = defaultdict(set)
for did, hset in disease_hpo_sets.items():
    for h in hset:
        hpo_to_dids[h].add(did)

# Partition (Seed 42)
import random
random.seed(42)
shuffled_dids = list(all_diseases)
random.shuffle(shuffled_dids)
n_train_d = int(0.70 * len(shuffled_dids))
n_val_d = int(0.15 * len(shuffled_dids))

train_dids = set(shuffled_dids[:n_train_d])
val_dids = set(shuffled_dids[n_train_d:n_train_d+n_val_d])
test_dids = set(shuffled_dids[n_train_d+n_val_d:])

test_cases = [c for c in valid_cases if c['diseases'][0] in test_dids]
N_t = len(test_cases)

def evaluate_baseline(gamma=0.0):
    h1, h5, h10 = 0, 0, 0
    mrr_sum = 0.0
    for c in test_cases:
        p_pos = set(c['hpos'])
        tgt = c['diseases'][0]
        overlap_ic = defaultdict(float)
        for h in p_pos:
            ic_val = hpo_ic.get(h, 1.0)
            for did in hpo_to_dids.get(h, set()):
                overlap_ic[did] += ic_val
        cand_scores = {}
        for did, raw_ic in overlap_ic.items():
            denom = (len(disease_hpo_sets[did]) ** gamma) if gamma > 0 else 1.0
            cand_scores[did] = raw_ic / denom
        tgt_score = cand_scores.get(tgt, -999.0)
        better_count = sum(1 for did, s in cand_scores.items() if s > tgt_score)
        r = better_count + 1
        if r == 1: h1 += 1
        if r <= 5: h5 += 1
        if r <= 10: h10 += 1
        mrr_sum += 1.0 / r
    return h1 / N_t * 100, h5 / N_t * 100, h10 / N_t * 100, mrr_sum / N_t

print("=== EXACT BASELINE VERIFICATION ===")
a0_h1, a0_h5, a0_h10, a0_mrr = evaluate_baseline(gamma=0.0)
a1_h1, a1_h5, a1_h10, a1_mrr = evaluate_baseline(gamma=0.10)

print(f"A0 Raw IC (gamma=0.0):   Hits@1 = {a0_h1:.2f}% | Hits@5 = {a0_h5:.2f}% | Hits@10 = {a0_h10:.2f}% | MRR = {a0_mrr:.4f}")
print(f"A1 Power-Norm (gamma=0.10): Hits@1 = {a1_h1:.2f}% | Hits@5 = {a1_h5:.2f}% | Hits@10 = {a1_h10:.2f}% | MRR = {a1_mrr:.4f}")
