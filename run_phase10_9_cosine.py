"""
Verify exact formulas for all historical baselines:
- Raw Sum IC (A0_raw)
- Cosine IC (A0_cosine)
- Power-norm gamma=0.10 (A1)
- Deterministic Semantic Residual (A2)
- Learned Symbolic Residual (A3)
"""
import json, math
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

total_kb = len(disease_to_hpos)
hpo_doc_freq = Counter(h for hset in disease_to_hpos.values() for h in hset)
hpo_ic = {h: -math.log((freq + 1)/(total_kb + 1)) for h, freq in hpo_doc_freq.items()}

hpo_to_dids = defaultdict(set)
for did, hset in disease_hpo_sets.items():
    for h in hset:
        hpo_to_dids[h].add(did)

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

# 1. Cosine IC Baseline
h1, h5, h10, mrr = 0, 0, 0, 0.0
for c in test_cases:
    p_pos, tgt = set(c['hpos']), c['diseases'][0]
    cand_scores = {}
    for h in p_pos:
        ic_val = hpo_ic.get(h, 1.0)
        for did in hpo_to_dids.get(h, set()):
            cand_scores[did] = cand_scores.get(did, 0.0) + ic_val
    for did in cand_scores:
        cand_scores[did] = cand_scores[did] / math.sqrt(len(p_pos) * len(disease_hpo_sets[did]))
    tgt_score = cand_scores.get(tgt, -999.0)
    r = sum(1 for did, s in cand_scores.items() if s > tgt_score) + 1
    if r == 1: h1 += 1
    if r <= 5: h5 += 1
    if r <= 10: h10 += 1
    mrr += 1.0 / r

print(f"Cosine IC Baseline:  Hits@1 = {h1/N_t*100:.2f}% | Hits@5 = {h5/N_t*100:.2f}% | Hits@10 = {h10/N_t*100:.2f}% | MRR = {mrr/N_t:.4f}")
