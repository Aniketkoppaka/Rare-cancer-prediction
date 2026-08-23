"""
Phase 10.8 — Lightweight Entropy Quartile Sensitivity & Provenance Generator
"""

import json
import math
from collections import defaultdict, Counter
import numpy as np

# Load Benchmark Data
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

def fast_score_and_rank(p_pos, p_neg, target_did, gamma=0.10, lambda_neg=0.50):
    overlap_ic = defaultdict(float)
    for h in p_pos:
        ic_val = hpo_ic.get(h, 1.0)
        for did in hpo_to_dids.get(h, set()):
            overlap_ic[did] += ic_val
    cand_scores = {}
    for did, raw_ic in overlap_ic.items():
        cand_scores[did] = raw_ic / (len(disease_hpo_sets[did]) ** gamma)
    if p_neg:
        for h in p_neg:
            ic_val = hpo_ic.get(h, 1.0)
            for did in hpo_to_dids.get(h, set()):
                if did in cand_scores:
                    cand_scores[did] -= lambda_neg * ic_val
    tgt_score = cand_scores.get(target_did, -999.0)
    sorted_items = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
    top1_did, top1_score = sorted_items[0] if sorted_items else (None, 0.0)
    better_count = sum(1 for did, s in cand_scores.items() if s > tgt_score)
    r = better_count + 1
    return r, top1_did, top1_score, sorted_items[:10]

def select_policy_query(p_pos, p_neg, top10_items, K=10):
    top_cands_items = top10_items[:K]
    top_cands = [d for d, s in top_cands_items]
    cand_phenos = set()
    for d in top_cands: cand_phenos |= (disease_hpo_sets[d] - p_pos - p_neg)
    if not cand_phenos: return None
    scores_arr = np.array([s for d, s in top_cands_items])
    exp_s = np.exp(scores_arr - np.max(scores_arr))
    p_dist = exp_s / np.sum(exp_s)
    best_h, best_score = None, -1.0
    for h in cand_phenos:
        p_h = sum(p_dist[i] for i, d in enumerate(top_cands) if h in disease_hpo_sets[d])
        if 0.001 < p_h < 0.999:
            ent = - (p_h * math.log2(p_h) + (1 - p_h) * math.log2(1 - p_h))
            score = ent * hpo_ic.get(h, 1.0)
            if score > best_score:
                best_score = score
                best_h = h
    return best_h or max(cand_phenos, key=lambda h: hpo_ic.get(h, 1.0))

# Run all test cases and collect ranks + initial entropy
ranks_0, ranks_1, ranks_5, entropies = [], [], [], []

for c in test_cases:
    p_pos, tgt = set(c['hpos']), c['diseases'][0]
    r0, _, _, top10_0 = fast_score_and_rank(p_pos, set(), tgt)
    ranks_0.append(r0)
    
    # Entropy
    top10_scores = np.array([s for d, s in top10_0])
    exp_s = np.exp(top10_scores - np.max(top10_scores))
    p_dist = exp_s / np.sum(exp_s)
    ent = -np.sum(p_dist * np.log2(p_dist + 1e-12))
    entropies.append(ent)
    
    # Q=1
    p1_pos, p1_neg = set(p_pos), set()
    hq1 = select_policy_query(p1_pos, p1_neg, top10_0)
    if hq1:
        if hq1 in disease_hpo_sets[tgt]: p1_pos.add(hq1)
        else: p1_neg.add(hq1)
    r1, _, _, _ = fast_score_and_rank(p1_pos, p1_neg, tgt)
    ranks_1.append(r1)
    
    # Q=5
    p5_pos, p5_neg = set(p_pos), set()
    for _ in range(5):
        _, _, _, top10_curr = fast_score_and_rank(p5_pos, p5_neg, tgt)
        hq = select_policy_query(p5_pos, p5_neg, top10_curr)
        if not hq: break
        if hq in disease_hpo_sets[tgt]: p5_pos.add(hq)
        else: p5_neg.add(hq)
    r5, _, _, _ = fast_score_and_rank(p5_pos, p5_neg, tgt)
    ranks_5.append(r5)

r0_arr, r1_arr, r5_arr, ent_arr = np.array(ranks_0), np.array(ranks_1), np.array(ranks_5), np.array(entropies)

print("=== ENTROPY MULTI-QUANTILE ROBUSTNESS AUDIT ===")
# Quartiles
q25, q50, q75 = np.percentile(ent_arr, [25, 50, 75])
print(f"Entropy Quartiles: Q25={q25:.3f}, Q50={q50:.3f}, Q75={q75:.3f} bits")

masks = [
    ("Quartile 1 (Lowest Ambiguity: H < Q25)", ent_arr < q25),
    ("Quartile 2 (Low-Mid Ambiguity: Q25 <= H < Q50)", (ent_arr >= q25) & (ent_arr < q50)),
    ("Quartile 3 (Mid-High Ambiguity: Q50 <= H < Q75)", (ent_arr >= q50) & (ent_arr < q75)),
    ("Quartile 4 (Highest Ambiguity: H >= Q75)", ent_arr >= q75),
]

for label, m in masks:
    s0 = np.mean(r0_arr[m] == 1) * 100
    s1 = np.mean(r1_arr[m] == 1) * 100
    s5 = np.mean(r5_arr[m] == 1) * 100
    print(f"  {label:<45} N={np.sum(m):>3} | Static: {s0:5.2f}% | Q=1: {s1:5.2f}% (+{s1-s0:4.2f} pp) | Q=5: {s5:5.2f}% (+{s5-s0:5.2f} pp)")

print("\n=== SENSITIVITY SPLITS ===")
# Lower 25% vs Upper 25%
m_bot25 = ent_arr < q25
m_top25 = ent_arr >= q75
print(f"  Bottom 25% vs Top 25% 5-Step Gain: +{np.mean(r5_arr[m_bot25]==1)*100 - np.mean(r0_arr[m_bot25]==1)*100:.2f} pp vs +{np.mean(r5_arr[m_top25]==1)*100 - np.mean(r0_arr[m_top25]==1)*100:.2f} pp")
