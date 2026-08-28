"""
Continuous Integer Step Evaluation (Q=0, 1, 2, 3, ..., 25)
Evaluates complete step-by-step resolution trajectory and theoretical upper limit.
"""

import json
import math
import random
import time
from collections import defaultdict, Counter
import numpy as np

print("=" * 85)
print("CONTINUOUS STEP BENCHMARK (Q = 0 to 25) ON LOCKED ZERO-SHOT TEST SET")
print("=" * 85)

# 1. Load Data
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

total_kb = len(disease_to_hpos)
hpo_doc_freq = Counter(h for hset in disease_to_hpos.values() for h in hset)
hpo_ic = {h: -math.log((freq + 1)/(total_kb + 1)) for h, freq in hpo_doc_freq.items()}

hpo_to_dids = defaultdict(set)
for did, hset in disease_hpo_sets.items():
    for h in hset:
        hpo_to_dids[h].add(did)

gamma_opt = 0.10
disease_len_pow = {did: len(hset) ** gamma_opt for did, hset in disease_hpo_sets.items()}

# Partition (Seed 42)
random.seed(42)
shuffled_dids = list(all_diseases)
random.shuffle(shuffled_dids)
test_dids = set(shuffled_dids[int(0.85 * len(shuffled_dids)):])
test_cases = [c for c in valid_cases if c['diseases'][0] in test_dids]
N_t = len(test_cases)

def fast_score_and_rank(p_pos, p_neg, target_did, gamma=0.10, lambda_neg=0.50):
    overlap_ic = defaultdict(float)
    for h in p_pos:
        ic_val = hpo_ic.get(h, 1.0)
        for did in hpo_to_dids.get(h, ()):
            overlap_ic[did] += ic_val
    cand_scores = {}
    for did, raw_ic in overlap_ic.items():
        cand_scores[did] = raw_ic / disease_len_pow[did]
    if p_neg:
        for h in p_neg:
            ic_val = hpo_ic.get(h, 1.0)
            for did in hpo_to_dids.get(h, ()):
                if did in cand_scores:
                    cand_scores[did] -= lambda_neg * ic_val
    tgt_score = cand_scores.get(target_did, -999.0)
    sorted_items = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
    top1_did, top1_score = sorted_items[0] if sorted_items else (None, 0.0)
    better_count = sum(1 for did, s in cand_scores.items() if s > tgt_score)
    r = better_count + 1
    return r, top1_did, top1_score, sorted_items[:20]

def select_adaptive_query(p_pos, p_neg, top_items, K=15, alpha_margin=0.35):
    top_cands_items = top_items[:K]
    top_cands = [d for d, s in top_cands_items]
    cand_phenos = set()
    for d in top_cands: 
        cand_phenos |= (disease_hpo_sets[d] - p_pos - p_neg)
    if not cand_phenos: return None
    
    scores_arr = np.array([s for d, s in top_cands_items])
    tau = max(0.5, np.std(scores_arr))
    exp_s = np.exp((scores_arr - np.max(scores_arr)) / tau)
    p_dist = exp_s / np.sum(exp_s)
    
    best_h, best_score = None, -1.0
    for h in cand_phenos:
        p_h = sum(p_dist[i] for i, d in enumerate(top_cands) if h in disease_hpo_sets[d])
        if 0.001 < p_h < 0.999:
            ent = - (p_h * math.log2(p_h) + (1 - p_h) * math.log2(1 - p_h))
            ic_val = hpo_ic.get(h, 1.0)
            in_top1 = (h in disease_hpo_sets[top_cands[0]])
            in_top2 = (len(top_cands) > 1 and h in disease_hpo_sets[top_cands[1]])
            margin_boost = 1.0 + (alpha_margin if (in_top1 != in_top2) else 0.0)
            score = ent * ic_val * margin_boost
            if score > best_score:
                best_score = score
                best_h = h
    return best_h or max(cand_phenos, key=lambda h: hpo_ic.get(h, 1.0))

MAX_Q = 25
results = {q: {'h1': 0, 'h5': 0, 'h10': 0, 'mrr': 0.0} for q in range(MAX_Q + 1)}

t0 = time.time()
for c in test_cases:
    p_pos, tgt = set(c['hpos']), c['diseases'][0]
    p_curr, n_curr = set(p_pos), set()
    
    for q in range(MAX_Q + 1):
        r, _, _, top_curr = fast_score_and_rank(p_curr, n_curr, tgt)
        if r == 1: results[q]['h1'] += 1
        if r <= 5: results[q]['h5'] += 1
        if r <= 10: results[q]['h10'] += 1
        results[q]['mrr'] += 1.0 / r
        
        if q < MAX_Q:
            hq = select_adaptive_query(p_curr, n_curr, top_curr)
            if not hq:
                # If no further informative questions exist, copy forward
                for rem_q in range(q + 1, MAX_Q + 1):
                    if r == 1: results[rem_q]['h1'] += 1
                    if r <= 5: results[rem_q]['h5'] += 1
                    if r <= 10: results[rem_q]['h10'] += 1
                    results[rem_q]['mrr'] += 1.0 / r
                break
            if hq in disease_hpo_sets[tgt]: p_curr.add(hq)
            else: n_curr.add(hq)

elapsed = time.time() - t0
print(f"Continuous evaluation completed in {elapsed:.2f}s")

print("\n" + "=" * 85)
print("CONTINUOUS STEP-BY-STEP TRAJECTORY (Q = 0 to 25)")
print("=" * 85)
print(f"{'Q':<5}{'Hits@1 (%)':<15}{'Hits@5 (%)':<15}{'Hits@10 (%)':<15}{'MRR':<15}{'Marginal Delta':<15}")
print("-" * 85)
prev_h1 = 0
for q in range(MAX_Q + 1):
    h1 = (results[q]['h1'] / N_t) * 100
    h5 = (results[q]['h5'] / N_t) * 100
    h10 = (results[q]['h10'] / N_t) * 100
    mrr = results[q]['mrr'] / N_t
    delta = f"+{h1 - prev_h1:.2f} pp" if q > 0 else "—"
    prev_h1 = h1
    print(f"Q={q:<3}{h1:>6.2f}%{h5:>14.2f}%{h10:>14.2f}%{mrr:>14.4f}{delta:>14}")
print("=" * 85)
