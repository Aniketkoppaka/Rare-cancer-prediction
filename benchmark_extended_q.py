"""
Extended Query Budget Benchmark (Q=0 to Q=15) on Held-Out Unseen Test & Validation Cohorts
"""

import json
import math
import random
import time
from collections import defaultdict, Counter
import numpy as np

print("=" * 85)
print("EXTENDED QUERY BUDGET BENCHMARK (Q = 0 to 15) ON UNSEEN DATA")
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
candidate_diseases = list(disease_to_hpos.keys())

total_kb = len(disease_to_hpos)
hpo_doc_freq = Counter(h for hset in disease_to_hpos.values() for h in hset)
hpo_ic = {h: -math.log((freq + 1)/(total_kb + 1)) for h, freq in hpo_doc_freq.items()}

# Inverted index
hpo_to_dids = defaultdict(set)
for did, hset in disease_hpo_sets.items():
    for h in hset:
        hpo_to_dids[h].add(did)

# Precompute disease lengths power
gamma_opt = 0.10
disease_len_pow = {did: len(hset) ** gamma_opt for did, hset in disease_hpo_sets.items()}

# Partition (Seed 42)
random.seed(42)
shuffled_dids = list(all_diseases)
random.shuffle(shuffled_dids)
n_train_d = int(0.70 * len(shuffled_dids))
n_val_d = int(0.15 * len(shuffled_dids))

train_dids = set(shuffled_dids[:n_train_d])
val_dids = set(shuffled_dids[n_train_d:n_train_d+n_val_d])
test_dids = set(shuffled_dids[n_train_d+n_val_d:])

val_cases = [c for c in valid_cases if c['diseases'][0] in val_dids]
test_cases = [c for c in valid_cases if c['diseases'][0] in test_dids]

print(f"[*] Candidate Universe:            {len(candidate_diseases)} diseases")
print(f"[*] Unseen Val Cohort:             {len(val_cases)} cases | {len(val_dids)} unseen diseases")
print(f"[*] Unseen Zero-Shot Test Cohort:  {len(test_cases)} cases | {len(test_dids)} unseen diseases")

# Fast scoring function
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

# Adaptive Query Policy
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

def run_cohort_evaluation(cases, max_Q=15, q_eval_points=[0, 1, 2, 3, 5, 7, 10, 12, 15]):
    N = len(cases)
    results = {q: {'h1': 0, 'h5': 0, 'h10': 0, 'mrr': 0.0} for q in q_eval_points}
    
    for c in cases:
        p_pos, tgt = set(c['hpos']), c['diseases'][0]
        p_curr, n_curr = set(p_pos), set()
        
        for q in range(max_Q + 1):
            r, _, _, top_curr = fast_score_and_rank(p_curr, n_curr, tgt)
            if q in results:
                if r == 1: results[q]['h1'] += 1
                if r <= 5: results[q]['h5'] += 1
                if r <= 10: results[q]['h10'] += 1
                results[q]['mrr'] += 1.0 / r
                
            if q < max_Q:
                hq = select_adaptive_query(p_curr, n_curr, top_curr, K=15)
                if not hq: 
                    # If exhausted, maintain current performance for remaining Q
                    for remaining_q in q_eval_points:
                        if remaining_q > q:
                            if r == 1: results[remaining_q]['h1'] += 1
                            if r <= 5: results[remaining_q]['h5'] += 1
                            if r <= 10: results[remaining_q]['h10'] += 1
                            results[remaining_q]['mrr'] += 1.0 / r
                    break
                if hq in disease_hpo_sets[tgt]: p_curr.add(hq)
                else: n_curr.add(hq)
                
    return results, N

q_points = [0, 1, 2, 3, 5, 7, 10, 12, 15]

# 1. Evaluate on Unseen Validation Cohort
print("\n[1/2] Evaluating on Unseen Validation Cohort (N=1,000)...")
t0 = time.time()
val_res, val_N = run_cohort_evaluation(val_cases, max_Q=15, q_eval_points=q_points)
print(f"Validation finished in {time.time()-t0:.2f}s")

# 2. Evaluate on Unseen Zero-Shot Test Cohort
print("\n[2/2] Evaluating on Unseen Zero-Shot Test Cohort (N=1,880)...")
t0 = time.time()
test_res, test_N = run_cohort_evaluation(test_cases, max_Q=15, q_eval_points=q_points)
print(f"Test finished in {time.time()-t0:.2f}s")

print("\n" + "=" * 85)
print("EXTENDED TRAJECTORY RESULTS ON UNSEEN DATA (Q = 0 to 15)")
print("=" * 85)
print(f"{'Query (Q)':<12}{'Test Hits@1':<15}{'Test Hits@5':<15}{'Test Hits@10':<15}{'Test MRR':<15}{'Val Hits@1':<12}")
print("-" * 85)
for q in q_points:
    th1 = (test_res[q]['h1'] / test_N) * 100
    th5 = (test_res[q]['h5'] / test_N) * 100
    th10 = (test_res[q]['h10'] / test_N) * 100
    tmrr = test_res[q]['mrr'] / test_N
    vh1 = (val_res[q]['h1'] / val_N) * 100
    print(f"Q = {q:<8}{th1:>6.2f}%{th5:>14.2f}%{th10:>14.2f}%{tmrr:>14.4f}{vh1:>11.2f}%")
print("=" * 85)
