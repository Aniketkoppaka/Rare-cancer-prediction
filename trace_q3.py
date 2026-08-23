"""
Trace exact Q=3 case counts (1500 vs 1501) between standalone and loop evaluator.
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

def select_policy_a_query(p_pos, p_neg, top10_items, K=10):
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

h3_count = 0
for c in test_cases:
    p_pos, tgt = set(c['hpos']), c['diseases'][0]
    p_curr, n_curr = set(p_pos), set()
    for q in range(3):
        r, _, _, top10_curr = fast_score_and_rank(p_curr, n_curr, tgt)
        hq = select_policy_a_query(p_curr, n_curr, top10_curr)
        if not hq: break
        if hq in disease_hpo_sets[tgt]: p_curr.add(hq)
        else: n_curr.add(hq)
    r3, _, _, _ = fast_score_and_rank(p_curr, n_curr, tgt)
    if r3 == 1: h3_count += 1

print(f"Exact Q=3 Correct Cases: {h3_count} / {N_t} = {h3_count / N_t * 100:.2f}% ({h3_count / N_t * 100:.4f}%)")
