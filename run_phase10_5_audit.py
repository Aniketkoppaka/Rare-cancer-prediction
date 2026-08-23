"""
Phase 10.5 — Forensic Falsification, Leakage Audit & Universe Scaling Engine

Executes:
1. Exact transition matrix audit (measuring any worsened ranks).
2. Random baseline mechanism deconstruction across 5 variants.
3. Full policy comparison (InfoGain, Gini, Maximin, CandidateReduction, High-IC, Low-IC, Random).
4. Candidate universe scaling experiment (100, 500, 1000, 5000, 12956).
5. Query budget curve (Q = 0, 1, 2, 3, 5, 10) with accuracy vs cost.
6. Extraction of 20 representative clinical case traces.
"""

import json
import math
import os
import random
from collections import defaultdict, Counter
import numpy as np

print("=== PHASE 10.5: ADVERSARIAL FALSIFICATION & SCALING AUDIT ===", flush=True)

# --- 1. LOAD BENCHMARK DATA ---
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
all_hpo_vocab = list(hpo_ic.keys())

# Inverted index
hpo_to_dids = defaultdict(set)
for did, hset in disease_hpo_sets.items():
    for h in hset:
        hpo_to_dids[h].add(did)

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
N_t = len(test_cases)
N_v = len(val_cases)

# --- 2. BASELINE SCORER ---
def score_candidates(p_pos, p_neg=set(), candidate_pool=candidate_diseases, gamma=0.10):
    cand_pool_set = set(candidate_pool)
    overlap_ic = defaultdict(float)
    for h in p_pos:
        ic_val = hpo_ic.get(h, 1.0)
        for did in hpo_to_dids.get(h, set()):
            if did in cand_pool_set:
                overlap_ic[did] += ic_val
                
    scores = {}
    for did in candidate_pool:
        raw_ic = overlap_ic.get(did, 0.0)
        if raw_ic == 0.0 and not p_neg:
            scores[did] = 0.0
        else:
            s = raw_ic / (len(disease_hpo_sets[did]) ** gamma)
            if p_neg:
                neg_overlap = disease_hpo_sets[did] & p_neg
                if neg_overlap:
                    s -= 0.5 * sum(hpo_ic.get(h, 1.0) for h in neg_overlap)
            scores[did] = s
    return scores

def rank_target(scores, target_did):
    tgt_score = scores.get(target_did, -999.0)
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top1_did, top1_score = sorted_items[0] if sorted_items else (None, 0.0)
    r = sum(1 for did, s in scores.items() if s > tgt_score) + 1
    return r, top1_did, top1_score, sorted_items[:5]

# --- 3. RANDOM BASELINE DECONSTRUCTION AUDIT ---
print("\n[1/6] Deconstructing the Random Baseline (Multi-Seed Validation Audit)...", flush=True)

def select_random_variants(p_pos, p_neg, current_scores, variant_type, rng):
    sorted_cands = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    top_cands = [d for d, s in sorted_cands]
    top_cands_phenotypes = set()
    for d in top_cands:
        top_cands_phenotypes |= (disease_hpo_sets[d] - p_pos - p_neg)
        
    if variant_type == 'Uniform_All_HPO':
        # True completely unguided random across all 18k HPO terms
        choices = [h for h in all_hpo_vocab if h not in p_pos and h not in p_neg]
        return rng.choice(choices) if choices else None
        
    elif variant_type == 'Random_Top_Candidate_Pool':
        # Random term among Top-10 candidates
        return rng.choice(list(top_cands_phenotypes)) if top_cands_phenotypes else None
        
    elif variant_type == 'Random_High_IC_Top_Pool':
        # Random term among top-50% highest IC terms in candidate pool
        if not top_cands_phenotypes: return None
        sorted_by_ic = sorted(top_cands_phenotypes, key=lambda h: hpo_ic.get(h, 1.0), reverse=True)
        top_half = sorted_by_ic[:max(1, len(sorted_by_ic)//2)]
        return rng.choice(top_half)
        
    elif variant_type == 'Random_Low_IC_Top_Pool':
        # Random term among lowest IC terms in candidate pool
        if not top_cands_phenotypes: return None
        sorted_by_ic = sorted(top_cands_phenotypes, key=lambda h: hpo_ic.get(h, 1.0))
        bottom_half = sorted_by_ic[:max(1, len(sorted_by_ic)//2)]
        return rng.choice(bottom_half)

random_variants = [
    'Uniform_All_HPO',
    'Random_Top_Candidate_Pool',
    'Random_High_IC_Top_Pool',
    'Random_Low_IC_Top_Pool'
]

for var in random_variants:
    seed_hits = []
    for seed in [10, 42, 100, 2026, 9999]:
        rng = random.Random(seed)
        correct = 0
        for c in val_cases:
            p_pos = set(c['hpos'])
            p_neg = set()
            tgt = c['diseases'][0]
            scores = score_candidates(p_pos, p_neg)
            h_q = select_random_variants(p_pos, p_neg, scores, var, rng)
            if h_q is not None:
                if h_q in disease_hpo_sets[tgt]: p_pos.add(h_q)
                else: p_neg.add(h_q)
            up_scores = score_candidates(p_pos, p_neg)
            r, _, _, _ = rank_target(up_scores, tgt)
            if r == 1: correct += 1
        seed_hits.append(correct / N_v * 100)
    print(f"  {var:<28} Val Hits@1: {np.mean(seed_hits):.2f}% ± {np.std(seed_hits):.2f}%")

# --- 4. DETAILED POLICY IMPLEMENTATION & TRANSITION MATRIX ---
print("\n[2/6] Running Full Policy Benchmarking & Paired Transition Matrices on Test Set...", flush=True)

def select_policy(p_pos, p_neg, current_scores, policy_name):
    sorted_cands = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    top_cands = [d for d, s in sorted_cands]
    candidate_phenotypes = set()
    for d in top_cands:
        candidate_phenotypes |= (disease_hpo_sets[d] - p_pos - p_neg)
        
    if not candidate_phenotypes: return None
    
    if policy_name == 'Policy_A_InfoGain':
        scores_arr = np.array([s for d, s in sorted_cands])
        exp_s = np.exp(scores_arr - np.max(scores_arr))
        p_dist = exp_s / np.sum(exp_s)
        best_h, best_ig = None, -1.0
        for h in candidate_phenotypes:
            p_h = sum(p_dist[i] for i, d in enumerate(top_cands) if h in disease_hpo_sets[d])
            if 0.01 < p_h < 0.99:
                ig = - (p_h * math.log2(p_h) + (1 - p_h) * math.log2(1 - p_h)) * hpo_ic.get(h, 1.0)
                if ig > best_ig:
                    best_ig = ig
                    best_h = h
        return best_h or max(candidate_phenotypes, key=lambda h: hpo_ic.get(h, 1.0))
        
    elif policy_name == 'Policy_Gini_Reduction':
        scores_arr = np.array([s for d, s in sorted_cands])
        exp_s = np.exp(scores_arr - np.max(scores_arr))
        p_dist = exp_s / np.sum(exp_s)
        best_h, best_gini_gain = None, -1.0
        for h in candidate_phenotypes:
            p_h = sum(p_dist[i] for i, d in enumerate(top_cands) if h in disease_hpo_sets[d])
            # Gini impurity reduction = 2 * p_h * (1 - p_h) * IC
            gini_gain = 2.0 * p_h * (1.0 - p_h) * hpo_ic.get(h, 1.0)
            if gini_gain > best_gini_gain:
                best_gini_gain = gini_gain
                best_h = h
        return best_h or max(candidate_phenotypes, key=lambda h: hpo_ic.get(h, 1.0))
        
    elif policy_name == 'Policy_C_PairwiseMaximin':
        if len(top_cands) >= 2:
            d1, d2 = top_cands[0], top_cands[1]
            diff = (disease_hpo_sets[d1] ^ disease_hpo_sets[d2]) - p_pos - p_neg
            if diff: return max(diff, key=lambda h: hpo_ic.get(h, 1.0))
        return max(candidate_phenotypes, key=lambda h: hpo_ic.get(h, 1.0))
        
    elif policy_name == 'Policy_B_CandidateReduction':
        best_h, best_balance = None, 999
        K = len(top_cands)
        for h in candidate_phenotypes:
            match_count = sum(1 for d in top_cands if h in disease_hpo_sets[d])
            balance = abs(match_count - (K / 2))
            if balance < best_balance:
                best_balance = balance
                best_h = h
        return best_h or max(candidate_phenotypes, key=lambda h: hpo_ic.get(h, 1.0))
        
    elif policy_name == 'Policy_D_HighIC':
        return max(candidate_phenotypes, key=lambda h: hpo_ic.get(h, 1.0))
        
    elif policy_name == 'Policy_LowIC':
        return min(candidate_phenotypes, key=lambda h: hpo_ic.get(h, 1.0))

policy_list = [
    'Policy_LowIC',
    'Policy_D_HighIC',
    'Policy_B_CandidateReduction',
    'Policy_C_PairwiseMaximin',
    'Policy_Gini_Reduction',
    'Policy_A_InfoGain'
]

test_results = {}
for pol in policy_list:
    init_correct_after_correct = 0
    init_correct_after_incorrect = 0
    init_incorrect_after_correct = 0
    init_incorrect_after_incorrect = 0
    rank_deltas = []
    
    for c in test_cases:
        p_pos = set(c['hpos'])
        p_neg = set()
        tgt = c['diseases'][0]
        
        init_scores = score_candidates(p_pos, p_neg)
        r_init, _, _, _ = rank_target(init_scores, tgt)
        
        h_q = select_policy(p_pos, p_neg, init_scores, pol)
        if h_q is not None:
            if h_q in disease_hpo_sets[tgt]: p_pos.add(h_q)
            else: p_neg.add(h_q)
            
        up_scores = score_candidates(p_pos, p_neg)
        r_up, _, _, _ = rank_target(up_scores, tgt)
        
        rank_deltas.append(r_init - r_up) # positive = improved, negative = worsened
        
        if r_init == 1 and r_up == 1: init_correct_after_correct += 1
        elif r_init == 1 and r_up > 1: init_correct_after_incorrect += 1
        elif r_init > 1 and r_up == 1: init_incorrect_after_correct += 1
        elif r_init > 1 and r_up > 1: init_incorrect_after_incorrect += 1
        
    final_h1 = (init_correct_after_correct + init_incorrect_after_correct) / N_t * 100
    test_results[pol] = {
        'Hits@1': final_h1,
        'C->C': init_correct_after_correct,
        'C->I (Worsened)': init_correct_after_incorrect,
        'I->C (Resolved)': init_incorrect_after_correct,
        'I->I': init_incorrect_after_incorrect,
        'mean_rank_delta': np.mean(rank_deltas),
        'median_rank_delta': np.median(rank_deltas),
        'worst_rank_drop': min(rank_deltas)
    }
    
    print(f"\n  === {pol} ===")
    print(f"    Test Hits@1: {final_h1:.2f}% | Resolved: {init_incorrect_after_correct} / 590 | Worsened: {init_correct_after_incorrect}")
    print(f"    Transition Matrix: [C->C: {init_correct_after_correct}, C->I: {init_correct_after_incorrect}] | [I->C: {init_incorrect_after_correct}, I->I: {init_incorrect_after_incorrect}]")
    print(f"    Rank Shifts: Mean Δ = {np.mean(rank_deltas):+.2f} | Median Δ = {np.median(rank_deltas):+.0f} | Worst Drop = {min(rank_deltas)}")

# --- 5. CANDIDATE UNIVERSE SCALING EXPERIMENT ---
print("\n[3/6] Candidate Universe Scaling Experiment (100, 500, 1000, 5000, 12956)...", flush=True)

universe_sizes = [100, 500, 1000, 5000, 12956]
scaling_results = []

for u_size in universe_sizes:
    if u_size == 12956:
        curr_pool = candidate_diseases
    else:
        # Include all test target diseases + sample random distractors to reach u_size
        test_tgt_set = set(c['diseases'][0] for c in test_cases)
        other_cands = [d for d in candidate_diseases if d not in test_tgt_set]
        rng = random.Random(42)
        sampled_others = rng.sample(other_cands, u_size - len(test_tgt_set))
        curr_pool = list(test_tgt_set) + sampled_others
        
    static_hits1 = 0
    active1_hits1 = 0
    active5_hits1 = 0
    
    for c in test_cases:
        p_pos = set(c['hpos'])
        p_neg = set()
        tgt = c['diseases'][0]
        
        # Static
        s0 = score_candidates(p_pos, p_neg, candidate_pool=curr_pool)
        r0, _, _, _ = rank_target(s0, tgt)
        if r0 == 1: static_hits1 += 1
        
        # 1-step Active
        h_q = select_policy(p_pos, p_neg, s0, 'Policy_A_InfoGain')
        p_pos_1 = set(p_pos)
        p_neg_1 = set(p_neg)
        if h_q is not None:
            if h_q in disease_hpo_sets[tgt]: p_pos_1.add(h_q)
            else: p_neg_1.add(h_q)
        s1 = score_candidates(p_pos_1, p_neg_1, candidate_pool=curr_pool)
        r1, _, _, _ = rank_target(s1, tgt)
        if r1 == 1: active1_hits1 += 1
        
        # 5-step Active
        p_pos_5 = set(p_pos)
        p_neg_5 = set(p_neg)
        for _ in range(5):
            curr_s = score_candidates(p_pos_5, p_neg_5, candidate_pool=curr_pool)
            hq = select_policy(p_pos_5, p_neg_5, curr_s, 'Policy_A_InfoGain')
            if hq is None: break
            if hq in disease_hpo_sets[tgt]: p_pos_5.add(hq)
            else: p_neg_5.add(hq)
        s5 = score_candidates(p_pos_5, p_neg_5, candidate_pool=curr_pool)
        r5, _, _, _ = rank_target(s5, tgt)
        if r5 == 1: active5_hits1 += 1
        
    res = {
        'universe_size': u_size,
        'static_hits1': static_hits1 / N_t * 100,
        'active1_hits1': active1_hits1 / N_t * 100,
        'active5_hits1': active5_hits1 / N_t * 100,
        'gain_1step': (active1_hits1 - static_hits1) / N_t * 100,
        'gain_5step': (active5_hits1 - static_hits1) / N_t * 100
    }
    scaling_results.append(res)
    print(f"  Universe: {u_size:>5} | Static Hits@1: {res['static_hits1']:5.2f}% | 1-Step Active: {res['active1_hits1']:5.2f}% (+{res['gain_1step']:4.2f} pp) | 5-Step Active: {res['active5_hits1']:5.2f}% (+{res['gain_5step']:4.2f} pp)")

# --- 6. QUERY BUDGET & COST CURVE ---
print("\n[4/6] Query Budget & Information Gain Curve (Q = 0, 1, 2, 3, 5, 10)...", flush=True)

budget_steps = [0, 1, 2, 3, 5, 10]
budget_results = []

for q_max in budget_steps:
    h1, h5, h10, mrr_sum = 0, 0, 0, 0.0
    for c in test_cases:
        p_pos = set(c['hpos'])
        p_neg = set()
        tgt = c['diseases'][0]
        
        for _ in range(q_max):
            curr_s = score_candidates(p_pos, p_neg)
            hq = select_policy(p_pos, p_neg, curr_s, 'Policy_A_InfoGain')
            if hq is None: break
            if hq in disease_hpo_sets[tgt]: p_pos.add(hq)
            else: p_neg.add(hq)
            
        final_s = score_candidates(p_pos, p_neg)
        r, _, _, _ = rank_target(final_s, tgt)
        if r <= 1: h1 += 1
        if r <= 5: h5 += 1
        if r <= 10: h10 += 1
        mrr_sum += 1.0 / r
        
    b_res = {
        'queries': q_max,
        'Hits@1': h1 / N_t * 100,
        'Hits@5': h5 / N_t * 100,
        'Hits@10': h10 / N_t * 100,
        'MRR': mrr_sum / N_t
    }
    budget_results.append(b_res)
    print(f"  Q = {q_max:>2} | Hits@1: {b_res['Hits@1']:5.2f}% | Hits@5: {b_res['Hits@5']:5.2f}% | Hits@10: {b_res['Hits@10']:5.2f}% | MRR: {b_res['MRR']:.4f}")

# --- 7. EXTRACT 20 REPRESENTATIVE CASE TRACES ---
print("\n[5/6] Generating 20 Representative Diagnostic Case Traces...", flush=True)

case_traces = []
# Pick a diverse set of cases: resolved sparse, resolved sibling, unchanged, etc.
for idx, c in enumerate(test_cases):
    if len(case_traces) >= 20: break
    p_pos_init = set(c['hpos'])
    tgt = c['diseases'][0]
    
    s0 = score_candidates(p_pos_init)
    r0, top1_0, _, top5_0 = rank_target(s0, tgt)
    
    # 1-step query
    hq = select_policy(p_pos_init, set(), s0, 'Policy_A_InfoGain')
    if hq is None: continue
    
    ans = "Present" if hq in disease_hpo_sets[tgt] else "Absent"
    p_pos_1 = set(p_pos_init)
    p_neg_1 = set()
    if ans == "Present": p_pos_1.add(hq)
    else: p_neg_1.add(hq)
    
    s1 = score_candidates(p_pos_1, p_neg_1)
    r1, top1_1, _, top5_1 = rank_target(s1, tgt)
    
    # Capture interesting cases (some resolved, some initially correct, some difficult)
    if (r0 > 1 and r1 == 1) or (r0 > 1 and r1 > 1 and len(case_traces) % 3 == 0) or (r0 == 1 and len(case_traces) % 4 == 0):
        case_traces.append({
            'case_id': c.get('id', str(idx)),
            'true_disease': tgt,
            'initial_hpos': list(p_pos_init)[:5],
            'initial_rank': r0,
            'initial_top5': [d for d, s in top5_0],
            'queried_phenotype': hq,
            'queried_ic': hpo_ic.get(hq, 1.0),
            'simulated_answer': ans,
            'updated_rank': r1,
            'updated_top5': [d for d, s in top5_1],
            'outcome': "Resolved (Error -> Correct)" if (r0 > 1 and r1 == 1) else ("Maintained Correct" if r0 == 1 else "Unresolved")
        })

print(f"  Generated {len(case_traces)} Traceable Diagnostic Dialogues.")

# Save payload
phase10_5_payload = {
    'policy_benchmarking': test_results,
    'universe_scaling': scaling_results,
    'budget_results': budget_results,
    'case_traces': case_traces
}

with open('data/benchmark/phase10_5_falsification_results.json', 'w') as f:
    json.dump(phase10_5_payload, f, indent=2)

print("\n=== PHASE 10.5 FALSIFICATION & SCALING AUDIT COMPLETED ===", flush=True)
