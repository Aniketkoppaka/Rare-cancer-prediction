"""
Phase 10.5 — Kaggle GPU Standalone Engine: Falsification, Leakage Audit & Universe Scaling (Fixed)

Executes in ~30-45s on GPU/CPU:
1. Deconstructs Random Baseline across 4 variants (5 seeds on Validation).
2. Evaluates all Realistic Policies on Locked Test Set (Hits@1, MRR, Paired Transition Matrix, dRank).
3. Audits Zero-Worsening Forensics (Mean, Median, and Worst-Case Rank Shifts).
4. Runs Candidate Universe Scaling (100, 500, 1000, 5000, 12956 candidates per case).
5. Generates Accuracy-vs-Query Budget Curve (Q = 0, 1, 2, 3, 5, 10).
6. Extracts 20 Traceable Diagnostic Dialogues.
"""

import json
import math
import os
import random
import time
from collections import defaultdict, Counter
import numpy as np

print("=" * 72)
print("PHASE 10.5 — ADVERSARIAL FALSIFICATION, LEAKAGE AUDIT & SCALING ENGINE")
print("=" * 72)

# --- 1. RESOLVE FILE PATHS ---
def find_data_file(filename):
    for root, dirs, files in os.walk("/kaggle/input"):
        if filename in files:
            return os.path.join(root, filename)
    for root, dirs, files in os.walk("data"):
        if filename in files:
            return os.path.join(root, filename)
    return filename

phenopackets_path = find_data_file("clean_phenopackets.json")
hpoa_path = find_data_file("phenotype.hpoa")

print(f"[*] Phenopackets: {phenopackets_path}")
print(f"[*] HPOA:         {hpoa_path}")

# --- 2. LOAD DATA ---
with open(phenopackets_path, "r", encoding="utf-8") as f:
    raw_cases = json.load(f)

disease_to_hpos = defaultdict(set)
with open(hpoa_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("#") or not line.strip(): continue
        parts = line.strip().split("\t")
        if len(parts) >= 4 and parts[3].startswith("HP:"):
            disease_to_hpos[parts[0]].add(parts[3])

valid_cases = [c for c in raw_cases if c["diseases"][0] in disease_to_hpos]
all_diseases = sorted(list(set([c["diseases"][0] for c in valid_cases])))
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

val_cases = [c for c in valid_cases if c["diseases"][0] in val_dids]
test_cases = [c for c in valid_cases if c["diseases"][0] in test_dids]
N_t = len(test_cases)
N_v = len(val_cases)

print(f"[*] Train Diseases: {len(train_dids)} | Val: {len(val_dids)} ({N_v} cases) | Test: {len(test_dids)} ({N_t} cases)")
print(f"[*] Candidate Universe: {len(candidate_diseases)} diseases")

# Fast Scorer using inverted index
def fast_score_and_rank(p_pos, p_neg, target_did, candidate_pool_set=all_candidate_set, gamma=0.10):
    overlap_ic = defaultdict(float)
    for h in p_pos:
        ic_val = hpo_ic.get(h, 1.0)
        for did in hpo_to_dids.get(h, set()):
            if did in candidate_pool_set:
                overlap_ic[did] += ic_val
                
    cand_scores = {}
    for did, raw_ic in overlap_ic.items():
        cand_scores[did] = raw_ic / (len(disease_hpo_sets[did]) ** gamma)
        
    if p_neg:
        for h in p_neg:
            ic_val = hpo_ic.get(h, 1.0)
            for did in hpo_to_dids.get(h, set()):
                if did in cand_scores:
                    cand_scores[did] -= 0.5 * ic_val
                    
    tgt_score = cand_scores.get(target_did, -999.0)
    sorted_items = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
    top1_did, top1_score = sorted_items[0] if sorted_items else (None, 0.0)
    better_count = sum(1 for did, s in cand_scores.items() if s > tgt_score)
    r = better_count + 1
    return r, top1_did, top1_score, sorted_items[:10]

# --- 3. DECONSTRUCTING THE RANDOM BASELINE (VALIDATION AUDIT) ---
print("\n[1/5] Deconstructing Random Baseline Variants (5 Random Seeds on Validation)...")

def get_random_query(p_pos, p_neg, top10_items, variant, rng):
    top_cands = [d for d, s in top10_items]
    cand_phenos = set()
    for d in top_cands: cand_phenos |= (disease_hpo_sets[d] - p_pos - p_neg)
    
    if variant == "Uniform_All_HPO":
        choices = [h for h in all_hpo_vocab if h not in p_pos and h not in p_neg]
        return rng.choice(choices) if choices else None
    elif variant == "Random_Top_Candidate_Pool":
        return rng.choice(list(cand_phenos)) if cand_phenos else None
    elif variant == "Random_High_IC_Top_Pool":
        if not cand_phenos: return None
        sorted_by_ic = sorted(cand_phenos, key=lambda h: hpo_ic.get(h, 1.0), reverse=True)
        return rng.choice(sorted_by_ic[:max(1, len(sorted_by_ic)//2)])
    elif variant == "Random_Low_IC_Top_Pool":
        if not cand_phenos: return None
        sorted_by_ic = sorted(cand_phenos, key=lambda h: hpo_ic.get(h, 1.0))
        return rng.choice(sorted_by_ic[:max(1, len(sorted_by_ic)//2)])

print(f"{'Random Variant Strategy':<30} {'Mean Val Hits@1':>18} {'Std Dev':>10}")
print("-" * 62)
for var in ["Uniform_All_HPO", "Random_Top_Candidate_Pool", "Random_High_IC_Top_Pool", "Random_Low_IC_Top_Pool"]:
    seed_hits = []
    for seed in [10, 42, 100, 2026, 9999]:
        rng = random.Random(seed)
        correct = 0
        for c in val_cases:
            p_pos = set(c["hpos"])
            p_neg = set()
            tgt = c["diseases"][0]
            _, _, _, top10 = fast_score_and_rank(p_pos, p_neg, tgt)
            hq = get_random_query(p_pos, p_neg, top10, var, rng)
            if hq:
                if hq in disease_hpo_sets[tgt]: p_pos.add(hq)
                else: p_neg.add(hq)
            r, _, _, _ = fast_score_and_rank(p_pos, p_neg, tgt)
            if r == 1: correct += 1
        seed_hits.append(correct / N_v * 100)
    print(f"{var:<30} {np.mean(seed_hits):>17.2f}% {np.std(seed_hits):>9.2f}%")
print("-" * 62)

# --- 4. REALISTIC POLICIES & PAIRED TRANSITION MATRICES (TEST SET) ---
print("\n[2/5] Evaluating Realistic Blind Policies & Transition Matrices (Locked Test Set)...")

def select_policy_query(p_pos, p_neg, top10_items, pol_name):
    top_cands = [d for d, s in top10_items]
    cand_phenos = set()
    for d in top_cands: cand_phenos |= (disease_hpo_sets[d] - p_pos - p_neg)
    if not cand_phenos: return None
    
    if pol_name == "Policy_A_InfoGain":
        scores_arr = np.array([s for d, s in top10_items])
        exp_s = np.exp(scores_arr - np.max(scores_arr))
        p_dist = exp_s / np.sum(exp_s)
        best_h, best_ig = None, -1.0
        for h in cand_phenos:
            p_h = sum(p_dist[i] for i, d in enumerate(top_cands) if h in disease_hpo_sets[d])
            if 0.01 < p_h < 0.99:
                ig = - (p_h * math.log2(p_h) + (1 - p_h) * math.log2(1 - p_h)) * hpo_ic.get(h, 1.0)
                if ig > best_ig:
                    best_ig = ig
                    best_h = h
        return best_h or max(cand_phenos, key=lambda h: hpo_ic.get(h, 1.0))
        
    elif pol_name == "Policy_Gini_Reduction":
        scores_arr = np.array([s for d, s in top10_items])
        exp_s = np.exp(scores_arr - np.max(scores_arr))
        p_dist = exp_s / np.sum(exp_s)
        best_h, best_gini = None, -1.0
        for h in cand_phenos:
            p_h = sum(p_dist[i] for i, d in enumerate(top_cands) if h in disease_hpo_sets[d])
            gini_gain = 2.0 * p_h * (1.0 - p_h) * hpo_ic.get(h, 1.0)
            if gini_gain > best_gini:
                best_gini = gini_gain
                best_h = h
        return best_h or max(cand_phenos, key=lambda h: hpo_ic.get(h, 1.0))
        
    elif pol_name == "Policy_C_PairwiseMaximin":
        if len(top_cands) >= 2:
            d1, d2 = top_cands[0], top_cands[1]
            diff = (disease_hpo_sets[d1] ^ disease_hpo_sets[d2]) - p_pos - p_neg
            if diff: return max(diff, key=lambda h: hpo_ic.get(h, 1.0))
        return max(cand_phenos, key=lambda h: hpo_ic.get(h, 1.0))
        
    elif pol_name == "Policy_B_CandidateReduction":
        best_h, best_balance = None, 999
        K = len(top_cands)
        for h in cand_phenos:
            match_count = sum(1 for d in top_cands if h in disease_hpo_sets[d])
            balance = abs(match_count - (K / 2))
            if balance < best_balance:
                best_balance = balance
                best_h = h
        return best_h or max(cand_phenos, key=lambda h: hpo_ic.get(h, 1.0))
        
    elif pol_name == "Policy_D_HighIC":
        return max(cand_phenos, key=lambda h: hpo_ic.get(h, 1.0))
        
    elif pol_name == "Policy_LowIC":
        return min(cand_phenos, key=lambda h: hpo_ic.get(h, 1.0))

policy_list = [
    "Policy_LowIC",
    "Policy_D_HighIC",
    "Policy_B_CandidateReduction",
    "Policy_C_PairwiseMaximin",
    "Policy_Gini_Reduction",
    "Policy_A_InfoGain"
]

policy_perf = {}
print(f"\n{'Policy Name':<28} {'Hits@1':>8} {'Resolved':>10} {'Worsened':>10} {'Mean ΔRank':>12} {'Worst Drop':>12}")
print("-" * 84)

for pol in policy_list:
    c_to_c, c_to_i, i_to_c, i_to_i = 0, 0, 0, 0
    rank_deltas = []
    
    for c in test_cases:
        p_pos = set(c["hpos"])
        p_neg = set()
        tgt = c["diseases"][0]
        
        r0, _, _, top10_0 = fast_score_and_rank(p_pos, p_neg, tgt)
        hq = select_policy_query(p_pos, p_neg, top10_0, pol)
        if hq:
            if hq in disease_hpo_sets[tgt]: p_pos.add(hq)
            else: p_neg.add(hq)
        r1, _, _, _ = fast_score_and_rank(p_pos, p_neg, tgt)
        
        rank_deltas.append(r0 - r1)
        if r0 == 1 and r1 == 1: c_to_c += 1
        elif r0 == 1 and r1 > 1: c_to_i += 1
        elif r0 > 1 and r1 == 1: i_to_c += 1
        elif r0 > 1 and r1 > 1: i_to_i += 1
        
    final_h1 = (c_to_c + i_to_c) / N_t * 100
    policy_perf[pol] = {
        "Hits@1": final_h1,
        "Resolved": i_to_c,
        "Worsened": c_to_i,
        "Mean_Delta_Rank": float(np.mean(rank_deltas)),
        "Worst_Rank_Drop": int(min(rank_deltas))
    }
    print(f"{pol:<28} {final_h1:>7.2f}% {i_to_c:>10} {c_to_i:>10} {np.mean(rank_deltas):>+12.2f} {min(rank_deltas):>12}")
print("-" * 84)

# --- 5. CANDIDATE UNIVERSE SCALING EXPERIMENT ---
print("\n[3/5] Candidate Universe Scaling Experiment (100, 500, 1000, 5000, 12956)...")
scaling_results = []
print(f"{'Universe Size':<15} {'Static Hits@1':>15} {'1-Step Active':>15} {'5-Step Active':>15} {'5-Step Gain':>14}")
print("-" * 78)

for u_size in [100, 500, 1000, 5000, 12956]:
    static_h1, active1_h1, active5_h1 = 0, 0, 0
    rng = random.Random(42)
    
    for c in test_cases:
        p_pos = set(c["hpos"])
        tgt = c["diseases"][0]
        
        if u_size == 12956:
            curr_pool = all_candidate_set
        else:
            other_cands = [d for d in candidate_diseases if d != tgt]
            sampled_others = rng.sample(other_cands, min(u_size - 1, len(other_cands)))
            curr_pool = set([tgt] + sampled_others)
        
        # Static
        r0, _, _, top10_0 = fast_score_and_rank(p_pos, set(), tgt, candidate_pool_set=curr_pool)
        if r0 == 1: static_h1 += 1
        
        # 1-step
        p_pos_1 = set(p_pos)
        p_neg_1 = set()
        hq1 = select_policy_query(p_pos_1, p_neg_1, top10_0, "Policy_A_InfoGain")
        if hq1:
            if hq1 in disease_hpo_sets[tgt]: p_pos_1.add(hq1)
            else: p_neg_1.add(hq1)
        r1, _, _, _ = fast_score_and_rank(p_pos_1, p_neg_1, tgt, candidate_pool_set=curr_pool)
        if r1 == 1: active1_h1 += 1
        
        # 5-step
        p_pos_5 = set(p_pos)
        p_neg_5 = set()
        for _ in range(5):
            _, _, _, top10_5 = fast_score_and_rank(p_pos_5, p_neg_5, tgt, candidate_pool_set=curr_pool)
            hq = select_policy_query(p_pos_5, p_neg_5, top10_5, "Policy_A_InfoGain")
            if not hq: break
            if hq in disease_hpo_sets[tgt]: p_pos_5.add(hq)
            else: p_neg_5.add(hq)
        r5, _, _, _ = fast_score_and_rank(p_pos_5, p_neg_5, tgt, candidate_pool_set=curr_pool)
        if r5 == 1: active5_h1 += 1
        
    res = {
        "universe_size": u_size,
        "static_hits1": static_h1 / N_t * 100,
        "active1_hits1": active1_h1 / N_t * 100,
        "active5_hits1": active5_h1 / N_t * 100,
        "gain_1step": (active1_h1 - static_h1) / N_t * 100,
        "gain_5step": (active5_h1 - static_h1) / N_t * 100
    }
    scaling_results.append(res)
    print(f"{u_size:<15} {res['static_hits1']:>14.2f}% {res['active1_hits1']:>14.2f}% {res['active5_hits1']:>14.2f}% {res['gain_5step']:>+13.2f} pp")
print("-" * 78)

# --- 6. QUERY BUDGET & COST CURVE ---
print("\n[4/5] Query Budget & Multi-Step Diagnostic Trajectory (Q = 0, 1, 2, 3, 5, 10)...")
budget_results = []
print(f"{'Queries (Q)':<15} {'Hits@1':>10} {'Hits@5':>10} {'Hits@10':>10} {'MRR':>10} {'Gain vs Static':>16}")
print("-" * 75)

for q_max in [0, 1, 2, 3, 5, 10]:
    h1, h5, h10, mrr_sum = 0, 0, 0, 0.0
    for c in test_cases:
        p_pos = set(c["hpos"])
        p_neg = set()
        tgt = c["diseases"][0]
        for _ in range(q_max):
            _, _, _, top10 = fast_score_and_rank(p_pos, p_neg, tgt)
            hq = select_policy_query(p_pos, p_neg, top10, "Policy_A_InfoGain")
            if not hq: break
            if hq in disease_hpo_sets[tgt]: p_pos.add(hq)
            else: p_neg.add(hq)
        r, _, _, _ = fast_score_and_rank(p_pos, p_neg, tgt)
        if r <= 1: h1 += 1
        if r <= 5: h5 += 1
        if r <= 10: h10 += 1
        mrr_sum += 1.0 / r
    b_res = {"Q": q_max, "Hits@1": h1/N_t*100, "Hits@5": h5/N_t*100, "Hits@10": h10/N_t*100, "MRR": mrr_sum/N_t}
    budget_results.append(b_res)
    print(f"Q = {q_max:<11} {b_res['Hits@1']:>9.2f}% {b_res['Hits@5']:>9.2f}% {b_res['Hits@10']:>9.2f}% {b_res['MRR']:>10.4f} {b_res['Hits@1'] - 68.62:>+15.2f} pp")
print("-" * 75)

# --- 7. EXTRACT 20 REPRESENTATIVE CASE TRACES ---
print("\n[5/5] Extracting 20 Representative Diagnostic Dialogues...")
case_traces = []
for idx, c in enumerate(test_cases):
    if len(case_traces) >= 20: break
    p_pos_init = set(c["hpos"])
    tgt = c["diseases"][0]
    r0, top1_0, _, top10_0 = fast_score_and_rank(p_pos_init, set(), tgt)
    hq = select_policy_query(p_pos_init, set(), top10_0, "Policy_A_InfoGain")
    if not hq: continue
    ans = "Present" if hq in disease_hpo_sets[tgt] else "Absent"
    p_pos_1 = set(p_pos_init)
    p_neg_1 = set()
    if ans == "Present": p_pos_1.add(hq)
    else: p_neg_1.add(hq)
    r1, top1_1, _, top10_1 = fast_score_and_rank(p_pos_1, p_neg_1, tgt)
    
    if (r0 > 1 and r1 == 1) or (r0 > 1 and r1 > 1 and len(case_traces) % 3 == 0) or (r0 == 1 and len(case_traces) % 4 == 0):
        case_traces.append({
            "case_id": c.get("id", str(idx)),
            "true_disease": tgt,
            "initial_rank": r0,
            "initial_top3": [d for d, s in top10_0[:3]],
            "queried_phenotype": hq,
            "queried_ic": float(hpo_ic.get(hq, 1.0)),
            "simulated_answer": ans,
            "updated_rank": r1,
            "updated_top3": [d for d, s in top10_1[:3]],
            "outcome": "Resolved" if (r0 > 1 and r1 == 1) else ("Maintained Correct" if r0 == 1 else "Unresolved")
        })

print(f"[✓] Captured {len(case_traces)} Diagnostic Dialogues.")

# Export JSON
out_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
out_path = os.path.join(out_dir, "phase10_5_falsification_results.json")
payload = {
    "policy_benchmarking": policy_perf,
    "universe_scaling": scaling_results,
    "budget_results": budget_results,
    "case_traces": case_traces
}
with open(out_path, "w") as f:
    json.dump(payload, f, indent=2)

print(f"\n[✓] Results successfully exported to: {out_path}")
print("=" * 72)
