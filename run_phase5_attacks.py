import json
import math
import random
from collections import defaultdict, Counter
import numpy as np

print("=== PHASE 5: ATTACKING THE RESEARCH HYPOTHESIS WITH CLASSICAL IR METHODS ===")

# 1. Load Data
with open('data/benchmark/clean_phenopackets.json', 'r', encoding='utf-8') as f:
    cases = json.load(f)

disease_to_hpos = defaultdict(set)
with open('data/benchmark/phenotype.hpoa', 'r', encoding='utf-8') as f:
    for line in f:
        if line.startswith('#') or not line.strip(): continue
        parts = line.strip().split('\t')
        if len(parts) >= 4:
            did = parts[0]
            hid = parts[3]
            if hid.startswith('HP:'):
                disease_to_hpos[did].add(hid)

total_kb = len(disease_to_hpos)
hpo_doc_freq = Counter()
for did, hset in disease_to_hpos.items():
    for h in hset:
        hpo_doc_freq[h] += 1

hpo_ic = {h: -math.log((freq + 1)/(total_kb + 1)) for h, freq in hpo_doc_freq.items()}

# Filter valid cases
valid_cases = [c for c in cases if c['diseases'][0] in disease_to_hpos]
all_diseases = sorted(list(set([c['diseases'][0] for c in valid_cases])))

# Fixed Seed 42 Split (70% Train, 15% Val, 15% Test)
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

candidate_diseases = list(disease_to_hpos.keys())
print(f"Validation Cohort: {len(val_cases)} cases across {len(val_dids)} diseases")
print(f"Zero-Shot Test Cohort: {len(test_cases)} cases across {len(test_dids)} diseases")
print(f"Candidate Universe: {len(candidate_diseases)} rare diseases")

# Precompute mean document length for BM25
avg_doc_len = np.mean([len(disease_to_hpos[d]) for d in candidate_diseases])

def evaluate_method(score_fn, cohort, name):
    hits1, hits5, hits10 = 0, 0, 0
    mrr_total = 0.0
    
    for c in cohort:
        p_hpos = set(c['hpos'])
        tgt = c['diseases'][0]
        
        scores = []
        for did in candidate_diseases:
            d_hpos = disease_to_hpos[did]
            s = score_fn(p_hpos, d_hpos, did)
            scores.append((s, did))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        ranked = [x[1] for x in scores]
        try:
            r = ranked.index(tgt) + 1
        except ValueError:
            r = len(candidate_diseases)
            
        if r <= 1: hits1 += 1
        if r <= 5: hits5 += 1
        if r <= 10: hits10 += 1
        mrr_total += 1.0 / r
        
    N = len(cohort)
    res = {
        'Method': name,
        'Hits@1': hits1/N*100,
        'Hits@5': hits5/N*100,
        'Hits@10': hits10/N*100,
        'MRR': mrr_total/N
    }
    return res

# 1. Baseline: Raw IC Overlap
def score_raw_ic(p_hpos, d_hpos, did):
    inter = p_hpos & d_hpos
    return sum(hpo_ic.get(h, 1.0) for h in inter) if inter else 0.0

# 2. Length Normalizations
def make_power_norm(alpha):
    def score_power(p_hpos, d_hpos, did):
        inter = p_hpos & d_hpos
        if not inter: return 0.0
        s = sum(hpo_ic.get(h, 1.0) for h in inter)
        denom = (len(d_hpos) ** alpha) if len(d_hpos) > 0 else 1.0
        return s / denom
    return score_power

# 3. Symmetric Normalization (Cosine-like in set space)
def score_symmetric(p_hpos, d_hpos, did):
    inter = p_hpos & d_hpos
    if not inter: return 0.0
    s = sum(hpo_ic.get(h, 1.0) for h in inter)
    return s / math.sqrt(len(p_hpos) * len(d_hpos))

# 4. BM25 scoring
def make_bm25(k1=1.5, b=0.75):
    def score_bm25(p_hpos, d_hpos, did):
        inter = p_hpos & d_hpos
        if not inter: return 0.0
        doc_len = len(d_hpos)
        score = 0.0
        for h in inter:
            idf = hpo_ic.get(h, 1.0)
            tf = 1.0 # binary term freq in HPOA profile
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
            score += idf * tf_norm
        return score
    return score_bm25

print("\n--- TUNING ON VALIDATION COHORT ---")
alphas = [0.1, 0.25, 0.35, 0.5, 0.75, 1.0]
for a in alphas:
    r = evaluate_method(make_power_norm(a), val_cases, f"Power Norm alpha={a}")
    print(f"Val alpha={a:4.2f}: Hits@1={r['Hits@1']:5.2f}%, Hits@5={r['Hits@5']:5.2f}%, Hits@10={r['Hits@10']:5.2f}%, MRR={r['MRR']:.4f}")

bm25_params = [(1.2, 0.3), (1.2, 0.5), (1.5, 0.5), (1.5, 0.75)]
for k1, b in bm25_params:
    r = evaluate_method(make_bm25(k1, b), val_cases, f"BM25 k1={k1}, b={b}")
    print(f"Val BM25 (k1={k1}, b={b}): Hits@1={r['Hits@1']:5.2f}%, Hits@5={r['Hits@5']:5.2f}%, Hits@10={r['Hits@10']:5.2f}%, MRR={r['MRR']:.4f}")

# Best validation alpha is chosen
best_alpha = 0.25
best_bm25 = (1.2, 0.3)

print("\n=== FINAL TEST EVALUATION ON ZERO-SHOT COHORT (N=1880) ===")
results = []
results.append(evaluate_method(score_raw_ic, test_cases, "1. Raw IC Overlap (Unnormalized)"))
results.append(evaluate_method(make_power_norm(0.25), test_cases, "2. Best Length-Normalized IC (alpha=0.25)"))
results.append(evaluate_method(make_power_norm(0.50), test_cases, "3. Sqrt Length-Normalized IC (alpha=0.50)"))
results.append(evaluate_method(make_power_norm(1.00), test_cases, "4. Full Mean IC (alpha=1.00)"))
results.append(evaluate_method(score_symmetric, test_cases, "5. Symmetric Normalized IC (Cosine-Set)"))
results.append(evaluate_method(make_bm25(1.2, 0.3), test_cases, "6. BM25 (k1=1.2, b=0.3)"))
results.append(evaluate_method(make_bm25(1.5, 0.75), test_cases, "7. Standard BM25 (k1=1.5, b=0.75)"))

for res in results:
    print(f"{res['Method']:<45} | Hits@1: {res['Hits@1']:5.2f}% | Hits@5: {res['Hits@5']:5.2f}% | Hits@10: {res['Hits@10']:5.2f}% | MRR: {res['MRR']:.4f}")

with open('data/benchmark/phase5_attack_results.json', 'w') as out:
    json.dump(results, out, indent=2)
print("\nSaved attack benchmark results to data/benchmark/phase5_attack_results.json")
