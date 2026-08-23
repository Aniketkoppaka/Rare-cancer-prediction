import json
import math
from collections import defaultdict, Counter
import numpy as np

print("=== RUNNING RARE DISEASE PHENOTYPE RETRIEVAL BASELINES ===")

# 1. Load patient cases
with open('data/benchmark/sample_cases.json', 'r') as f:
    cases = json.load(f)
print(f"Loaded {len(cases)} real patient profiles.")

# 2. Load disease knowledge base from phenotype.hpoa
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

print(f"Loaded disease knowledge base: {len(disease_to_hpos)} diseases with verified HPO annotations.")

# 3. Compute Information Content (IC) for every HPO term
# IC(t) = -log( freq(t) / total_diseases )
total_d = len(disease_to_hpos)
hpo_doc_freq = Counter()
for did, hset in disease_to_hpos.items():
    for h in hset:
        hpo_doc_freq[h] += 1

hpo_ic = {}
for h, freq in hpo_doc_freq.items():
    hpo_ic[h] = -math.log((freq + 1) / (total_d + 1))

# 4. Filter cases that have ground-truth disease present in knowledge base
eval_cases = []
for c in cases:
    # check if at least one disease is in disease_to_hpos
    valid_d = [d for d in c['diseases'] if d in disease_to_hpos]
    if len(valid_d) > 0:
        eval_cases.append({'id': c['id'], 'hpos': set(c['hpos']), 'target': valid_d[0]})

print(f"Evaluation cohort: {len(eval_cases)} patients with ground-truth disease in KB.")
candidate_diseases = list(disease_to_hpos.keys())
print(f"Candidate search space: {len(candidate_diseases)} rare diseases.")

# 5. Evaluate Method 1: Classical Jaccard Similarity Baseline
# Sim(Patient, Disease) = |Patient_HPOs ∩ Disease_HPOs| / |Patient_HPOs ∪ Disease_HPOs|
# Evaluate Method 2: Information Content (IC) Weighted Jaccard
# Sim_IC(Patient, Disease) = sum(IC(h) for h in Patient ∩ Disease) / sum(IC(h) for h in Patient ∪ Disease)

def evaluate_retrieval(sim_func_name):
    hits1, hits5, hits10 = 0, 0, 0
    mrr_total = 0.0
    
    for c in eval_cases:
        p_hpos = c['hpos']
        target = c['target']
        
        scores = []
        for did in candidate_diseases:
            d_hpos = disease_to_hpos[did]
            intersection = p_hpos & d_hpos
            if not intersection:
                score = 0.0
            else:
                if sim_func_name == 'jaccard':
                    union = p_hpos | d_hpos
                    score = len(intersection) / len(union)
                elif sim_func_name == 'ic_overlap':
                    score = sum(hpo_ic.get(h, 1.0) for h in intersection)
                elif sim_func_name == 'ic_weighted_jaccard':
                    union = p_hpos | d_hpos
                    score = sum(hpo_ic.get(h, 1.0) for h in intersection) / sum(hpo_ic.get(h, 1.0) for h in union)
            scores.append((score, did))
        
        # Rank descending
        scores.sort(key=lambda x: x[0], reverse=True)
        ranked_dids = [x[1] for x in scores]
        
        try:
            rank = ranked_dids.index(target) + 1
        except ValueError:
            rank = len(candidate_diseases)
            
        if rank <= 1: hits1 += 1
        if rank <= 5: hits5 += 1
        if rank <= 10: hits10 += 1
        mrr_total += 1.0 / rank

    N = len(eval_cases)
    return {
        'Method': sim_func_name,
        'Hits@1 (%)': (hits1 / N) * 100,
        'Hits@5 (%)': (hits5 / N) * 100,
        'Hits@10 (%)': (hits10 / N) * 100,
        'MRR': mrr_total / N
    }

print("\n--- BENCHMARK RESULTS ACROSS ALL CANDIDATE DISEASES ---")
res_jaccard = evaluate_retrieval('jaccard')
print(f"Jaccard Similarity: Hits@1={res_jaccard['Hits@1 (%)']:.2f}%, Hits@5={res_jaccard['Hits@5 (%)']:.2f}%, Hits@10={res_jaccard['Hits@10 (%)']:.2f}%, MRR={res_jaccard['MRR']:.4f}")

res_ic_overlap = evaluate_retrieval('ic_overlap')
print(f"IC Overlap Sum:     Hits@1={res_ic_overlap['Hits@1 (%)']:.2f}%, Hits@5={res_ic_overlap['Hits@5 (%)']:.2f}%, Hits@10={res_ic_overlap['Hits@10 (%)']:.2f}%, MRR={res_ic_overlap['MRR']:.4f}")

res_ic_jaccard = evaluate_retrieval('ic_weighted_jaccard')
print(f"IC Weighted Jaccard:Hits@1={res_ic_jaccard['Hits@1 (%)']:.2f}%, Hits@5={res_ic_jaccard['Hits@5 (%)']:.2f}%, Hits@10={res_ic_jaccard['Hits@10 (%)']:.2f}%, MRR={res_ic_jaccard['MRR']:.4f}")
