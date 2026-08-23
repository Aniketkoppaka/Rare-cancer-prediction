import json
import math
from collections import defaultdict, Counter
import numpy as np

print("=== EXACT MATHEMATICAL AUDIT OF IC OVERLAP BASELINE ===")

# 1. Load data
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

total_kb_diseases = len(disease_to_hpos)
hpo_doc_freq = Counter()
for did, hset in disease_to_hpos.items():
    for h in hset:
        hpo_doc_freq[h] += 1

# Mathematical definition: IC(h) = -ln( (freq(h) + 1) / (N_diseases + 1) )
hpo_ic = {h: -math.log((freq + 1)/(total_kb_diseases + 1)) for h, freq in hpo_doc_freq.items()}

# Sample 1,000 cases to audit profile size correlation with false positives
eval_cases = [c for c in cases if c['diseases'][0] in disease_to_hpos][:1000]
candidate_diseases = list(disease_to_hpos.keys())

# Compare score distributions
top1_pred_sizes = []
target_sizes = []
failure_target_sizes = []
failure_pred_sizes = []

for c in eval_cases:
    p_hpos = set(c['hpos'])
    target = c['diseases'][0]
    
    scores = []
    for did in candidate_diseases:
        d_hpos = disease_to_hpos[did]
        inter = p_hpos & d_hpos
        s = sum(hpo_ic.get(h, 1.0) for h in inter)
        scores.append((s, did))
    scores.sort(key=lambda x: x[0], reverse=True)
    
    top1 = scores[0][1]
    rank = [x[1] for x in scores].index(target) + 1
    
    top1_pred_sizes.append(len(disease_to_hpos[top1]))
    target_sizes.append(len(disease_to_hpos[target]))
    
    if rank > 10:
        failure_target_sizes.append(len(disease_to_hpos[target]))
        failure_pred_sizes.append(len(disease_to_hpos[top1]))

print("\n--- PROFILE SIZE COMPARISON ---")
print(f"Overall Target Mean Profile Size: {np.mean(target_sizes):.2f} HPOs")
print(f"Overall Top-1 Prediction Mean Profile Size: {np.mean(top1_pred_sizes):.2f} HPOs")
print(f"In Failures (Rank > 10):")
print(f"  Target Disease Mean Profile Size: {np.mean(failure_target_sizes):.2f} HPOs")
print(f"  False-Positive Top-1 Disease Mean Profile Size: {np.mean(failure_pred_sizes):.2f} HPOs")
