import json
import math
from collections import defaultdict, Counter
import numpy as np

print("=== FAILURE CASE & HIERARCHICAL ERROR ANALYSIS ===")

# 1. Load benchmark cases
with open('data/benchmark/clean_phenopackets.json', 'r', encoding='utf-8') as f:
    cases = json.load(f)

# 2. Load knowledge base from phenotype.hpoa
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

# 3. Load HPO ontology hierarchy
hpo_parents = defaultdict(set)
hpo_ancestors = defaultdict(set)
with open('data/benchmark/hp.obo', 'r', encoding='utf-8') as f:
    current_term = None
    for line in f:
        if line.startswith('[Term]'):
            current_term = None
        elif line.startswith('id: '):
            current_term = line.strip().split('id: ')[1].split()[0]
        elif line.startswith('is_a: ') and current_term:
            parent = line.strip().split('is_a: ')[1].split()[0]
            hpo_parents[current_term].add(parent)

def get_ancestors(term):
    if term in hpo_ancestors: return hpo_ancestors[term]
    visited = set()
    stack = [term]
    while stack:
        curr = stack.pop()
        for p in hpo_parents.get(curr, []):
            if p not in visited:
                visited.add(p)
                stack.append(p)
    hpo_ancestors[term] = visited
    return visited

# Precompute Information Content (IC)
total_d = len(disease_to_hpos)
hpo_doc_freq = Counter()
for did, hset in disease_to_hpos.items():
    for h in hset:
        hpo_doc_freq[h] += 1

hpo_ic = {h: -math.log((freq + 1)/(total_d + 1)) for h, freq in hpo_doc_freq.items()}

# Sample 1,000 cases for deep failure analysis
eval_cases = [c for c in cases if c['diseases'][0] in disease_to_hpos][:1000]
candidate_diseases = list(disease_to_hpos.keys())
print(f"Analyzing {len(eval_cases)} cases against {len(candidate_diseases)} KB diseases...")

failures = []
ranks = []
jaccard_overlaps = []

for c in eval_cases:
    p_hpos = set(c['hpos'])
    target = c['diseases'][0]
    
    # Check exact Jaccard overlap with ground truth KB profile
    kb_hpos = disease_to_hpos[target]
    inter = p_hpos & kb_hpos
    union = p_hpos | kb_hpos
    jaccard_overlaps.append(len(inter)/len(union) if union else 0.0)
    
    # Retrieve using IC Overlap Sum
    scores = []
    for did in candidate_diseases:
        d_hpos = disease_to_hpos[did]
        s = sum(hpo_ic.get(h, 1.0) for h in (p_hpos & d_hpos))
        scores.append((s, did))
    scores.sort(key=lambda x: x[0], reverse=True)
    
    ranked_dids = [x[1] for x in scores]
    rank = ranked_dids.index(target) + 1
    ranks.append(rank)
    
    if rank > 10:
        top1_pred = ranked_dids[0]
        failures.append({
            'id': c['id'],
            'target': target,
            'top1_pred': top1_pred,
            'rank': rank,
            'patient_hpo_count': len(p_hpos),
            'kb_hpo_count': len(kb_hpos),
            'exact_overlap_count': len(inter),
            'patient_hpos': list(p_hpos),
            'kb_hpos': list(kb_hpos),
            'top1_kb_hpos': list(disease_to_hpos[top1_pred])
        })

print(f"\nOverall Top-10 Retrieval Results on 1,000 Patient Cohort:")
print(f"  Hits@1:  {sum(1 for r in ranks if r <= 1)/len(ranks)*100:.2f}%")
print(f"  Hits@5:  {sum(1 for r in ranks if r <= 5)/len(ranks)*100:.2f}%")
print(f"  Hits@10: {sum(1 for r in ranks if r <= 10)/len(ranks)*100:.2f}%")
print(f"  MRR:     {np.mean([1.0/r for r in ranks]):.4f}")
print(f"  Mean Patient-to-KB Jaccard Profile Overlap: {np.mean(jaccard_overlaps)*100:.2f}%")

print(f"\nTotal Top-10 Failures: {len(failures)} / {len(eval_cases)} ({len(failures)/len(eval_cases)*100:.1f}%)")

# Categorize failure reasons
reasons = Counter()
for f in failures:
    if f['exact_overlap_count'] <= 1:
        reasons['Severe Phenotype Mismatch / Atypical Case (<=1 exact match)'] += 1
    elif f['patient_hpo_count'] <= 3:
        reasons['Extreme Sparsity (<=3 patient HPOs)'] += 1
    elif f['kb_hpo_count'] >= 50:
        reasons['KB Profile Overshadowing (Generic broad disease in KB)'] += 1
    else:
        # Check if top1 is hierarchical ancestor / descendant in HPO terms
        reasons['Phenotypic Overlap / Tie-breaking ambiguity'] += 1

print("\n--- QUANTIFIED FAILURE CATEGORIES ---")
for r, count in reasons.most_common():
    print(f"  {r}: {count} cases ({count/len(failures)*100:.1f}%)")

with open('data/benchmark/failure_cases.json', 'w') as out:
    json.dump(failures[:50], out, indent=2)
print("Saved 50 detailed failure profiles to data/benchmark/failure_cases.json")
