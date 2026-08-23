import json
import math
import random
from collections import defaultdict, Counter
import numpy as np

print("=== PHASE 5.5: IMPLEMENTING ONTOLOGY-AWARE SEMANTIC BASELINES (RESNIK, MICA, ANCESTOR PROPAGATION, BMA) ===")

# 1. Load Data
with open('data/benchmark/clean_phenopackets.json', 'r', encoding='utf-8') as f:
    cases = json.load(f)

disease_to_hpos = defaultdict(set)
with open('data/benchmark/phenotype.hpoa', 'r', encoding='utf-8') as f:
    for line in f:
        if line.startswith('#') or not line.strip(): continue
        parts = line.strip().split('\t')
        if len(parts) >= 4:
            did, hid = parts[0], parts[3]
            if hid.startswith('HP:'):
                disease_to_hpos[did].add(hid)

# 2. Parse HPO DAG and build Ancestor Mapping
hpo_parents = defaultdict(set)
with open('data/benchmark/hp.obo', 'r', encoding='utf-8') as f:
    curr_term = None
    for line in f:
        if line.startswith('[Term]'):
            curr_term = None
        elif line.startswith('id: '):
            curr_term = line.strip().split('id: ')[1].split()[0]
        elif line.startswith('is_a: ') and curr_term:
            p = line.strip().split('is_a: ')[1].split()[0]
            hpo_parents[curr_term].add(p)

ancestor_cache = {}
def get_ancestors(term):
    if term in ancestor_cache: return ancestor_cache[term]
    visited = set([term]) # include self
    stack = [term]
    while stack:
        curr = stack.pop()
        for p in hpo_parents.get(curr, []):
            if p not in visited:
                visited.add(p)
                stack.append(p)
    ancestor_cache[term] = visited
    return visited

# Precompute propagated disease profiles
print("Propagating HPO ancestors for Knowledge Base diseases...")
disease_to_propagated_hpos = {}
for did, hset in disease_to_hpos.items():
    prop = set()
    for h in hset:
        prop.update(get_ancestors(h))
    disease_to_propagated_hpos[did] = prop

# Compute IC based on propagated frequencies (standard Phenomizer/Resnik definition)
total_kb = len(disease_to_propagated_hpos)
prop_hpo_freq = Counter([h for hset in disease_to_propagated_hpos.values() for h in hset])
hpo_ic = {h: -math.log((freq + 1)/(total_kb + 1)) for h, freq in prop_hpo_freq.items()}

# Lock cohorts
valid_cases = [c for c in cases if c['diseases'][0] in disease_to_hpos]
all_diseases = sorted(list(set([c['diseases'][0] for c in valid_cases])))

random.seed(42)
shuffled_dids = list(all_diseases)
random.shuffle(shuffled_dids)

n_train_d = int(0.70 * len(shuffled_dids))
n_val_d = int(0.15 * len(shuffled_dids))

val_dids = set(shuffled_dids[n_train_d:n_train_d+n_val_d])
test_dids = set(shuffled_dids[n_train_d+n_val_d:])

val_cases = [c for c in valid_cases if c['diseases'][0] in val_dids]
test_cases = [c for c in valid_cases if c['diseases'][0] in test_dids]
candidate_diseases = list(disease_to_hpos.keys())

print(f"Validation cases: {len(val_cases)}, Zero-Shot Test cases: {len(test_cases)}")

# 3. Define Advanced Ontology Similarity Functions

# A. Exact IC Overlap (Unpropagated)
def score_raw_ic(p_hpos, did):
    d_hpos = disease_to_hpos[did]
    inter = p_hpos & d_hpos
    return sum(hpo_ic.get(h, 1.0) for h in inter) if inter else 0.0

# B. Ancestor-Propagated Overlap (Patient propagated & Disease propagated)
# S(P, D) = sum(IC(h) for h in Prop(P) ∩ Prop(D)) with Information Content
def score_propagated_overlap(p_prop_hpos, did):
    d_prop = disease_to_propagated_hpos[did]
    inter = p_prop_hpos & d_prop
    return sum(hpo_ic.get(h, 1.0) for h in inter) if inter else 0.0

# C. Asymmetric Directional Subsumption (Patient -> Disease Coverage)
# Measures what fraction of the patient's information is covered by the disease
def score_patient_subsumption(p_hpos, did):
    d_prop = disease_to_propagated_hpos[did]
    covered_ic = sum(hpo_ic.get(h, 1.0) for h in p_hpos if h in d_prop)
    total_p_ic = sum(hpo_ic.get(h, 1.0) for h in p_hpos)
    return (covered_ic / total_p_ic) if total_p_ic > 0 else 0.0

# D. Best Match Average (BMA) Resnik Similarity (Sample-based for ranking feasibility)
# For each patient term p, find max IC(MICA(p, d)) in disease D
# sim(p, d) = max_{a in Anc(p) ∩ Anc(d)} IC(a)
def score_resnik_bma_approx(p_hpos, p_prop_hpos, did):
    d_prop = disease_to_propagated_hpos[did]
    # For every p in patient, find highest IC ancestor shared with disease
    score = 0.0
    for p in p_hpos:
        p_anc = get_ancestors(p)
        shared = p_anc & d_prop
        if shared:
            score += max(hpo_ic.get(a, 0.0) for a in shared)
    return score

# Evaluation Harness
def run_eval(score_type, cohort, name):
    hits1, hits5, hits10 = 0, 0, 0
    mrr_total = 0.0
    
    for c in cohort:
        p_hpos = set(c['hpos'])
        tgt = c['diseases'][0]
        
        # Precompute patient ancestors
        p_prop = set()
        for h in p_hpos:
            p_prop.update(get_ancestors(h))
            
        scores = []
        for did in candidate_diseases:
            if score_type == 'raw_ic':
                s = score_raw_ic(p_hpos, did)
            elif score_type == 'prop_ic':
                s = score_propagated_overlap(p_prop, did)
            elif score_type == 'subsumption':
                s = score_patient_subsumption(p_hpos, did)
            elif score_type == 'resnik_bma':
                s = score_resnik_bma_approx(p_hpos, p_prop, did)
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
    return {
        'Method': name,
        'Hits@1': hits1/N*100,
        'Hits@5': hits5/N*100,
        'Hits@10': hits10/N*100,
        'MRR': mrr_total/N
    }

print("\n--- EVALUATING ON VALIDATION COHORT (N=1000) ---")
res_val_raw = run_eval('raw_ic', val_cases, '1. Raw Exact IC Overlap')
print(f"Val Raw IC:         Hits@1={res_val_raw['Hits@1']:5.2f}%, Hits@5={res_val_raw['Hits@5']:5.2f}%, Hits@10={res_val_raw['Hits@10']:5.2f}%, MRR={res_val_raw['MRR']:.4f}")

res_val_resnik = run_eval('resnik_bma', val_cases, '2. Resnik MICA Best-Match-Average (BMA)')
print(f"Val Resnik BMA:     Hits@1={res_val_resnik['Hits@1']:5.2f}%, Hits@5={res_val_resnik['Hits@5']:5.2f}%, Hits@10={res_val_resnik['Hits@10']:5.2f}%, MRR={res_val_resnik['MRR']:.4f}")

res_val_prop = run_eval('prop_ic', val_cases, '3. Full Ancestor Propagated IC Overlap')
print(f"Val Propagated IC:  Hits@1={res_val_prop['Hits@1']:5.2f}%, Hits@5={res_val_prop['Hits@5']:5.2f}%, Hits@10={res_val_prop['Hits@10']:5.2f}%, MRR={res_val_prop['MRR']:.4f}")

# Locked Test Evaluation on the 1,880 Zero-Shot Holdout
print("\n=== FINAL TEST EVALUATION ON ZERO-SHOT HOLDOUT (N=1880) ===")
res_test_raw = run_eval('raw_ic', test_cases, '1. Raw Exact IC Overlap')
res_test_resnik = run_eval('resnik_bma', test_cases, '2. Resnik MICA Best-Match-Average (BMA)')
res_test_prop = run_eval('prop_ic', test_cases, '3. Full Ancestor Propagated IC Overlap')

test_summary = [res_test_raw, res_test_resnik, res_test_prop]
for r in test_summary:
    print(f"{r['Method']:<45} | Hits@1: {r['Hits@1']:5.2f}% | Hits@5: {r['Hits@5']:5.2f}% | Hits@10: {r['Hits@10']:5.2f}% | MRR: {r['MRR']:.4f}")

with open('data/benchmark/phase5_5_semantic_results.json', 'w') as out:
    json.dump(test_summary, out, indent=2)
print("Saved data/benchmark/phase5_5_semantic_results.json successfully!")
