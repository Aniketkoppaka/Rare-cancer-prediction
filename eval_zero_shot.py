import json
import random
import math
from collections import defaultdict, Counter
import numpy as np

print("=== RIGOROUS ZERO-SHOT & DISEASE-DISJOINT SPLIT EVALUATION ===")

# 1. Load benchmark cases
with open('data/benchmark/clean_phenopackets.json', 'r', encoding='utf-8') as f:
    cases = json.load(f)

# 2. Load KB
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

total_d = len(disease_to_hpos)
hpo_doc_freq = Counter()
for did, hset in disease_to_hpos.items():
    for h in hset:
        hpo_doc_freq[h] += 1
hpo_ic = {h: -math.log((freq + 1)/(total_d + 1)) for h, freq in hpo_doc_freq.items()}

# Filter cases where disease in KB
valid_cases = [c for c in cases if c['diseases'][0] in disease_to_hpos]
all_diseases = sorted(list(set([c['diseases'][0] for c in valid_cases])))

# Set seed for reproducible split
random.seed(42)
shuffled_dids = list(all_diseases)
random.shuffle(shuffled_dids)

# 70% Train Diseases, 15% Val Diseases, 15% Zero-Shot Test Diseases
n_train_d = int(0.70 * len(shuffled_dids))
n_val_d = int(0.15 * len(shuffled_dids))

train_dids = set(shuffled_dids[:n_train_d])
val_dids = set(shuffled_dids[n_train_d:n_train_d+n_val_d])
test_dids = set(shuffled_dids[n_train_d+n_val_d:])

train_cases = [c for c in valid_cases if c['diseases'][0] in train_dids]
val_cases = [c for c in valid_cases if c['diseases'][0] in val_dids]
test_cases = [c for c in valid_cases if c['diseases'][0] in test_dids]

print(f"Total Unique Valid Diseases: {len(all_diseases)}")
print(f"  Train Diseases: {len(train_dids)} ({len(train_cases)} patient cases)")
print(f"  Val Diseases:   {len(val_dids)} ({len(val_cases)} patient cases)")
print(f"  Test Diseases (ZERO-SHOT): {len(test_dids)} ({len(test_cases)} patient cases)")

# Evaluate Baseline Ladder on ZERO-SHOT Test Cohort against full 12,956 candidate diseases
candidate_diseases = list(disease_to_hpos.keys())
print(f"\nEvaluating Baseline Ladder on ZERO-SHOT Test Cohort (N={len(test_cases)} patients across {len(test_dids)} unseen diseases)...")

# Baseline 0: Random Ranking Expectation
# E[Hits@1] = 1/12956 = 0.0077%, E[Hits@10] = 10/12956 = 0.077%, E[MRR] = sum(1/i)/N ≈ 0.00077

# Baseline 1: Disease Prior (annotation frequency)
d_freq_ranked = sorted(candidate_diseases, key=lambda d: len(disease_to_hpos[d]), reverse=True)
b1_hits1, b1_hits5, b1_hits10 = 0, 0, 0
b1_mrr = 0.0
for c in test_cases:
    tgt = c['diseases'][0]
    rank = d_freq_ranked.index(tgt) + 1 if tgt in d_freq_ranked else len(candidate_diseases)
    if rank <= 1: b1_hits1 += 1
    if rank <= 5: b1_hits5 += 1
    if rank <= 10: b1_hits10 += 1
    b1_mrr += 1.0 / rank

# Baseline 2 & 3: Jaccard & IC Overlap
jacc_hits1, jacc_hits5, jacc_hits10, jacc_mrr = 0, 0, 0, 0.0
ic_hits1, ic_hits5, ic_hits10, ic_mrr = 0, 0, 0, 0.0

for c in test_cases:
    p_hpos = set(c['hpos'])
    tgt = c['diseases'][0]
    
    # Jaccard scores
    j_scores = []
    # IC scores
    ic_scores = []
    
    for did in candidate_diseases:
        d_hpos = disease_to_hpos[did]
        inter = p_hpos & d_hpos
        if not inter:
            j_scores.append((0.0, did))
            ic_scores.append((0.0, did))
        else:
            union_len = len(p_hpos | d_hpos)
            j_scores.append((len(inter)/union_len, did))
            ic_scores.append((sum(hpo_ic.get(h, 1.0) for h in inter), did))
            
    j_scores.sort(key=lambda x: x[0], reverse=True)
    ic_scores.sort(key=lambda x: x[0], reverse=True)
    
    r_j = [x[1] for x in j_scores].index(tgt) + 1
    r_ic = [x[1] for x in ic_scores].index(tgt) + 1
    
    if r_j <= 1: jacc_hits1 += 1
    if r_j <= 5: jacc_hits5 += 1
    if r_j <= 10: jacc_hits10 += 1
    jacc_mrr += 1.0 / r_j
    
    if r_ic <= 1: ic_hits1 += 1
    if r_ic <= 5: ic_hits5 += 1
    if r_ic <= 10: ic_hits10 += 1
    ic_mrr += 1.0 / r_ic

N = len(test_cases)
print("\n=== ZERO-SHOT BASELINE LADDER RESULTS ===")
print(f"Baseline 0 (Random Ranking):     Hits@1=0.01%,  Hits@5=0.04%,  Hits@10=0.08%,  MRR=0.0008")
print(f"Baseline 1 (Disease Frequency):   Hits@1={b1_hits1/N*100:.2f}%, Hits@5={b1_hits5/N*100:.2f}%, Hits@10={b1_hits10/N*100:.2f}%, MRR={b1_mrr/N:.4f}")
print(f"Baseline 2 (Unweighted Jaccard):  Hits@1={jacc_hits1/N*100:.2f}%, Hits@5={jacc_hits5/N*100:.2f}%, Hits@10={jacc_hits10/N*100:.2f}%, MRR={jacc_mrr/N:.4f}")
print(f"Baseline 3 (IC Overlap Sum):      Hits@1={ic_hits1/N*100:.2f}%, Hits@5={ic_hits5/N*100:.2f}%, Hits@10={ic_hits10/N*100:.2f}%, MRR={ic_mrr/N:.4f}")
