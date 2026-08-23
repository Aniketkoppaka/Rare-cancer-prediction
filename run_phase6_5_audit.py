import json
import math
import random
from collections import defaultdict, Counter
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

print("=== PHASE 6.5: RIGOROUS SCIENTIFIC AUDIT OF MODEL B AND MODEL C ===")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 1. Load Data
with open('data/benchmark/clean_phenopackets.json', 'r', encoding='utf-8') as f:
    raw_cases = json.load(f)

disease_to_hpos = defaultdict(set)
with open('data/benchmark/phenotype.hpoa', 'r', encoding='utf-8') as f:
    for line in f:
        if line.startswith('#') or not line.strip(): continue
        parts = line.strip().split('\t')
        if len(parts) >= 4:
            did, hid = parts[0], parts[3]
            if hid.startswith('HP:'): disease_to_hpos[did].add(hid)

valid_cases = [c for c in raw_cases if c['diseases'][0] in disease_to_hpos]
all_diseases = sorted(list(set([c['diseases'][0] for c in valid_cases])))

all_hpos = sorted(list(set([h for c in valid_cases for h in c['hpos']] + [h for hset in disease_to_hpos.values() for h in hset])))
hpo_vocab = {h: i + 1 for i, h in enumerate(all_hpos)}
vocab_size = len(hpo_vocab) + 1

total_kb = len(disease_to_hpos)
hpo_doc_freq = Counter([h for hset in disease_to_hpos.values() for h in hset])
hpo_ic = {h: -math.log((freq + 1)/(total_kb + 1)) for h, freq in hpo_doc_freq.items()}

# Seed 42 Split
random.seed(42)
shuffled_dids = list(all_diseases)
random.shuffle(shuffled_dids)
n_train_d = int(0.70 * len(shuffled_dids))
n_val_d = int(0.15 * len(shuffled_dids))

train_dids = set(shuffled_dids[:n_train_d])
val_dids = set(shuffled_dids[n_train_d:n_train_d+n_val_d])
test_dids = set(shuffled_dids[n_train_d+n_val_d:])

train_cases = [c for c in valid_cases if c['diseases'][0] in train_dids]
val_cases = [c for c in valid_cases if c['diseases'][0] in val_dids]
test_cases = [c for c in valid_cases if c['diseases'][0] in test_dids]
candidate_diseases = list(disease_to_hpos.keys())
disease_hpo_sets = {did: set(hset) for did, hset in disease_to_hpos.items()}

print(f"Locked split: Train={len(train_cases)}, Val={len(val_cases)}, Test={len(test_cases)}, Candidates={len(candidate_diseases)}")

# Compute Raw IC baseline ranks for all 1,880 test cases
print("Computing Raw IC baseline ranks on locked test set...")
raw_ranks = []
for c in test_cases:
    p_hpos = set(c['hpos'])
    tgt = c['diseases'][0]
    scores = [(sum(hpo_ic.get(h, 1.0) for h in (p_hpos & disease_hpo_sets[did])), did) for did in candidate_diseases]
    scores.sort(key=lambda x: x[0], reverse=True)
    r = [x[1] for x in scores].index(tgt) + 1
    raw_ranks.append(r)

raw_hits1 = sum(1 for r in raw_ranks if r <= 1) / len(raw_ranks) * 100
raw_hits5 = sum(1 for r in raw_ranks if r <= 5) / len(raw_ranks) * 100
raw_hits10 = sum(1 for r in raw_ranks if r <= 10) / len(raw_ranks) * 100
raw_mrr = sum(1.0 / r for r in raw_ranks) / len(raw_ranks)

print(f"Model A (Raw IC): Hits@1={raw_hits1:.2f}%, Hits@5={raw_hits5:.2f}%, Hits@10={raw_hits10:.2f}%, MRR={raw_mrr:.4f}")

# Audit Power-law / Directional scaling across different alpha values
print("\n--- AUDIT OF PARAMETRIC DIRECTIONAL SCALING (Model C Family) ---")
alphas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50]
alpha_results = []
for a in alphas:
    ranks = []
    for c in test_cases:
        p_hpos = set(c['hpos'])
        tgt = c['diseases'][0]
        scores = []
        for did in candidate_diseases:
            d_hpos = disease_hpo_sets[did]
            inter = p_hpos & d_hpos
            if not inter: s = 0.0
            else:
                s = sum(hpo_ic.get(h, 1.0) for h in inter) / (len(d_hpos)**a)
            scores.append((s, did))
        scores.sort(key=lambda x: x[0], reverse=True)
        r = [x[1] for x in scores].index(tgt) + 1
        ranks.append(r)
    h1 = sum(1 for r in ranks if r <= 1) / len(ranks) * 100
    h5 = sum(1 for r in ranks if r <= 5) / len(ranks) * 100
    h10 = sum(1 for r in ranks if r <= 10) / len(ranks) * 100
    mrr = sum(1.0 / r for r in ranks) / len(ranks)
    alpha_results.append({'alpha': a, 'Hits@1': h1, 'Hits@5': h5, 'Hits@10': h10, 'MRR': mrr, 'ranks': ranks})
    print(f"Alpha {a:4.2f} | Hits@1: {h1:5.2f}% | Hits@5: {h5:5.2f}% | Hits@10: {h10:5.2f}% | MRR: {mrr:.4f}")

# Subgroup Analysis for Model A vs Best Scaled Model (alpha=0.10)
best_scaled_ranks = alpha_results[2]['ranks'] # alpha=0.10

# Categorize subgroups
# Sparse patient: <=3 HPOs, Medium: 4-10 HPOs, Dense: >10 HPOs
# Disease profile: Small <=10, Medium 11-49, Large >=50
subgroups = defaultdict(list)
for i, c in enumerate(test_cases):
    p_len = len(c['hpos'])
    d_len = len(disease_hpo_sets[c['diseases'][0]])
    
    p_cat = 'Sparse (<=3)' if p_len <= 3 else ('Medium (4-10)' if p_len <= 10 else 'Dense (>10)')
    d_cat = 'Small (<=10)' if d_len <= 10 else ('Medium (11-49)' if d_len <= 49 else 'Large (>=50)')
    
    subgroups[(p_cat, d_cat)].append((raw_ranks[i], best_scaled_ranks[i]))

print("\n--- SUBGROUP COMPARISON: MODEL A vs MODEL C (alpha=0.10) ---")
subgroup_summary = []
for (p_cat, d_cat), pair_list in sorted(subgroups.items()):
    n = len(pair_list)
    a_h1 = sum(1 for a, c in pair_list if a <= 1) / n * 100
    c_h1 = sum(1 for a, c in pair_list if c <= 1) / n * 100
    a_mrr = sum(1.0/a for a, c in pair_list) / n
    c_mrr = sum(1.0/c for a, c in pair_list) / n
    subgroup_summary.append({
        'Patient': p_cat, 'Disease': d_cat, 'N': n,
        'A_Hits@1': a_h1, 'C_Hits@1': c_h1, 'Diff_Hits@1': c_h1 - a_h1,
        'A_MRR': a_mrr, 'C_MRR': c_mrr, 'Diff_MRR': c_mrr - a_mrr
    })
    print(f"Patient {p_cat:<14} | Disease {d_cat:<14} | N={n:4d} | A Hits@1: {a_h1:5.2f}% -> C: {c_h1:5.2f}% (Diff: {c_h1-a_h1:+5.2f}%) | MRR: {a_mrr:.4f} -> {c_mrr:.4f}")

# Paired McNemar Test & Bootstrap
a_correct = np.array([r <= 1 for r in raw_ranks])
c_correct = np.array([r <= 1 for r in best_scaled_ranks])

b_only = np.sum(a_correct & ~c_correct) # A right, C wrong
c_only = np.sum(~a_correct & c_correct) # C right, A wrong
both = np.sum(a_correct & c_correct)
neither = np.sum(~a_correct & ~c_correct)

# McNemar test
mcnemar_stat = (abs(b_only - c_only) - 1)**2 / (b_only + c_only) if (b_only + c_only) > 0 else 0.0
p_val = stats.chi2.sf(mcnemar_stat, 1)

print(f"\nPaired Contingency Table: Both Correct={both}, Neither={neither}, A-only={b_only}, C-only={c_only}")
print(f"McNemar Chi2 Stat = {mcnemar_stat:.4f}, p-value = {p_val:.4f}")

audit_payload = {
    'raw_metrics': {'Hits@1': raw_hits1, 'Hits@5': raw_hits5, 'Hits@10': raw_hits10, 'MRR': raw_mrr},
    'alpha_sweep': [{'alpha': r['alpha'], 'Hits@1': r['Hits@1'], 'Hits@5': r['Hits@5'], 'Hits@10': r['Hits@10'], 'MRR': r['MRR']} for r in alpha_results],
    'subgroups': subgroup_summary,
    'mcnemar': {'b_only': int(b_only), 'c_only': int(c_only), 'both': int(both), 'neither': int(neither), 'p_value': float(p_val)}
}

with open('data/benchmark/phase6_5_audit_stats.json', 'w') as out:
    json.dump(audit_payload, out, indent=2)
print("Saved data/benchmark/phase6_5_audit_stats.json successfully!")
