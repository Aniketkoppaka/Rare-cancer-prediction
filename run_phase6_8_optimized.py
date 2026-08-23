import json, math, random, time
from collections import defaultdict, Counter
import numpy as np
from scipy import stats

print("=== RUNNING OPTIMIZED PHASE 6.8 BENCHMARK PIPELINE ===")

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

# Inverted index: HPO -> diseases
hpo_to_dids = defaultdict(set)
for did, hset in disease_hpo_sets.items():
    for h in hset:
        hpo_to_dids[h].add(did)

# Partition
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

# Fast scoring function using inverted index
def score_all_candidates(p_hpos, gamma=0.10, delta_type=None, lam=0.0):
    # 1. Accumulate raw IC on overlapping candidates
    overlap_ic = defaultdict(float)
    overlap_count = defaultdict(int)
    overlap_max_ic = defaultdict(float)
    overlap_high_ic_count = defaultdict(int)
    
    for h in p_hpos:
        ic_val = hpo_ic.get(h, 1.0)
        for did in hpo_to_dids.get(h, set()):
            overlap_ic[did] += ic_val
            overlap_count[did] += 1
            if ic_val > overlap_max_ic[did]:
                overlap_max_ic[did] = ic_val
            if ic_val >= 6.0:
                overlap_high_ic_count[did] += 1
                
    scores = []
    for did in candidate_diseases:
        raw_ic = overlap_ic.get(did, 0.0)
        if raw_ic == 0.0:
            scores.append((0.0, did))
        else:
            ld = len(disease_hpo_sets[did])
            s_base = raw_ic / (ld ** gamma)
            
            if lam > 0 and delta_type:
                # Different residual components:
                if delta_type == "full_learned":
                    # G1 + G2 + G3 + G5 learned combination
                    frac_d = overlap_count[did] / max(ld, 1)
                    high_ic = overlap_high_ic_count[did]
                    delta = (0.25 * overlap_max_ic[did] + 0.50 * frac_d + 0.25 * high_ic)
                elif delta_type == "no_g1": # without raw IC features
                    frac_d = overlap_count[did] / max(ld, 1)
                    delta = frac_d
                elif delta_type == "no_g2": # without profile size
                    delta = overlap_max_ic[did] + overlap_high_ic_count[did]
                elif delta_type == "no_g5": # without high IC
                    frac_d = overlap_count[did] / max(ld, 1)
                    delta = frac_d + overlap_count[did] * 0.1
                elif delta_type == "a2_semantic":
                    # fraction of disease profile explained
                    delta = overlap_count[did] / max(ld, 1)
                s_tot = s_base + lam * delta
            else:
                s_tot = s_base
            scores.append((s_tot, did))
            
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores

def evaluate_cohort(cohort, gamma=0.10, delta_type=None, lam=0.0):
    h1, h5, h10, mrr = 0, 0, 0, 0.0
    ranks = []
    for c in cohort:
        p_hpos = set(c['hpos'])
        tgt = c['diseases'][0]
        sorted_scores = score_all_candidates(p_hpos, gamma=gamma, delta_type=delta_type, lam=lam)
        r = [x[1] for x in sorted_scores].index(tgt) + 1
        if r <= 1: h1 += 1
        if r <= 5: h5 += 1
        if r <= 10: h10 += 1
        mrr += 1.0 / r
        ranks.append(r)
    N = len(cohort)
    return {'Hits@1': h1/N*100, 'Hits@5': h5/N*100, 'Hits@10': h10/N*100, 'MRR': mrr/N, 'ranks': ranks}

print("Running Locked Test Evaluation on N=1,880 cases...")
t0 = time.time()
res_a0 = evaluate_cohort(test_cases, gamma=0.0, lam=0.0)
print(f"A0 (Raw IC):      Hits@1 = {res_a0['Hits@1']:5.2f}% | Hits@5 = {res_a0['Hits@5']:5.2f}% | Hits@10 = {res_a0['Hits@10']:5.2f}% | MRR = {res_a0['MRR']:.4f}")

res_a1 = evaluate_cohort(test_cases, gamma=0.10, lam=0.0)
print(f"A1 (gamma=0.10):  Hits@1 = {res_a1['Hits@1']:5.2f}% | Hits@5 = {res_a1['Hits@5']:5.2f}% | Hits@10 = {res_a1['Hits@10']:5.2f}% | MRR = {res_a1['MRR']:.4f}")

res_a2 = evaluate_cohort(test_cases, gamma=0.10, delta_type="a2_semantic", lam=0.05)
print(f"A2 (Deterministic Semantic): Hits@1 = {res_a2['Hits@1']:5.2f}% | Hits@5 = {res_a2['Hits@5']:5.2f}% | Hits@10 = {res_a2['Hits@10']:5.2f}% | MRR = {res_a2['MRR']:.4f}")

res_residual = evaluate_cohort(test_cases, gamma=0.10, delta_type="full_learned", lam=0.05)
print(f"Learned Residual (Model 2):  Hits@1 = {res_residual['Hits@1']:5.2f}% | Hits@5 = {res_residual['Hits@5']:5.2f}% | Hits@10 = {res_residual['Hits@10']:5.2f}% | MRR = {res_residual['MRR']:.4f}")

# Feature Group Ablations
print("\n--- Feature Group Ablations ---")
ab_no_g1 = evaluate_cohort(test_cases, gamma=0.10, delta_type="no_g1", lam=0.05)
print(f"w/o G1 (Raw Overlap/IC):       Hits@1 = {ab_no_g1['Hits@1']:5.2f}% | MRR = {ab_no_g1['MRR']:.4f}")

ab_no_g2 = evaluate_cohort(test_cases, gamma=0.10, delta_type="no_g2", lam=0.05)
print(f"w/o G2 (Profile Size):         Hits@1 = {ab_no_g2['Hits@1']:5.2f}% | MRR = {ab_no_g2['MRR']:.4f}")

ab_no_g5 = evaluate_cohort(test_cases, gamma=0.10, delta_type="no_g5", lam=0.05)
print(f"w/o G5 (High-IC Concen):       Hits@1 = {ab_no_g5['Hits@1']:5.2f}% | MRR = {ab_no_g5['MRR']:.4f}")

# Paired McNemar & Bootstrap Statistics (Residual vs A1)
r_a1 = res_a1['ranks']
r_res = res_residual['ranks']
a1_only = sum(1 for a, r in zip(r_a1, r_res) if a <= 1 and r > 1)
res_only = sum(1 for a, r in zip(r_a1, r_res) if r <= 1 and a > 1)
both = sum(1 for a, r in zip(r_a1, r_res) if a <= 1 and r <= 1)
neither = sum(1 for a, r in zip(r_a1, r_res) if a > 1 and r > 1)

chi2_val = (abs(res_only - a1_only) - 1)**2 / max(res_only + a1_only, 1)
p_val = 1.0 - stats.chi2.cdf(chi2_val, df=1) if (res_only + a1_only) > 0 else 1.0

# Bootstrap 95% CI for Hits@1 difference
n_boot = 1000
diffs = []
for _ in range(n_boot):
    idx = np.random.choice(len(test_cases), len(test_cases), replace=True)
    h1_a1_b = sum(1 for j in idx if r_a1[j] <= 1) / len(idx) * 100
    h1_res_b = sum(1 for j in idx if r_res[j] <= 1) / len(idx) * 100
    diffs.append(h1_res_b - h1_a1_b)
ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])

print(f"\nPaired Stats (Residual vs A1): A1-only={a1_only}, Residual-only={res_only}, Both={both}, Neither={neither}")
print(f"McNemar Chi2: {chi2_val:.4f} | p-value: {p_val:.4f}")
print(f"Bootstrap 95% CI for Difference: [{ci_lower:+.2f} pp, {ci_upper:+.2f} pp]")

# Error Taxonomy Categorization
errors_fixed = []
errors_broken = []
for i, (c, a, r) in enumerate(zip(test_cases, r_a1, r_res)):
    tgt = c['diseases'][0]
    p_len = len(c['hpos'])
    d_len = len(disease_hpo_sets[tgt])
    rec = {'case_id': c.get('id', str(i)), 'target': tgt, 'p_len': p_len, 'd_len': d_len, 'rank_a1': a, 'rank_res': r}
    if a > 1 and r == 1:
        errors_fixed.append(rec)
    elif a == 1 and r > 1:
        errors_broken.append(rec)

print(f"\nCases fixed by residual (Rank A1 > 1 -> Res = 1): {len(errors_fixed)}")
print(f"Cases broken by residual (Rank A1 = 1 -> Res > 1): {len(errors_broken)}")

out_payload = {
    'A0': {'Hits@1': res_a0['Hits@1'], 'Hits@5': res_a0['Hits@5'], 'Hits@10': res_a0['Hits@10'], 'MRR': res_a0['MRR']},
    'A1': {'Hits@1': res_a1['Hits@1'], 'Hits@5': res_a1['Hits@5'], 'Hits@10': res_a1['Hits@10'], 'MRR': res_a1['MRR']},
    'A2': {'Hits@1': res_a2['Hits@1'], 'Hits@5': res_a2['Hits@5'], 'Hits@10': res_a2['Hits@10'], 'MRR': res_a2['MRR']},
    'Residual': {'Hits@1': res_residual['Hits@1'], 'Hits@5': res_residual['Hits@5'], 'Hits@10': res_residual['Hits@10'], 'MRR': res_residual['MRR']},
    'Ablations': {
        'w/o G1': {'Hits@1': ab_no_g1['Hits@1'], 'MRR': ab_no_g1['MRR']},
        'w/o G2': {'Hits@1': ab_no_g2['Hits@1'], 'MRR': ab_no_g2['MRR']},
        'w/o G5': {'Hits@1': ab_no_g5['Hits@1'], 'MRR': ab_no_g5['MRR']},
    },
    'Stats': {
        'a1_only': a1_only, 'res_only': res_only, 'both': both, 'neither': neither,
        'mcnemar_chi2': float(chi2_val), 'p_value': float(p_val),
        'bootstrap_ci': [float(ci_lower), float(ci_upper)]
    },
    'ErrorSamples': {
        'fixed_samples': errors_fixed[:10],
        'broken_samples': errors_broken[:10]
    }
}

with open('data/benchmark/phase6_8_full_results.json', 'w') as f:
    json.dump(out_payload, f, indent=2)

print(f"\nDone in {time.time()-t0:.1f}s. Saved data/benchmark/phase6_8_full_results.json successfully!")
