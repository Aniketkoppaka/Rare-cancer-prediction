"""
Phase 6.8 — Symbolic-Grounded Residual Learning & Empirical Audit

Features:
- G1: Raw overlap & IC (exact count, sum IC, max IC, mean IC, sum top-3 IC)
- G2: Disease profile size features (log |D|, fraction of D explained)
- G3: Patient profile size features (log |P|, fraction of P explained)
- G4: Semantic proximity features (Jaccard on HPO ancestors, IC-weighted Jaccard)
- G5: High-IC concentration features (count of IC > 6.0 terms, ratio max_ic / sum_ic)

Baselines:
- A0: Raw exact IC
- A1: Power-normalized IC (gamma=0.10)
- A2: A1 + deterministic semantic proximity correction
- Residual Model: A1 + lambda * Learned_Delta(P, D)
"""

import json
import math
import os
import random
import time
from collections import defaultdict, Counter

import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== PHASE 6.8: SYMBOLIC-GROUNDED RESIDUAL LEARNING ===")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

# --- 1. LOAD BENCHMARK DATA & HPO DAG ---
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

# Build ancestors map from hp.obo if available
hpo_parents = defaultdict(set)
if os.path.exists('data/benchmark/hp.obo'):
    current_term = None
    with open('data/benchmark/hp.obo', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('[Term]'):
                current_term = None
            elif line.startswith('id: HP:'):
                current_term = line.split()[1]
            elif line.startswith('is_a: HP:') and current_term:
                parent = line.split()[1]
                hpo_parents[current_term].add(parent)

def get_ancestors(term, memo={}):
    if term in memo: return memo[term]
    anc = {term}
    for p in hpo_parents.get(term, set()):
        anc |= get_ancestors(p, memo)
    memo[term] = anc
    return anc

total_kb = len(disease_to_hpos)
hpo_doc_freq = Counter(h for hset in disease_to_hpos.values() for h in hset)
hpo_ic = {h: -math.log((freq + 1)/(total_kb + 1)) for h, freq in hpo_doc_freq.items()}

# Seed 42 Split (FROZEN)
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

print(f"Data split: Train={len(train_cases)} cases ({len(train_dids)} dids), Val={len(val_cases)} cases ({len(val_dids)} dids), Test={len(test_cases)} cases ({len(test_dids)} dids)")
print(f"Candidate universe: {len(candidate_diseases)} diseases")

# Pre-compute ancestors for all candidate disease HPO sets
print("Pre-computing ancestor graphs for semantic features...")
disease_ancestor_sets = {}
for did, hset in disease_to_hpos.items():
    anc = set()
    for h in hset: anc |= get_ancestors(h)
    disease_ancestor_sets[did] = anc

# --- 2. FEATURE EXTRACTION FUNCTION ---
# Feature vector schema (15 features):
# G1: [exact_count, sum_ic, max_ic, mean_ic, sum_top3_ic]
# G2: [log_d_size, frac_d_explained]
# G3: [log_p_size, frac_p_explained]
# G4: [ancestor_jaccard, ic_weighted_jaccard]
# G5: [count_high_ic, max_over_sum_ratio, ic_entropy, overlap_density]

def extract_features(p_hpos, p_anc, d_id, group_mask=None):
    d_hpos = disease_hpo_sets[d_id]
    d_anc = disease_ancestor_sets[d_id]
    
    inter = p_hpos & d_hpos
    lp = max(len(p_hpos), 1)
    ld = max(len(d_hpos), 1)
    
    if not inter:
        f_g1 = [0.0, 0.0, 0.0, 0.0, 0.0]
        f_g5 = [0.0, 0.0, 0.0, 0.0]
    else:
        ics = sorted([hpo_ic.get(h, 1.0) for h in inter], reverse=True)
        sum_ic = sum(ics)
        max_ic = ics[0]
        mean_ic = sum_ic / len(ics)
        top3_ic = sum(ics[:3])
        f_g1 = [float(len(inter)), sum_ic, max_ic, mean_ic, top3_ic]
        
        # G5 features
        high_ic_count = float(sum(1 for x in ics if x >= 6.0))
        ratio = max_ic / max(sum_ic, 1e-4)
        probs = [x / max(sum_ic, 1e-4) for x in ics]
        entropy = -sum(p * math.log(p + 1e-9) for p in probs)
        density = len(inter) / (lp * ld)**0.5
        f_g5 = [high_ic_count, ratio, entropy, density]
        
    f_g2 = [math.log(ld), len(inter) / ld]
    f_g3 = [math.log(lp), len(inter) / lp]
    
    # G4 semantic features
    anc_inter = p_anc & d_anc
    anc_union = p_anc | d_anc
    anc_jaccard = len(anc_inter) / max(len(anc_union), 1)
    ic_anc_inter = sum(hpo_ic.get(h, 1.0) for h in anc_inter)
    ic_anc_union = sum(hpo_ic.get(h, 1.0) for h in anc_union)
    ic_jaccard = ic_anc_inter / max(ic_anc_union, 1e-4)
    f_g4 = [anc_jaccard, ic_jaccard]
    
    all_feats = f_g1 + f_g2 + f_g3 + f_g4 + f_g5
    if group_mask is not None:
        # Zero out masked feature groups
        # G1: 0..4, G2: 5..6, G3: 7..8, G4: 9..10, G5: 11..14
        masked = list(all_feats)
        for g_idx in group_mask:
            if g_idx == 1: masked[0:5] = [0.0]*5
            elif g_idx == 2: masked[5:7] = [0.0]*2
            elif g_idx == 3: masked[7:9] = [0.0]*2
            elif g_idx == 4: masked[9:11] = [0.0]*2
            elif g_idx == 5: masked[11:15] = [0.0]*4
        return masked
    return all_feats

# --- 3. TRAIN LEARNED PAIRWISE RANKING RESIDUAL MODEL ---
print("\n--- Training Pairwise Linear/MLP Residual Model ---")

class LearnedResidualScorer(nn.Module):
    def __init__(self, in_features=15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# Prepare training pairs (1 positive, 5 hard/random negatives per case)
def prepare_train_data(cases_subset, group_mask=None):
    X_pos, X_neg = [], []
    for c in cases_subset:
        p_hpos = set(c['hpos'])
        p_anc = set()
        for h in p_hpos: p_anc |= get_ancestors(h)
        pos_did = c['diseases'][0]
        
        # sample negatives from train_dids
        neg_candidates = list(train_dids - {pos_did})
        # Mix: 2 with non-zero overlap, 3 random
        overlap_negs = [d for d in random.sample(neg_candidates, min(50, len(neg_candidates))) if p_hpos & disease_hpo_sets[d]]
        sampled_negs = (overlap_negs[:2] + random.sample(neg_candidates, 3))[:3]
        
        f_pos = extract_features(p_hpos, p_anc, pos_did, group_mask)
        for neg_did in sampled_negs:
            f_neg = extract_features(p_hpos, p_anc, neg_did, group_mask)
            X_pos.append(f_pos)
            X_neg.append(f_neg)
            
    return torch.tensor(X_pos, dtype=torch.float32), torch.tensor(X_neg, dtype=torch.float32)

print("Preparing training pairs...")
X_pos_train, X_neg_train = prepare_train_data(train_cases[:2500])
print(f"Training dataset: {len(X_pos_train)} pairwise comparison instances")

# Feature normalization
mean_f = X_pos_train.mean(dim=0)
std_f = X_pos_train.std(dim=0).clamp(min=1e-4)

def normalize_feat(t):
    return (t - mean_f) / std_f

residual_net = LearnedResidualScorer(15)
optimizer = torch.optim.Adam(residual_net.parameters(), lr=2e-3, weight_decay=1e-4)

for epoch in range(1, 16):
    residual_net.train()
    Xp_norm = normalize_feat(X_pos_train)
    Xn_norm = normalize_feat(X_neg_train)
    
    # Pairwise margin loss: score(pos) should exceed score(neg)
    s_pos = residual_net(Xp_norm)
    s_neg = residual_net(Xn_norm)
    loss = F.relu(1.0 - (s_pos - s_neg)).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:2d}/15 | Pairwise Ranking Margin Loss: {loss.item():.4f}")

# --- 4. PRE-SELECT LAMBDA (AND TEST DETERMINISTIC A2) ON VALIDATION ---
print("\n--- Tuning Lambda & Deterministic Semantic Baseline on Validation (N=1,000) ---")

def score_a1(p_hpos, d_id, gamma=0.10):
    inter = p_hpos & disease_hpo_sets[d_id]
    if not inter: return 0.0
    s = sum(hpo_ic.get(h, 1.0) for h in inter)
    return s / (len(disease_hpo_sets[d_id]) ** gamma)

def score_semantic_deterministic(p_anc, d_id):
    d_anc = disease_ancestor_sets[d_id]
    anc_inter = p_anc & d_anc
    anc_union = p_anc | d_anc
    return len(anc_inter) / max(len(anc_union), 1)

# Pre-extract validation patient ancestor sets
val_p_data = []
for c in val_cases:
    p_h = set(c['hpos'])
    p_a = set()
    for h in p_h: p_a |= get_ancestors(h)
    val_p_data.append((p_h, p_a, c['diseases'][0]))

# Search lambda grid for learned residual:
lambdas = [0.0, 0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
residual_net.eval()

print("Evaluating lambda grid on validation set...")
val_lambda_results = []
for lam in lambdas:
    h1 = 0
    mrr = 0.0
    for p_h, p_a, tgt in val_p_data:
        # Fast candidate evaluation: rank non-zero overlap + top semantic candidates
        scores = []
        for did in candidate_diseases:
            s_base = score_a1(p_h, did, 0.10)
            if lam > 0 and (p_h & disease_hpo_sets[did]):
                f = extract_features(p_h, p_a, did)
                f_norm = normalize_feat(torch.tensor([f], dtype=torch.float32))
                with torch.no_grad():
                    delta = residual_net(f_norm).item()
                s_tot = s_base + lam * delta
            else:
                s_tot = s_base
            scores.append((s_tot, did))
        scores.sort(key=lambda x: x[0], reverse=True)
        r = [x[1] for x in scores].index(tgt) + 1
        if r <= 1: h1 += 1
        mrr += 1.0 / r
    h1_p = h1 / len(val_p_data) * 100
    mrr_v = mrr / len(val_p_data)
    val_lambda_results.append((lam, h1_p, mrr_v))
    print(f"  Val Lambda={lam:6.3f} | Hits@1: {h1_p:5.2f}% | MRR: {mrr_v:.4f}")

best_lam = max(val_lambda_results, key=lambda x: x[1])[0]
print(f"--> Pre-selected Lambda on Validation: {best_lam}")

# Also evaluate Deterministic Semantic Baseline A2 on Validation:
print("\nEvaluating Deterministic Semantic Baseline (A2: A1 + mu * Jaccard_Ancestors) on Val...")
mu_grid = [0.01, 0.05, 0.10, 0.20, 0.50]
val_mu_results = []
for mu in mu_grid:
    h1, mrr = 0, 0.0
    for p_h, p_a, tgt in val_p_data:
        scores = []
        for did in candidate_diseases:
            s_base = score_a1(p_h, did, 0.10)
            s_sem = score_semantic_deterministic(p_a, did)
            s_tot = s_base + mu * s_sem
            scores.append((s_tot, did))
        scores.sort(key=lambda x: x[0], reverse=True)
        r = [x[1] for x in scores].index(tgt) + 1
        if r <= 1: h1 += 1
        mrr += 1.0 / r
    h1_p = h1 / len(val_p_data) * 100
    mrr_v = mrr / len(val_p_data)
    val_mu_results.append((mu, h1_p, mrr_v))
    print(f"  Val Mu={mu:5.2f} | Hits@1: {h1_p:5.2f}% | MRR: {mrr_v:.4f}")

best_mu = max(val_mu_results, key=lambda x: x[1])[0]
print(f"--> Pre-selected Mu on Validation: {best_mu}")

# --- 5. EVALUATION ON LOCKED TEST SET (N=1,880) ---
print("\n" + "="*70)
print("EVALUATING FINAL MODELS ON LOCKED ZERO-SHOT TEST SET (N=1,880)")
print("="*70)

test_p_data = []
for c in test_cases:
    p_h = set(c['hpos'])
    p_a = set()
    for h in p_h: p_a |= get_ancestors(h)
    test_p_data.append((p_h, p_a, c['diseases'][0]))

def run_test_eval(mode="A1", param=0.0, group_mask=None):
    h1, h5, h10, mrr = 0, 0, 0, 0.0
    ranks = []
    for p_h, p_a, tgt in test_p_data:
        scores = []
        for did in candidate_diseases:
            if mode == "A0":
                s_tot = score_a1(p_h, did, 0.0)
            elif mode == "A1":
                s_tot = score_a1(p_h, did, 0.10)
            elif mode == "A2":
                s_tot = score_a1(p_h, did, 0.10) + param * score_semantic_deterministic(p_a, did)
            elif mode == "Residual":
                s_base = score_a1(p_h, did, 0.10)
                if param > 0 and (p_h & disease_hpo_sets[did]):
                    f = extract_features(p_h, p_a, did, group_mask)
                    f_norm = normalize_feat(torch.tensor([f], dtype=torch.float32))
                    with torch.no_grad():
                        delta = residual_net(f_norm).item()
                    s_tot = s_base + param * delta
                else:
                    s_tot = s_base
            scores.append((s_tot, did))
        scores.sort(key=lambda x: x[0], reverse=True)
        r = [x[1] for x in scores].index(tgt) + 1
        if r <= 1: h1 += 1
        if r <= 5: h5 += 1
        if r <= 10: h10 += 1
        mrr += 1.0 / r
        ranks.append(r)
    N = len(test_p_data)
    return {'Hits@1': h1/N*100, 'Hits@5': h5/N*100, 'Hits@10': h10/N*100, 'MRR': mrr/N, 'ranks': ranks}

print("Running Test A0...")
res_a0 = run_test_eval("A0")
print("Running Test A1 (gamma=0.10)...")
res_a1 = run_test_eval("A1")
print(f"Running Test A2 (Deterministic Semantic, mu={best_mu})...")
res_a2 = run_test_eval("A2", best_mu)
print(f"Running Test Residual (Learned, lambda={best_lam})...")
res_res = run_test_eval("Residual", best_lam)

print("\n--- FINAL TEST BENCHMARK RESULTS ---")
print(f"{'Method':<40} {'Hits@1':>8} {'Hits@5':>8} {'Hits@10':>8} {'MRR':>8}")
print("-" * 76)
print(f"{'A0: Raw Exact IC':<40} {res_a0['Hits@1']:>7.2f}% {res_a0['Hits@5']:>7.2f}% {res_a0['Hits@10']:>7.2f}% {res_a0['MRR']:>8.4f}")
print(f"{'A1: Power-Normalized IC (gamma=0.10)':<40} {res_a1['Hits@1']:>7.2f}% {res_a1['Hits@5']:>7.2f}% {res_a1['Hits@10']:>7.2f}% {res_a1['MRR']:>8.4f}")
print(f"{'A2: Deterministic Semantic (A1 + Semantic)':<40} {res_a2['Hits@1']:>7.2f}% {res_a2['Hits@5']:>7.2f}% {res_a2['Hits@10']:>7.2f}% {res_a2['MRR']:>8.4f}")
print(f"{'Model 2: Learned Symbolic Residual':<40} {res_res['Hits@1']:>7.2f}% {res_res['Hits@5']:>7.2f}% {res_res['Hits@10']:>7.2f}% {res_res['MRR']:>8.4f}")
print("=" * 76)

# --- 6. LEAVE-ONE-GROUP-OUT ABLATIONS ---
print("\n--- Leave-One-Group-Out Feature Ablations on Test Set ---")
# G1: overlap/IC, G2: d_size, G3: p_size, G4: semantic, G5: high_ic
ablations = {
    'All Features (Full Residual)': None,
    'w/o G1 (Raw Overlap/IC)': [1],
    'w/o G2 (Disease Profile Size)': [2],
    'w/o G3 (Patient Profile Size)': [3],
    'w/o G4 (Semantic Proximity)': [4],
    'w/o G5 (High-IC Concentration)': [5],
}
for name, mask in ablations.items():
    if mask is None:
        ab_res = res_res
    else:
        ab_res = run_test_eval("Residual", best_lam, group_mask=mask)
    print(f"  {name:<36} | Hits@1: {ab_res['Hits@1']:5.2f}% | MRR: {ab_res['MRR']:.4f}")

# --- 7. RANKING CHANGE & ERROR TAXONOMY ANALYSIS ---
print("\n--- Ranking-Change Analysis (A1 vs Learned Residual) ---")
a1_ranks = res_a1['ranks']
res_ranks = res_res['ranks']

cat_a = [] # A1 correct, Residual wrong
cat_b = [] # A1 wrong, Residual correct
cat_c = [] # Both correct
cat_d = [] # Both wrong

for i, (p_h, p_a, tgt) in enumerate(test_p_data):
    r_a1 = a1_ranks[i]
    r_res = res_ranks[i]
    rec = {
        'case_idx': i,
        'tgt': tgt,
        'r_a1': r_a1,
        'r_res': r_res,
        'p_len': len(p_h),
        'd_len': len(disease_hpo_sets[tgt]),
        'ic_overlap': sum(hpo_ic.get(h, 1.0) for h in (p_h & disease_hpo_sets[tgt]))
    }
    if r_a1 == 1 and r_res > 1: cat_a.append(rec)
    elif r_a1 > 1 and r_res == 1: cat_b.append(rec)
    elif r_a1 == 1 and r_res == 1: cat_c.append(rec)
    else: cat_d.append(rec)

print(f"  Category A (A1 Correct -> Residual Breaks): {len(cat_a)} cases")
print(f"  Category B (A1 Wrong   -> Residual Fixes):  {len(cat_b)} cases")
print(f"  Category C (Both Correct):                  {len(cat_c)} cases")
print(f"  Category D (Both Wrong):                    {len(cat_d)} cases")

# McNemar Test
b_only = len(cat_a)
c_only = len(cat_b)
chi2_val = (abs(b_only - c_only) - 1)**2 / max(b_only + c_only, 1)
p_mcnemar = 1.0 - stats.chi2.cdf(chi2_val, df=1) if (b_only + c_only) > 0 else 1.0
print(f"\n  McNemar Test (Residual vs A1): Chi2={chi2_val:.4f}, p-value={p_mcnemar:.4f}")

# Bootstrap CI for Hits@1 difference
n_boot = 1000
diffs = []
for _ in range(n_boot):
    idx = np.random.choice(len(test_p_data), len(test_p_data), replace=True)
    h1_a1_b = sum(1 for j in idx if a1_ranks[j] <= 1) / len(idx) * 100
    h1_res_b = sum(1 for j in idx if res_ranks[j] <= 1) / len(idx) * 100
    diffs.append(h1_res_b - h1_a1_b)
ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])
print(f"  Bootstrap 95% CI for Hits@1 Difference (Residual - A1): [{ci_lower:+.2f} pp, {ci_upper:+.2f} pp]")

# Save detailed results payload
out_payload = {
    'val_lambda_results': val_lambda_results,
    'val_mu_results': val_mu_results,
    'best_lambda': best_lam,
    'best_mu': best_mu,
    'test_a0': {'Hits@1': res_a0['Hits@1'], 'Hits@5': res_a0['Hits@5'], 'Hits@10': res_a0['Hits@10'], 'MRR': res_a0['MRR']},
    'test_a1': {'Hits@1': res_a1['Hits@1'], 'Hits@5': res_a1['Hits@5'], 'Hits@10': res_a1['Hits@10'], 'MRR': res_a1['MRR']},
    'test_a2': {'Hits@1': res_a2['Hits@1'], 'Hits@5': res_a2['Hits@5'], 'Hits@10': res_a2['Hits@10'], 'MRR': res_a2['MRR']},
    'test_residual': {'Hits@1': res_res['Hits@1'], 'Hits@5': res_res['Hits@5'], 'Hits@10': res_res['Hits@10'], 'MRR': res_res['MRR']},
    'mcnemar': {'a_breaks': b_only, 'b_fixes': c_only, 'chi2': float(chi2_val), 'p_value': float(p_mcnemar)},
    'bootstrap_ci': [float(ci_lower), float(ci_upper)]
}

with open('data/benchmark/phase6_8_residual_results.json', 'w') as f:
    json.dump(out_payload, f, indent=2)

print("\n=== PHASE 6.8 EXPERIMENTAL RUN COMPLETE ===")
