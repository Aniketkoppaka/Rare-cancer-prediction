import json, math, random, time
from collections import defaultdict, Counter
import numpy as np
from scipy import stats
import torch, torch.nn as nn, torch.nn.functional as F

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

total_kb = len(disease_to_hpos)
hpo_doc_freq = Counter(h for hset in disease_to_hpos.values() for h in hset)
hpo_ic = {h: -math.log((freq + 1)/(total_kb + 1)) for h, freq in hpo_doc_freq.items()}

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

def extract_case_feats(p_hpos, did):
    d_hpos = disease_hpo_sets[did]
    inter = p_hpos & d_hpos
    if not inter: return None
    ics = sorted([hpo_ic.get(h, 1.0) for h in inter], reverse=True)
    sum_ic = sum(ics)
    max_ic = ics[0]
    mean_ic = sum_ic / len(ics)
    lp = len(p_hpos)
    ld = len(d_hpos)
    high_ic_count = float(sum(1 for x in ics if x >= 6.0))
    ratio = max_ic / max(sum_ic, 1e-4)
    return [
        float(len(inter)), sum_ic, max_ic, mean_ic,
        math.log(max(ld, 1)), len(inter)/max(ld, 1),
        high_ic_count, ratio
    ]

# Train linear residual model
X_p, X_n = [], []
for c in train_cases[:2000]:
    p_hpos = set(c['hpos'])
    pos_did = c['diseases'][0]
    f_pos = extract_case_feats(p_hpos, pos_did)
    if not f_pos: continue
    
    neg_cands = [d for d in train_dids - {pos_did} if p_hpos & disease_hpo_sets[d]]
    if not neg_cands: continue
    for neg_did in random.sample(neg_cands, min(3, len(neg_cands))):
        f_neg = extract_case_feats(p_hpos, neg_did)
        if f_neg:
            X_p.append(f_pos)
            X_n.append(f_neg)

X_p = torch.tensor(X_p, dtype=torch.float32)
X_n = torch.tensor(X_n, dtype=torch.float32)
mean_f = X_p.mean(dim=0)
std_f = X_p.std(dim=0).clamp(min=1e-4)

class LinearResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(8, 1, bias=False)
    def forward(self, x):
        return self.w(x).squeeze(-1)

net = LinearResidual()
opt = torch.optim.Adam(net.parameters(), lr=5e-3)
for ep in range(1, 21):
    s_p = net((X_p - mean_f)/std_f)
    s_n = net((X_n - mean_f)/std_f)
    loss = F.relu(1.0 - (s_p - s_n)).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

weights = net.w.weight.data[0].numpy()
print("Learned Feature Weights:", [round(float(x), 3) for x in weights])

def eval_pipeline(cohort, lam=0.0):
    h1, h5, h10, mrr = 0, 0, 0, 0.0
    ranks = []
    for c in cohort:
        p_hpos = set(c['hpos'])
        tgt = c['diseases'][0]
        scores = []
        for did in candidate_diseases:
            inter = p_hpos & disease_hpo_sets[did]
            if not inter:
                s = 0.0
            else:
                s_a1 = sum(hpo_ic.get(h, 1.0) for h in inter) / (len(disease_hpo_sets[did])**0.10)
                if lam > 0:
                    f = extract_case_feats(p_hpos, did)
                    f_norm = (np.array(f) - mean_f.numpy()) / std_f.numpy()
                    delta = float(np.dot(weights, f_norm))
                    s = s_a1 + lam * delta
                else:
                    s = s_a1
            scores.append((s, did))
        scores.sort(key=lambda x: x[0], reverse=True)
        r = [x[1] for x in scores].index(tgt) + 1
        if r <= 1: h1 += 1
        if r <= 5: h5 += 1
        if r <= 10: h10 += 1
        mrr += 1.0 / r
        ranks.append(r)
    N = len(cohort)
    return {'Hits@1': h1/N*100, 'Hits@5': h5/N*100, 'Hits@10': h10/N*100, 'MRR': mrr/N, 'ranks': ranks}

print("\n--- Validation Sweep (N=1000) ---")
for l in [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]:
    v_res = eval_pipeline(val_cases, lam=l)
    print(f"Val Lambda={l:5.2f} | Hits@1: {v_res['Hits@1']:5.2f}% | MRR: {v_res['MRR']:.4f}")

# Test set evaluation
print("\n--- Locked Test Set Evaluation (N=1880) ---")
# A0 (Raw IC)
h1_a0, h5_a0, h10_a0, mrr_a0 = 0, 0, 0, 0.0
ranks_a0 = []
for c in test_cases:
    p_hpos = set(c['hpos'])
    tgt = c['diseases'][0]
    scores = [(sum(hpo_ic.get(h, 1.0) for h in (p_hpos & disease_hpo_sets[did])), did) for did in candidate_diseases]
    scores.sort(key=lambda x: x[0], reverse=True)
    r = [x[1] for x in scores].index(tgt) + 1
    if r <= 1: h1_a0 += 1
    if r <= 5: h5_a0 += 1
    if r <= 10: h10_a0 += 1
    mrr_a0 += 1.0 / r
    ranks_a0.append(r)

test_a1 = eval_pipeline(test_cases, lam=0.0)
test_res = eval_pipeline(test_cases, lam=0.02) # optimal val lambda

print(f"Test A0 (Raw IC):               Hits@1 = {h1_a0/1880*100:5.2f}% | Hits@5 = {h5_a0/1880*100:5.2f}% | Hits@10 = {h10_a0/1880*100:5.2f}% | MRR = {mrr_a0/1880:.4f}")
print(f"Test A1 (Gamma=0.10):           Hits@1 = {test_a1['Hits@1']:5.2f}% | Hits@5 = {test_a1['Hits@5']:5.2f}% | Hits@10 = {test_a1['Hits@10']:5.2f}% | MRR = {test_a1['MRR']:.4f}")
print(f"Test Residual (A1 + 0.02*Delta): Hits@1 = {test_res['Hits@1']:5.2f}% | Hits@5 = {test_res['Hits@5']:5.2f}% | Hits@10 = {test_res['Hits@10']:5.2f}% | MRR = {test_res['MRR']:.4f}")

# Paired McNemar Test (Residual vs A1)
r_a1 = test_a1['ranks']
r_res = test_res['ranks']
a1_only = sum(1 for a, r in zip(r_a1, r_res) if a <= 1 and r > 1)
res_only = sum(1 for a, r in zip(r_a1, r_res) if r <= 1 and a > 1)
both = sum(1 for a, r in zip(r_a1, r_res) if a <= 1 and r <= 1)
neither = sum(1 for a, r in zip(r_a1, r_res) if a > 1 and r > 1)

chi2_val = (abs(res_only - a1_only) - 1)**2 / max(res_only + a1_only, 1)
p_val = 1.0 - stats.chi2.cdf(chi2_val, df=1) if (res_only + a1_only) > 0 else 1.0

print(f"\nPaired Stats (Residual vs A1): A1-only={a1_only}, Residual-only={res_only}, Both={both}, Neither={neither}")
print(f"McNemar Chi2: {chi2_val:.4f} | p-value: {p_val:.4f}")

out_stats = {
    'val_results': 'done',
    'test_a0': {'Hits@1': h1_a0/1880*100, 'Hits@5': h5_a0/1880*100, 'Hits@10': h10_a0/1880*100, 'MRR': mrr_a0/1880},
    'test_a1': test_a1,
    'test_residual': test_res,
    'paired': {'a1_only': a1_only, 'res_only': res_only, 'both': both, 'neither': neither, 'p_value': float(p_val)}
}
with open('data/benchmark/phase6_8_fast_results.json', 'w') as f:
    json.dump(out_stats, f, indent=2)
print("Saved data/benchmark/phase6_8_fast_results.json successfully!")
