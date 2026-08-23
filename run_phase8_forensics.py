"""
Phase 8 — Forensic Verification & Information Sufficiency Engine

Computes:
1. Exact cumulative differential diagnosis curve (Top-1, 2, 3, 5, 10, 20, 50, 100).
2. Fine-grained Information Sufficiency Breakdown for all 590 errors (Categories I - VIII).
3. Raw Phenopacket feature audit (negated phenotypes, onset, modifiers, evidence).
4. Pairwise Ambiguity & Discriminating Phenotype Analysis.
"""

import json
import math
import os
import random
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
import numpy as np

print("=== RUNNING PHASE 8 FORENSIC & INFORMATION SUFFICIENCY ENGINE ===")

# --- 1. LOAD DATA ---
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
all_candidate_set = set(candidate_diseases)

total_kb = len(disease_to_hpos)
hpo_doc_freq = Counter(h for hset in disease_to_hpos.values() for h in hset)
hpo_ic = {h: -math.log((freq + 1)/(total_kb + 1)) for h, freq in hpo_doc_freq.items()}

# Inverted index
hpo_to_dids = defaultdict(set)
for did, hset in disease_hpo_sets.items():
    for h in hset:
        hpo_to_dids[h].add(did)

# Frozen Partition (Seed 42)
random.seed(42)
shuffled_dids = list(all_diseases)
random.shuffle(shuffled_dids)
n_train_d = int(0.70 * len(shuffled_dids))
n_val_d = int(0.15 * len(shuffled_dids))

train_dids = set(shuffled_dids[:n_train_d])
val_dids = set(shuffled_dids[n_train_d:n_train_d+n_val_d])
test_dids = set(shuffled_dids[n_train_d+n_val_d:])

test_cases = [c for c in valid_cases if c['diseases'][0] in test_dids]

# Parse Orphadata XMLs
tree1 = ET.parse('data/orphadata_xml/en_product1.xml')
disorders1 = tree1.getroot().find('DisorderList')

omim_to_orpha = defaultdict(set)
orpha_to_omim = defaultdict(set)
orpha_to_group = {}
orpha_to_type = {}
orpha_to_name = {}

for d in disorders1:
    code = d.find('OrphaCode').text
    name = d.find('Name').text if d.find('Name') is not None else ''
    orpha_to_name[f'ORPHA:{code}'] = name
    dtype = d.find('DisorderType/Name')
    if dtype is not None: orpha_to_type[f'ORPHA:{code}'] = dtype.text
    dgroup = d.find('DisorderGroup/Name')
    if dgroup is not None: orpha_to_group[f'ORPHA:{code}'] = dgroup.text
    ext_refs = d.find('ExternalReferenceList')
    if ext_refs is not None:
        for ref in ext_refs.findall('ExternalReference'):
            src = ref.find('Source').text
            ref_id = ref.find('Reference').text
            if src == 'OMIM':
                orpha_to_omim[f'ORPHA:{code}'].add(f'OMIM:{ref_id}')
                omim_to_orpha[f'OMIM:{ref_id}'].add(f'ORPHA:{code}')

tree6 = ET.parse('data/orphadata_xml/en_product6.xml')
disorders6 = tree6.getroot().find('DisorderList')
orpha_to_genes = defaultdict(set)
for d in disorders6:
    code = d.find('OrphaCode').text
    genes = d.find('DisorderGeneAssociationList')
    if genes is not None:
        for g in genes.findall('DisorderGeneAssociation'):
            sym = g.find('Gene/Symbol')
            if sym is not None and sym.text:
                orpha_to_genes[f'ORPHA:{code}'].add(sym.text)

def get_orpha_codes(did):
    if did.startswith('ORPHA:') or did.startswith('ORPHANET:'): return {f'ORPHA:{did.split(":")[-1]}'}
    elif did.startswith('OMIM:'): return omim_to_orpha.get(did, set())
    return set()

def get_genes_for_disease(did):
    orpha_codes = get_orpha_codes(did)
    genes = set()
    for oc in orpha_codes: genes |= orpha_to_genes.get(oc, set())
    return genes

def get_group_for_disease(did):
    orpha_codes = get_orpha_codes(did)
    for oc in orpha_codes:
        if oc in orpha_to_group: return orpha_to_group[oc]
    return 'Unknown'

# --- 2. FAST RANKING & DIFFERENTIAL DIAGNOSIS CURVE ---
print("Computing Cumulative Differential Diagnosis Curve on Locked Test Set (N=1,880)...")

test_ranks = []
all_test_records = []

for idx, c in enumerate(test_cases):
    p_hpos = set(c['hpos'])
    tgt = c['diseases'][0]
    
    overlap_ic = defaultdict(float)
    for h in p_hpos:
        ic_val = hpo_ic.get(h, 1.0)
        for did in hpo_to_dids.get(h, set()):
            overlap_ic[did] += ic_val
            
    cand_scores = {}
    for did, raw_ic in overlap_ic.items():
        cand_scores[did] = raw_ic / (len(disease_hpo_sets[did]) ** 0.10)
        
    tgt_score = cand_scores.get(tgt, 0.0)
    top1_did, top1_score = max(cand_scores.items(), key=lambda x: x[1]) if cand_scores else (None, 0.0)
    better_count = sum(1 for did, s in cand_scores.items() if s > tgt_score)
    r = better_count + 1
    
    test_ranks.append(r)
    all_test_records.append({
        'case_idx': idx,
        'case_id': c.get('id', str(idx)),
        'true_did': tgt,
        'pred_did': top1_did,
        'rank': r,
        'p_hpos': p_hpos,
        'p_len': len(p_hpos),
        'true_score': tgt_score,
        'pred_score': top1_score
    })

N_t = len(test_cases)
hits1 = sum(1 for r in test_ranks if r <= 1) / N_t * 100
hits2 = sum(1 for r in test_ranks if r <= 2) / N_t * 100
hits3 = sum(1 for r in test_ranks if r <= 3) / N_t * 100
hits5 = sum(1 for r in test_ranks if r <= 5) / N_t * 100
hits10 = sum(1 for r in test_ranks if r <= 10) / N_t * 100
hits20 = sum(1 for r in test_ranks if r <= 20) / N_t * 100
hits50 = sum(1 for r in test_ranks if r <= 50) / N_t * 100
hits100 = sum(1 for r in test_ranks if r <= 100) / N_t * 100
mrr = sum(1.0 / r for r in test_ranks) / N_t

print(f"\n--- Cumulative Differential Diagnosis Curve ---")
print(f"  Top-1  (Exact Diagnosis):   {hits1:6.2f}% ({sum(1 for r in test_ranks if r <= 1)}/{N_t})")
print(f"  Top-2  Differential:        {hits2:6.2f}% ({sum(1 for r in test_ranks if r <= 2)}/{N_t})")
print(f"  Top-3  Differential:        {hits3:6.2f}% ({sum(1 for r in test_ranks if r <= 3)}/{N_t})")
print(f"  Top-5  Differential:        {hits5:6.2f}% ({sum(1 for r in test_ranks if r <= 5)}/{N_t})")
print(f"  Top-10 Differential:        {hits10:6.2f}% ({sum(1 for r in test_ranks if r <= 10)}/{N_t})")
print(f"  Top-20 Differential:        {hits20:6.2f}% ({sum(1 for r in test_ranks if r <= 20)}/{N_t})")
print(f"  Top-50 Differential:        {hits50:6.2f}% ({sum(1 for r in test_ranks if r <= 50)}/{N_t})")
print(f"  Top-100 Differential:       {hits100:6.2f}% ({sum(1 for r in test_ranks if r <= 100)}/{N_t})")
print(f"  MRR:                        {mrr:.4f}")

# --- 3. INFORMATION SUFFICIENCY AUDIT FOR ALL 590 ERRORS ---
print("\n--- Information Sufficiency Breakdown for 590 Error Cases ---")
error_records = [r for r in all_test_records if r['rank'] > 1]
print(f"Total Errors to Classify: {len(error_records)}")

info_categories = Counter()
category_case_examples = defaultdict(list)

for err in error_records:
    p_hpos = err['p_hpos']
    true_hpos = disease_hpo_sets[err['true_did']]
    pred_hpos = disease_hpo_sets.get(err['pred_did'], set())
    
    # Check differentiating terms present in disease profiles
    d_true_only = true_hpos - pred_hpos
    d_pred_only = pred_hpos - true_hpos
    
    # Terms in patient matching True only vs Pred only
    p_matching_true_only = p_hpos & d_true_only
    p_matching_pred_only = p_hpos & d_pred_only
    
    true_orpha = get_orpha_codes(err['true_did'])
    pred_orpha = get_orpha_codes(err['pred_did'])
    true_genes = get_genes_for_disease(err['true_did'])
    pred_genes = get_genes_for_disease(err['pred_did'])
    true_grp = get_group_for_disease(err['true_did'])
    pred_grp = get_group_for_disease(err['pred_did'])
    
    # 1. Same concept / Ontology Artifact
    if bool(true_orpha & pred_orpha):
        cat = "VII. Same-concept / Ontology synonym artifact"
    # 2. Molecular / Genetic information strictly required
    elif (true_genes and pred_genes and bool(true_genes & pred_genes)) or (true_grp != 'Unknown' and true_grp == pred_grp and len(d_true_only) == 0):
        cat = "VI. Genetic/molecular distinction required (Shared locus/gene family)"
    # 3. Patient representation contains terms matching True ONLY (Method failed to rank)
    elif len(p_matching_true_only) > len(p_matching_pred_only) and sum(hpo_ic.get(h,1.0) for h in p_matching_true_only) > sum(hpo_ic.get(h,1.0) for h in p_matching_pred_only):
        cat = "I. Phenotype information theoretically sufficient (Scoring failure)"
    # 4. Extreme Sparsity (<= 2 patient terms or 0 overlap with true)
    elif len(p_hpos) <= 2 or len(p_hpos & true_hpos) == 0:
        cat = "III. Severe patient phenotype sparsity (<= 2 terms)"
    # 5. Patient has 0 discriminating terms (Phenotype set is pure subset of both diseases)
    elif len(p_matching_true_only) == 0 and len(p_matching_pred_only) == 0:
        cat = "II. Phenotype information insufficient in case (Pure overlapping subset)"
    # 6. Candidate distractor matches patient on non-specific common terms
    elif len(p_matching_pred_only) > 0 and all(hpo_ic.get(h, 1.0) < 4.0 for h in p_matching_pred_only):
        cat = "IV. Negative phenotype / exclusion required (Common symptom distractor)"
    # 7. Sibling disorders with subtle phenotypic differences
    elif true_grp != 'Unknown' and true_grp == pred_grp:
        cat = "V. Subtype/severity/onset modifiers required (Orphanet sibling syndrome)"
    else:
        cat = "VIII. Complex multi-system overlap / Unclear"
        
    info_categories[cat] += 1
    if len(category_case_examples[cat]) < 3:
        category_case_examples[cat].append({
            'case_id': err['case_id'],
            'true_did': err['true_did'],
            'pred_did': err['pred_did'],
            'rank': err['rank'],
            'p_len': err['p_len']
        })

print(f"\n{'Information Sufficiency Category':<72} {'Count':>6} {'Percent':>8}")
print("-" * 88)
for cat, count in sorted(info_categories.items(), key=lambda x: x[1], reverse=True):
    print(f"{cat:<72} {count:>6} {count/len(error_records)*100:>7.2f}%")
print("-" * 88)
print(f"Total Classified Errors: {sum(info_categories.values())} (Matches 590: {sum(info_categories.values()) == 590})")

# --- 4. RAW PHENOPACKET INFORMATION AUDIT ---
print("\n--- Auditing Information Lost in Reduction to P = set(HPO) ---")
# Check how many raw phenopackets contained age of onset, negated phenotypes, etc.
raw_onset_count = 0
raw_negated_count = 0
raw_severity_count = 0
raw_metadata_count = 0

for c in raw_cases:
    # check keys in dict
    if 'onset' in c or 'age_of_onset' in c or 'age' in c: raw_onset_count += 1
    if 'negated_hpos' in c or 'excluded_hpos' in c or 'negated' in c: raw_negated_count += 1
    if 'severity' in c or 'modifiers' in c: raw_severity_count += 1
    if 'sex' in c or 'gender' in c: raw_metadata_count += 1

print(f"Total Raw Cases: {len(raw_cases)}")
print(f"  Cases with Sex/Gender Metadata:   {raw_metadata_count:>6} ({raw_metadata_count/len(raw_cases)*100:.2f}%)")
print(f"  Cases with Age/Onset Metadata:    {raw_onset_count:>6} ({raw_onset_count/len(raw_cases)*100:.2f}%)")
print(f"  Cases with Negated/Excluded HPOs: {raw_negated_count:>6} ({raw_negated_count/len(raw_cases)*100:.2f}%)")
print(f"  Cases with Modifier/Severity:     {raw_severity_count:>6} ({raw_severity_count/len(raw_cases)*100:.2f}%)")

# --- 5. PAIRWISE AMBIGUITY EXPERIMENT ---
print("\n--- Pairwise Ambiguity Analysis: Top Confused Disease Pairs ---")
pair_counts = Counter((e['true_did'], e['pred_did']) for e in error_records)
top_confused_pairs = pair_counts.most_common(10)

pairwise_stats = []
for (t_did, p_did), count in top_confused_pairs:
    t_hpos = disease_hpo_sets[t_did]
    p_hpos = disease_hpo_sets.get(p_did, set())
    inter = t_hpos & p_hpos
    union = t_hpos | p_hpos
    jacc = len(inter) / max(len(union), 1)
    
    # Discriminating terms
    t_diff = t_hpos - p_hpos
    p_diff = p_hpos - t_hpos
    max_ic_t = max([hpo_ic.get(h, 1.0) for h in t_diff], default=0.0)
    max_ic_p = max([hpo_ic.get(h, 1.0) for h in p_diff], default=0.0)
    
    t_name = orpha_to_name.get(t_did, t_did)
    p_name = orpha_to_name.get(p_did, p_did)
    
    pairwise_stats.append({
        'true': t_did, 'pred': p_did, 'count': count,
        'jaccard': jacc, 'shared_hpos': len(inter),
        'true_exclusive_hpos': len(t_diff),
        'pred_exclusive_hpos': len(p_diff),
        'max_discriminating_ic': max(max_ic_t, max_ic_p)
    })
    print(f"  Pair: {t_did} vs {p_did} (Confused in {count} patients)")
    print(f"    Shared HPOs: {len(inter)} (Jaccard: {jacc:.2f}) | Discriminating terms: True={len(t_diff)}, Pred={len(p_diff)} (Max IC: {max(max_ic_t, max_ic_p):.2f})")

# Save Phase 8 stats
payload_p8 = {
    'diff_dx_curve': {
        'Top1': hits1, 'Top2': hits2, 'Top3': hits3, 'Top5': hits5,
        'Top10': hits10, 'Top20': hits20, 'Top50': hits50, 'Top100': hits100, 'MRR': mrr
    },
    'info_sufficiency': {k: int(v) for k, v in info_categories.items()},
    'pairwise_confusions': pairwise_stats
}

with open('data/benchmark/phase8_forensic_stats.json', 'w') as f:
    json.dump(payload_p8, f, indent=2)

print("\n=== PHASE 8 FORENSIC ENGINE COMPLETED SUCCESSFULLY ===")
