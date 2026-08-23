"""
Phase 7 — Kaggle GPU Standalone Runner: Orphadata Feasibility & Ambiguity Analysis

Executes in ~1-2 minutes:
1. Recomputes exact Orphadata mapping coverage.
2. Isolates the A1 test error population (591 errors across 12,956 candidate universe).
3. Classifies errors into Ambiguity Categories A-H.
4. Computes Phenotype Upper-Bounds (Oracle Distinguishability).
5. Runs Causal Hierarchy Re-ranking (O1-O4) on Full Universe & Mapped Subset.
6. Evaluates Hierarchical Error Distance (Clinical Proximity).
"""

import json
import math
import os
import random
import time
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
import numpy as np

print("=" * 70)
print("PHASE 7 — ORPHADATA FEASIBILITY, AMBIGUITY & CAUSAL INTEGRATION")
print("=" * 70)

# --- 1. RESOLVE FILE PATHS ---
def find_data_file(filename):
    for root, dirs, files in os.walk("/kaggle/input"):
        if filename in files:
            return os.path.join(root, filename)
    for root, dirs, files in os.walk("data"):
        if filename in files:
            return os.path.join(root, filename)
    return filename

phenopackets_path = find_data_file("clean_phenopackets.json")
hpoa_path = find_data_file("phenotype.hpoa")
product1_path = find_data_file("en_product1.xml")
product6_path = find_data_file("en_product6.xml")

print(f"[*] Phenopackets: {phenopackets_path}")
print(f"[*] HPOA:         {hpoa_path}")
print(f"[*] Product 1:    {product1_path}")
print(f"[*] Product 6:    {product6_path}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

# --- 2. LOAD DATA ---
with open(phenopackets_path, "r", encoding="utf-8") as f:
    raw_cases = json.load(f)

disease_to_hpos = defaultdict(set)
with open(hpoa_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("#") or not line.strip(): continue
        parts = line.strip().split("\t")
        if len(parts) >= 4 and parts[3].startswith("HP:"):
            disease_to_hpos[parts[0]].add(parts[3])

valid_cases = [c for c in raw_cases if c["diseases"][0] in disease_to_hpos]
all_diseases = sorted(list(set([c["diseases"][0] for c in valid_cases])))
disease_hpo_sets = {did: set(hset) for did, hset in disease_to_hpos.items()}
candidate_diseases = list(disease_to_hpos.keys())

total_kb = len(disease_to_hpos)
hpo_doc_freq = Counter(h for hset in disease_to_hpos.values() for h in hset)
hpo_ic = {h: -math.log((freq + 1)/(total_kb + 1)) for h, freq in hpo_doc_freq.items()}

# Inverted index
hpo_to_dids = defaultdict(set)
for did, hset in disease_hpo_sets.items():
    for h in hset:
        hpo_to_dids[h].add(did)

# Frozen Partition
random.seed(42)
shuffled_dids = list(all_diseases)
random.shuffle(shuffled_dids)
n_train_d = int(0.70 * len(shuffled_dids))
n_val_d = int(0.15 * len(shuffled_dids))

train_dids = set(shuffled_dids[:n_train_d])
val_dids = set(shuffled_dids[n_train_d:n_train_d+n_val_d])
test_dids = set(shuffled_dids[n_train_d+n_val_d:])

train_cases = [c for c in valid_cases if c["diseases"][0] in train_dids]
val_cases = [c for c in valid_cases if c["diseases"][0] in val_dids]
test_cases = [c for c in valid_cases if c["diseases"][0] in test_dids]

# --- 3. PARSE ORPHADATA XMLs ---
print("\n[1/6] Parsing Orphadata XML structures...")
tree1 = ET.parse(product1_path)
disorders1 = tree1.getroot().find("DisorderList")

omim_to_orpha = defaultdict(set)
orpha_to_omim = defaultdict(set)
orpha_to_group = {}
orpha_to_type = {}
orpha_to_name = {}

for d in disorders1:
    code = d.find("OrphaCode").text
    name = d.find("Name").text if d.find("Name") is not None else ""
    orpha_to_name[f"ORPHA:{code}"] = name
    
    dtype = d.find("DisorderType/Name")
    if dtype is not None: orpha_to_type[f"ORPHA:{code}"] = dtype.text
    
    dgroup = d.find("DisorderGroup/Name")
    if dgroup is not None: orpha_to_group[f"ORPHA:{code}"] = dgroup.text
    
    ext_refs = d.find("ExternalReferenceList")
    if ext_refs is not None:
        for ref in ext_refs.findall("ExternalReference"):
            src = ref.find("Source").text
            ref_id = ref.find("Reference").text
            if src == "OMIM":
                orpha_to_omim[f"ORPHA:{code}"].add(f"OMIM:{ref_id}")
                omim_to_orpha[f"OMIM:{ref_id}"].add(f"ORPHA:{code}")

orpha_to_genes = defaultdict(set)
if os.path.exists(product6_path) and os.path.getsize(product6_path) > 1000:
    tree6 = ET.parse(product6_path)
    disorders6 = tree6.getroot().find("DisorderList")
    for d in disorders6:
        code = d.find("OrphaCode").text
        genes = d.find("DisorderGeneAssociationList")
        if genes is not None:
            for g in genes.findall("DisorderGeneAssociation"):
                sym = g.find("Gene/Symbol")
                if sym is not None and sym.text:
                    orpha_to_genes[f"ORPHA:{code}"].add(sym.text)

def get_orpha_codes(did):
    if did.startswith("ORPHA:") or did.startswith("ORPHANET:"):
        return {f"ORPHA:{did.split(':')[-1]}"}
    elif did.startswith("OMIM:"):
        return omim_to_orpha.get(did, set())
    return set()

def get_genes_for_disease(did):
    orpha_codes = get_orpha_codes(did)
    genes = set()
    for oc in orpha_codes:
        genes |= orpha_to_genes.get(oc, set())
    return genes

def get_group_for_disease(did):
    orpha_codes = get_orpha_codes(did)
    for oc in orpha_codes:
        if oc in orpha_to_group: return orpha_to_group[oc]
    return "Unknown"

# Mapping Coverage Reconciliation
mapped_bench_dids = [d for d in all_diseases if len(get_orpha_codes(d)) > 0]
mapped_bench_cases = [c for c in valid_cases if len(get_orpha_codes(c["diseases"][0])) > 0]
mapped_cands = [d for d in candidate_diseases if len(get_orpha_codes(d)) > 0]

print(f"  Benchmark Diseases Mapped: {len(mapped_bench_dids)} / {len(all_diseases)} ({len(mapped_bench_dids)/len(all_diseases)*100:.2f}%)")
print(f"  Benchmark Cases Mapped:    {len(mapped_bench_cases)} / {len(valid_cases)} ({len(mapped_bench_cases)/len(valid_cases)*100:.2f}%)")
print(f"  Candidate Universe Mapped: {len(mapped_cands)} / {len(candidate_diseases)} ({len(mapped_cands)/len(candidate_diseases)*100:.2f}%)")

# --- 4. FAST SCORING ENGINE ---
def score_patient_fast(p_hpos, gamma=0.10, lam=0.0, use_gene=False):
    overlap_ic = defaultdict(float)
    for h in p_hpos:
        ic_val = hpo_ic.get(h, 1.0)
        for did in hpo_to_dids.get(h, set()):
            overlap_ic[did] += ic_val
            
    cand_scores = {}
    for did, raw_ic in overlap_ic.items():
        cand_scores[did] = raw_ic / (len(disease_hpo_sets[did]) ** gamma)
        
    if lam == 0.0 or not cand_scores:
        scores = [(cand_scores.get(did, 0.0), did) for did in candidate_diseases]
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores
        
    sorted_base = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)[:50]
    group_counts = Counter(get_group_for_disease(d) for d, s in sorted_base if get_group_for_disease(d) != "Unknown")
    tot_grp = sum(group_counts.values()) or 1
    
    gene_counts = Counter()
    if use_gene:
        for d, s in sorted_base:
            for g in get_genes_for_disease(d):
                gene_counts[g] += 1
                
    scores = []
    for did in candidate_diseases:
        s_base = cand_scores.get(did, 0.0)
        if s_base == 0.0:
            scores.append((0.0, did))
        else:
            grp = get_group_for_disease(did)
            grp_density = group_counts.get(grp, 0) / tot_grp
            gene_b = 0.0
            if use_gene:
                d_genes = get_genes_for_disease(did)
                if d_genes:
                    gene_b = sum(gene_counts.get(g, 0) for g in d_genes) / len(d_genes)
            s_final = s_base + lam * (grp_density * 2.0 + gene_b * 0.2)
            scores.append((s_final, did))
            
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores

# --- 5. ISOLATE A1 TEST ERROR POPULATION (N=591) ---
print("\n[2/6] Running A1 baseline on locked test set (N=1,880)...")
a1_errors = []
a1_correct = []
for idx, c in enumerate(test_cases):
    p_hpos = set(c["hpos"])
    tgt = c["diseases"][0]
    ranked = score_patient_fast(p_hpos, gamma=0.10, lam=0.0)
    top1_did = ranked[0][1]
    top1_score = ranked[0][0]
    r = [x[1] for x in ranked].index(tgt) + 1
    tgt_score = [x[0] for x in ranked if x[1] == tgt][0]
    rec = {
        "case_id": c.get("id", str(idx)),
        "true_did": tgt,
        "pred_did": top1_did,
        "true_score": tgt_score,
        "pred_score": top1_score,
        "rank": r,
        "p_hpos": p_hpos,
        "true_len": len(disease_hpo_sets[tgt]),
        "pred_len": len(disease_hpo_sets[top1_did])
    }
    if r == 1: a1_correct.append(rec)
    else: a1_errors.append(rec)

print(f"  Test A1 Hits@1: {len(a1_correct)} ({len(a1_correct)/len(test_cases)*100:.2f}%)")
print(f"  Test A1 Errors: {len(a1_errors)} ({len(a1_errors)/len(test_cases)*100:.2f}%)")

# --- 6. PHENOTYPE AMBIGUITY CATEGORIZATION (A-H) ---
print("\n[3/6] Categorizing 591 A1 Errors into Formal Ambiguity Categories (A-H)...")
error_categories = Counter()
for err in a1_errors:
    p_hpos = err["p_hpos"]
    true_hpos = disease_hpo_sets[err["true_did"]]
    pred_hpos = disease_hpo_sets[err["pred_did"]]
    
    t_inter = true_hpos & pred_hpos
    t_union = true_hpos | pred_hpos
    jaccard_d = len(t_inter) / max(len(t_union), 1)
    
    true_orpha = get_orpha_codes(err["true_did"])
    pred_orpha = get_orpha_codes(err["pred_did"])
    true_genes = get_genes_for_disease(err["true_did"])
    pred_genes = get_genes_for_disease(err["pred_did"])
    true_group = get_group_for_disease(err["true_did"])
    pred_group = get_group_for_disease(err["pred_did"])
    
    shared_genes = true_genes & pred_genes
    same_orpha = bool(true_orpha & pred_orpha)
    same_group = (true_group != "Unknown" and true_group == pred_group)
    
    p_true_inter = p_hpos & true_hpos
    p_pred_inter = p_hpos & pred_hpos
    p_true_ic = sum(hpo_ic.get(h, 1.0) for h in p_true_inter)
    p_pred_ic = sum(hpo_ic.get(h, 1.0) for h in p_pred_inter)
    
    if same_orpha or jaccard_d >= 0.70:
        cat = "A: Near-identical phenotype profiles / Same concept"
    elif same_group and (true_genes and pred_genes and shared_genes):
        cat = "C: Related diseases in same genetic/disease family"
    elif jaccard_d >= 0.35:
        cat = "B: Strong HPO overlap but distinct diseases"
    elif same_group:
        cat = "D: Same Orphanet clinical classification branch"
    elif err["pred_len"] >= 80 and p_pred_ic >= p_true_ic and len(p_true_inter) <= 2:
        cat = "F: Candidate caused by broad-profile accumulation"
    elif len(p_true_inter) == 0 or len(p_hpos) <= 2:
        cat = "G: Annotation mismatch / Sparse patient presentation"
    elif len(p_pred_inter & p_true_inter) > 0:
        cat = "E: Weak phenotype overlap on common symptoms"
    else:
        cat = "H: Unknown / Other"
    error_categories[cat] += 1

print(f"\n{'Category':<54} {'Count':>6} {'Percent':>8}")
print("-" * 70)
for cat, count in sorted(error_categories.items(), key=lambda x: x[1], reverse=True):
    print(f"{cat:<54} {count:>6} {count/len(a1_errors)*100:>7.2f}%")
print("-" * 70)

# --- 7. PHENOTYPE UPPER BOUND / ORACLE DISTINGUISHABILITY ---
print("\n[4/6] Phenotype Upper-Bound & Distinguishability Study...")
tied = sum(1 for e in a1_errors if math.isclose(e["true_score"], e["pred_score"], rel_tol=1e-5))
top5 = sum(1 for e in a1_errors if e["rank"] <= 5)
top10 = sum(1 for e in a1_errors if e["rank"] <= 10)
rank_gt_10 = len(a1_errors) - top10

print(f"  Errors strictly tied with Top-1 score: {tied:>4} ({tied/len(a1_errors)*100:5.2f}%)")
print(f"  Errors where True Disease is Rank 2-5:  {top5:>4} ({top5/len(a1_errors)*100:5.2f}%)")
print(f"  Errors where True Disease is Rank 6-10: {top10-top5:>4} ({(top10-top5)/len(a1_errors)*100:5.2f}%)")
print(f"  Errors where True Disease is Rank > 10: {rank_gt_10:>4} ({rank_gt_10/len(a1_errors)*100:5.2f}%)")

# --- 8. CAUSAL HIERARCHY RE-RANKING EXPERIMENTS ---
print("\n[5/6] Full-Universe & Mapped-Subset Evaluation (Methods O1 - O4)...")

def eval_test_mode(cohort, candidate_pool, lam=0.0, use_gene=False):
    h1, h5, h10, mrr = 0, 0, 0, 0.0
    for c in cohort:
        p_hpos = set(c["hpos"])
        tgt = c["diseases"][0]
        ranked = score_patient_fast(p_hpos, gamma=0.10, lam=lam, use_gene=use_gene)
        if candidate_pool != candidate_diseases:
            ranked = [x for x in ranked if x[1] in set(candidate_pool)]
        r = [x[1] for x in ranked].index(tgt) + 1
        if r <= 1: h1 += 1
        if r <= 5: h5 += 1
        if r <= 10: h10 += 1
        mrr += 1.0 / r
    N = len(cohort)
    return {"Hits@1": h1/N*100, "Hits@5": h5/N*100, "Hits@10": h10/N*100, "MRR": mrr/N}

# Full universe (12,956 candidates)
res_o1 = eval_test_mode(test_cases, candidate_diseases, lam=0.0)
res_o2 = eval_test_mode(test_cases, candidate_diseases, lam=0.01, use_gene=False)
res_o3 = eval_test_mode(test_cases, candidate_diseases, lam=0.01, use_gene=True)

# Mapped subset
mapped_test_cases = [c for c in test_cases if len(get_orpha_codes(c["diseases"][0])) > 0]
mapped_candidates_pool = [d for d in candidate_diseases if len(get_orpha_codes(d)) > 0]
res_m_o1 = eval_test_mode(mapped_test_cases, mapped_candidates_pool, lam=0.0)
res_m_o2 = eval_test_mode(mapped_test_cases, mapped_candidates_pool, lam=0.01, use_gene=False)

print("\n" + "=" * 76)
print("FINAL TEST BENCHMARK RESULTS (FULL UNIVERSE & MAPPED SUBSET)")
print("=" * 76)
print(f"{'Evaluation Setting / Method':<44} {'Hits@1':>8} {'Hits@5':>8} {'Hits@10':>8} {'MRR':>8}")
print("-" * 76)
print(f"{'Full Universe — O1: A1 Baseline':<44} {res_o1['Hits@1']:>7.2f}% {res_o1['Hits@5']:>7.2f}% {res_o1['Hits@10']:>7.2f}% {res_o1['MRR']:>8.4f}")
print(f"{'Full Universe — O2: A1 + Hierarchy Density':<44} {res_o2['Hits@1']:>7.2f}% {res_o2['Hits@5']:>7.2f}% {res_o2['Hits@10']:>7.2f}% {res_o2['MRR']:>8.4f}")
print(f"{'Full Universe — O3: A1 + Hierarchy + Genes':<44} {res_o3['Hits@1']:>7.2f}% {res_o3['Hits@5']:>7.2f}% {res_o3['Hits@10']:>7.2f}% {res_o3['MRR']:>8.4f}")
print("-" * 76)
print(f"{'Mapped Subset — O1: A1 Baseline':<44} {res_m_o1['Hits@1']:>7.2f}% {res_m_o1['Hits@5']:>7.2f}% {res_m_o1['Hits@10']:>7.2f}% {res_m_o1['MRR']:>8.4f}")
print(f"{'Mapped Subset — O2: A1 + Hierarchy Density':<44} {res_m_o2['Hits@1']:>7.2f}% {res_m_o2['Hits@5']:>7.2f}% {res_m_o2['Hits@10']:>7.2f}% {res_m_o2['MRR']:>8.4f}")
print("=" * 76)

# --- 9. HIERARCHICAL ERROR DISTANCE ---
print("\n[6/6] Hierarchical Error Distance (Clinical Diagnostic Proximity)...")
same_group_count = sum(1 for e in a1_errors if get_group_for_disease(e["true_did"]) != "Unknown" and get_group_for_disease(e["true_did"]) == get_group_for_disease(e["pred_did"]))
print(f"  A1 Top-1 Errors Sharing Exact Same Orphadata Group: {same_group_count} / {len(a1_errors)} ({same_group_count/len(a1_errors)*100:5.2f}%)")

# Save Phase 7 results
out_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
out_path = os.path.join(out_dir, "phase7_results.json")

payload = {
    "mapping_coverage": {
        "benchmark_diseases_mapped": len(mapped_bench_dids),
        "benchmark_diseases_total": len(all_diseases),
        "benchmark_cases_mapped": len(mapped_bench_cases),
        "benchmark_cases_total": len(valid_cases),
        "candidate_universe_mapped": len(mapped_cands),
        "candidate_universe_total": len(candidate_diseases)
    },
    "error_taxonomy": {k: int(v) for k, v in error_categories.items()},
    "upper_bound": {"tied": tied, "top5": top5, "top10": top10, "rank_gt_10": rank_gt_10, "total": len(a1_errors)},
    "full_universe": {"O1": res_o1, "O2": res_o2, "O3": res_o3},
    "mapped_subset": {"O1": res_m_o1, "O2": res_m_o2},
    "same_group_errors": {"count": same_group_count, "pct": float(same_group_count/len(a1_errors)*100)}
}

with open(out_path, "w") as f:
    json.dump(payload, f, indent=2)

print(f"\n[✓] Results successfully exported to: {out_path}")
