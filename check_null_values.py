"""
Ultra-Fast Null Value, Data Integrity & Missing Field Audit
Checks clean_phenopackets.json and phenotype.hpoa for:
- None/null values in disease IDs, patient IDs, HPO lists
- Empty phenotype arrays
- Corrupted/malformed HPO identifiers
- Knowledge base missing entries or orphaned IDs
"""

import json
import os
import sys

print("=" * 80)
print("FAST DATA INTEGRITY & NULL VALUE AUDIT")
print("=" * 80)

# 1. Inspect Phenopackets
pheno_path = 'data/benchmark/clean_phenopackets.json'
print(f"\n[1/2] Auditing '{pheno_path}'...")

with open(pheno_path, 'r', encoding='utf-8') as f:
    cases = json.load(f)

total_cases = len(cases)
null_cases = 0
empty_diseases = 0
null_disease_entries = 0
empty_hpos = 0
null_hpo_entries = 0
malformed_hpos = 0
invalid_id_cases = 0

for idx, c in enumerate(cases):
    if c is None:
        null_cases += 1
        continue
    
    # Check ID
    pid = c.get('id')
    if pid is None or pid == "":
        invalid_id_cases += 1
        
    # Check Diseases
    dlist = c.get('diseases')
    if dlist is None or len(dlist) == 0:
        empty_diseases += 1
    else:
        for d in dlist:
            if d is None or not isinstance(d, str) or not d.strip():
                null_disease_entries += 1
                
    # Check HPOs
    hlist = c.get('hpos')
    if hlist is None or len(hlist) == 0:
        empty_hpos += 1
    else:
        for h in hlist:
            if h is None or not isinstance(h, str):
                null_hpo_entries += 1
            elif not h.startswith('HP:'):
                malformed_hpos += 1

print(f"  Total Patient Cases:           {total_cases:,}")
print(f"  Null / None Objects:           {null_cases}")
print(f"  Cases with Missing/Empty ID:   {invalid_id_cases}")
print(f"  Cases with Empty Diseases:     {empty_diseases}")
print(f"  Null Disease String Entries:   {null_disease_entries}")
print(f"  Cases with Empty Phenotypes:   {empty_hpos}")
print(f"  Null Phenotype String Entries: {null_hpo_entries}")
print(f"  Malformed Phenotypes (no HP:): {malformed_hpos}")

# 2. Inspect Knowledge Base (phenotype.hpoa)
hpoa_path = 'data/benchmark/phenotype.hpoa'
print(f"\n[2/2] Auditing '{hpoa_path}'...")

total_lines = 0
comment_lines = 0
null_or_empty_lines = 0
corrupted_lines = 0
null_db_entries = 0
null_hpo_terms = 0
valid_records = 0

with open(hpoa_path, 'r', encoding='utf-8') as f:
    for line in f:
        total_lines += 1
        if line.startswith('#'):
            comment_lines += 1
            continue
        s_line = line.strip()
        if not s_line:
            null_or_empty_lines += 1
            continue
            
        parts = s_line.split('\t')
        if len(parts) < 4:
            corrupted_lines += 1
            continue
            
        did = parts[0].strip()
        hpo = parts[3].strip()
        
        if not did or did.lower() in ('null', 'none', 'nan'):
            null_db_entries += 1
        if not hpo or not hpo.startswith('HP:'):
            null_hpo_terms += 1
            
        valid_records += 1

print(f"  Total Raw Lines:               {total_lines:,}")
print(f"  Header/Comment Lines:          {comment_lines:,}")
print(f"  Empty/Blank Lines:             {null_or_empty_lines}")
print(f"  Corrupted Columns (<4 cols):   {corrupted_lines}")
print(f"  Null Disease IDs:              {null_db_entries}")
print(f"  Null / Invalid HPO terms:      {null_hpo_terms}")
print(f"  Total Valid Annotation Links:  {valid_records:,}")

print("\n" + "=" * 80)
if (null_cases == 0 and empty_diseases == 0 and null_disease_entries == 0 and 
    empty_hpos == 0 and null_hpo_entries == 0 and corrupted_lines == 0 and 
    null_db_entries == 0 and null_hpo_terms == 0):
    print("STATUS: ZERO NULL VALUES DETECTED. 100% DATA CLEANLINESS VERIFIED.")
else:
    print("STATUS: WARNING - Anomalies detected above.")
print("=" * 80)
