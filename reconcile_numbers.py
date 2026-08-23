import json
from collections import defaultdict, Counter

print("=== RECONCILING ARITHMETIC DISCREPANCY ===")

with open('data/benchmark/clean_phenopackets.json', 'r', encoding='utf-8') as f:
    cases = json.load(f)

# Load KB
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

all_patient_diseases = set([c['diseases'][0] for c in cases])
print(f"Total raw parsed patient cases: {len(cases)}")
print(f"Total unique diagnosed diseases in patients: {len(all_patient_diseases)}")

# Check which diseases are in the knowledge base
in_kb_diseases = set([d for d in all_patient_diseases if d in disease_to_hpos])
not_in_kb_diseases = set([d for d in all_patient_diseases if d not in disease_to_hpos])

print(f"Diseases PRESENT in KB (phenotype.hpoa): {len(in_kb_diseases)}")
print(f"Diseases ABSENT from KB (phenotype.hpoa): {len(not_in_kb_diseases)}")
print(f"List of 4 absent diseases: {not_in_kb_diseases}")

# Cases breakdown
in_kb_cases = [c for c in cases if c['diseases'][0] in in_kb_diseases]
not_in_kb_cases = [c for c in cases if c['diseases'][0] in not_in_kb_diseases]

print(f"\nCases with disease IN KB: {len(in_kb_cases)}")
print(f"Cases with disease ABSENT from KB: {len(not_in_kb_cases)}")

# Detailed breakdown of absent cases
absent_counts = Counter([c['diseases'][0] for c in not_in_kb_cases])
print("Absent disease case counts:")
for d, cnt in absent_counts.items():
    print(f"  {d}: {cnt} cases")

print(f"Check sum: {len(in_kb_cases)} + {len(not_in_kb_cases)} = {len(in_kb_cases) + len(not_in_kb_cases)} (Total: {len(cases)})")
print(f"Check sum diseases: {len(in_kb_diseases)} + {len(not_in_kb_diseases)} = {len(in_kb_diseases) + len(not_in_kb_diseases)} (Total: {len(all_patient_diseases)})")
