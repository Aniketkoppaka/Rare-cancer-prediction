import json
import os
import math
from collections import defaultdict, Counter
import numpy as np

print("=== DEEP RECORD-LEVEL AUDIT & BENCHMARK REPRODUCTION ===")

# 1. Inspect clean_phenopackets.json
with open('data/benchmark/clean_phenopackets.json', 'r', encoding='utf-8') as f:
    cases = json.load(f)

print(f"Total raw parsed patient cases: {len(cases)}")

# 2. Detailed audit of cases
patient_ids = [c['id'] for c in cases]
files = [c['file'] for c in cases]
print(f"Unique case IDs: {len(set(patient_ids))}")
print(f"Unique JSON source files: {len(set(files))}")

# Check diseases per case
diseases_per_case = [len(c['diseases']) for c in cases]
hpos_per_case = [len(c['hpos']) for c in cases]

print(f"Diseases per case: mean={np.mean(diseases_per_case):.2f}, min={min(diseases_per_case)}, max={max(diseases_per_case)}")
print(f"HPOs per case: mean={np.mean(hpos_per_case):.2f}, min={min(hpos_per_case)}, max={max(hpos_per_case)}")

# Check duplicate phenotype profiles across cases
hpo_tuples = [tuple(sorted(c['hpos'])) for c in cases]
unique_hpo_profiles = len(set(hpo_tuples))
print(f"Unique distinct HPO symptom sets: {unique_hpo_profiles} / {len(cases)} ({unique_hpo_profiles/len(cases)*100:.1f}%)")

# Cases per disease distribution
d_to_cases = defaultdict(list)
for idx, c in enumerate(cases):
    for d in c['diseases']:
        d_to_cases[d].append(idx)

total_diseases = len(d_to_cases)
counts = [len(clist) for clist in d_to_cases.values()]

print(f"\nTotal unique diagnosed diseases: {total_diseases}")
print(f"1-case diseases (Singletons): {sum(1 for c in counts if c == 1)} ({sum(1 for c in counts if c == 1)/total_diseases*100:.1f}%)")
print(f"2-case diseases: {sum(1 for c in counts if c == 2)} ({sum(1 for c in counts if c == 2)/total_diseases*100:.1f}%)")
print(f"3-5 cases: {sum(1 for c in counts if 3 <= c <= 5)} ({sum(1 for c in counts if 3 <= c <= 5)/total_diseases*100:.1f}%)")
print(f"6-10 cases: {sum(1 for c in counts if 6 <= c <= 10)} ({sum(1 for c in counts if 6 <= c <= 10)/total_diseases*100:.1f}%)")
print(f"11-50 cases: {sum(1 for c in counts if 11 <= c <= 50)} ({sum(1 for c in counts if 11 <= c <= 50)/total_diseases*100:.1f}%)")
print(f">50 cases: {sum(1 for c in counts if c > 50)} ({sum(1 for c in counts if c > 50)/total_diseases*100:.1f}%)")

# Save record summary
summary = {
    'total_cases': len(cases),
    'unique_diseases': total_diseases,
    'unique_hpos': len(set([h for c in cases for h in c['hpos']])),
    'singletons_pct': sum(1 for c in counts if c == 1)/total_diseases*100,
    'few_shot_le_3_pct': sum(1 for c in counts if c <= 3)/total_diseases*100
}
with open('data/benchmark/audit_summary.json', 'w') as out:
    json.dump(summary, out, indent=2)
print("\nSaved audit summary to data/benchmark/audit_summary.json")
