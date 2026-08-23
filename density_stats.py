import json
import numpy as np
from collections import Counter

print("=== RECOMPUTING PHENOTYPE DENSITY & DISTRIBUTION ===")

with open('data/benchmark/clean_phenopackets.json', 'r', encoding='utf-8') as f:
    cases = json.load(f)

# Focus on the valid 10,601 cases present in KB
hpo_counts = [len(c['hpos']) for c in cases]

print(f"Total Cohort (N = {len(cases)}):")
print(f"  Min: {min(hpo_counts)}")
print(f"  Max: {max(hpo_counts)}")
print(f"  Mean: {np.mean(hpo_counts):.4f}")
print(f"  Median: {np.median(hpo_counts):.1f}")
print(f"  Std Dev: {np.std(hpo_counts):.4f}")
print(f"  Q1 (25th percentile): {np.percentile(hpo_counts, 25):.1f}")
print(f"  Q3 (75th percentile): {np.percentile(hpo_counts, 75):.1f}")

# Distribution buckets
le3 = sum(1 for c in hpo_counts if c <= 3)
le5 = sum(1 for c in hpo_counts if c <= 5)
le10 = sum(1 for c in hpo_counts if c <= 10)
gt20 = sum(1 for c in hpo_counts if c > 20)

print(f"\nBucket Breakdown:")
print(f"  Cases with <= 3 HPOs: {le3} ({le3/len(cases)*100:.2f}%)")
print(f"  Cases with <= 5 HPOs: {le5} ({le5/len(cases)*100:.2f}%)")
print(f"  Cases with <= 10 HPOs: {le10} ({le10/len(cases)*100:.2f}%)")
print(f"  Cases with > 20 HPOs: {gt20} ({gt20/len(cases)*100:.2f}%)")
