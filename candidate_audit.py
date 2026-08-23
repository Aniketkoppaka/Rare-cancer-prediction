import json
from collections import defaultdict, Counter

print("=== AUDITING 12,956 CANDIDATE DISEASES IN HPOA ===")

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

total_kb_diseases = len(disease_to_hpos)
print(f"Total Diseases in HPOA: {total_kb_diseases}")

omim_kb = [d for d in disease_to_hpos if d.startswith('OMIM:')]
orpha_kb = [d for d in disease_to_hpos if d.startswith('ORPHA:')]
decipher_kb = [d for d in disease_to_hpos if d.startswith('DECIPHER:')]

print(f"  OMIM: {len(omim_kb)}")
print(f"  ORPHA: {len(orpha_kb)}")
print(f"  DECIPHER: {len(decipher_kb)}")

# Profile size distribution
profile_sizes = [len(hset) for hset in disease_to_hpos.values()]
print(f"\nKB Disease Profile Sizes:")
print(f"  Min: {min(profile_sizes)}")
print(f"  Max: {max(profile_sizes)}")
print(f"  Mean: {sum(profile_sizes)/len(profile_sizes):.2f}")
print(f"  Diseases with <= 3 HPOs: {sum(1 for s in profile_sizes if s <= 3)} ({sum(1 for s in profile_sizes if s <= 3)/total_kb_diseases*100:.1f}%)")
print(f"  Diseases with <= 5 HPOs: {sum(1 for s in profile_sizes if s <= 5)} ({sum(1 for s in profile_sizes if s <= 5)/total_kb_diseases*100:.1f}%)")
print(f"  Diseases with >= 50 HPOs: {sum(1 for s in profile_sizes if s >= 50)} ({sum(1 for s in profile_sizes if s >= 50)/total_kb_diseases*100:.1f}%)")
