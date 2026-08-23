import json
import os
import glob
from collections import Counter

base_dir = 'data/benchmark/phenopacket-store/notebooks'
dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
print(f"Total disease directories: {len(dirs)}")

cases = []
for d in dirs:
    files = glob.glob(os.path.join(d, '**/*.json'), recursive=True)
    for fpath in files:
        if 'metadata' in fpath or 'schema' in fpath:
            continue
        try:
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
            
            pfs = data.get('phenotypicFeatures', [])
            hpos = [p.get('type', {}).get('id') for p in pfs if p.get('type', {}).get('id') and not p.get('excluded', False)]
            
            diags = [x.get('term', {}).get('id') for x in data.get('diseases', []) if x.get('term', {}).get('id')]
            for interp in data.get('interpretations', []):
                did = interp.get('diagnosis', {}).get('disease', {}).get('id')
                if did:
                    diags.append(did)
                    
            hpos = list(set([h for h in hpos if str(h).startswith('HP:')]))
            diags = list(set([str(x) for x in diags if x]))
            
            if len(hpos) > 0 and len(diags) > 0:
                cases.append({'id': data.get('id', os.path.basename(fpath)), 'hpos': hpos, 'diseases': diags})
        except Exception:
            pass

all_diseases = [d for c in cases for d in c['diseases']]
all_hpos = [h for c in cases for h in c['hpos']]
print(f"Extracted {len(cases)} total valid patient cases!")
print(f"Unique diseases: {len(set(all_diseases))}")
print(f"Unique HPO terms: {len(set(all_hpos))}")
print(f"Mean HPO terms per patient: {len(all_hpos)/len(cases):.2f}")

with open('data/benchmark/parsed_phenopackets.json', 'w', encoding='utf-8') as out:
    json.dump(cases, out, indent=2)
print("Saved data/benchmark/parsed_phenopackets.json!")
