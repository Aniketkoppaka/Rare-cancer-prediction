import json
import os
import glob
import time
from collections import Counter

start_time = time.time()
files = glob.glob('data/benchmark/phenopacket-store/notebooks/**/*.json', recursive=True)
print(f"Discovered {len(files)} JSON files. Parsing...")

cases = []
diseases_all = []
hpos_all = []

for i, fpath in enumerate(files):
    fname = os.path.basename(fpath)
    if fname.startswith('metadata') or 'schema' in fpath.lower():
        continue
    try:
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as jf:
            data = json.load(jf)
        
        # Check phenotypic features
        pfs = data.get('phenotypicFeatures', [])
        hpos = [p.get('type', {}).get('id') for p in pfs if p.get('type', {}).get('id') and not p.get('excluded', False)]
        
        # Check diagnoses
        diags = [d.get('term', {}).get('id') for d in data.get('diseases', []) if d.get('term', {}).get('id')]
        for interp in data.get('interpretations', []):
            d = interp.get('diagnosis', {}).get('disease', {}).get('id')
            if d:
                diags.append(d)
        
        hpos = list(set([h for h in hpos if str(h).startswith('HP:')]))
        diags = list(set([str(d) for d in diags if d]))
        
        if len(hpos) > 0 and len(diags) > 0:
            cases.append({
                'id': data.get('id', fname),
                'hpos': hpos,
                'diseases': diags
            })
            for d in diags:
                diseases_all.append(d)
            for h in hpos:
                hpos_all.append(h)
    except Exception:
        pass

print(f"Finished parsing in {time.time() - start_time:.2f}s!")
print(f"Total Valid Patient Cases: {len(cases)}")
print(f"Total Unique Diagnosed Diseases: {len(set(diseases_all))}")
print(f"Total Unique HPO Terms: {len(set(hpos_all))}")

# Breakdown by disease vocabulary
vocab_counter = Counter([d.split(':')[0] for d in diseases_all])
print("Vocabulary distribution:", dict(vocab_counter))

# Save
with open('data/benchmark/parsed_phenopackets.json', 'w', encoding='utf-8') as out:
    json.dump(cases, out)
print("Saved data/benchmark/parsed_phenopackets.json")
