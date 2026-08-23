import json
import os
import glob
import time

start = time.time()
base_dir = 'data/benchmark/phenopacket-store/notebooks'
dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

cases = []
seen_ids = set()

for d in dirs:
    for fpath in glob.glob(os.path.join(d, '**/*.json'), recursive=True):
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
                    
            hpos = sorted(list(set([str(h).strip() for h in hpos if str(h).startswith('HP:')])))
            diags = sorted(list(set([str(x).strip() for x in diags if x and (str(x).startswith('OMIM:') or str(x).startswith('ORPHA:'))])))
            
            cid = str(data.get('id', os.path.basename(fpath))).strip()
            
            if len(hpos) > 0 and len(diags) > 0:
                cases.append({
                    'id': cid,
                    'file': os.path.basename(fpath),
                    'hpos': hpos,
                    'diseases': diags
                })
        except Exception:
            pass

print(f"Extraction complete in {time.time() - start:.2f}s!")
print(f"Total Clean Patient Cases: {len(cases)}")
all_d = set([d for c in cases for d in c['diseases']])
all_h = set([h for c in cases for h in c['hpos']])
print(f"Total Unique Diseases: {len(all_d)}")
print(f"Total Unique HPO Terms: {len(all_h)}")

with open('data/benchmark/clean_phenopackets.json', 'w', encoding='utf-8') as out:
    json.dump(cases, out, indent=2)
print("Saved data/benchmark/clean_phenopackets.json cleanly!")
