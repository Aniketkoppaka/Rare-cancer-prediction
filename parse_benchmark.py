import json
import os
import glob
from collections import Counter

print("Starting Phenopacket Store parsing...")
files = glob.glob('data/benchmark/phenopacket-store/**/*.json', recursive=True)
print(f"Discovered {len(files)} total JSON files.")

cases = []
diseases = []
hpos = []
hpos_per_case = []
skipped_no_diag = 0
skipped_no_hpo = 0

for i, fpath in enumerate(files):
    fname = os.path.basename(fpath)
    if fname.startswith('metadata') or 'schema' in fpath.lower() or 'template' in fpath.lower():
        continue
    
    try:
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as jf:
            data = json.load(jf)
        
        # Must be a phenopacket schema
        if not isinstance(data, dict):
            continue
        
        subject_id = data.get('id', fname)
        phenotypic_features = data.get('phenotypicFeatures', [])
        interpretations = data.get('interpretations', [])
        diseases_list = data.get('diseases', [])
        
        # 1. Collect non-negated HPO terms
        case_hpos = []
        for pf in phenotypic_features:
            type_obj = pf.get('type', {})
            hp_id = type_obj.get('id')
            if hp_id and str(hp_id).startswith('HP:') and not pf.get('excluded', False):
                case_hpos.append(str(hp_id))
        
        # 2. Collect ground truth diagnosed disease
        case_diseases = []
        for d in diseases_list:
            term = d.get('term', {})
            did = term.get('id')
            if did:
                case_diseases.append(str(did))
        
        for interp in interpretations:
            diag = interp.get('diagnosis', {})
            disease = diag.get('disease', {})
            did = disease.get('id')
            if did:
                case_diseases.append(str(did))
        
        case_hpos = list(set(case_hpos))
        case_diseases = list(set(case_diseases))
        
        if len(case_diseases) == 0:
            skipped_no_diag += 1
            continue
        if len(case_hpos) == 0:
            skipped_no_hpo += 1
            continue
        
        cases.append({
            'id': str(subject_id),
            'file': fname,
            'hpos': case_hpos,
            'diseases': case_diseases
        })
        hpos_per_case.append(len(case_hpos))
        for d in case_diseases:
            diseases.append(d)
        for h in case_hpos:
            hpos.append(h)
            
    except Exception:
        pass

print(f"\nSuccessfully Parsed {len(cases)} Valid Patient Profiles!")
print(f"Skipped without diagnosis: {skipped_no_diag}, Skipped without HPO: {skipped_no_hpo}")
print(f"Total Unique Diagnosed Diseases: {len(set(diseases))}")
print(f"Total Unique HPO Terms: {len(set(hpos))}")
print(f"Average HPOs per Patient Case: {sum(hpos_per_case)/len(hpos_per_case):.2f} (Range: {min(hpos_per_case)} - {max(hpos_per_case)})")

# Breakdown of disease prefixes (OMIM vs ORPHA vs MONDO)
d_prefixes = Counter([d.split(':')[0] for d in diseases])
print("Disease Vocabulary Prefix Counts in Cases:", dict(d_prefixes))

# Top 10 diseases
d_counts = Counter(diseases)
print("\nTop 10 Most Represented Rare Diseases:")
for d, count in d_counts.most_common(10):
    print(f"  {d}: {count} patient cases")

# Save structured dataset
os.makedirs('data/benchmark', exist_ok=True)
with open('data/benchmark/parsed_phenopackets.json', 'w', encoding='utf-8') as out:
    json.dump(cases, out, indent=2)
print("Saved data/benchmark/parsed_phenopackets.json successfully!")
