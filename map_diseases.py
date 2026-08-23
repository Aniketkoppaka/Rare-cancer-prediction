import json
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter

print("=== MAPPING DISEASE IDS & ORPHADATA CROSS-REFERENCES ===")

# 1. Load OMIM diseases in Phenopackets
with open('data/benchmark/clean_phenopackets.json', 'r', encoding='utf-8') as f:
    cases = json.load(f)

patient_diseases = set([c['diseases'][0] for c in cases])
print(f"Total Unique Diagnosed Disease IDs in Patients: {len(patient_diseases)}")

# 2. Parse Orphadata en_product1.xml cross-references
orpha_xml = 'data/orphadata_xml/en_product1.xml'
omim_to_orpha = defaultdict(set)
orpha_to_omim = defaultdict(set)
orpha_to_name = {}

context = ET.iterparse(orpha_xml, events=('end',))
for event, elem in context:
    if elem.tag == 'Disorder':
        orpha_code = elem.findtext('OrphaCode')
        name = elem.findtext('Name')
        if orpha_code:
            orpha_id = f"ORPHA:{orpha_code}"
            orpha_to_name[orpha_id] = name
            
            # Look for ExternalReference
            for ext in elem.findall('.//ExternalReference'):
                src = ext.findtext('Source')
                ref = ext.findtext('Reference')
                if src == 'OMIM' and ref:
                    omim_id = f"OMIM:{ref}"
                    omim_to_orpha[omim_id].add(orpha_id)
                    orpha_to_omim[orpha_id].add(omim_id)
        elem.clear()

print(f"Orphadata Knowledge Base: {len(orpha_to_name)} distinct ORPHA disorders.")
print(f"Orphadata Cross-References: {len(omim_to_orpha)} unique OMIM IDs mapped to ORPHA codes.")

# 3. Check mapping rate for benchmark diseases
mapped_omim = set()
unmapped_omim = set()

for did in patient_diseases:
    if did in omim_to_orpha:
        mapped_omim.add(did)
    else:
        unmapped_omim.add(did)

print(f"\nBenchmark Mapping Results:")
print(f"  Mapped cleanly to ORPHA: {len(mapped_omim)} / {len(patient_diseases)} ({len(mapped_omim)/len(patient_diseases)*100:.1f}%)")
print(f"  Unmapped (OMIM only): {len(unmapped_omim)} / {len(patient_diseases)} ({len(unmapped_omim)/len(patient_diseases)*100:.1f}%)")

# How many patient cases map to ORPHA?
mapped_cases = sum(1 for c in cases if c['diseases'][0] in omim_to_orpha)
print(f"  Patient cases mapping to ORPHA: {mapped_cases} / {len(cases)} ({mapped_cases/len(cases)*100:.1f}%)")
