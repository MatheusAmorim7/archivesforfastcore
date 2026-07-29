from pathlib import Path
import csv
import xml.etree.ElementTree as ET

ROOT = Path('.').resolve()
SETTINGS = ROOT / 'config' / 'PhysiCell_settings.xml'
REPORT = ROOT / 'notebooks' / 'fastcore_outputs' / 'official_fastcore' / 'validated' / 'physicell_fastcore_static_coupling_check.csv'
DETAILS = ROOT / 'notebooks' / 'fastcore_outputs' / 'official_fastcore' / 'validated' / 'physicell_fastcore_static_coupling_check.txt'


def sbml_reaction_ids(path):
    tree = ET.parse(path)
    root = tree.getroot()
    ids = set()
    for elem in root.iter():
        if elem.tag.endswith('reaction') and 'id' in elem.attrib:
            ids.add(elem.attrib['id'])
    return ids


def norm_candidates(rid):
    ids = {rid}
    if rid.startswith('R_'):
        ids.add(rid[2:])
    else:
        ids.add('R_' + rid)
    return ids

settings_tree = ET.parse(SETTINGS)
settings_root = settings_tree.getroot()
rows = []
lines = []
cache = {}
for cd in settings_root.findall('.//cell_definition'):
    cell = cd.attrib.get('name', '')
    if not cell.startswith('zone_'):
        continue
    sbml_node = cd.find('.//intracellular/settings/sbml_filename')
    obj_node = cd.find('.//intracellular/growth_model/objective_reaction')
    if sbml_node is None:
        continue
    sbml_rel = sbml_node.text.strip()
    sbml_path = (ROOT / sbml_rel.replace('./', '')).resolve()
    exists = sbml_path.exists()
    if exists and sbml_path not in cache:
        cache[sbml_path] = sbml_reaction_ids(sbml_path)
    reactions = cache.get(sbml_path, set())
    lines.append(f'[{cell}] {sbml_rel}')
    rows.append({'cell_definition': cell, 'check': 'sbml_file', 'id': sbml_rel, 'status': 'present' if exists else 'missing'})

    objective = obj_node.text.strip() if obj_node is not None and obj_node.text else ''
    obj_present = any(x in reactions for x in norm_candidates(objective))
    rows.append({'cell_definition': cell, 'check': 'objective_reaction', 'id': objective, 'status': 'present' if obj_present else 'missing'})
    lines.append(f'objective {objective}: {"present" if obj_present else "MISSING"}')

    for ex in cd.findall('.//intracellular/transport_model/exchange'):
        substrate = ex.attrib.get('substrate', '')
        flux_node = ex.find('fba_flux')
        rid = flux_node.text.strip() if flux_node is not None and flux_node.text else ''
        present = any(x in reactions for x in norm_candidates(rid))
        rows.append({'cell_definition': cell, 'check': f'exchange:{substrate}', 'id': rid, 'status': 'present' if present else 'missing'})
        lines.append(f'exchange {substrate} {rid}: {"present" if present else "MISSING"}')
    lines.append('')

REPORT.parent.mkdir(parents=True, exist_ok=True)
with REPORT.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['cell_definition', 'check', 'id', 'status'])
    writer.writeheader()
    writer.writerows(rows)
DETAILS.write_text('\n'.join(lines))
print(REPORT)
print(DETAILS)
for row in rows:
    print(row)