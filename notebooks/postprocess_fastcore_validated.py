from pathlib import Path
import pandas as pd
from cobra.io import read_sbml_model, write_sbml_model

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_official_fastcore_literature as r

OUT = r.FASTCORE_OUT / "validated"
OUT.mkdir(parents=True, exist_ok=True)


def read_ids(path):
    df = pd.read_csv(path)
    col = "reaction_id" if "reaction_id" in df.columns else df.columns[0]
    return set(str(x) for x in df[col].dropna().tolist())


def make_validated(label, sbml_path, prepare_fn, fastcore_csv, compact_csv, objective, direction):
    model = read_sbml_model(str(sbml_path))
    model.solver = "glpk"
    model = prepare_fn(model, sbml_path)

    fastcore_ids = read_ids(fastcore_csv)
    compact_ids = read_ids(compact_csv)
    required_exchange_ids = {
        "EX_HC00017_s", "EX_HC00040_s", "EX_HC00021_s", "EX_HC00177_s", "EX_HC00011_s",
        "EX_HC02131_c", "EX_HC02134_c", objective,
    }
    keep = {rid for rid in (fastcore_ids | compact_ids | required_exchange_ids) if rid in model.reactions}

    reduced = r.build_reduced_model(model, keep)
    xml_path = OUT / f"{label}_official_fastcore_literature_validated_reduced.xml"
    write_sbml_model(reduced, str(xml_path))

    pd.DataFrame({"reaction_id": sorted(keep)}).to_csv(OUT / f"{label}_official_fastcore_literature_validated_reactions.csv", index=False)
    summary = {
        "network": label,
        "fastcore_reactions_input": len(fastcore_ids),
        "compact_core_added": len(compact_ids - fastcore_ids),
        "validated_reactions": len(reduced.reactions),
        "validated_metabolites": len(reduced.metabolites),
        "xml_path": str(xml_path),
    }
    rows = r.validate_reduced(f"{label}_official_fastcore_literature_validated", xml_path, objective, direction)
    return summary, rows


def main():
    summaries = []
    rows = []
    s, v = make_validated(
        "zone12",
        r.HEPATONET,
        r.prepare_zone12_model,
        r.FASTCORE_OUT / "zone12" / "zone12_official_fastcore_reactions.csv",
        r.OUT / "zone12_compact_functional_core.csv",
        "r1032",
        "min",
    )
    summaries.append(s)
    rows.extend(v)

    s, v = make_validated(
        "zone3",
        r.HEPATONET_ZONE3,
        r.prepare_zone3_model,
        r.FASTCORE_OUT / "zone3" / "zone3_official_fastcore_reactions.csv",
        r.OUT / "zone3" / "zone3_compact_functional_core.csv",
        "r1389",
        "max",
    )
    summaries.append(s)
    rows.extend(v)

    pd.DataFrame(summaries).to_csv(OUT / "official_fastcore_literature_validated_summary.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "official_fastcore_literature_validated_validation_report.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()