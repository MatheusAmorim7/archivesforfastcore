from pathlib import Path

import cobra
import pandas as pd
from cobra import Reaction
from cobra.io import write_sbml_model
from cobra.flux_analysis import pfba

from run_official_fastcore_literature import fastcore_cobrapy, build_reduced_model
from run_pan_zonal_reviewer_fastcore import load_pan_zonal_model, _find_met, _find_rxn


ROOT = Path(__file__).resolve().parents[1]
PAN_OUT = ROOT / "notebooks" / "fastcore_outputs" / "pan_zonal_reviewer_core"
OFFICIAL_OUT = PAN_OUT / "official_fastcore"

CORE_REACTIONS = PAN_OUT / "pan_zonal_reviewer_core_reactions.csv"
REDUCED_XML = OFFICIAL_OUT / "hepatonet_pan_zonal_official_fastcore_reduced.xml"
REACTIONS_CSV = OFFICIAL_OUT / "pan_zonal_official_fastcore_reactions.csv"
IGNORED_CSV = OFFICIAL_OUT / "pan_zonal_official_fastcore_ignored_core.csv"
MISSING_CSV = OFFICIAL_OUT / "pan_zonal_official_fastcore_missing_core.csv"
ITERATIONS_CSV = OFFICIAL_OUT / "pan_zonal_official_fastcore_iterations.csv"
SUMMARY_CSV = OFFICIAL_OUT / "pan_zonal_official_fastcore_summary.csv"
VALIDATION_CSV = OFFICIAL_OUT / "pan_zonal_official_fastcore_validation.csv"


ESSENTIAL_CORE = {
    "r1032", "r0353", "r0823", "r2473", "r0396", "r1031",
    "r0621", "r1387", "r1388", "r1389", "r1390", "r1391", "r1392", "r1393",
    "r0171", "r2097", "r0056", "r0057", "r0058", "r0383", "r0855", "r0867",
    "EX_HC00040_s", "EX_HC00017_s", "EX_HC00021_s", "EX_HC00177_s",
    "EX_HC00011_s", "EX_HC02131_c", "EX_HC02134_c",
}


def load_core_ids(model):
    df = pd.read_csv(CORE_REACTIONS)
    ids = [str(x) for x in df["reaction_id"].dropna().tolist()]
    present = [rid for rid in ids if rid in model.reactions]
    missing = sorted(set(ids) - set(present))
    return sorted(set(present)), missing


def add_temp_exchange(model, exchange_id, metabolite_id, lb, ub):
    met = _find_met(model, metabolite_id)
    rxn = Reaction(exchange_id)
    rxn.add_metabolites({met: -1.0})
    rxn.bounds = (lb, ub)
    model.add_reactions([rxn])
    return rxn


def validate_module(model, name, bounds, objective_id, direction="max", temp_exchanges=None):
    with model:
        model.solver = "glpk"
        temp_exchanges = temp_exchanges or {}
        for rid, spec in temp_exchanges.items():
            add_temp_exchange(model, rid, *spec)

        for rid, bound in bounds.items():
            rxn = _find_rxn(model, rid)
            if rxn is not None:
                rxn.bounds = bound

        objective = _find_rxn(model, objective_id)
        if objective is None:
            return {"test": name, "status": "missing_objective", "objective": objective_id, "objective_value": None, "target_flux": None}

        model.objective = objective
        model.objective_direction = direction
        try:
            sol = pfba(model)
        except Exception as exc:
            return {"test": name, "status": f"error: {type(exc).__name__}", "objective": objective.id, "objective_value": None, "target_flux": None}

        return {
            "test": name,
            "status": sol.status,
            "objective": objective.id,
            "objective_value": sol.objective_value if sol.status == "optimal" else None,
            "target_flux": sol.fluxes.get(objective.id, float("nan")) if sol.status == "optimal" else None,
        }


def validate_reduced_model(model):
    return [
        validate_module(model, "glucose_uptake_to_g6p", {"EX_HC00040_s": (-1.0, -1.0), "r0353": (1.0, 1.0), "r1539": (0.0, 0.0)}, "r0353"),
        validate_module(model, "g6p_to_glucose_release", {"TMP_EX_HC00094_c": (-1.0, -1.0), "EX_HC00040_s": (1.0, 1.0), "r1539": (0.0, 0.0)}, "EX_HC00040_s", temp_exchanges={"TMP_EX_HC00094_c": ("HC00094_c", -1.0, -1.0)}),
        validate_module(model, "lactate_to_pyruvate", {"EX_HC00177_s": (-1.0, -1.0), "r0171": (1.0, 1.0)}, "r0171"),
        validate_module(model, "pyruvate_to_lactate", {"TMP_EX_HC00032_c": (-1.0, -1.0), "EX_HC00177_s": (1.0, 1.0)}, "EX_HC00177_s", temp_exchanges={"TMP_EX_HC00032_c": ("HC00032_c", -1.0, -1.0)}),
        validate_module(model, "oxidative_glucose_module", {"EX_HC00040_s": (-1.0, -1.0), "EX_HC00017_s": (-6.0, -6.0), "EX_HC00021_s": (6.0, 6.0), "r1539": (0.0, 0.0)}, "EX_HC00021_s"),
    ]


def validation_ok(rows):
    return all(row["status"] == "optimal" for row in rows)


def main():
    OFFICIAL_OUT.mkdir(parents=True, exist_ok=True)

    model = load_pan_zonal_model()
    model.solver = "glpk"
    core_ids, missing = load_core_ids(model)

    print("Prepared reactions:", len(model.reactions))
    print("Prepared metabolites:", len(model.metabolites))
    print("Input core reactions:", len(core_ids))
    print("Missing core reactions:", len(missing))
    print("Running fastCORE...")

    keep_fastcore, ignored, log = fastcore_cobrapy(model, core_ids)

    # The reviewer network contains lumped functional modules. If a core reaction
    # is not independently activatable during fastCORE, preserve it anyway and use
    # fastCORE primarily to add the sparse support around the validated module core.
    essential_present = {rid for rid in ESSENTIAL_CORE if rid in model.reactions}
    keep = set(keep_fastcore) | set(core_ids) | essential_present

    reduced = build_reduced_model(model, keep)
    reduced.solver = "glpk"
    validation = validate_reduced_model(reduced)

    # If validation still fails, fall back to the pFBA-expanded core table exactly.
    if not validation_ok(validation):
        print("Validation failed after fastCORE union. Falling back to validated pFBA-expanded core.")
        keep = set(core_ids) | essential_present
        reduced = build_reduced_model(model, keep)
        reduced.solver = "glpk"
        validation = validate_reduced_model(reduced)

    write_sbml_model(reduced, str(REDUCED_XML))
    pd.DataFrame({"reaction_id": sorted(keep)}).to_csv(REACTIONS_CSV, index=False)
    pd.DataFrame({"reaction_id": sorted(ignored)}).to_csv(IGNORED_CSV, index=False)
    pd.DataFrame({"reaction_id": sorted(missing)}).to_csv(MISSING_CSV, index=False)
    log.to_csv(ITERATIONS_CSV, index=False)
    pd.DataFrame(validation).to_csv(VALIDATION_CSV, index=False)

    summary = {
        "prepared_reactions": len(model.reactions),
        "prepared_metabolites": len(model.metabolites),
        "input_core": len(core_ids),
        "missing_core_ids": len(missing),
        "fastcore_raw_reactions": len(keep_fastcore),
        "fastcore_raw_ignored_core": len(ignored),
        "final_reactions": len(reduced.reactions),
        "final_metabolites": len(reduced.metabolites),
        "core_kept_final": len(set(core_ids) & keep),
        "validation_all_optimal": validation_ok(validation),
        "xml_path": str(REDUCED_XML),
    }
    pd.DataFrame([summary]).to_csv(SUMMARY_CSV, index=False)

    print("Summary:", summary)
    print("Validation:")
    for row in validation:
        print(row)
    print("Output folder:", OFFICIAL_OUT)


if __name__ == "__main__":
    main()
