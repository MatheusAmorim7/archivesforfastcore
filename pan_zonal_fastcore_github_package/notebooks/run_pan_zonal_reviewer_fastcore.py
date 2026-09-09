from pathlib import Path

import cobra
import pandas as pd
from cobra import Reaction
from cobra.flux_analysis import pfba


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
OUT = ROOT / "notebooks" / "fastcore_outputs" / "pan_zonal_reviewer_core"

HEPATONET_ORIGINAL = CONFIG / "hepatonet_original_13_08.xml"
CORE_CSV = ROOT / "notebooks" / "fastcore_outputs" / "reviewer_minimal_carbohydrate_core.csv"

PAN_ZONAL_CORE_CSV = OUT / "pan_zonal_reviewer_core_reactions.csv"
PAN_ZONAL_VALIDATION_CSV = OUT / "pan_zonal_reviewer_validation.csv"
PAN_ZONAL_MODEL_XML = OUT / "hepatonet_pan_zonal_reviewer_candidate.xml"


PHYSICELL_EXCHANGES = {
    "EX_HC00040_s": ("HC00040_s", -1000.0, 1000.0),  # glucose
    "EX_HC00017_s": ("HC00017_s", -1000.0, 0.0),     # O2 uptake only
    "EX_HC00021_s": ("HC00021_s", 0.0, 1000.0),      # CO2 release only
    "EX_HC00177_s": ("HC00177_s", -1000.0, 1000.0),  # lactate reversible for module tests
    "EX_HC00011_s": ("HC00011_s", -1000.0, 1000.0),  # water support
    "EX_HC02131_c": ("HC02131_c", -1000.0, 1000.0),  # glycogenin support
    "EX_HC02134_c": ("HC02134_c", -1000.0, 1000.0),  # glycogenin-G4G7 support
}


def _find_met(model, metabolite_id):
    """COBRA may keep or strip SBML prefixes depending on the source file."""
    candidates = [metabolite_id, f"M_{metabolite_id}"]
    for mid in candidates:
        if mid in model.metabolites:
            return model.metabolites.get_by_id(mid)
    raise KeyError(f"Metabolite not found: {metabolite_id}")


def _find_rxn(model, reaction_id):
    candidates = [reaction_id, f"R_{reaction_id}"]
    for rid in candidates:
        if rid in model.reactions:
            return model.reactions.get_by_id(rid)
    return None


def add_or_update_exchange(model, exchange_id, metabolite_id, lb, ub):
    rxn = _find_rxn(model, exchange_id)
    if rxn is None:
        met = _find_met(model, metabolite_id)
        rxn = Reaction(exchange_id)
        rxn.add_metabolites({met: -1.0})
        model.add_reactions([rxn])
    rxn.bounds = (lb, ub)
    return rxn


def load_pan_zonal_model():
    model = cobra.io.read_sbml_model(str(HEPATONET_ORIGINAL))

    for exchange_id, (metabolite_id, lb, ub) in PHYSICELL_EXCHANGES.items():
        add_or_update_exchange(model, exchange_id, metabolite_id, lb, ub)

    # Keep the sodium-coupled glucose route closed unless explicitly tested.
    r1539 = _find_rxn(model, "r1539")
    if r1539 is not None:
        r1539.bounds = (0.0, 0.0)

    return model


def load_core_reactions(model):
    core = pd.read_csv(CORE_CSV)
    present = []
    missing = []

    for rid in core["reaction_id"].dropna().astype(str):
        rxn = _find_rxn(model, rid)
        if rxn is None:
            missing.append(rid)
        else:
            present.append(rxn.id)

    return sorted(set(present)), sorted(set(missing))


def active_reactions_from_tests(model):
    """Add pFBA-active support reactions for reviewer modules.

    The reviewer scheme is partly lumped. These tests force small normalized
    fluxes through each module and then use pFBA to find parsimonious support
    routes that can be used as a seed for official fastCORE.
    """
    active = set()
    tests = []

    def add_temp_exchange(local_model, exchange_id, metabolite_id, lb, ub):
        met = _find_met(local_model, metabolite_id)
        rxn = Reaction(exchange_id)
        rxn.add_metabolites({met: -1.0})
        rxn.bounds = (lb, ub)
        local_model.add_reactions([rxn])
        return rxn

    def run_test(name, bounds, objective_id, direction="max", temp_exchanges=None):
        with model:
            temp_exchanges = temp_exchanges or {}
            for rid, spec in temp_exchanges.items():
                add_temp_exchange(model, rid, *spec)

            for rid, bound in bounds.items():
                rxn = _find_rxn(model, rid)
                if rxn is not None:
                    rxn.bounds = bound

            objective = _find_rxn(model, objective_id)
            if objective is None:
                tests.append({
                    "test": name,
                    "status": "missing_objective",
                    "objective": objective_id,
                    "objective_value": float("nan"),
                    "target_flux": float("nan"),
                })
                return

            model.objective = objective
            model.objective_direction = direction
            sol = pfba(model)
            tests.append({
                "test": name,
                "status": sol.status,
                "objective": objective.id,
                "objective_value": sol.objective_value,
                "target_flux": sol.fluxes.get(objective.id, float("nan")) if sol.status == "optimal" else float("nan"),
            })
            if sol.status == "optimal":
                for rid, flux in sol.fluxes.items():
                    if abs(flux) > 1e-9 and not rid.startswith("TMP_"):
                        active.add(rid)

    run_test(
        "glucose_uptake_to_g6p",
        {
            "EX_HC00040_s": (-1.0, -1.0),
            "r0353": (1.0, 1.0),
            "r1539": (0.0, 0.0),
        },
        "r0353",
    )
    run_test(
        "g6p_to_glucose_release",
        {
            "TMP_EX_HC00094_c": (-1.0, -1.0),
            "EX_HC00040_s": (1.0, 1.0),
            "r1539": (0.0, 0.0),
        },
        "EX_HC00040_s",
        temp_exchanges={"TMP_EX_HC00094_c": ("HC00094_c", -1.0, -1.0)},
    )
    run_test(
        "lactate_to_pyruvate",
        {"EX_HC00177_s": (-1.0, -1.0), "r0171": (1.0, 1.0)},
        "r0171",
    )
    run_test(
        "pyruvate_to_lactate",
        {
            "TMP_EX_HC00032_c": (-1.0, -1.0),
            "EX_HC00177_s": (1.0, 1.0),
        },
        "EX_HC00177_s",
        temp_exchanges={"TMP_EX_HC00032_c": ("HC00032_c", -1.0, -1.0)},
    )
    run_test(
        "oxidative_glucose_module",
        {
            "EX_HC00040_s": (-1.0, -1.0),
            "EX_HC00017_s": (-6.0, -6.0),
            "EX_HC00021_s": (6.0, 6.0),
            "r1539": (0.0, 0.0),
        },
        "EX_HC00021_s",
    )

    return active, tests


def write_outputs(model, core_present, core_missing, active_support, tests):
    OUT.mkdir(parents=True, exist_ok=True)

    all_core = sorted(set(core_present) | set(active_support))
    rows = []
    for rid in all_core:
        rxn = model.reactions.get_by_id(rid)
        rows.append({
            "reaction_id": rid,
            "lower_bound": rxn.lower_bound,
            "upper_bound": rxn.upper_bound,
            "equation": rxn.reaction,
            "source": "reviewer_core" if rid in core_present else "pfba_support",
        })

    pd.DataFrame(rows).to_csv(PAN_ZONAL_CORE_CSV, index=False)
    pd.DataFrame(tests).to_csv(PAN_ZONAL_VALIDATION_CSV, index=False)
    pd.DataFrame({"missing_reaction_id": core_missing}).to_csv(OUT / "missing_core_reactions.csv", index=False)

    # This is not a pruned fastCORE result yet. It is the pan-zonal candidate
    # model with original HepatoNet + PhysiCell exchanges + calibrated bounds.
    cobra.io.write_sbml_model(model, str(PAN_ZONAL_MODEL_XML))


def main():
    model = load_pan_zonal_model()
    core_present, core_missing = load_core_reactions(model)
    active_support, tests = active_reactions_from_tests(model)
    write_outputs(model, core_present, core_missing, active_support, tests)

    print("Pan-zonal reviewer core reactions:", len(core_present))
    print("pFBA support reactions:", len(active_support))
    print("Missing core reactions:", len(core_missing))
    print("Output folder:", OUT)


if __name__ == "__main__":
    main()
