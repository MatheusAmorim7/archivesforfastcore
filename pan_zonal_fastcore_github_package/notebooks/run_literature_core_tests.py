from pathlib import Path

import cobra
import libsbml
import pandas as pd
from cobra import Reaction
from cobra.flux_analysis import pfba
from cobra.io import read_sbml_model, write_sbml_model


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "fastcore_outputs"
OUT.mkdir(parents=True, exist_ok=True)

HEPATONET = ROOT / "config" / "hepatonet.xml"
HEPATONET_ZONE3 = ROOT / "config" / "hepatonet_zone3.xml"
MANUAL_REDUCED = Path("/home/matheus/PhysiCelldFBA - Backup rede sbml reduzida/config/hepato_reduced.xml")


def safe_set_bounds(model, rid, lb, ub):
    if rid in model.reactions:
        model.reactions.get_by_id(rid).bounds = (lb, ub)
        return True
    print(f"[missing] {rid}")
    return False


def add_exchange_if_missing(model, rid, met_id, bounds):
    if rid in model.reactions:
        model.reactions.get_by_id(rid).bounds = bounds
        return
    met = model.metabolites.get_by_id(met_id)
    rxn = Reaction(rid)
    rxn.add_metabolites({met: -1})
    rxn.bounds = bounds
    model.add_reactions([rxn])


def reaction_table(model, rxn_ids, fluxes=None):
    rows = []
    for rid in sorted(rxn_ids):
        if rid not in model.reactions:
            continue
        rxn = model.reactions.get_by_id(rid)
        row = {
            "id": rxn.id,
            "name": rxn.name,
            "reaction": rxn.reaction,
            "lower_bound": rxn.lower_bound,
            "upper_bound": rxn.upper_bound,
        }
        if fluxes is not None and rid in fluxes.index:
            row["flux"] = fluxes[rid]
        rows.append(row)
    return pd.DataFrame(rows)


def find_reactions_by_patterns(model, patterns):
    found = set()
    for pattern in patterns:
        pattern = pattern.lower()
        for rxn in model.reactions:
            text = " ".join(
                [
                    rxn.id or "",
                    rxn.name or "",
                    rxn.reaction or "",
                    " ".join(m.id for m in rxn.metabolites),
                    " ".join((m.name or "") for m in rxn.metabolites),
                ]
            ).lower()
            if pattern in text:
                found.add(rxn.id)
    return found


def prepare_zone12_model(model, sbml_file):
    sbml_doc = libsbml.readSBMLFromFile(str(sbml_file))
    sbml_model = sbml_doc.getModel()
    for sbml_rxn in sbml_model.getListOfReactions():
        if not sbml_rxn.getReversible():
            rid = sbml_rxn.getId()
            if rid in model.reactions:
                rxn = model.reactions.get_by_id(rid)
                if rxn.lower_bound < 0:
                    rxn.lower_bound = 0

    for rxn in model.exchanges:
        rxn.bounds = (0, 0)

    add_exchange_if_missing(model, "EX_HC00011_s", "HC00011_s", (-1000, 1000))
    add_exchange_if_missing(model, "EX_HC02131_c", "HC02131_c", (0, 1000))

    safe_set_bounds(model, "EX_HC02134_c", -1000, 0)
    safe_set_bounds(model, "EX_HC00040_s", 0, 1000)
    safe_set_bounds(model, "EX_HC00017_s", -1000, 0)
    safe_set_bounds(model, "EX_HC00021_s", 0, 1000)
    safe_set_bounds(model, "EX_HC00011_s", -1000, 1000)
    safe_set_bounds(model, "EX_HC02131_c", 0, 1000)
    safe_set_bounds(model, "r1539", 0, 0)
    safe_set_bounds(model, "r1032", -1, 1000)

    model.objective = "r1032"
    model.objective.direction = "min"
    return model



def prepare_zone3_model(model, sbml_file):
    sbml_doc = libsbml.readSBMLFromFile(str(sbml_file))
    sbml_model = sbml_doc.getModel()
    for sbml_rxn in sbml_model.getListOfReactions():
        if not sbml_rxn.getReversible():
            rid = sbml_rxn.getId()
            if rid in model.reactions:
                rxn = model.reactions.get_by_id(rid)
                if rxn.lower_bound < 0:
                    rxn.lower_bound = 0

    for rxn in model.exchanges:
        rxn.bounds = (0, 0)

    add_exchange_if_missing(model, "EX_HC00011_s", "HC00011_s", (-1000, 1000))
    add_exchange_if_missing(model, "EX_HC02131_c", "HC02131_c", (-1000, 0))

    safe_set_bounds(model, "EX_HC00040_s", -1000, 0)
    safe_set_bounds(model, "EX_HC00017_s", -1000, 0)
    safe_set_bounds(model, "EX_HC00021_s", 0, 1000)
    safe_set_bounds(model, "EX_HC00011_s", -1000, 1000)
    safe_set_bounds(model, "EX_HC02134_c", 0, 1000)

    for rid in ["r1390", "r1391", "r1392", "r1393"]:
        safe_set_bounds(model, rid, 0, 0)
    safe_set_bounds(model, "r0208", 0, 0)
    safe_set_bounds(model, "r0801", -1.419354, -1.419354)
    safe_set_bounds(model, "r0054", -1.419354, -1.419354)
    safe_set_bounds(model, "EX_HC02131_c", -1, 0)
    safe_set_bounds(model, "r1389", 1, 1000)

    model.objective = "r1389"
    model.objective.direction = "max"
    return model

def run_pfba(model, objective, direction, bounds=None):
    with model:
        model.solver = "glpk"
        if bounds:
            for rid, bd in bounds.items():
                safe_set_bounds(model, rid, bd[0], bd[1])
        model.objective = objective
        model.objective.direction = direction
        try:
            sol = pfba(model)
        except Exception as e:
            print(f"[pfba failed] objective={objective} direction={direction}: {e}")
            sol = model.optimize()
    active = set(sol.fluxes[sol.fluxes.abs() > 1e-9].index) if sol.status == "optimal" else set()
    return sol, active


def print_fluxes(title, sol, ids):
    print("\n" + title)
    print("status:", sol.status)
    print("objective:", sol.objective_value)
    for rid in ids:
        if rid in sol.fluxes.index:
            print(f"{rid}: {sol.fluxes[rid]}")


def build_reduced_from_core(model, core_ids):
    reduced = model.copy()
    remove_rxns = [rxn for rxn in reduced.reactions if rxn.id not in core_ids]
    reduced.remove_reactions(remove_rxns, remove_orphans=True)
    return reduced


def test_hepatonet_literature_core():
    print("=== Loading HepatoNet ===")
    model = read_sbml_model(str(HEPATONET))
    model.solver = "glpk"
    print("original reactions:", len(model.reactions))
    print("original metabolites:", len(model.metabolites))

    model = prepare_zone12_model(model, HEPATONET)
    write_sbml_model(model, str(OUT / "hepatonet_zone12_prepared.xml"))

    sol_r1032, r1032_active = run_pfba(model, "r1032", "min")
    print_fluxes("Zone 1/2 r1032 module", sol_r1032, ["r1032", "EX_HC00040_s", "EX_HC00017_s", "EX_HC00021_s"])
    reaction_table(model, r1032_active, sol_r1032.fluxes).to_csv(OUT / "module_r1032_active.csv", index=False)

    glc_ox_bounds = {
        "EX_HC00040_s": (-1, -1),
        "EX_HC00017_s": (-6, -6),
        "EX_HC00021_s": (6, 6),
        "EX_HC00011_s": (-1000, 1000),
    }
    sol_glc_ox, glucose_oxidation_active = run_pfba(model, "EX_HC00021_s", "max", glc_ox_bounds)
    print_fluxes(
        "Glucose oxidation module",
        sol_glc_ox,
        ["EX_HC00040_s", "EX_HC00017_s", "EX_HC00021_s", "EX_HC00011_s", "r1032"],
    )
    reaction_table(model, glucose_oxidation_active, sol_glc_ox.fluxes).to_csv(
        OUT / "module_glucose_oxidation_active.csv", index=False
    )

    lactate_bounds = {
        "EX_HC00040_s": (-1, -1),
        "EX_HC00177_s": (0, 2),
        "EX_HC00017_s": (0, 0),
        "EX_HC00021_s": (0, 0),
    }
    sol_lac, lactate_active = run_pfba(model, "EX_HC00177_s", "max", lactate_bounds)
    print_fluxes(
        "Glucose to lactate module",
        sol_lac,
        ["EX_HC00040_s", "EX_HC00177_s", "EX_HC00017_s", "EX_HC00021_s", "r1032"],
    )
    reaction_table(model, lactate_active, sol_lac.fluxes).to_csv(OUT / "module_lactate_active.csv", index=False)

    carbohydrate_patterns = [
        "glucose", "glc", "HC00040", "oxygen", "o2", "HC00017", "carbon dioxide", "co2", "HC00021",
        "water", "h2o", "HC00011", "lactate", "lac", "HC00177", "pyruvate", "pyr", "glycogen",
        "glycogenin", "HC02131", "HC02134", "citrate", "succinate", "fumarate", "malate", "oxaloacetate",
        "ketoglutarate",
    ]
    nitrogen_patterns = [
        "glutamine", "gln", "glutamate", "glu", "alpha-ketoglutarate", "2-oxoglutarate",
        "ketoglutarate", "ammonia", "ammonium", "nh3", "nh4", "urea", "carbamoyl", "ornithine",
        "citrulline", "arginine", "argininosuccinate", "aspartate", "alanine", "transaminase",
        "aminotransferase", "nitrogen",
    ]
    carb_candidates = find_reactions_by_patterns(model, carbohydrate_patterns)
    nitrogen_candidates = find_reactions_by_patterns(model, nitrogen_patterns)
    reaction_table(model, carb_candidates).to_csv(OUT / "core_carbohydrate_keyword_candidates.csv", index=False)
    reaction_table(model, nitrogen_candidates).to_csv(OUT / "core_nitrogen_keyword_candidates.csv", index=False)

    must_keep = {
        "r1032", "EX_HC00040_s", "EX_HC00017_s", "EX_HC00021_s", "EX_HC00177_s",
        "EX_HC00011_s", "EX_HC02131_c", "EX_HC02134_c",
    }
    must_keep = {rid for rid in must_keep if rid in model.reactions}

    compact_core = sorted(must_keep | r1032_active | glucose_oxidation_active | lactate_active)
    literature_core = sorted(set(compact_core) | carb_candidates | nitrogen_candidates)
    pd.DataFrame({"reaction_id": compact_core}).to_csv(OUT / "zone12_compact_functional_core.csv", index=False)
    pd.DataFrame({"reaction_id": literature_core}).to_csv(OUT / "zone12_literature_guided_core.csv", index=False)

    reduced = build_reduced_from_core(model, compact_core)
    write_sbml_model(reduced, str(OUT / "hepatonet_zone12_compact_reduced_candidate.xml"))

    red_sol_r1032, _ = run_pfba(reduced, "r1032", "min")
    print_fluxes("Reduced candidate r1032", red_sol_r1032, ["r1032", "EX_HC00040_s", "EX_HC00017_s", "EX_HC00021_s"])
    red_sol_ox, _ = run_pfba(reduced, "EX_HC00021_s", "max", glc_ox_bounds)
    print_fluxes(
        "Reduced candidate glucose oxidation",
        red_sol_ox,
        ["EX_HC00040_s", "EX_HC00017_s", "EX_HC00021_s", "EX_HC00011_s", "r1032"],
    )

    summary = {
        "original_reactions": len(model.reactions),
        "original_metabolites": len(model.metabolites),
        "r1032_active": len(r1032_active),
        "glucose_oxidation_active": len(glucose_oxidation_active),
        "lactate_active": len(lactate_active),
        "carbohydrate_keyword_candidates": len(carb_candidates),
        "nitrogen_keyword_candidates": len(nitrogen_candidates),
        "compact_core": len(compact_core),
        "literature_core": len(literature_core),
        "reduced_reactions": len(reduced.reactions),
        "reduced_metabolites": len(reduced.metabolites),
        "reduced_r1032_status": red_sol_r1032.status,
        "reduced_oxidation_status": red_sol_ox.status,
    }
    pd.DataFrame([summary]).to_csv(OUT / "test_summary.csv", index=False)
    print("\nSUMMARY")
    for k, v in summary.items():
        print(f"{k}: {v}")



def test_hepatonet_zone3_literature_core():
    zone3_out = OUT / "zone3"
    zone3_out.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading HepatoNet Zone 3 ===")
    model = read_sbml_model(str(HEPATONET_ZONE3))
    model.solver = "glpk"
    print("zone3 original reactions:", len(model.reactions))
    print("zone3 original metabolites:", len(model.metabolites))

    model = prepare_zone3_model(model, HEPATONET_ZONE3)
    write_sbml_model(model, str(zone3_out / "hepatonet_zone3_prepared.xml"))

    sol_r1389, r1389_active = run_pfba(model, "r1389", "max")
    print_fluxes("Zone 3 r1389 module", sol_r1389, ["r1389", "EX_HC00040_s", "EX_HC00017_s", "EX_HC00021_s", "EX_HC02131_c", "EX_HC02134_c"])
    reaction_table(model, r1389_active, sol_r1389.fluxes).to_csv(zone3_out / "module_r1389_active.csv", index=False)

    glc_ox_bounds = {
        "EX_HC00040_s": (-1, -1),
        "EX_HC00017_s": (-6, -6),
        "EX_HC00021_s": (6, 6),
        "EX_HC00011_s": (-1000, 1000),
        "r1389": (0, 1000),
        "r0801": (-1000, 1000),
        "r0054": (-1000, 1000),
        "EX_HC02131_c": (-1000, 1000),
        "EX_HC02134_c": (-1000, 1000),
    }
    sol_glc_ox, glucose_oxidation_active = run_pfba(model, "EX_HC00021_s", "max", glc_ox_bounds)
    print_fluxes(
        "Zone 3 glucose oxidation module",
        sol_glc_ox,
        ["EX_HC00040_s", "EX_HC00017_s", "EX_HC00021_s", "EX_HC00011_s", "r1389"],
    )
    reaction_table(model, glucose_oxidation_active, sol_glc_ox.fluxes).to_csv(
        zone3_out / "module_glucose_oxidation_active.csv", index=False
    )

    lactate_bounds = {
        "EX_HC00040_s": (-1, -1),
        "EX_HC00177_s": (0, 2),
        "EX_HC00017_s": (0, 0),
        "EX_HC00021_s": (0, 0),
        "r1389": (0, 1000),
        "r0801": (-1000, 1000),
        "r0054": (-1000, 1000),
        "EX_HC02131_c": (-1000, 1000),
        "EX_HC02134_c": (-1000, 1000),
    }
    sol_lac, lactate_active = run_pfba(model, "EX_HC00177_s", "max", lactate_bounds)
    print_fluxes(
        "Zone 3 glucose to lactate module",
        sol_lac,
        ["EX_HC00040_s", "EX_HC00177_s", "EX_HC00017_s", "EX_HC00021_s", "r1389"],
    )
    reaction_table(model, lactate_active, sol_lac.fluxes).to_csv(zone3_out / "module_lactate_active.csv", index=False)

    carbohydrate_patterns = [
        "glucose", "glc", "HC00040", "oxygen", "o2", "HC00017", "carbon dioxide", "co2", "HC00021",
        "water", "h2o", "HC00011", "lactate", "lac", "HC00177", "pyruvate", "pyr", "glycogen",
        "glycogenin", "HC02131", "HC02134", "citrate", "succinate", "fumarate", "malate", "oxaloacetate",
        "ketoglutarate",
    ]
    nitrogen_patterns = [
        "glutamine", "gln", "glutamate", "glu", "alpha-ketoglutarate", "2-oxoglutarate",
        "ketoglutarate", "ammonia", "ammonium", "nh3", "nh4", "urea", "carbamoyl", "ornithine",
        "citrulline", "arginine", "argininosuccinate", "aspartate", "alanine", "transaminase",
        "aminotransferase", "nitrogen",
    ]
    carb_candidates = find_reactions_by_patterns(model, carbohydrate_patterns)
    nitrogen_candidates = find_reactions_by_patterns(model, nitrogen_patterns)
    reaction_table(model, carb_candidates).to_csv(zone3_out / "core_carbohydrate_keyword_candidates.csv", index=False)
    reaction_table(model, nitrogen_candidates).to_csv(zone3_out / "core_nitrogen_keyword_candidates.csv", index=False)

    must_keep = {
        "r1389", "EX_HC00040_s", "EX_HC00017_s", "EX_HC00021_s", "EX_HC00177_s",
        "EX_HC00011_s", "EX_HC02131_c", "EX_HC02134_c", "r1390", "r1391", "r1392", "r1393",
        "r0208", "r0801", "r0054",
    }
    must_keep = {rid for rid in must_keep if rid in model.reactions}

    compact_core = sorted(must_keep | r1389_active | glucose_oxidation_active | lactate_active)
    literature_core = sorted(set(compact_core) | carb_candidates | nitrogen_candidates)
    pd.DataFrame({"reaction_id": compact_core}).to_csv(zone3_out / "zone3_compact_functional_core.csv", index=False)
    pd.DataFrame({"reaction_id": literature_core}).to_csv(zone3_out / "zone3_literature_guided_core.csv", index=False)

    reduced = build_reduced_from_core(model, compact_core)
    write_sbml_model(reduced, str(zone3_out / "hepatonet_zone3_compact_reduced_candidate.xml"))

    red_sol_r1389, _ = run_pfba(reduced, "r1389", "max")
    print_fluxes("Zone 3 reduced candidate r1389", red_sol_r1389, ["r1389", "EX_HC00040_s", "EX_HC00017_s", "EX_HC00021_s", "EX_HC02131_c", "EX_HC02134_c"])
    red_sol_ox, _ = run_pfba(reduced, "EX_HC00021_s", "max", glc_ox_bounds)
    print_fluxes(
        "Zone 3 reduced candidate glucose oxidation",
        red_sol_ox,
        ["EX_HC00040_s", "EX_HC00017_s", "EX_HC00021_s", "EX_HC00011_s", "r1389"],
    )

    summary = {
        "zone3_original_reactions": len(model.reactions),
        "zone3_original_metabolites": len(model.metabolites),
        "r1389_active": len(r1389_active),
        "glucose_oxidation_active": len(glucose_oxidation_active),
        "lactate_active": len(lactate_active),
        "carbohydrate_keyword_candidates": len(carb_candidates),
        "nitrogen_keyword_candidates": len(nitrogen_candidates),
        "compact_core": len(compact_core),
        "literature_core": len(literature_core),
        "reduced_reactions": len(reduced.reactions),
        "reduced_metabolites": len(reduced.metabolites),
        "reduced_r1389_status": red_sol_r1389.status,
        "reduced_oxidation_status": red_sol_ox.status,
    }
    pd.DataFrame([summary]).to_csv(zone3_out / "zone3_test_summary.csv", index=False)
    print("\nZONE 3 SUMMARY")
    for k, v in summary.items():
        print(f"{k}: {v}")

def test_manual_reduced_model():
    if not MANUAL_REDUCED.exists():
        print(f"Manual reduced model not found: {MANUAL_REDUCED}")
        return
    print("\n=== Loading manual reduced model ===")
    model = read_sbml_model(str(MANUAL_REDUCED))
    model.solver = "glpk"
    print("manual reactions:", len(model.reactions))
    print("manual metabolites:", len(model.metabolites))
    reaction_table(model, [rxn.id for rxn in model.reactions]).to_csv(OUT / "manual_reduced_reactions.csv", index=False)
    sol = model.optimize()
    print_fluxes("Manual reduced default objective", sol, [rxn.id for rxn in model.reactions if "EX" in rxn.id])


if __name__ == "__main__":
    test_hepatonet_literature_core()
    test_hepatonet_zone3_literature_core()
    test_manual_reduced_model()
