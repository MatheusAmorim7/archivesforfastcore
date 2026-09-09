from pathlib import Path
import csv
import xml.etree.ElementTree as ET

import pandas as pd
from optlang.symbolics import Zero
from cobra.io import read_sbml_model, write_sbml_model
from cobra.flux_analysis import pfba

from run_literature_core_tests import (
    HEPATONET,
    HEPATONET_ZONE3,
    OUT,
    prepare_zone12_model,
    prepare_zone3_model,
    safe_set_bounds,
)

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ROOT / "config" / "PhysiCell_settings.xml"
FASTCORE_OUT = OUT / "official_fastcore"
FASTCORE_OUT.mkdir(parents=True, exist_ok=True)

EPSILON = 1e-6
SUPPORT_EPSILON = 1e-9
MAX_ITERATIONS = 200

EXCHANGE_FLUX_IDS = {
    "glucose": "EX_HC00040_s",
    "oxygen": "EX_HC00017_s",
    "CO2": "EX_HC00021_s",
    "lactate": "EX_HC00177_s",
    "H2O": "EX_HC00011_s",
}


def load_core_ids(path, model):
    df = pd.read_csv(path)
    col = "reaction_id" if "reaction_id" in df.columns else df.columns[0]
    ids = [str(x) for x in df[col].dropna().tolist()]
    existing = [rid for rid in ids if rid in model.reactions]
    missing = sorted(set(ids) - set(existing))
    return existing, missing


def get_flux(rxn):
    return float(rxn.forward_variable.primal or 0.0) - float(rxn.reverse_variable.primal or 0.0)


def add_flux_direction_constraint(model, rxn, direction, epsilon, tag):
    interface = model.solver.interface
    if direction >= 0:
        cons = interface.Constraint(rxn.flux_expression, lb=epsilon, name=f"fc_active_pos_{tag}_{rxn.id}")
    else:
        cons = interface.Constraint(rxn.flux_expression, ub=-epsilon, name=f"fc_active_neg_{tag}_{rxn.id}")
    model.add_cons_vars([cons])
    return cons


def lp7_find_active(model, target_ids, epsilon):
    with model:
        model.solver = "glpk"
        interface = model.solver.interface
        z_vars = []
        z_meta = []
        cons = []
        for rid in sorted(target_ids):
            if rid not in model.reactions:
                continue
            rxn = model.reactions.get_by_id(rid)
            if rxn.upper_bound >= epsilon:
                z = interface.Variable(f"fc_z_pos_{rid}", lb=0, ub=1)
                cons.append(interface.Constraint(rxn.flux_expression - epsilon * z, lb=0, name=f"fc_lp7_pos_{rid}"))
                z_vars.append(z)
                z_meta.append((rid, 1, z))
            if rxn.lower_bound <= -epsilon:
                z = interface.Variable(f"fc_z_neg_{rid}", lb=0, ub=1)
                cons.append(interface.Constraint(-rxn.flux_expression - epsilon * z, lb=0, name=f"fc_lp7_neg_{rid}"))
                z_vars.append(z)
                z_meta.append((rid, -1, z))
        if not z_vars:
            return {}, "no_direction", 0.0
        model.add_cons_vars(z_vars + cons)
        model.objective = interface.Objective(sum(z_vars), direction="max")
        status = model.solver.optimize()
        if status != "optimal":
            return {}, status, 0.0
        active = {}
        for rid, direction, z in z_meta:
            z_value = float(z.primal or 0.0)
            if z_value >= 0.5 and rid not in active:
                active[rid] = direction
        return active, status, float(model.objective.value or 0.0)


def lp10_sparse_support(model, active_dirs, penalty_ids, epsilon):
    with model:
        model.solver = "glpk"
        for idx, (rid, direction) in enumerate(sorted(active_dirs.items())):
            if rid in model.reactions:
                add_flux_direction_constraint(model, model.reactions.get_by_id(rid), direction, epsilon, str(idx))
        try:
            sol = pfba(model)
        except Exception:
            sol = model.optimize()
        if sol.status != "optimal":
            return set(active_dirs), {}, sol.status, None
        fluxes = {rid: float(value) for rid, value in sol.fluxes.items()}
        support = {rid for rid, value in fluxes.items() if abs(value) > SUPPORT_EPSILON}
        support |= set(active_dirs)
        return support, fluxes, sol.status, float(sol.objective_value or 0.0)

def fastcore_cobrapy(model, core_ids, epsilon=EPSILON, max_iterations=MAX_ITERATIONS):
    all_ids = {rxn.id for rxn in model.reactions}
    core = set(rid for rid in core_ids if rid in all_ids)
    penalty = all_ids - core
    reconstruction = set()
    ignored = set()
    logs = []
    singleton = False
    iteration = 0

    while core - reconstruction - ignored and iteration < max_iterations:
        iteration += 1
        remaining = sorted(core - reconstruction - ignored)
        targets = remaining[:1] if singleton else remaining
        active_dirs, lp7_status, lp7_value = lp7_find_active(model, targets, epsilon)

        if not active_dirs:
            if not singleton:
                singleton = True
                logs.append({
                    "iteration": iteration,
                    "mode": "switch_to_singleton",
                    "remaining_core": len(remaining),
                    "lp7_status": lp7_status,
                    "lp7_value": lp7_value,
                    "active_targets": 0,
                    "support_added": 0,
                    "reconstruction_size": len(reconstruction),
                })
                continue
            ignored.add(remaining[0])
            logs.append({
                "iteration": iteration,
                "mode": "ignore_inconsistent_core",
                "remaining_core": len(remaining),
                "lp7_status": lp7_status,
                "lp7_value": lp7_value,
                "active_targets": 0,
                "support_added": 0,
                "reconstruction_size": len(reconstruction),
                "ignored_reaction": remaining[0],
            })
            continue

        support, fluxes, lp10_status, lp10_value = lp10_sparse_support(model, active_dirs, penalty - reconstruction, epsilon)
        before = len(reconstruction)
        reconstruction |= support
        support_added = len(reconstruction) - before
        covered = len(core & reconstruction)
        logs.append({
            "iteration": iteration,
            "mode": "singleton" if singleton else "batch",
            "remaining_core": len(remaining),
            "lp7_status": lp7_status,
            "lp7_value": lp7_value,
            "lp10_status": lp10_status,
            "lp10_value": lp10_value,
            "active_targets": len(active_dirs),
            "support_added": support_added,
            "reconstruction_size": len(reconstruction),
            "core_covered": covered,
            "ignored_core": len(ignored),
        })

        if support_added == 0:
            if singleton:
                ignored.add(remaining[0])
            else:
                singleton = True
        elif support_added > 0:
            singleton = False

    return reconstruction, ignored, pd.DataFrame(logs)


def build_reduced_model(model, keep_ids):
    reduced = model.copy()
    remove = [rxn for rxn in reduced.reactions if rxn.id not in keep_ids]
    reduced.remove_reactions(remove, remove_orphans=True)
    return reduced


def safe_flux(sol, rid):
    if sol is None or sol.status != "optimal" or rid not in sol.fluxes.index:
        return None
    return float(sol.fluxes[rid])


def run_with_bounds(model, objective, direction, bounds):
    with model:
        model.solver = "glpk"
        for rid, bd in bounds.items():
            if rid in model.reactions:
                model.reactions.get_by_id(rid).bounds = bd
        if objective not in model.reactions:
            return None
        if objective in model.reactions:
            model.objective = objective
            model.objective.direction = direction
        try:
            sol = model.optimize()
        except Exception:
            sol = None
    return sol


def get_physicell_exchange_ids(settings_path):
    tree = ET.parse(settings_path)
    root = tree.getroot()
    ids = []
    for ex in root.findall(".//transport_model/exchange"):
        substrate = ex.attrib.get("substrate", "")
        flux_node = ex.find("fba_flux")
        if flux_node is not None and flux_node.text:
            ids.append((substrate, flux_node.text.strip()))
    return sorted(set(ids))


def get_sbml_reaction_ids(path):
    tree = ET.parse(path)
    root = tree.getroot()
    ids = set()
    for elem in root.iter():
        if elem.tag.endswith("reaction") and "id" in elem.attrib:
            ids.add(elem.attrib["id"])
    return ids


def validate_reduced(label, path, objective_id, objective_direction):
    model = read_sbml_model(str(path))
    model.solver = "glpk"
    rows = []

    tests = [
        (f"{objective_id} objective", objective_id, objective_direction, {}),
        ("Glc + 6O2 -> 6CO2 + 6H2O", "EX_HC00021_s", "max", {
            "EX_HC00040_s": (-1, -1),
            "EX_HC00017_s": (-6, -6),
            "EX_HC00021_s": (6, 6),
            "EX_HC00011_s": (-1000, 1000),
            "r1032": (-1000, 1000),
            "r1389": (0, 1000),
            "r0801": (-1000, 1000),
            "r0054": (-1000, 1000),
            "EX_HC02131_c": (-1000, 1000),
            "EX_HC02134_c": (-1000, 1000),
        }),
        ("Glc -> 2 lactate", "EX_HC00177_s", "max", {
            "EX_HC00040_s": (-1, -1),
            "EX_HC00177_s": (0, 2),
            "EX_HC00017_s": (0, 0),
            "EX_HC00021_s": (0, 0),
            "r1032": (-1000, 1000),
            "r1389": (0, 1000),
            "r0801": (-1000, 1000),
            "r0054": (-1000, 1000),
            "EX_HC02131_c": (-1000, 1000),
            "EX_HC02134_c": (-1000, 1000),
        }),
    ]

    for test_name, objective, direction, bounds in tests:
        sol = run_with_bounds(model, objective, direction, bounds)
        rows.append({
            "network": label,
            "test": test_name,
            "status": sol.status if sol is not None else "error",
            "objective_value": sol.objective_value if sol is not None and sol.status == "optimal" else None,
            "r1032": safe_flux(sol, "r1032"),
            "r1389": safe_flux(sol, "r1389"),
            "glucose": safe_flux(sol, "EX_HC00040_s"),
            "oxygen": safe_flux(sol, "EX_HC00017_s"),
            "CO2": safe_flux(sol, "EX_HC00021_s"),
            "lactate": safe_flux(sol, "EX_HC00177_s"),
            "H2O": safe_flux(sol, "EX_HC00011_s"),
            "exchange_id": "",
        })

    reaction_ids = {rxn.id for rxn in model.reactions}
    sbml_ids = get_sbml_reaction_ids(path)
    for substrate, rid in get_physicell_exchange_ids(SETTINGS):
        if substrate not in {"oxygen", "glucose", "CO2", "lactate"}:
            continue
        cobra_id = rid[2:] if rid.startswith("R_") else rid
        present = rid in sbml_ids or cobra_id in reaction_ids
        rows.append({
            "network": label,
            "test": f"PhysiCell exchange ID: {substrate}",
            "status": "present" if present else "missing",
            "objective_value": "",
            "r1032": "",
            "r1389": "",
            "glucose": "",
            "oxygen": "",
            "CO2": "",
            "lactate": "",
            "H2O": "",
            "exchange_id": rid,
        })
    return rows


def run_case(label, sbml_path, core_csv, prepare_fn, objective_id, objective_direction, out_dir):
    print(f"\n=== {label} ===")
    model = read_sbml_model(str(sbml_path))
    model.solver = "glpk"
    model = prepare_fn(model, sbml_path)
    core_ids, missing = load_core_ids(core_csv, model)
    print("prepared reactions:", len(model.reactions))
    print("prepared metabolites:", len(model.metabolites))
    print("input core:", len(core_ids), "missing:", len(missing))

    keep, ignored, log = fastcore_cobrapy(model, core_ids)
    reduced = build_reduced_model(model, keep)
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = out_dir / f"{label}_official_fastcore_literature_reduced.xml"
    write_sbml_model(reduced, str(xml_path))

    pd.DataFrame({"reaction_id": sorted(keep)}).to_csv(out_dir / f"{label}_official_fastcore_reactions.csv", index=False)
    pd.DataFrame({"reaction_id": sorted(ignored)}).to_csv(out_dir / f"{label}_official_fastcore_ignored_core.csv", index=False)
    pd.DataFrame({"reaction_id": sorted(missing)}).to_csv(out_dir / f"{label}_official_fastcore_missing_core.csv", index=False)
    log.to_csv(out_dir / f"{label}_official_fastcore_iterations.csv", index=False)

    summary = {
        "network": label,
        "prepared_reactions": len(model.reactions),
        "prepared_metabolites": len(model.metabolites),
        "input_core": len(core_ids),
        "missing_core_ids": len(missing),
        "fastcore_reactions": len(reduced.reactions),
        "fastcore_metabolites": len(reduced.metabolites),
        "core_kept": len(set(core_ids) & keep),
        "core_ignored": len(ignored),
        "xml_path": str(xml_path),
    }
    print(summary)
    return xml_path, summary


def main():
    zone12_xml, zone12_summary = run_case(
        "zone12",
        HEPATONET,
        OUT / "zone12_literature_guided_core.csv",
        prepare_zone12_model,
        "r1032",
        "min",
        FASTCORE_OUT / "zone12",
    )
    zone3_xml, zone3_summary = run_case(
        "zone3",
        HEPATONET_ZONE3,
        OUT / "zone3" / "zone3_literature_guided_core.csv",
        prepare_zone3_model,
        "r1389",
        "max",
        FASTCORE_OUT / "zone3",
    )

    validation_rows = []
    validation_rows.extend(validate_reduced("zone12_official_fastcore_literature", zone12_xml, "r1032", "min"))
    validation_rows.extend(validate_reduced("zone3_official_fastcore_literature", zone3_xml, "r1389", "max"))

    pd.DataFrame([zone12_summary, zone3_summary]).to_csv(FASTCORE_OUT / "official_fastcore_summary.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(FASTCORE_OUT / "official_fastcore_validation_report.csv", index=False)

    print("\nValidation report:")
    for row in validation_rows:
        print(row)


if __name__ == "__main__":
    main()
