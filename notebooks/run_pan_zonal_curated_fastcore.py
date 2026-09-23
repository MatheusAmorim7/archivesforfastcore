#!/usr/bin/env python3
"""Build a validated boundary-curated pan-zonal FastCORE model from HepatoNet1.

This pipeline corrects two issues identified by FVA of the first pan-zonal
candidate:

1. COBRApy assigns default reversible bounds when the source SBML has no FBC
   bounds, so irreversible flags from the original SBML are restored.
2. Automatically generated boundary reactions are closed by default. Only the
   PhysiCell substrates and a small, documented support set remain available.

The current PhysiCell configuration is not modified by this script.
"""

import logging
import warnings
from pathlib import Path

import cobra
import libsbml
import pandas as pd
from cobra import Reaction
from cobra.flux_analysis import pfba
from cobra.io import write_sbml_model

from fastcore_utils import (
    build_reduced_model,
    fastcore_cobrapy,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
SOURCE_SBML = CONFIG / "hepatonet_original_13_08.xml"
REVIEWER_CORE = (
    ROOT
    / "notebooks"
    / "fastcore_outputs"
    / "reviewer_minimal_carbohydrate_core.csv"
)
OUT = (
    ROOT
    / "notebooks"
    / "fastcore_outputs"
    / "pan_zonal_reviewer_core"
    / "curated_fastcore"
)
OUTPUT_SBML = OUT / "hepatonet_pan_zonal_curated_fastcore_reduced.xml"
CONFIG_MODEL = CONFIG / "hepatonet_pan_zonal_curated_fastcore_reduced.xml"


# These are the only boundary reactions allowed while constructing the model.
# Bounds follow the standard COBRA convention: negative uptake, positive release.
CURATED_BOUNDARIES = {
    "EX_HC00017_s": (-1000.0, 0.0, "dynamic_physicell", "O2 uptake"),
    "EX_HC00040_s": (-1000.0, 0.0, "dynamic_physicell", "glucose uptake"),
    "EX_HC00177_s": (-1000.0, 1000.0, "dynamic_physicell", "lactate uptake/release"),
    "EX_HC00021_s": (0.0, 1000.0, "dynamic_physicell", "CO2 release"),
    "EX_HC00011_s": (-1000.0, 1000.0, "buffered_support", "water balance"),
    "EX_HC00019_s": (0.0, 1000.0, "buffered_support", "phosphate release"),
    "EX_HC02131_c": (-1000.0, 0.0, "pool_support", "glycogenin seed uptake"),
    "EX_HC02134_c": (0.0, 1000.0, "pool_support", "glycogen product release"),
}


def quiet_cobra_logging():
    logging.getLogger("cobra.io.sbml").setLevel(logging.ERROR)
    logging.getLogger("cobra.core.model").setLevel(logging.ERROR)


def restore_sbml_irreversibility(model, sbml_path):
    document = libsbml.readSBMLFromFile(str(sbml_path))
    sbml_model = document.getModel()
    if sbml_model is None:
        raise ValueError(f"Could not read SBML model from {sbml_path}")

    corrected = []
    for sbml_reaction in sbml_model.getListOfReactions():
        reaction_id = sbml_reaction.getId()
        if (
            not sbml_reaction.getReversible()
            and reaction_id in model.reactions
            and model.reactions.get_by_id(reaction_id).lower_bound < 0.0
        ):
            reaction = model.reactions.get_by_id(reaction_id)
            old_lower_bound = reaction.lower_bound
            reaction.lower_bound = 0.0
            corrected.append({
                "reaction_id": reaction_id,
                "old_lower_bound": old_lower_bound,
                "new_lower_bound": 0.0,
                "upper_bound": reaction.upper_bound,
            })
    return pd.DataFrame(corrected)


def configure_curated_boundaries(model):
    for reaction_id in CURATED_BOUNDARIES:
        if reaction_id in model.reactions:
            continue
        metabolite_id = reaction_id.removeprefix("EX_")
        if metabolite_id not in model.metabolites:
            raise KeyError(
                f"Missing metabolite {metabolite_id} for boundary {reaction_id}"
            )
        reaction = Reaction(reaction_id)
        reaction.add_metabolites({model.metabolites.get_by_id(metabolite_id): -1.0})
        model.add_reactions([reaction])

    for reaction in model.reactions:
        if reaction.id.startswith("EX_"):
            reaction.bounds = (0.0, 0.0)

    missing = []
    for reaction_id, (lower, upper, _, _) in CURATED_BOUNDARIES.items():
        if reaction_id not in model.reactions:
            missing.append(reaction_id)
            continue
        model.reactions.get_by_id(reaction_id).bounds = (lower, upper)

    if missing:
        raise KeyError(f"Missing curated boundary reactions: {missing}")

    if "r1539" in model.reactions:
        model.reactions.get_by_id("r1539").bounds = (0.0, 0.0)


def load_prepared_model():
    quiet_cobra_logging()
    model = cobra.io.read_sbml_model(str(SOURCE_SBML))
    model.solver = "glpk"
    corrections = restore_sbml_irreversibility(model, SOURCE_SBML)
    configure_curated_boundaries(model)
    return model, corrections


def add_temporary_exchange(model, reaction_id, metabolite_id, bounds):
    reaction = Reaction(reaction_id)
    reaction.add_metabolites({model.metabolites.get_by_id(metabolite_id): -1.0})
    reaction.bounds = bounds
    model.add_reactions([reaction])


def functional_test_definitions():
    return [
        {
            "test": "glucose_uptake_to_g6p",
            "bounds": {"EX_HC00040_s": (-1.0, -1.0), "r0353": (1.0, 1.0)},
            "objective": "r0353",
        },
        {
            "test": "g6p_to_glucose_release",
            "bounds": {
                "EX_HC00040_s": (1.0, 1.0),
                "r1032": (-1000.0, 1000.0),
            },
            "objective": "EX_HC00040_s",
            "temporary_exchange": (
                "TMP_EX_HC00094_c",
                "HC00094_c",
                (-1.0, -1.0),
            ),
        },
        {
            "test": "lactate_to_pyruvate",
            "bounds": {"EX_HC00177_s": (-1.0, -1.0), "r0171": (1.0, 1.0)},
            "objective": "r0171",
        },
        {
            "test": "pyruvate_to_lactate",
            "bounds": {"EX_HC00177_s": (1.0, 1.0)},
            "objective": "EX_HC00177_s",
            "temporary_exchange": (
                "TMP_EX_HC00032_c",
                "HC00032_c",
                (-1.0, -1.0),
            ),
        },
        {
            "test": "oxidative_glucose_module",
            "bounds": {
                "EX_HC00040_s": (-1.0, -1.0),
                "EX_HC00017_s": (-6.0, -6.0),
                "EX_HC00021_s": (6.0, 6.0),
            },
            "objective": "EX_HC00021_s",
        },
    ]


def run_functional_tests(model, collect_support=False):
    rows = []
    support = set()

    for test in functional_test_definitions():
        with model:
            temporary = test.get("temporary_exchange")
            if temporary:
                add_temporary_exchange(model, *temporary)

            missing = [rid for rid in test["bounds"] if rid not in model.reactions]
            if test["objective"] not in model.reactions:
                missing.append(test["objective"])
            if missing:
                rows.append({
                    "test": test["test"],
                    "status": "missing_reactions",
                    "objective": test["objective"],
                    "target_flux": None,
                    "pfba_total_flux": None,
                    "active_reactions": None,
                    "details": ";".join(sorted(set(missing))),
                })
                continue

            for reaction_id, bounds in test["bounds"].items():
                model.reactions.get_by_id(reaction_id).bounds = bounds

            model.objective = test["objective"]
            model.objective_direction = "max"
            try:
                solution = pfba(model, fraction_of_optimum=1.0)
            except Exception as exc:
                rows.append({
                    "test": test["test"],
                    "status": f"error:{type(exc).__name__}",
                    "objective": test["objective"],
                    "target_flux": None,
                    "pfba_total_flux": None,
                    "active_reactions": None,
                    "details": str(exc),
                })
                continue

            active = {
                reaction_id
                for reaction_id, flux in solution.fluxes.items()
                if abs(float(flux)) > 1e-9
                and not reaction_id.startswith("TMP_")
            }
            if collect_support:
                support.update(active)
            rows.append({
                "test": test["test"],
                "status": solution.status,
                "objective": test["objective"],
                "target_flux": float(solution.fluxes[test["objective"]]),
                "pfba_total_flux": float(solution.objective_value),
                "active_reactions": len(active),
                "details": "",
            })

    return pd.DataFrame(rows), support


def load_reviewer_core(model):
    frame = pd.read_csv(REVIEWER_CORE)
    present = []
    missing = []
    for reaction_id in frame["reaction_id"].dropna().astype(str):
        if reaction_id in model.reactions:
            present.append(reaction_id)
        else:
            missing.append(reaction_id)
    return set(present), sorted(set(missing))


def validate_runtime_objective(model):
    with model:
        if "r1032" not in model.reactions:
            return "missing", None
        model.reactions.get_by_id("r1032").bounds = (0.0, 1.0)
        model.objective = "r1032"
        model.objective_direction = "max"
        solution = model.optimize()
        return solution.status, (
            float(solution.objective_value)
            if solution.status == "optimal"
            else None
        )


def quick_validation_passes(model):
    """Fast feasibility check used while pruning redundant reactions."""
    reaction_ids = {reaction.id for reaction in model.reactions}
    metabolite_ids = {metabolite.id for metabolite in model.metabolites}

    for test in functional_test_definitions():
        with model:
            temporary = test.get("temporary_exchange")
            if temporary:
                _, metabolite_id, _ = temporary
                if metabolite_id not in metabolite_ids:
                    return False
                add_temporary_exchange(model, *temporary)

            required = set(test["bounds"]) | {test["objective"]}
            if not required.issubset(reaction_ids):
                return False
            for reaction_id, bounds in test["bounds"].items():
                model.reactions.get_by_id(reaction_id).bounds = bounds
            model.objective = test["objective"]
            model.objective_direction = "max"
            solution = model.optimize()
            if solution.status != "optimal":
                return False

    runtime_status, runtime_value = validate_runtime_objective(model)
    return (
        runtime_status == "optimal"
        and runtime_value is not None
        and runtime_value > 1e-9
    )


def prune_redundant_reactions(model, protected_reactions, expanded_core):
    """Greedily remove non-protected reactions while preserving all tests."""
    pruned = model.copy()
    candidates = sorted(
        (
            reaction.id
            for reaction in pruned.reactions
            if reaction.id not in protected_reactions
        ),
        # Try FastCORE-only support first, then pFBA support reactions.
        key=lambda reaction_id: (reaction_id in expanded_core, reaction_id),
    )
    removed = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Solver status is 'infeasible'")
        for reaction_id in candidates:
            if reaction_id not in pruned.reactions:
                continue
            trial = pruned.copy()
            trial.remove_reactions([reaction_id], remove_orphans=True)
            if quick_validation_passes(trial):
                pruned = trial
                removed.append(reaction_id)

    return pruned, pd.DataFrame({"reaction_id": removed})


def boundary_audit(model):
    rows = []
    for reaction in model.reactions:
        if not reaction.id.startswith("EX_"):
            continue
        metabolite = next(iter(reaction.metabolites), None)
        curated = CURATED_BOUNDARIES.get(reaction.id)
        rows.append({
            "reaction_id": reaction.id,
            "metabolite_id": metabolite.id if metabolite else "",
            "metabolite_name": metabolite.name if metabolite else "",
            "lower_bound": reaction.lower_bound,
            "upper_bound": reaction.upper_bound,
            "category": curated[2] if curated else "unexpected",
            "rationale": curated[3] if curated else "not retained by curated core",
        })
    return pd.DataFrame(rows).sort_values("reaction_id")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model, corrections = load_prepared_model()
    reviewer_core, missing_core = load_reviewer_core(model)

    prepared_validation, functional_support = run_functional_tests(
        model, collect_support=True
    )
    if not (prepared_validation["status"] == "optimal").all():
        raise RuntimeError(
            "The curated full model did not pass all functional tests.\n"
            + prepared_validation.to_string(index=False)
        )

    expanded_core = reviewer_core | functional_support
    fastcore_keep, fastcore_ignored, iterations = fastcore_cobrapy(
        model, sorted(expanded_core)
    )

    # Preserve the validated support paths even if individual reactions are not
    # independently activatable in the FastCORE LP7 step.
    final_keep = set(fastcore_keep) | expanded_core
    reduced = build_reduced_model(model, final_keep)
    reduced.solver = "glpk"
    configure_curated_boundaries(reduced)
    preprune_reactions = len(reduced.reactions)
    preprune_metabolites = len(reduced.metabolites)

    protected_reactions = reviewer_core | set(CURATED_BOUNDARIES)
    reduced, removed_reactions = prune_redundant_reactions(
        reduced,
        protected_reactions,
        expanded_core,
    )
    configure_curated_boundaries(reduced)

    reduced_validation, _ = run_functional_tests(reduced)
    runtime_status, runtime_objective = validate_runtime_objective(reduced)
    all_tests_optimal = bool((reduced_validation["status"] == "optimal").all())
    if not all_tests_optimal or runtime_status != "optimal":
        raise RuntimeError(
            "The reduced model failed validation.\n"
            + reduced_validation.to_string(index=False)
            + f"\nRuntime objective: {runtime_status} {runtime_objective}"
        )

    reduced.reactions.get_by_id("r1032").bounds = (0.0, 1.0)
    reduced.objective = "r1032"
    reduced.objective_direction = "max"

    write_sbml_model(reduced, str(OUTPUT_SBML))
    write_sbml_model(reduced, str(CONFIG_MODEL))

    corrections.to_csv(OUT / "restored_irreversible_reactions.csv", index=False)
    prepared_validation.to_csv(OUT / "prepared_model_validation.csv", index=False)
    reduced_validation.to_csv(OUT / "curated_reduced_validation.csv", index=False)
    iterations.to_csv(OUT / "fastcore_iterations.csv", index=False)
    removed_reactions.to_csv(OUT / "greedy_pruned_reactions.csv", index=False)
    boundary_audit(reduced).to_csv(OUT / "curated_boundary_audit.csv", index=False)
    pd.DataFrame({"reaction_id": sorted(missing_core)}).to_csv(
        OUT / "missing_reviewer_core.csv", index=False
    )

    core_rows = []
    for reaction_id in sorted(expanded_core):
        sources = []
        if reaction_id in reviewer_core:
            sources.append("reviewer_core")
        if reaction_id in functional_support:
            sources.append("curated_pfba_support")
        core_rows.append({
            "reaction_id": reaction_id,
            "source": "+".join(sources),
        })
    pd.DataFrame(core_rows).to_csv(OUT / "curated_expanded_core.csv", index=False)
    pd.DataFrame({
        "reaction_id": sorted(reaction.id for reaction in reduced.reactions)
    }).to_csv(
        OUT / "curated_final_reactions.csv", index=False
    )
    pd.DataFrame({"reaction_id": sorted(fastcore_ignored)}).to_csv(
        OUT / "fastcore_ignored_core.csv", index=False
    )

    summary = pd.DataFrame([{
        "prepared_reactions": len(model.reactions),
        "prepared_metabolites": len(model.metabolites),
        "restored_irreversible_reactions": len(corrections),
        "reviewer_core_reactions": len(reviewer_core),
        "curated_pfba_support_reactions": len(functional_support),
        "expanded_core_reactions": len(expanded_core),
        "fastcore_raw_reactions": len(fastcore_keep),
        "fastcore_ignored_core": len(fastcore_ignored),
        "preprune_reactions": preprune_reactions,
        "preprune_metabolites": preprune_metabolites,
        "greedy_pruned_reactions": len(removed_reactions),
        "final_reactions": len(reduced.reactions),
        "final_metabolites": len(reduced.metabolites),
        "final_boundary_reactions": len(boundary_audit(reduced)),
        "functional_validation_all_optimal": all_tests_optimal,
        "runtime_objective_status": runtime_status,
        "runtime_objective_value": runtime_objective,
        "output_sbml": str(OUTPUT_SBML),
        "config_model": str(CONFIG_MODEL),
    }])
    summary.to_csv(OUT / "curated_fastcore_summary.csv", index=False)

    print("=== Curated FastCORE summary ===")
    print(summary.to_string(index=False))
    print("\n=== Reduced-model validation ===")
    print(reduced_validation.to_string(index=False))
    print("\n=== Retained boundaries ===")
    print(boundary_audit(reduced).to_string(index=False))
    print(f"\nValidated SBML: {CONFIG_MODEL}")


if __name__ == "__main__":
    main()
