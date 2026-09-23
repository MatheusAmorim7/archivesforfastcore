"""Small COBRApy implementation helpers used by the curated FastCORE workflow."""

import pandas as pd
from cobra.flux_analysis import pfba


EPSILON = 1e-6
SUPPORT_EPSILON = 1e-9
MAX_ITERATIONS = 200


def add_flux_direction_constraint(model, reaction, direction, epsilon, tag):
    interface = model.solver.interface
    if direction >= 0:
        constraint = interface.Constraint(
            reaction.flux_expression - epsilon,
            lb=0,
            name=f"fc_active_pos_{tag}_{reaction.id}",
        )
    else:
        constraint = interface.Constraint(
            -reaction.flux_expression - epsilon,
            lb=0,
            name=f"fc_active_neg_{tag}_{reaction.id}",
        )
    model.add_cons_vars([constraint])
    return constraint


def lp7_find_active(model, target_ids, epsilon):
    with model:
        model.solver = "glpk"
        interface = model.solver.interface
        indicator_variables = []
        indicator_metadata = []
        constraints = []

        for reaction_id in sorted(target_ids):
            if reaction_id not in model.reactions:
                continue
            reaction = model.reactions.get_by_id(reaction_id)
            if reaction.upper_bound >= epsilon:
                indicator = interface.Variable(
                    f"fc_z_pos_{reaction_id}", lb=0, ub=1
                )
                constraints.append(
                    interface.Constraint(
                        reaction.flux_expression - epsilon * indicator,
                        lb=0,
                        name=f"fc_lp7_pos_{reaction_id}",
                    )
                )
                indicator_variables.append(indicator)
                indicator_metadata.append((reaction_id, 1, indicator))
            if reaction.lower_bound <= -epsilon:
                indicator = interface.Variable(
                    f"fc_z_neg_{reaction_id}", lb=0, ub=1
                )
                constraints.append(
                    interface.Constraint(
                        -reaction.flux_expression - epsilon * indicator,
                        lb=0,
                        name=f"fc_lp7_neg_{reaction_id}",
                    )
                )
                indicator_variables.append(indicator)
                indicator_metadata.append((reaction_id, -1, indicator))

        if not indicator_variables:
            return {}, "no_direction", 0.0

        model.add_cons_vars(indicator_variables + constraints)
        model.objective = interface.Objective(
            sum(indicator_variables), direction="max"
        )
        status = model.solver.optimize()
        if status != "optimal":
            return {}, status, 0.0

        active = {}
        for reaction_id, direction, indicator in indicator_metadata:
            if float(indicator.primal or 0.0) >= 0.5 and reaction_id not in active:
                active[reaction_id] = direction
        return active, status, float(model.objective.value or 0.0)


def lp10_sparse_support(model, active_directions, epsilon):
    with model:
        model.solver = "glpk"
        for index, (reaction_id, direction) in enumerate(
            sorted(active_directions.items())
        ):
            if reaction_id in model.reactions:
                add_flux_direction_constraint(
                    model,
                    model.reactions.get_by_id(reaction_id),
                    direction,
                    epsilon,
                    str(index),
                )
        try:
            solution = pfba(model)
        except Exception:
            solution = model.optimize()
        if solution.status != "optimal":
            return set(active_directions), solution.status, None

        support = {
            reaction_id
            for reaction_id, value in solution.fluxes.items()
            if abs(float(value)) > SUPPORT_EPSILON
        }
        support.update(active_directions)
        return support, solution.status, float(solution.objective_value or 0.0)


def fastcore_cobrapy(
    model,
    core_ids,
    epsilon=EPSILON,
    max_iterations=MAX_ITERATIONS,
):
    all_ids = {reaction.id for reaction in model.reactions}
    core = {reaction_id for reaction_id in core_ids if reaction_id in all_ids}
    reconstruction = set()
    ignored = set()
    logs = []
    singleton = False
    iteration = 0

    while core - reconstruction - ignored and iteration < max_iterations:
        iteration += 1
        remaining = sorted(core - reconstruction - ignored)
        targets = remaining[:1] if singleton else remaining
        active, lp7_status, lp7_value = lp7_find_active(
            model, targets, epsilon
        )

        if not active:
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

        support, lp10_status, lp10_value = lp10_sparse_support(
            model, active, epsilon
        )
        before = len(reconstruction)
        reconstruction.update(support)
        support_added = len(reconstruction) - before
        logs.append({
            "iteration": iteration,
            "mode": "singleton" if singleton else "batch",
            "remaining_core": len(remaining),
            "lp7_status": lp7_status,
            "lp7_value": lp7_value,
            "lp10_status": lp10_status,
            "lp10_value": lp10_value,
            "active_targets": len(active),
            "support_added": support_added,
            "reconstruction_size": len(reconstruction),
            "core_covered": len(core & reconstruction),
            "ignored_core": len(ignored),
        })

        if support_added == 0:
            if singleton:
                ignored.add(remaining[0])
            else:
                singleton = True
        else:
            singleton = False

    return reconstruction, ignored, pd.DataFrame(logs)


def build_reduced_model(model, keep_ids):
    reduced = model.copy()
    remove = [
        reaction for reaction in reduced.reactions if reaction.id not in keep_ids
    ]
    reduced.remove_reactions(remove, remove_orphans=True)
    return reduced
