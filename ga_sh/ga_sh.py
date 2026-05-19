# ga_sh.py
# ============================================================
# Minimal GA orchestration module for the self-healing project
# Current version:
# - context
# - persistent cache in disk
# - result structure
# - candidate evaluation
# - timing by stage
# - mono-objective DEAP GA
# ============================================================

from __future__ import annotations

import copy
import pickle
import random
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from deap import base, creator, tools

from binary_encoder import (
    encode,
    decode,
    sanitize_code,
    random_valid_code,
    RANGES,
)
from ml_filter import evaluate_ml_filter
from surface_mc import get_surface_composition_wt
from go_nogo import evaluate_go_nogo
from fitness import evaluate_fitness

from tqdm import tqdm

# ============================================================
# Persistent cache utilities
# ============================================================

def load_persistent_cache(cache_path: str | None) -> dict:
    """
    Load persistent cache from disk.
    Returns an empty dict if the file does not exist.
    """
    if cache_path is None:
        return {}

    path = Path(cache_path)
    if not path.exists():
        return {}

    with open(path, "rb") as f:
        cache = pickle.load(f)

    if not isinstance(cache, dict):
        raise ValueError(f"Persistent cache must be a dict. Got: {type(cache)}")

    return cache


def save_persistent_cache(cache: dict, cache_path: str | None) -> None:
    """
    Save persistent cache to disk using an atomic replace.
    """
    if cache_path is None:
        return

    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    tmp_path.replace(path)


# ============================================================
# Context / result templates
# ============================================================

def build_context(
    mode: str,
    population_size: int,
    n_generations: int,
    crossover_prob: float,
    mutation_prob: float,
    elite_size: int,
    random_seed: int,
    penalty_fitness: float,
    t_eval_h: float,
    ml_bulk_model,
    ml_surf_model,
    ml_thresholds: dict,
    ml_domain_bounds: dict,
    surface_mc_kwargs: Optional[dict] = None,
    gonogo_kwargs: Optional[dict] = None,
    fitness_kwargs: Optional[dict] = None,
    cache_path: str | None = None,
    cache_autosave_every: int = 50,
) -> Dict[str, Any]:
    """
    Build the execution context for the GA.
    """
    if mode not in {"bulk_dominant", "surface_enhanced"}:
        raise ValueError(
            f"Invalid mode: {mode!r}. Expected 'bulk_dominant' or 'surface_enhanced'."
        )

    loaded_cache = load_persistent_cache(cache_path)

    return {
        "ga_config": {
            "population_size": int(population_size),
            "n_generations": int(n_generations),
            "crossover_prob": float(crossover_prob),
            "mutation_prob": float(mutation_prob),
            "elite_size": int(elite_size),
            "random_seed": int(random_seed),
        },
        "study_config": {
            "mode": mode,
            "penalty_fitness": float(penalty_fitness),
            "t_eval_h": float(t_eval_h),
        },
        "models": {
            "ml_bulk_model": ml_bulk_model,
            "ml_surf_model": ml_surf_model,
        },
        "module_config": {
            "ml_thresholds": dict(ml_thresholds),
            "ml_domain_bounds": dict(ml_domain_bounds),
            "surface_mc_kwargs": dict(surface_mc_kwargs or {}),
            "gonogo_kwargs": dict(gonogo_kwargs or {}),
            "fitness_kwargs": dict(fitness_kwargs or {}),
        },
        "runtime": {
            "current_generation": 0,
            "evaluation_counter": 0,
            "candidate_counter": 0,
        },
        "cache_config": {
            "cache_path": cache_path,
            "cache_autosave_every": int(cache_autosave_every),
        },
        "storage": {
            "cache": loaded_cache,        # code -> result_dict
            "history_candidates": [],     # list[result_dict]
            "history_generations": [],    # list[generation_summary_dict]
        },
    }


def make_empty_result_dict() -> Dict[str, Any]:
    """
    Predefine all result fields so every evaluated individual
    has the same structure, even if rejected early.
    """
    return {
        "candidate_id": None,
        "generation": None,
        "mode": None,
        "code": None,
        "from_cache": None,

        "decoded_values": None,
        "composition_bulk_wt": None,
        "T_sol": None,
        "T_aging": None,

        "DF_bulk_ml": None,
        "DF_surf_ml": None,
        "ml_status": None,
        "ml_reason": None,
        "ml_in_domain": None,
        "ml_out_of_domain_features": None,

        "surface_valid": None,
        "surface_reason": None,
        "surface_mean_wt_substitutional": None,
        "surface_mean_wt_solutes": None,
        "surface_gamma_Jm2": None,
        "surface_acceptance_ratio": None,

        "gonogo_valid": None,
        "gonogo_reason": None,

        "Fe_balance_wt": None,

        "LIQ_Tsol": None,
        "BCC_A2_Tsol": None,
        "FCC_A1_Tsol": None,
        "matrix_total_Tsol": None,
        "Laves_Tsol": None,
        "M23C6_Tsol": None,
        "MX_Tsol": None,

        "BCC_A2_Taging": None,
        "FCC_A1_Taging": None,
        "Laves_Taging": None,
        "M23C6_Taging": None,
        "MX_Taging": None,

        "DF_bulk_tc": None,
        "DF_surf_tc": None,
        "delta_DF_surf_minus_bulk": None,
        "f_laves_bulk": None,
        "tc_error": None,

        "fitness": None,
        "fitness_valid": None,
        "fitness_reason": None,

        "final_status": None,

        # Timing fields
        "time_ml_s": None,
        "time_surface_mc_s": None,
        "time_gonogo_s": None,
        "time_fitness_s": None,
        "time_total_s": None,

        "T_aging_C": None,
        "coarsening_x_matrix": None,
        "coarsening_x_laves": None,
        "coarsening_diffusivities": None,

        "K_m3_s": None,
        "rt_m": None,
        "rt_nm": None,
        "DGm_J_mol": None,
        "DGv_J_m3": None,
        "r0_m": None,
        "r0_nm": None,
        "r0_cubed_m3": None,
        "Kt_m3": None,
        "Kt_over_r0_cubed": None,
        "coarsening_sum": None,
        "coarsening_elements_used": None,
        "coarsening_elements_skipped": None,
        "coarsening_terms": None
    }


# ============================================================
# Candidate evaluation
# ============================================================

def _autosave_cache_if_needed(ctx: Dict[str, Any]) -> None:
    """
    Autosave persistent cache if the configured interval is reached.
    """
    autosave_every = ctx["cache_config"]["cache_autosave_every"]
    cache = ctx["storage"]["cache"]

    if autosave_every > 0 and len(cache) > 0 and len(cache) % autosave_every == 0:
        save_persistent_cache(cache, ctx["cache_config"]["cache_path"])


def evaluate_candidate(ind, ctx: Dict[str, Any]):
    """
    Evaluate one candidate through the current pipeline:

    decode
    -> ml_filter
    -> surface_mc
    -> go_nogo (placeholder)
    -> fitness (temporary)

    Parameters
    ----------
    ind
        Individual representation. Can be an int or a DEAP individual
        convertible to int.
    ctx : dict
        Context returned by build_context().

    Returns
    -------
    fitness_value : float
    result_dict : dict
    """
    t_total_start = time.perf_counter()

    code = int(ind)
    cache = ctx["storage"]["cache"]

    # --------------------------------------------------------
    # Cache
    # --------------------------------------------------------
    if code in cache:
        cached = copy.deepcopy(cache[code])
        cached["from_cache"] = True
        cached["generation"] = ctx["runtime"]["current_generation"]

        ctx["runtime"]["candidate_counter"] += 1
        cached["candidate_id"] = ctx["runtime"]["candidate_counter"]

        ctx["storage"]["history_candidates"].append(cached)
        return cached["fitness"], cached

    # --------------------------------------------------------
    # Init result structure
    # --------------------------------------------------------
    ctx["runtime"]["evaluation_counter"] += 1
    ctx["runtime"]["candidate_counter"] += 1

    result = make_empty_result_dict()
    result["candidate_id"] = ctx["runtime"]["candidate_counter"]
    result["generation"] = ctx["runtime"]["current_generation"]
    result["mode"] = ctx["study_config"]["mode"]
    result["code"] = code
    result["from_cache"] = False

    penalty = ctx["study_config"]["penalty_fitness"]

    # --------------------------------------------------------
    # 1) Decode individual
    # --------------------------------------------------------
    decoded = decode(code)
    result["decoded_values"] = decoded

    composition_bulk_wt = {
        "Cr": decoded["Cr"],
        "Mn": decoded["Mn"],
        "Si": decoded["Si"],
        "Mo": decoded["Mo"],
        "Nb": decoded["Nb"],
        "Ti": decoded["Ti"],
        "W": decoded["W"],
        "Ni": decoded["Ni"],
        "Cu": decoded["Cu"],
    }

    result["composition_bulk_wt"] = composition_bulk_wt
    result["T_sol"] = decoded["T_sol"]
    result["T_aging"] = decoded["T_aging"]

    # --------------------------------------------------------
    # 2) ML filter
    # --------------------------------------------------------
    t_ml_start = time.perf_counter()
    ml_result = evaluate_ml_filter(
        composition_bulk=composition_bulk_wt,
        mode=ctx["study_config"]["mode"],
        thresholds=ctx["module_config"]["ml_thresholds"],
        bulk_model=ctx["models"]["ml_bulk_model"],
        surf_model=ctx["models"]["ml_surf_model"],
        domain_bounds=ctx["module_config"]["ml_domain_bounds"],
    )
    result["time_ml_s"] = time.perf_counter() - t_ml_start

    result["DF_bulk_ml"] = ml_result["DF_bulk_ml"]
    result["DF_surf_ml"] = ml_result["DF_surf_ml"]
    result["ml_status"] = ml_result["ml_status"]
    result["ml_reason"] = ml_result["reason"]
    result["ml_in_domain"] = ml_result["in_domain"]
    result["ml_out_of_domain_features"] = ml_result["out_of_domain_features"]

    if ml_result["ml_status"] == "reject":
        result["fitness"] = penalty
        result["fitness_valid"] = False
        result["fitness_reason"] = "rejected_by_ml_filter"

        result["gonogo_valid"] = False
        result["gonogo_reason"] = "not_evaluated_after_ml_reject"

        result["final_status"] = "rejected_ml"
        result["time_surface_mc_s"] = 0.0
        result["time_gonogo_s"] = 0.0
        result["time_fitness_s"] = 0.0
        result["time_total_s"] = time.perf_counter() - t_total_start

        cache[code] = copy.deepcopy(result)
        _autosave_cache_if_needed(ctx)
        ctx["storage"]["history_candidates"].append(result)
        return result["fitness"], result

    # --------------------------------------------------------
    # 3) Surface MC
    # --------------------------------------------------------
    t_surface_start = time.perf_counter()
    surface_result = get_surface_composition_wt(
        composition_bulk_wt=composition_bulk_wt,
        T_celsius=decoded["T_aging"],
        **ctx["module_config"]["surface_mc_kwargs"],
    )
    result["time_surface_mc_s"] = time.perf_counter() - t_surface_start

    result["surface_valid"] = surface_result["surface_valid"]
    result["surface_reason"] = surface_result["surface_reason"]

    if not surface_result["surface_valid"]:
        result["fitness"] = penalty
        result["fitness_valid"] = False
        result["fitness_reason"] = "surface_mc_failed"

        result["gonogo_valid"] = False
        result["gonogo_reason"] = "not_evaluated_after_surface_failure"

        result["final_status"] = "failed_surface_mc"
        result["time_gonogo_s"] = 0.0
        result["time_fitness_s"] = 0.0
        result["time_total_s"] = time.perf_counter() - t_total_start

        cache[code] = copy.deepcopy(result)
        _autosave_cache_if_needed(ctx)
        ctx["storage"]["history_candidates"].append(result)
        return result["fitness"], result

    result["surface_mean_wt_substitutional"] = surface_result["surface_mean_wt_substitutional"]
    result["surface_mean_wt_solutes"] = surface_result["surface_mean_wt_solutes"]
    result["surface_gamma_Jm2"] = surface_result["gamma_Jm2"]
    result["surface_acceptance_ratio"] = surface_result.get("acceptance_ratio", None)

    # --------------------------------------------------------
    # 4) Go/No-Go placeholder
    # --------------------------------------------------------
    t_gonogo_start = time.perf_counter()
    gonogo_result = evaluate_go_nogo(
        composition_bulk_wt=result["composition_bulk_wt"],
        composition_surface_wt=result["surface_mean_wt_substitutional"],
        T_sol=result["T_sol"],
        T_aging=result["T_aging"],
        mode=result["mode"],
        **ctx["module_config"]["gonogo_kwargs"],
    )
    result["time_gonogo_s"] = time.perf_counter() - t_gonogo_start

    gonogo_keys = [
        "gonogo_valid",
        "gonogo_reason",

        "Fe_balance_wt",

        "LIQ_Tsol",
        "BCC_A2_Tsol",
        "FCC_A1_Tsol",
        "matrix_total_Tsol",
        "Laves_Tsol",
        "M23C6_Tsol",
        "MX_Tsol",

        "BCC_A2_Taging",
        "FCC_A1_Taging",
        "Laves_Taging",
        "M23C6_Taging",
        "MX_Taging",

        "DF_bulk_tc",
        "DF_surf_tc",
        "delta_DF_surf_minus_bulk",
        "f_laves_bulk",
        "tc_error",

        "T_aging_C",
        "coarsening_x_matrix",
        "coarsening_x_laves",
        "coarsening_diffusivities",
    ]

    for key in gonogo_keys:
        result[key] = gonogo_result.get(key)

    if not gonogo_result["gonogo_valid"]:
        result["fitness"] = penalty
        result["fitness_valid"] = False
        result["fitness_reason"] = "failed_gonogo"
        result["final_status"] = "failed_gonogo"
        result["time_fitness_s"] = 0.0
        result["time_total_s"] = time.perf_counter() - t_total_start

        cache[code] = copy.deepcopy(result)
        _autosave_cache_if_needed(ctx)
        ctx["storage"]["history_candidates"].append(result)
        return result["fitness"], result

    # --------------------------------------------------------
    # 5) Fitness (temporary: -DF_bulk_ml)
    # --------------------------------------------------------
    t_fitness_start = time.perf_counter()
    fitness_result = evaluate_fitness(
        gonogo_result=gonogo_result,
        t_eval_h=ctx["study_config"]["t_eval_h"],
        df_bulk_ml=result["DF_bulk_ml"],
        **ctx["module_config"]["fitness_kwargs"],
    )
    result["time_fitness_s"] = time.perf_counter() - t_fitness_start

    result["fitness"] = fitness_result["fitness"]
    result["fitness_valid"] = fitness_result["fitness_valid"]
    result["fitness_reason"] = fitness_result["fitness_reason"]
    result["final_status"] = "evaluated"
    result["time_total_s"] = time.perf_counter() - t_total_start
    fitness_extra_keys = [
        "K_m3_s",
        "rt_m",
        "rt_nm",
        "r0_m",
        "r0_nm",
        "DGm_J_mol",
        "DGv_J_m3",
        "r0_cubed_m3",
        "Kt_m3",
        "Kt_over_r0_cubed",
        "coarsening_sum",
        "coarsening_terms",
        "coarsening_elements_used",
        "coarsening_elements_skipped",
    ]

    for key in fitness_extra_keys:
        result[key] = fitness_result.get(key)

    cache[code] = copy.deepcopy(result)
    _autosave_cache_if_needed(ctx)
    ctx["storage"]["history_candidates"].append(result)
    return result["fitness"], result


# ============================================================
# GA operators and execution
# ============================================================

MUTATION_RADIUS = {
    "Cr": 1,
    "Mn": 1,
    "Si": 1,
    "Mo": 1,
    "Nb": 1,
    "Ti": 1,
    "W": 1,
    "Ni": 1,
    "Cu": 1,
    "T_sol": 1,
    "T_aging": 1,
}


def make_individual():
    """
    Create one valid encoded individual.
    """
    return random_valid_code()


def initialize_population(toolbox, n: int):
    """
    Create the initial population.
    """
    return [toolbox.individual() for _ in range(n)]


def mate_individuals(ind1, ind2, swap_prob: float = 0.5):
    """
    Uniform crossover at field level:
    - decode parents
    - swap genes independently
    - re-encode and sanitize
    """
    vals1 = decode(int(ind1))
    vals2 = decode(int(ind2))

    for key in RANGES.keys():
        if random.random() < swap_prob:
            vals1[key], vals2[key] = vals2[key], vals1[key]

    child1 = sanitize_code(encode(vals1))
    child2 = sanitize_code(encode(vals2))

    return creator.Individual(child1), creator.Individual(child2)


def mutate_individual(ind, indpb: float = 0.20):
    """
    Mutate one individual by perturbing genes on the discretised grid.
    """
    vals = decode(int(ind))
    changed = False

    for key, (lo, hi, step) in RANGES.items():
        if random.random() < indpb:
            radius = MUTATION_RADIUS.get(key, 1)
            dq = random.randint(-radius, radius)

            if dq != 0:
                vals[key] = float(vals[key]) + dq * step
                changed = True

    if changed:
        new_code = sanitize_code(encode(vals))
        return (creator.Individual(new_code),)

    return (creator.Individual(sanitize_code(int(ind))),)


def evaluate_population(population, ctx):
    """
    Evaluate all invalid individuals in the population.

    Adds a progress bar for monitoring long Thermo-Calc/Surface-MC evaluations.
    Does not change GA logic.
    """

    generation = ctx.get("runtime", {}).get("current_generation", "?")

    iterator = tqdm(
        population,
        desc=f"Gen {generation}",
        unit="ind",
        dynamic_ncols=True,
        leave=True,
    )

    for ind in iterator:
        fitness_value, result = evaluate_candidate(ind, ctx)
        ind.fitness.values = (fitness_value,)
        ind.result = result

        iterator.set_postfix(
            {
                "fit": f"{fitness_value:.3g}" if fitness_value is not None else "None",
                "cache": result.get("from_cache"),
                "status": result.get("final_status"),
            }
        )


def summarize_generation(population, ctx):
    """
    Build a compact summary for the current generation.

    Robust behaviour:
    - fitness statistics are taken from ind.fitness.values when available;
    - result dictionaries are used for status counts and timing diagnostics;
    - penalised individuals are included in best/mean fitness;
    - no-valid-generation cases report penalty fitness instead of NaN when
      the population has valid DEAP fitness values.
    """

    penalty = float(ctx["study_config"]["penalty_fitness"])

    results = [getattr(ind, "result", None) for ind in population]
    results = [r for r in results if r is not None]

    # --------------------------------------------------------
    # Fitness values from DEAP individuals, fallback to result.
    # --------------------------------------------------------
    fitness_values = []

    for ind in population:
        value = None

        if getattr(ind, "fitness", None) is not None and ind.fitness.valid:
            try:
                value = float(ind.fitness.values[0])
            except Exception:
                value = None

        if value is None:
            r = getattr(ind, "result", None)
            if r is not None:
                try:
                    value = float(r.get("fitness"))
                except Exception:
                    value = None

        if value is not None and np.isfinite(value):
            fitness_values.append(value)

    # --------------------------------------------------------
    # Timing diagnostics from result dictionaries only.
    # --------------------------------------------------------
    time_ml = [
        r["time_ml_s"] for r in results
        if r.get("time_ml_s") is not None and np.isfinite(r["time_ml_s"])
    ]

    time_surface = [
        r["time_surface_mc_s"] for r in results
        if r.get("time_surface_mc_s") is not None and np.isfinite(r["time_surface_mc_s"])
    ]

    time_gonogo = [
        r["time_gonogo_s"] for r in results
        if r.get("time_gonogo_s") is not None and np.isfinite(r["time_gonogo_s"])
    ]

    time_fitness = [
        r["time_fitness_s"] for r in results
        if r.get("time_fitness_s") is not None and np.isfinite(r["time_fitness_s"])
    ]

    time_total = [
        r["time_total_s"] for r in results
        if r.get("time_total_s") is not None and np.isfinite(r["time_total_s"])
    ]

    new_results = [r for r in results if r.get("from_cache") is False]

    time_total_new = [
        r["time_total_s"] for r in new_results
        if r.get("time_total_s") is not None and np.isfinite(r["time_total_s"])
    ]

    time_surface_new = [
        r["time_surface_mc_s"] for r in new_results
        if r.get("time_surface_mc_s") is not None and np.isfinite(r["time_surface_mc_s"])
    ]

    # --------------------------------------------------------
    # Diversity diagnostics.
    # --------------------------------------------------------
    codes = [int(ind) for ind in population]
    n_unique_codes = len(set(codes))
    duplicate_fraction = 1.0 - (n_unique_codes / len(population)) if population else None

    if fitness_values:
        best_fitness = float(np.min(fitness_values))
        mean_fitness = float(np.mean(fitness_values))
        n_penalty_fitness = sum(v >= penalty for v in fitness_values)
    else:
        best_fitness = None
        mean_fitness = None
        n_penalty_fitness = None

    best_code = None
    if fitness_values:
        best_idx = int(np.argmin(fitness_values))
        best_code = int(population[best_idx])

    summary = {
        "generation": ctx["runtime"]["current_generation"],
        "n_population": len(population),
        "n_results_available": len(results),

        "n_unique_codes": n_unique_codes,
        "duplicate_fraction": duplicate_fraction,
        "best_code": best_code,

        "n_from_cache": sum(r.get("from_cache") is True for r in results),
        "n_new_evaluations": sum(r.get("from_cache") is False for r in results),

        "n_ml_accept": sum(r.get("ml_status") == "accept" for r in results),
        "n_ml_reject": sum(r.get("ml_status") == "reject" for r in results),
        "n_ml_uncertain_model": sum(r.get("ml_status") == "uncertain_model" for r in results),
        "n_ml_uncertain_domain": sum(r.get("ml_status") == "uncertain_domain" for r in results),

        "n_surface_fail": sum(r.get("final_status") == "failed_surface_mc" for r in results),
        "n_gonogo_fail": sum(r.get("final_status") == "failed_gonogo" for r in results),
        "n_final_valid": sum(r.get("final_status") == "evaluated" for r in results),
        "n_rejected_ml": sum(r.get("final_status") == "rejected_ml" for r in results),
        "n_penalty_fitness": n_penalty_fitness,

        "best_fitness": best_fitness,
        "mean_fitness": mean_fitness,

        # Timing summary for all individuals with result dictionaries
        "time_ml_mean_s": float(np.mean(time_ml)) if time_ml else None,
        "time_surface_mc_mean_s": float(np.mean(time_surface)) if time_surface else None,
        "time_gonogo_mean_s": float(np.mean(time_gonogo)) if time_gonogo else None,
        "time_fitness_mean_s": float(np.mean(time_fitness)) if time_fitness else None,
        "time_total_mean_s": float(np.mean(time_total)) if time_total else None,

        "time_ml_sum_s": float(np.sum(time_ml)) if time_ml else None,
        "time_surface_mc_sum_s": float(np.sum(time_surface)) if time_surface else None,
        "time_gonogo_sum_s": float(np.sum(time_gonogo)) if time_gonogo else None,
        "time_fitness_sum_s": float(np.sum(time_fitness)) if time_fitness else None,
        "time_total_sum_s": float(np.sum(time_total)) if time_total else None,

        # Timing summary for genuinely new evaluations only
        "time_surface_mc_mean_new_s": float(np.mean(time_surface_new)) if time_surface_new else None,
        "time_total_mean_new_s": float(np.mean(time_total_new)) if time_total_new else None,
        "time_surface_mc_sum_new_s": float(np.sum(time_surface_new)) if time_surface_new else None,
        "time_total_sum_new_s": float(np.sum(time_total_new)) if time_total_new else None,
    }

    ctx["storage"]["history_generations"].append(summary)
    return summary

def build_deap_toolbox(ctx):
    """
    Create the DEAP toolbox for the current GA configuration.
    """
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    if not hasattr(creator, "Individual"):
        creator.create("Individual", int, fitness=creator.FitnessMin, result=None)

    toolbox = base.Toolbox()

    toolbox.register("attr_code", make_individual)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_code)
    toolbox.register("population", initialize_population, toolbox)

    toolbox.register("mate", mate_individuals)
    toolbox.register("mutate", mutate_individual, indpb=0.20)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("clone", copy.deepcopy)

    return toolbox


def run_ga(ctx):
    """
    Run the mono-objective GA.

    Returns
    -------
    population : list
        Final population.
    best_individual : Individual
        Best individual found.
    history_generations : list[dict]
        Per-generation summary records.
    """
    random.seed(ctx["ga_config"]["random_seed"])
    np.random.seed(ctx["ga_config"]["random_seed"])

    toolbox = build_deap_toolbox(ctx)

    pop_size = ctx["ga_config"]["population_size"]
    n_generations = ctx["ga_config"]["n_generations"]
    crossover_prob = ctx["ga_config"]["crossover_prob"]
    mutation_prob = ctx["ga_config"]["mutation_prob"]
    elite_size = ctx["ga_config"]["elite_size"]

    # --------------------------------------------------------
    # Initial population
    # --------------------------------------------------------
    ctx["runtime"]["current_generation"] = 0
    population = toolbox.population(n=pop_size)

    evaluate_population(population, ctx)
    summary0 = summarize_generation(population, ctx)
    print(
        f"[Gen 0] "
        f"best={summary0['best_fitness']} "
        f"mean={summary0['mean_fitness']} "
        f"valid={summary0['n_final_valid']} "
        f"time_total_mean={summary0['time_total_mean_s']}"
    )

    # --------------------------------------------------------
    # Generational loop
    # --------------------------------------------------------
    for gen in range(1, n_generations + 1):
        ctx["runtime"]["current_generation"] = gen

        # Elitism
        elites = tools.selBest(population, elite_size)
        elites = [toolbox.clone(ind) for ind in elites]

        # Parent selection
        offspring = toolbox.select(population, pop_size - elite_size)
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Crossover
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < crossover_prob:
                c1, c2 = toolbox.mate(offspring[i], offspring[i + 1])
                offspring[i] = c1
                offspring[i + 1] = c2

                if hasattr(offspring[i].fitness, "values"):
                    del offspring[i].fitness.values
                if hasattr(offspring[i + 1].fitness, "values"):
                    del offspring[i + 1].fitness.values

        # Mutation
        for i in range(len(offspring)):
            if random.random() < mutation_prob:
                (mutant,) = toolbox.mutate(offspring[i])
                offspring[i] = mutant
                if hasattr(offspring[i].fitness, "values"):
                    del offspring[i].fitness.values

        # Evaluate offspring
        evaluate_population(offspring, ctx)

        # New population
        population = elites + offspring

        # Summary
        summary = summarize_generation(population, ctx)
        print(
            f"[Gen {gen}] "
            f"best={summary['best_fitness']} "
            f"mean={summary['mean_fitness']} "
            f"valid={summary['n_final_valid']} "
            f"ml_reject={summary['n_ml_reject']} "
            f"cache={summary['n_from_cache']} "
            f"new={summary['n_new_evaluations']} "
            f"time_total_mean={summary['time_total_mean_s']} "
            f"time_surface_mean={summary['time_surface_mc_mean_s']}"
        )

    # Final cache save
    save_persistent_cache(
        ctx["storage"]["cache"],
        ctx["cache_config"]["cache_path"],
    )

    best_individual = tools.selBest(population, 1)[0]
    return population, best_individual, ctx["storage"]["history_generations"]