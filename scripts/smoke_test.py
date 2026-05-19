from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable when running from scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "ga_sh"))

from config import load_config
from binary_encoder import random_valid_code, decode
from ml_filter import load_models
from ga_sh import build_context, evaluate_candidate


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "config" / "config.yaml")

    bulk_model, surf_model = load_models(
        bulk_model_path=cfg["ml"]["bulk_model_path"],
        surf_model_path=cfg["ml"]["surface_model_path"],
    )

    surface_mc_kwargs = {
        **cfg["surface_mc"],
        "surface_json_path": cfg["thermo"]["surface_json_path"],
        "omega_table_path": cfg["thermo"]["omega_table_path"],
    }

    ctx = build_context(
        mode=cfg["ga"]["mode"],
        population_size=cfg["ga"]["population_size"],
        n_generations=cfg["ga"]["n_generations"],
        crossover_prob=cfg["ga"]["crossover_prob"],
        mutation_prob=cfg["ga"]["mutation_prob"],
        elite_size=cfg["ga"]["elite_size"],
        random_seed=cfg["ga"]["random_seed"],
        penalty_fitness=cfg["ga"]["penalty_fitness"],
        t_eval_h=cfg["ga"]["t_eval_h"],
        ml_bulk_model=bulk_model,
        ml_surf_model=surf_model,
        ml_thresholds=cfg["ml"].get("thresholds", {}),
        ml_domain_bounds=cfg["ml"].get("domain_bounds", {}),
        surface_mc_kwargs=surface_mc_kwargs,
        gonogo_kwargs=cfg["go_nogo"],
        fitness_kwargs={},
        cache_path=None,
    )

    code = random_valid_code()
    decoded = decode(code)

    fitness, result = evaluate_candidate(code, ctx)

    print("SMOKE TEST OK")
    print("decoded:", decoded)
    print("fitness:", fitness)
    print("final_status:", result["final_status"])
    print("ml_status:", result["ml_status"])
    print("surface_valid:", result["surface_valid"])
    print("surface_reason:", result["surface_reason"])
    print("gonogo_valid:", result["gonogo_valid"])
    print("gonogo_reason:", result["gonogo_reason"])
    print("fitness_reason:", result["fitness_reason"])


if __name__ == "__main__":
    main()