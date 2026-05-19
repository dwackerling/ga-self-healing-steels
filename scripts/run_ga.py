from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------
# Suppress noisy TC-Python / Thermo-Calc warnings/logs
# Must be configured before importing tc_backend/tc_python.
# ------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    message=r".*pkg_resources is deprecated as an API.*",
)

warnings.filterwarnings(
    "ignore",
    message=r".*sklearn.utils.parallel.delayed.*",
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"sklearn\.utils\.parallel",
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s:%(name)s:%(message)s",
)

for logger_name in [
    "",
    "tc_python",
    "CalculationEngine",
    "JavaWrapper",
    "SystemBuilder",
    "Py4JPropertyModel",
    "Calculation",
]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)


# ------------------------------------------------------------
# Project imports
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "ga_sh"))

from config import load_config
from ml_filter import load_models
from ga_sh import build_context, run_ga
from tc_backend import ThermoCalcBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run self-healing alloy GA.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "config.yaml"),
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    bulk_model, surf_model = load_models(
        bulk_model_path=cfg["ml"]["bulk_model_path"],
        surf_model_path=cfg["ml"]["surface_model_path"],
    )

    # Avoid joblib worker warnings/logging conflicts during long GA runs.
    if hasattr(bulk_model, "n_jobs"):
        bulk_model.n_jobs = 1

    if hasattr(surf_model, "n_jobs"):
        surf_model.n_jobs = 1

    surface_mc_kwargs = {
        **cfg["surface_mc"],
        "surface_json_path": cfg["thermo"]["surface_json_path"],
        "omega_table_path": cfg["thermo"]["omega_table_path"],
    }

    cache_path = Path(cfg["project"]["cache_dir"]) / f"ga_cache_{cfg['ga']['mode']}.pkl"

    # Include W in the Thermo-Calc system. This does not make W an optimisation
    # variable unless the encoder/composition pipeline also includes W.
    elements = [
        "Fe",
        "Cr",
        "Mn",
        "Si",
        "Mo",
        "Nb",
        "Ti",
        "W",
        "Ni",
        "Cu",
        "C",
    ]

    # Avoid passing t_eval_h twice to evaluate_fitness().
    # ga_sh.py already passes t_eval_h explicitly from cfg["ga"]["t_eval_h"].
    fitness_kwargs = {
        k: v
        for k, v in cfg.get("fitness", {}).items()
        if k != "t_eval_h"
    }

    with ThermoCalcBackend(
        thermo_database=cfg["thermo"]["thermo_database"],
        mobility_database=cfg["thermo"]["mobility_database"],
        elements=elements,
        matrix_phase=cfg["go_nogo"]["phases"]["matrix_bcc"],
        precip_phase=cfg["go_nogo"]["phases"]["laves"],
        suppress_tc_output=True,
    ) as tc_backend:

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
            ml_thresholds=cfg["ml"]["thresholds"],
            ml_domain_bounds=cfg["ml"]["domain_bounds"],
            surface_mc_kwargs=surface_mc_kwargs,
            gonogo_kwargs={
                **cfg["go_nogo"],
                "tc_backend": tc_backend,
                "fitness_config": cfg.get("fitness", {}),
            },
            fitness_kwargs=fitness_kwargs,
            cache_path=str(cache_path),
            cache_autosave_every=50,
        )

        result = run_ga(ctx)

        run_name = (
            f"{cfg['ga']['mode']}"
            f"_pop{cfg['ga']['population_size']}"
            f"_gen{cfg['ga']['n_generations']}"
            f"_seed{cfg['ga']['random_seed']}"
        )

        output_dir = Path(cfg["project"]["output_dir"]) / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        candidates_path = output_dir / "history_candidates.csv"
        generations_path = output_dir / "history_generations.csv"
        config_snapshot_path = output_dir / "config_resolved.json"

        pd.DataFrame(ctx["storage"]["history_candidates"]).to_csv(
            candidates_path,
            index=False,
        )

        pd.DataFrame(ctx["storage"]["history_generations"]).to_csv(
            generations_path,
            index=False,
        )

        with open(config_snapshot_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        print("GA RUN COMPLETE")
        print("mode:", cfg["ga"]["mode"])
        print("run_name:", run_name)
        print("cache_path:", cache_path)
        print("candidates_path:", candidates_path)
        print("generations_path:", generations_path)
        print("config_snapshot_path:", config_snapshot_path)

        if result is not None:
            print("result type:", type(result))


if __name__ == "__main__":
    main()