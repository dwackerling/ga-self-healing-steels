from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "ga_sh"))

from config import load_config
from tc_backend import ThermoCalcBackend


def load_best_valid_candidate() -> tuple[dict[str, float], float, int]:
    """
    Load the best valid candidate from the latest surface_enhanced CSV.

    Best here means lowest current fitness among candidates with
    gonogo_reason == 'pass'.
    """
    path = PROJECT_ROOT / "outputs" / "runs" / "surface_enhanced" / "history_candidates.csv"
    df = pd.read_csv(path)

    valid = df[df["gonogo_reason"] == "pass"].copy()
    if valid.empty:
        raise RuntimeError("No valid candidate with gonogo_reason == 'pass' found.")

    valid["fitness_num"] = pd.to_numeric(valid["fitness"], errors="coerce")
    valid = valid.sort_values("fitness_num")

    row = valid.iloc[0]
    decoded = ast.literal_eval(row["decoded_values"])

    bulk = {
        "Cr": float(decoded.get("Cr", 0.0)),
        "Mn": float(decoded.get("Mn", 0.0)),
        "Si": float(decoded.get("Si", 0.0)),
        "Mo": float(decoded.get("Mo", 0.0)),
        "Nb": float(decoded.get("Nb", 0.0)),
        "Ti": float(decoded.get("Ti", 0.0)),
        "Ni": float(decoded.get("Ni", 0.0)),
        "Cu": float(decoded.get("Cu", 0.0)),
        "C": float(decoded.get("C", 0.05)),
    }

    return bulk, float(decoded["T_aging"]), int(row["candidate_id"])


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "config" / "config.yaml")

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

    bulk, T_aging, candidate_id = load_best_valid_candidate()

    phase_matrix = cfg["go_nogo"]["phases"]["matrix_bcc"]
    phase_laves = cfg["go_nogo"]["phases"]["laves"]

    coarsening_elements = cfg["fitness"]["coarsening"]["elements"]

    with ThermoCalcBackend(
        thermo_database=cfg["thermo"]["thermo_database"],
        mobility_database=cfg["thermo"]["mobility_database"],
        elements=elements,
        matrix_phase=phase_matrix,
        precip_phase=phase_laves,
    ) as backend:
        result = backend.calculate_equilibrium_result(
            composition_wt=bulk,
            temperature_c=T_aging,
        )

        x_matrix = backend.get_phase_mole_fractions(
            result=result,
            phase_name=phase_matrix,
            elements=coarsening_elements,
        )

        x_laves = backend.get_phase_mole_fractions(
            result=result,
            phase_name=phase_laves,
            elements=coarsening_elements,
        )

        diffusivities = backend.get_diffusivities(
            result=result,
            phase_name=phase_matrix,
            elements=coarsening_elements,
            dependent_element="Fe",
        )

    print("candidate_id:", candidate_id)
    print("T_aging:", T_aging)
    print("bulk_wt:", bulk)

    print("\nx_matrix_BCC_A2:")
    print(x_matrix)

    print("\nx_laves_C14_LAVES:")
    print(x_laves)

    print("\ndiffusivities_BCC_A2:")
    print(diffusivities)


if __name__ == "__main__":
    main()