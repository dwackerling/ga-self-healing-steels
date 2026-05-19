from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "ga_sh"))

from config import load_config
from tc_backend import ThermoCalcBackend


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

    # Candidate 12 from the GA output
    bulk = {
        "Cr": 16.0,   # cambia estos valores si quieres copiar exactamente candidate 12
        "Mn": 0.5,
        "Si": 0.5,
        "Mo": 2.0,
        "Nb": 0.5,
        "Ti": 0.2,
        "Ni": 2.0,
        "Cu": 0.5,
        "C": 0.05,
    }

    # Example surface composition from MC format should be solute-only or substitutional.
    # For this first test, use the same composition as bulk to validate the API.
    surface = dict(bulk)

    T_aging = 650.0

    with ThermoCalcBackend(
        thermo_database=cfg["thermo"]["thermo_database"],
        mobility_database=cfg["thermo"]["mobility_database"],
        elements=elements,
        matrix_phase=cfg["go_nogo"]["phases"]["matrix_bcc"],
        precip_phase=cfg["go_nogo"]["phases"]["laves"],
    ) as backend:
        df_bulk = backend.calculate_driving_force(
            composition_wt=bulk,
            temperature_c=T_aging,
            precip_phase=cfg["go_nogo"]["phases"]["laves"],
        )

        df_surface = backend.calculate_driving_force(
            composition_wt=surface,
            temperature_c=T_aging,
            precip_phase=cfg["go_nogo"]["phases"]["laves"],
        )

    print("DF_bulk:", df_bulk)
    print("DF_surface:", df_surface)
    print("delta_surface_minus_bulk:", df_surface - df_bulk)


if __name__ == "__main__":
    main()