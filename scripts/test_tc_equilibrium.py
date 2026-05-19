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

    comp = {
        "Cr": 16.0,
        "Mn": 0.5,
        "Si": 0.5,
        "Mo": 2.0,
        "Nb": 0.5,
        "Ti": 0.2,
        "Ni": 2.0,
        "Cu": 0.5,
        "C": 0.05,
        "W": 2
    }

    with ThermoCalcBackend(
        thermo_database=cfg["thermo"]["thermo_database"],
        mobility_database=cfg["thermo"]["mobility_database"],
        elements=elements,
        matrix_phase=cfg["go_nogo"]["phases"]["matrix_bcc"],
        precip_phase=cfg["go_nogo"]["phases"]["laves"],
    ) as backend:
        sol = backend.calculate_equilibrium_snapshot(
            composition_wt=comp,
            temperature_c=1000.0,
            phases=cfg["go_nogo"]["phases"],
        )

        aging = backend.calculate_equilibrium_snapshot(
            composition_wt=comp,
            temperature_c=650.0,
            phases=cfg["go_nogo"]["phases"],
        )

    print("T_sol snapshot:")
    print(sol)
    print("T_aging snapshot:")
    print(aging)


if __name__ == "__main__":
    main()