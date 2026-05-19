# build_omega_table.py
# ============================================================
# Precompute binary omega_nm table for surface_mc.py
# ============================================================
#
# Generates omega_nm(T) for all substitutional binary pairs
# used by the self-healing GA, over the T_aging grid of the AG.
#
# Output:
#   omega_pairs_table.csv
#
# This is an offline preprocessing step intended to avoid
# repeated pycalphad calls inside the AG loop.
# ============================================================

from __future__ import annotations

from pathlib import Path
from itertools import combinations
from typing import List, Tuple

import pandas as pd
from pycalphad import Database, equilibrium, variables as v
import numpy as np


# ============================================================
# CONFIG
# ============================================================

TDB_PATH = Path(
    r"D:\scripts_dw\sh_ga\data\thermo\steel_database_fix.tdb"
)

OUT_PATH = Path(
    r"D:\scripts_dw\sh_ga\ga_sh\omega_pairs_table.csv"
)

PHASE_NAME = "BCC_A2"
P_PA = 101325.0

# Elements relevant to current AG version
ELEMENTS = ["FE", "CR", "MN", "SI", "MO", "NB", "TI", "W", "NI", "CU"]

# T_aging grid from the encoder / AG
T_C_GRID = list(range(550, 751, 20))

# LSQ / regular solution settings
Z_BCC = 8
LSQ_X_MIN = 0.05
LSQ_X_MAX = 0.95
LSQ_NPTS = 19


# ============================================================
# THERMODYNAMIC HELPERS
# ============================================================

def delta_H_mix_Jmol(
    db: Database,
    phase: str,
    elemA: str,
    elemB: str,
    T_K: float,
    P: float,
    xA: float,
) -> float:
    elemA = elemA.upper()
    elemB = elemB.upper()
    xA = float(xA)
    xB = 1.0 - xA

    if not (0.0 < xA < 1.0):
        raise ValueError("xA must be in (0,1).")

    comps = [elemA, elemB, "VA"]

    H_sys = equilibrium(
        db, comps, [phase],
        {v.T: T_K, v.P: P, v.X(elemA): xA},
        output="HM"
    ).HM.values.flatten()[0]

    H_A = equilibrium(
        db, comps, [phase],
        {v.T: T_K, v.P: P, v.X(elemA): 0.99999},
        output="HM"
    ).HM.values.flatten()[0]

    H_B = equilibrium(
        db, comps, [phase],
        {v.T: T_K, v.P: P, v.X(elemA): 0.00001},
        output="HM"
    ).HM.values.flatten()[0]

    return float(H_sys - (xA * H_A + xB * H_B))


def omega_LSQ_Jmol(
    db: Database,
    phase: str,
    elemA: str,
    elemB: str,
    T_K: float,
    P: float,
    z: int = Z_BCC,
    x_min: float = LSQ_X_MIN,
    x_max: float = LSQ_X_MAX,
    npts: int = LSQ_NPTS,
) -> float:
    elemA = elemA.upper()
    elemB = elemB.upper()

    xs = np.linspace(float(x_min), float(x_max), int(npts))
    xs = np.clip(xs, 1e-3, 1.0 - 1e-3)

    dH = np.array(
        [delta_H_mix_Jmol(db, phase, elemA, elemB, T_K, P, xA=x) for x in xs],
        dtype=float
    )

    f = float(z) * xs * (1.0 - xs)

    denom = float(np.dot(f, f))
    if denom <= 0:
        return 0.0

    return float(np.dot(dH, f) / denom)


# ============================================================
# TABLE BUILDER
# ============================================================

def build_omega_table(
    db: Database,
    elements: List[str],
    t_c_grid: List[int],
    phase_name: str,
    pressure_pa: float,
    z_bcc: int,
    lsq_xmin: float,
    lsq_xmax: float,
    lsq_npts: int,
) -> pd.DataFrame:
    rows = []

    pairs: List[Tuple[str, str]] = list(combinations(sorted(set(elements)), 2))

    total = len(t_c_grid) * len(pairs)
    counter = 0

    for t_c in t_c_grid:
        t_k = float(t_c) + 273.15

        for elem_a, elem_b in pairs:
            counter += 1
            print(f"[{counter}/{total}] T={t_c} °C, pair=({elem_a}, {elem_b})")

            omega = omega_LSQ_Jmol(
                db=db,
                phase=phase_name,
                elemA=elem_a,
                elemB=elem_b,
                T_K=t_k,
                P=pressure_pa,
                z=z_bcc,
                x_min=lsq_xmin,
                x_max=lsq_xmax,
                npts=lsq_npts,
            )

            rows.append({
                "phase": phase_name,
                "T_celsius": int(t_c),
                "T_K": float(t_k),
                "elem_a": elem_a,
                "elem_b": elem_b,
                "omega_Jmol": float(omega),
                "z_bcc": int(z_bcc),
                "lsq_xmin": float(lsq_xmin),
                "lsq_xmax": float(lsq_xmax),
                "lsq_npts": int(lsq_npts),
            })

    df = pd.DataFrame(rows)
    return df.sort_values(["T_celsius", "elem_a", "elem_b"]).reset_index(drop=True)


def main():
    if not TDB_PATH.exists():
        raise FileNotFoundError(f"TDB not found: {TDB_PATH}")

    db = Database(str(TDB_PATH))

    df = build_omega_table(
        db=db,
        elements=ELEMENTS,
        t_c_grid=T_C_GRID,
        phase_name=PHASE_NAME,
        pressure_pa=P_PA,
        z_bcc=Z_BCC,
        lsq_xmin=LSQ_X_MIN,
        lsq_xmax=LSQ_X_MAX,
        lsq_npts=LSQ_NPTS,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nSaved omega table to:\n{OUT_PATH}")
    print(f"Rows: {len(df)}")
    print(df.head())


if __name__ == "__main__":
    main()