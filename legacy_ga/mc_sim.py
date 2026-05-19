# mc_sim.py
# Surface segregation Monte Carlo for multicomponent Fe-based alloys.
# Input: bulk composition in wt% (dict). Output: surface composition (layer-averaged) in wt% (dict).
# Interstitials (C, N, B) are excluded from MC and passed through unchanged.

import os
import numpy as np
from functools import lru_cache
from pycalphad import Database, equilibrium, variables as v

# ---------------------------
# Constants & caches
# ---------------------------

_INTERSTITIALS = {"C", "N", "B"}
_VAC = "VA"
_FE = "FE"

_DB_CACHE = {}  # lazy-loaded Database objects by absolute path

# Atomic masses (g/mol) used for wt%↔atomic fractions.
# Covers Fe plus: Cr, Mn, Ni, Mo, V, W, Si, Ti, Al, Nb, C, B, N
DEFAULT_MASSES = {
    "FE": 55.845,
    "CR": 52.00,
    "MN": 54.938,  # 54.938044
    "NI": 58.693,  # 58.6934
    "MO": 95.95,
    "V":  50.942,  # 50.9415
    "W":  183.84,
    "SI": 28.085,
    "TI": 47.867,
    "AL": 26.982,
    "NB": 92.906,  # 92.90637
    "C":  12.011,
    "B":  10.81,
    "N":  14.007,
}

# ---------------------------
# Utilities
# ---------------------------

def _to_upper(d: dict) -> dict:
    return {str(k).upper(): float(v) for k, v in d.items()}

def get_database(db_path: str) -> Database:
    """Lazy-load and cache a pycalphad Database by absolute path."""
    apath = os.path.abspath(db_path)
    db = _DB_CACHE.get(apath)
    if db is None:
        db = Database(apath)
        _DB_CACHE[apath] = db
    return db

def order_pairs_by_composition(elem1: str, x1: float, elem2: str, x2: float):
    """Return (elemA, elemB, xA, xB) with FE preferred as A; otherwise by higher x."""
    e1 = elem1.upper()
    e2 = elem2.upper()
    if e1 == _FE:
        return e1, e2, x1, x2
    if e2 == _FE:
        return e2, e1, x2, x1
    return (e1, e2, x1, x2) if x1 >= x2 else (e2, e1, x2, x1)

def wt_to_atomic_fraction(wt: np.ndarray, elems: list[str], masses: dict[str, float]) -> np.ndarray:
    m = np.array([masses[e] for e in elems], dtype=float)
    n = np.array(wt, dtype=float) / m
    s = n.sum()
    if s <= 0.0:
        raise ValueError("Non-positive total amount in wt_to_atomic_fraction.")
    return n / s

# ---------------------------
# Thermodynamic pieces (Ω via pycalphad)
# ---------------------------

@lru_cache(maxsize=200_000)
def _enthalpy(db_path: str, phase: str, elemA: str, elemB: str, T: float, P: float, xA: float) -> float:
    """Cached enthalpy HM for a binary A-B-(VA) system at composition xA of A (J/mol)."""
    db = get_database(db_path)
    comps = [elemA, elemB, _VAC]
    conds = {v.T: T, v.P: P, v.X(elemA): xA}
    H = equilibrium(db, comps, phase, conds, output='HM').HM.values.flatten()[0]
    return float(H)

def compute_binary_omega(elem1: str, x1: float, elem2: str, x2: float,
                         db_path: str,
                         phase: str = 'BCC_A2',
                         T: float = 650.0 + 273.15,
                         P: float = 101325.0,
                         z: int = 8) -> float | None:
    """
    Ω_AB from ΔH_mix:
      ΔH_mix = H(xA) - [xA*H(A) + (1-xA)*H(B)]
      Ω = ΔH_mix / (z * xA * (1-xA))
    Returns None on failure; 0.0 is a valid physical value.
    """
    elemA, elemB, xA_in, xB_in = order_pairs_by_composition(elem1, x1, elem2, x2)
    if elemA != _FE and elemB != _FE:
        total = xA_in + xB_in
        if total <= 0.0:
            return None
        xB = xB_in / total
        xA = 1.0 - xB
    else:
        xA = 1.0 - xB_in
    xA = float(np.clip(xA, 1e-12, 1 - 1e-12))
    try:
        H_sys = _enthalpy(db_path, phase, elemA, elemB, T, P, xA)
        H_A   = _enthalpy(db_path, phase, elemA, elemB, T, P, 1.0)
        H_B   = _enthalpy(db_path, phase, elemA, elemB, T, P, 0.0)
        dHmix = H_sys - (xA * H_A + (1.0 - xA) * H_B)
        Omega = dHmix / (z * xA * (1.0 - xA))
        return float(Omega) if np.isfinite(Omega) else None
    except Exception as e:
        print(f"[Ω] Error for pair {elemA}-{elemB}: {e}")
        return None

# ---------------------------
# MC energy functional
# ---------------------------

def gamma_Omega(Y: np.ndarray, x_bulk: np.ndarray, gamma0: np.ndarray,
                omega_nm: np.ndarray, Omega: float, R: float, T: float,
                zl: float, zv: float) -> float:
    """
    Interfacial free energy functional (per area).
    NOTE: If gamma0 == 0 and Omega == 1, only entropic + pair terms drive segregation.
    """
    N, L = Y.shape
    # 1) Base surface energy (first layer)
    term1 = float(np.dot(Y[:, 0], gamma0)) * Omega
    # 2) Entropic term
    ratio = np.clip(Y / x_bulk[:, None], 1e-12, None)
    term2 = R * T * float(np.sum(Y * np.log(ratio)))
    # 3–5) Pair terms
    term3 = 0.0
    for n in range(N):
        for m in range(n + 1, N):
            term3 += omega_nm[n, m] * (x_bulk[n]*x_bulk[m] - x_bulk[n]*Y[m, 0] - x_bulk[m]*Y[n, 0])
    term3 *= zv

    term4 = 0.0
    for i in range(L - 1):
        for n in range(N):
            for m in range(n + 1, N):
                term4 += omega_nm[n, m] * ((Y[m, i] - x_bulk[m]) * (Y[n, i+1] - x_bulk[n]) +
                                           (Y[n, i] - x_bulk[n]) * (Y[m, i+1] - x_bulk[m]))
    term4 *= zv

    term5 = 0.0
    for i in range(L):
        for n in range(N):
            for m in range(n + 1, N):
                term5 += omega_nm[n, m] * (Y[n, i] - x_bulk[n]) * (Y[m, i] - x_bulk[m])
    term5 *= zl

    return (term1 + term2 + term3 + term4 + term5) / Omega

def monte_carlo_multicomponent(Y_init: np.ndarray, x_bulk: np.ndarray, gamma0: np.ndarray,
                               omega_nm: np.ndarray, Omega: float, R: float, T: float,
                               zl: float, zv: float, steps: int = 100_000,
                               temp: float = 1e-4, delta_max: float = 1e-4,
                               seed: int | None = None):
    """Metropolis MC to minimize gamma_Omega. Deterministic if 'seed' is set."""
    rng = np.random.default_rng(seed)
    Y = Y_init.copy()
    best_Y = Y.copy()
    best_energy = gamma_Omega(Y, x_bulk, gamma0, omega_nm, Omega, R, T, zl, zv)
    energy_history = [best_energy]
    N, L = Y.shape

    layer_weights = np.exp(-np.linspace(0.0, 3.0, L))
    layer_weights /= layer_weights.sum()

    for _ in range(steps):
        layer = rng.choice(L, p=layer_weights)
        i, j = rng.choice(N, 2, replace=False)
        d = rng.uniform(-delta_max, delta_max)

        xi = Y[i, layer] - d
        xj = Y[j, layer] + d
        if 0.0 < xi < 1.0 and 0.0 < xj < 1.0:
            Y_trial = Y.copy()
            Y_trial[i, layer] = xi
            Y_trial[j, layer] = xj

            e_trial = gamma_Omega(Y_trial, x_bulk, gamma0, omega_nm, Omega, R, T, zl, zv)
            de = e_trial - best_energy
            if de < 0.0 or rng.random() < np.exp(-de / temp):
                Y = Y_trial
                if e_trial < best_energy:
                    best_energy = e_trial
                    best_Y = Y.copy()
            energy_history.append(e_trial)

    return best_Y, best_energy, energy_history

# ---------------------------
# Public API
# ---------------------------

def compute_surface_composition_from_bulk(
    bulk_wt: dict,
    db_path: str,
    masses: dict | None = None,
    *,
    phase: str = 'BCC_A2',
    temperature_K: float = 650.0 + 273.15,
    pressure_Pa: float = 101325.0,
    z_coordination: int = 8,
    layers: int = 6,
    avg_top_layers: int | None = None,   # None => promedio entre TODAS las capas
    zl: float = 0.0,
    zv: float = 4.0,
    mc_steps: int = 100_000,
    mc_temp: float = 1e-4,
    mc_delta_max: float = 1e-4,
    seed: int | None = None,
) -> dict:
    """
    Bulk wt% -> surface wt% via Monte Carlo segregation.
    - Uses ONLY the TDB (pycalphad enthalpies) + atomic masses.
    - Base surface energies (gamma0) set to 0 and Omega=1 for scale-invariance.

    Returns dict with {'SURF_<EL>': wt%, ..., 'SURF_ENERGY': best_energy}.
    Interstitials (C,N,B) are copied through as-is from bulk input.
    """
    masses = _to_upper(masses or DEFAULT_MASSES)
    bulk_wt = _to_upper(bulk_wt)

    # substitutionals (exclude interstitials)
    subs = {el: wt for el, wt in bulk_wt.items() if el not in _INTERSTITIALS and wt > 0.0}
    sum_subs = float(np.sum(list(subs.values()))) if subs else 0.0
    fe_wt = 100.0 - sum_subs
    if fe_wt < -1e-9:
        raise ValueError(f"Negative Fe balance: {fe_wt} wt%. Check bulk_wt sum.")
    subs[_FE] = max(fe_wt, 0.0)

    elements = list(subs.keys())
    wt_subs = np.array([subs[e] for e in elements], dtype=float)

    # atomic fractions
    x_bulk = wt_to_atomic_fraction(wt_subs, elements, masses)
    N = len(elements)

    # Ω matrix
    omega_nm = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            w_ij = compute_binary_omega(elements[i], x_bulk[i],
                                        elements[j], x_bulk[j],
                                        db_path=db_path,
                                        phase=phase, T=temperature_K,
                                        P=pressure_Pa, z=z_coordination)
            if w_ij is None or not np.isfinite(w_ij):
                continue
            omega_nm[i, j] = w_ij
            omega_nm[j, i] = w_ij

    # no external surface-energies/diameters: gamma0=0, Omega=1
    gamma0 = np.zeros(N, dtype=float)
    Omega = 1.0

    # init layers with bulk
    L = int(layers)
    Y0 = np.tile(x_bulk[:, None], (1, L))

    # MC
    Y_best, best_energy, _ = monte_carlo_multicomponent(
        Y0, x_bulk, gamma0, omega_nm, Omega, R=8.314462618, T=temperature_K,
        zl=zl, zv=zv, steps=mc_steps, temp=mc_temp, delta_max=mc_delta_max, seed=seed
    )

    # Average between layers:
    #  - if avg_top_layers is None -> average ACROSS ALL LAYERS
    #  - if int k -> average first k surface layers
    if avg_top_layers is None:
        Y_avg = np.mean(Y_best, axis=1)          # promedio entre las capas (todas)
    else:
        k = max(1, min(L, int(avg_top_layers)))
        Y_avg = np.mean(Y_best[:, :k], axis=1)   # promedio de las primeras k capas

    # atomic -> wt%
    m = np.array([masses[e] for e in elements], dtype=float)
    mass_total = float(np.sum(Y_avg * m))
    wt_surface = (Y_avg * m) / mass_total * 100.0

    # pack result
    result = {f"SURF_{e}": float(w) for e, w in zip(elements, wt_surface)}
    result['SURF_ENERGY'] = float(best_energy)

    # pass-through interstitials
    for e in _INTERSTITIALS:
        if e in bulk_wt:
            result[e] = float(bulk_wt[e])

    return result
