# fitness.py
# ============================================================
# Mono-objective fitness module for the self-healing GA project
# ============================================================
#
# Objective:
#   minimise the coarsened Laves precipitate radius:
#
#       r_t = (r0^3 + K t)^(1/3)
#
# Current implementation:
#   - gamma is fixed from YAML.
#   - Vm_laves is fixed from YAML.
#   - r0 is calculated from TC-Python normalizedDrivingForce:
#
#       normalizedDrivingForce = DeltaG_m / (R T)
#       DeltaG_m = normalizedDrivingForce * R * T
#       r0 = 2 gamma Vm_laves / DeltaG_m
#
#   - K is computed using TC-derived phase compositions and diffusivities.
#
# Output fitness unit:
#   nm
# ============================================================

from __future__ import annotations

import math
from typing import Any, Dict

R_GAS = 8.31446261815324  # J mol-1 K-1


def _to_float(value: Any, default=None):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(x):
        return default

    return x


def evaluate_fitness(
    gonogo_result: Dict[str, Any],
    t_eval_h: float,
    df_bulk_ml: float | None = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Mono-objective fitness evaluation.

    The GA minimises the returned scalar fitness.

    Parameters
    ----------
    gonogo_result
        Output from evaluate_go_nogo().
    t_eval_h
        Exposure/evaluation time in hours.
    df_bulk_ml
        Deprecated. Kept for interface compatibility.
    kwargs
        Fitness configuration from config.yaml.
    """

    if not gonogo_result.get("gonogo_valid", False):
        return {
            "fitness": None,
            "fitness_valid": False,
            "fitness_reason": "gonogo_invalid",
        }

    objective = kwargs.get("objective", "coarsened_laves_radius")

    if objective != "coarsened_laves_radius":
        return {
            "fitness": None,
            "fitness_valid": False,
            "fitness_reason": f"unsupported_objective_{objective}",
        }

    gamma_cfg = kwargs.get("gamma", {})
    initial_radius_cfg = kwargs.get("initial_radius", {})
    coarsening_cfg = kwargs.get("coarsening", {})
    molar_volume_cfg = kwargs.get("molar_volume", {})

    gamma_J_m2 = _to_float(gamma_cfg.get("gamma_J_m2"))
    Vm_laves_m3_mol = _to_float(molar_volume_cfg.get("Vm_laves_m3_mol"))

    if gamma_J_m2 is None or gamma_J_m2 <= 0.0:
        return {
            "fitness": None,
            "fitness_valid": False,
            "fitness_reason": "invalid_gamma_J_m2",
        }

    if Vm_laves_m3_mol is None or Vm_laves_m3_mol <= 0.0:
        return {
            "fitness": None,
            "fitness_valid": False,
            "fitness_reason": "invalid_Vm_laves_m3_mol",
        }

    T_aging_C = _to_float(gonogo_result.get("T_aging_C"))
    if T_aging_C is None:
        return {
            "fitness": None,
            "fitness_valid": False,
            "fitness_reason": "missing_T_aging_C",
        }

    T_K = T_aging_C + 273.15
    if T_K <= 0.0:
        return {
            "fitness": None,
            "fitness_valid": False,
            "fitness_reason": "invalid_T_K",
        }

    # --------------------------------------------------------
    # Initial radius r0
    # --------------------------------------------------------
    r0_source = initial_radius_cfg.get("source", "constant")

    DGm_J_mol = None
    DGv_J_m3 = None

    if r0_source == "normalized_driving_force":
        df_key = initial_radius_cfg.get("driving_force_key", "DF_surf_tc")
        normalized_df = _to_float(gonogo_result.get(df_key))

        if normalized_df is None or normalized_df <= 0.0:
            fallback_r0_nm = _to_float(initial_radius_cfg.get("fallback_r0_nm"))

            if fallback_r0_nm is None or fallback_r0_nm <= 0.0:
                return {
                    "fitness": None,
                    "fitness_valid": False,
                    "fitness_reason": f"invalid_or_missing_{df_key}_and_no_valid_fallback_r0",
                }

            r0_nm = fallback_r0_nm
            r0_m = r0_nm * 1.0e-9

        else:
            DGm_J_mol = abs(normalized_df) * R_GAS * T_K
            DGv_J_m3 = DGm_J_mol / Vm_laves_m3_mol

            if DGm_J_mol <= 0.0 or DGv_J_m3 <= 0.0:
                return {
                    "fitness": None,
                    "fitness_valid": False,
                    "fitness_reason": "invalid_driving_force_conversion",
                    "DGm_J_mol": DGm_J_mol,
                    "DGv_J_m3": DGv_J_m3,
                }

            r0_m = (2.0 * gamma_J_m2) / DGv_J_m3
            r0_nm = r0_m * 1.0e9

    elif r0_source == "constant":
        r0_nm = _to_float(initial_radius_cfg.get("r0_nm"))

        if r0_nm is None or r0_nm <= 0.0:
            return {
                "fitness": None,
                "fitness_valid": False,
                "fitness_reason": "invalid_r0_nm",
            }

        r0_m = r0_nm * 1.0e-9

    else:
        return {
            "fitness": None,
            "fitness_valid": False,
            "fitness_reason": f"unsupported_r0_source_{r0_source}",
        }

    if r0_m <= 0.0 or not math.isfinite(r0_m):
        return {
            "fitness": None,
            "fitness_valid": False,
            "fitness_reason": "invalid_r0_m",
            "r0_m": r0_m,
            "r0_nm": r0_nm,
        }

    # --------------------------------------------------------
    # Coarsening inputs
    # --------------------------------------------------------
    x_matrix = gonogo_result.get("coarsening_x_matrix")
    x_laves = gonogo_result.get("coarsening_x_laves")
    diffusivities = gonogo_result.get("coarsening_diffusivities")

    if not isinstance(x_matrix, dict):
        return {
            "fitness": None,
            "fitness_valid": False,
            "fitness_reason": "missing_coarsening_x_matrix",
        }

    if not isinstance(x_laves, dict):
        return {
            "fitness": None,
            "fitness_valid": False,
            "fitness_reason": "missing_coarsening_x_laves",
        }

    if not isinstance(diffusivities, dict):
        return {
            "fitness": None,
            "fitness_valid": False,
            "fitness_reason": "missing_coarsening_diffusivities",
        }

    elements = coarsening_cfg.get("elements", [])
    require_all = bool(coarsening_cfg.get("require_all_diffusivities", False))
    min_x_matrix = float(coarsening_cfg.get("min_matrix_mole_fraction", 1.0e-10))
    min_x_laves = float(coarsening_cfg.get("min_precipitate_mole_fraction", 1.0e-10))
    min_D = float(coarsening_cfg.get("min_diffusivity_m2_s", 1.0e-30))

    coarsening_sum = 0.0
    used: list[str] = []
    skipped: dict[str, str] = {}
    terms: dict[str, float] = {}

    for element in elements:
        el = str(element)
        key = el[0].upper() + el[1:].lower()

        xm = _to_float(x_matrix.get(key))
        xp = _to_float(x_laves.get(key))
        D = _to_float(diffusivities.get(key))

        if xm is None or xp is None or D is None:
            skipped[key] = "missing_value"
            if require_all:
                return {
                    "fitness": None,
                    "fitness_valid": False,
                    "fitness_reason": f"missing_coarsening_value_{key}",
                }
            continue

        if xm < min_x_matrix and xp < min_x_laves:
            skipped[key] = "trace_element"
            continue

        if xm < min_x_matrix:
            skipped[key] = "matrix_mole_fraction_below_min"
            if require_all:
                return {
                    "fitness": None,
                    "fitness_valid": False,
                    "fitness_reason": f"matrix_mole_fraction_below_min_{key}",
                }
            continue

        if xp < min_x_laves:
            skipped[key] = "precipitate_mole_fraction_below_min"
            continue

        if D < min_D:
            skipped[key] = "diffusivity_below_min"
            if require_all:
                return {
                    "fitness": None,
                    "fitness_valid": False,
                    "fitness_reason": f"diffusivity_below_min_{key}",
                }
            continue

        delta_x = xp - xm
        term = (delta_x * delta_x) / (xm * D)

        if not math.isfinite(term) or term <= 0.0:
            skipped[key] = "invalid_coarsening_term"
            if require_all:
                return {
                    "fitness": None,
                    "fitness_valid": False,
                    "fitness_reason": f"invalid_coarsening_term_{key}",
                }
            continue

        terms[key] = term
        coarsening_sum += term
        used.append(key)

    if not used:
        return {
            "fitness": None,
            "fitness_valid": False,
            "fitness_reason": "no_valid_coarsening_elements",
            "coarsening_elements_skipped": skipped,
        }

    if coarsening_sum <= 0.0 or not math.isfinite(coarsening_sum):
        return {
            "fitness": None,
            "fitness_valid": False,
            "fitness_reason": "invalid_coarsening_sum",
            "coarsening_elements_used": used,
            "coarsening_elements_skipped": skipped,
            "coarsening_terms": terms,
        }

    K_m3_s = (8.0 * gamma_J_m2 * Vm_laves_m3_mol) / (
        9.0 * R_GAS * T_K * coarsening_sum
    )

    if K_m3_s < 0.0 or not math.isfinite(K_m3_s):
        return {
            "fitness": None,
            "fitness_valid": False,
            "fitness_reason": "invalid_K_m3_s",
            "K_m3_s": K_m3_s,
            "coarsening_sum": coarsening_sum,
            "coarsening_terms": terms,
            "coarsening_elements_used": used,
            "coarsening_elements_skipped": skipped,
        }

    t_s = float(t_eval_h) * 3600.0
    if t_s <= 0.0:
        return {
            "fitness": None,
            "fitness_valid": False,
            "fitness_reason": "invalid_t_eval_h",
        }

    r0_cubed_m3 = r0_m**3
    Kt_m3 = K_m3_s * t_s

    if r0_cubed_m3 > 0.0:
        Kt_over_r0_cubed = Kt_m3 / r0_cubed_m3
    else:
        Kt_over_r0_cubed = None

    rt_m = (r0_cubed_m3 + Kt_m3) ** (1.0 / 3.0)
    rt_nm = rt_m * 1.0e9

    return {
        "fitness": rt_nm,
        "fitness_valid": True,
        "fitness_reason": "coarsened_laves_radius_nm",
        "K_m3_s": K_m3_s,
        "rt_m": rt_m,
        "rt_nm": rt_nm,
        "r0_m": r0_m,
        "r0_nm": r0_nm,
        "DGm_J_mol": DGm_J_mol,
        "DGv_J_m3": DGv_J_m3,
        "r0_cubed_m3": r0_cubed_m3,
        "Kt_m3": Kt_m3,
        "Kt_over_r0_cubed": Kt_over_r0_cubed,
        "coarsening_sum": coarsening_sum,
        "coarsening_terms": terms,
        "coarsening_elements_used": used,
        "coarsening_elements_skipped": skipped,
    }