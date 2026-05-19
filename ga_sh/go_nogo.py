from __future__ import annotations

from typing import Dict, Any


def _fail(reason: str, out: Dict[str, Any]) -> Dict[str, Any]:
    out["gonogo_valid"] = False
    out["gonogo_reason"] = reason
    return out


def _pass(out: Dict[str, Any]) -> Dict[str, Any]:
    out["gonogo_valid"] = True
    out["gonogo_reason"] = "pass"
    return out


def evaluate_go_nogo(
    composition_bulk_wt: Dict[str, float],
    composition_surface_wt: Dict[str, float],
    T_sol: float,
    T_aging: float,
    mode: str,
    tc_backend=None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Thermo-Calc go/no-go evaluation for the self-healing GA.

    Current implementation:
    - equilibrium at T_sol using bulk composition
    - equilibrium at T_aging using bulk composition
    - no diffusion criterion
    - no driving-force criterion yet
    """

    use_placeholder = bool(kwargs.get("use_placeholder", False))

    out: Dict[str, Any] = {
        "gonogo_valid": False,
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

        "T_aging_C": None,
        "coarsening_x_matrix": None,
        "coarsening_x_laves": None,
        "coarsening_diffusivities": None,
    }

    if use_placeholder:
        out.update({
            "gonogo_valid": True,
            "gonogo_reason": "placeholder_pass",
            "f_laves_bulk": None,
        })
        return out

    if tc_backend is None:
        return _fail("missing_tc_backend", out)

    if composition_bulk_wt is None:
        return _fail("missing_bulk_composition", out)

    if composition_surface_wt is None:
        return _fail("missing_surface_composition", out)

    comp_cfg = kwargs.get("composition", {})
    tsol_cfg = kwargs.get("t_sol", {})
    tage_cfg = kwargs.get("t_aging", {})
    phases = kwargs.get("phases", {})

    fe_min_wt = float(comp_cfg.get("fe_min_wt", 60.0))

    liquid_max = float(tsol_cfg.get("liquid_max", 1e-6))
    matrix_total_min = float(tsol_cfg.get("matrix_total_min", 0.98))
    laves_max_tsol = float(tsol_cfg.get("laves_max", 1e-4))
    m23c6_max_tsol = float(tsol_cfg.get("m23c6_max", 1e-4))
    mx_max_tsol = float(tsol_cfg.get("mx_max", 0.02))

    bcc_min = float(tage_cfg.get("bcc_min", 0.80))
    fcc_matrix_max = float(tage_cfg.get("fcc_matrix_max", 0.05))
    laves_min = float(tage_cfg.get("laves_min", 0.01))
    m23c6_max_tage = float(tage_cfg.get("m23c6_max", 0.02))
    mx_max_tage = float(tage_cfg.get("mx_max", 0.02))

    bulk = dict(composition_bulk_wt)
    bulk.setdefault("C", 0.05)

    total_non_fe = sum(float(v) for k, v in bulk.items() if str(k).upper() != "FE")
    fe_balance = 100.0 - total_non_fe
    out["Fe_balance_wt"] = fe_balance

    if fe_balance < fe_min_wt:
        return _fail("fe_below_minimum", out)

    try:
        sol = tc_backend.calculate_equilibrium_snapshot(
            composition_wt=bulk,
            temperature_c=float(T_sol),
            phases=phases,
        )
    except Exception as exc:
        out["tc_error"] = repr(exc)
        return _fail("tc_equilibrium_failed_Tsol", out)

    out["LIQ_Tsol"] = sol["LIQUID"]
    out["BCC_A2_Tsol"] = sol["BCC_A2"]
    out["FCC_A1_Tsol"] = sol["FCC_A1"]
    out["matrix_total_Tsol"] = sol["MATRIX_TOTAL_BCC_FCC"]
    out["Laves_Tsol"] = sol["LAVES"]
    out["M23C6_Tsol"] = sol["M23C6"]
    out["MX_Tsol"] = sol["MX"]

    if out["LIQ_Tsol"] > liquid_max:
        return _fail("liquid_present_Tsol", out)

    if out["matrix_total_Tsol"] < matrix_total_min:
        return _fail("matrix_total_bcc_fcc_below_min_Tsol", out)

    if out["Laves_Tsol"] > laves_max_tsol:
        return _fail("laves_present_Tsol", out)

    if out["M23C6_Tsol"] > m23c6_max_tsol:
        return _fail("m23c6_above_max_Tsol", out)

    if out["MX_Tsol"] > mx_max_tsol:
        return _fail("mx_above_max_Tsol", out)

    try:
        aging = tc_backend.calculate_equilibrium_snapshot(
            composition_wt=bulk,
            temperature_c=float(T_aging),
            phases=phases,
        )
    except Exception as exc:
        out["tc_error"] = repr(exc)
        return _fail("tc_equilibrium_failed_Taging", out)

    out["BCC_A2_Taging"] = aging["BCC_A2"]
    out["FCC_A1_Taging"] = aging["FCC_A1"]
    out["Laves_Taging"] = aging["LAVES"]
    out["M23C6_Taging"] = aging["M23C6"]
    out["MX_Taging"] = aging["MX"]
    out["f_laves_bulk"] = aging["LAVES"]
    out["T_aging_C"] = float(T_aging)

    if out["BCC_A2_Taging"] < bcc_min:
        return _fail("bcc_below_min_Taging", out)

    if out["FCC_A1_Taging"] > fcc_matrix_max:
        return _fail("fcc_above_max_Taging", out)

    if out["Laves_Taging"] < laves_min:
        return _fail("laves_below_min_Taging", out)

    if out["M23C6_Taging"] > m23c6_max_tage:
        return _fail("m23c6_above_max_Taging", out)

    if out["MX_Taging"] > mx_max_tage:
        return _fail("mx_above_max_Taging", out)

    # --------------------------------------------------------
    # Driving force: bulk vs surface
    # --------------------------------------------------------
    df_cfg = kwargs.get("driving_force", {})
    delta_df_min = float(
        df_cfg.get(
            "delta_normalized_df_min",
            df_cfg.get("delta_df_min_jmol", 0.01),
        )
    )
    
    require_positive_surface_df = bool(df_cfg.get("require_positive_surface_df", True))
    require_positive_bulk_df = bool(df_cfg.get("require_positive_bulk_df", True))

    precip_phase = phases.get("laves", "C14_LAVES")

    bulk_df_comp = dict(bulk)
    bulk_df_comp.setdefault("C", 0.05)

    surface_df_comp = dict(composition_surface_wt)
    surface_df_comp.setdefault("C", 0.05)

    try:
        df_bulk = tc_backend.calculate_driving_force(
            composition_wt=bulk_df_comp,
            temperature_c=float(T_aging),
            precip_phase=precip_phase,
        )
    except Exception as exc:
        out["tc_error"] = repr(exc)
        return _fail("tc_driving_force_failed_bulk", out)

    try:
        df_surf = tc_backend.calculate_driving_force(
            composition_wt=surface_df_comp,
            temperature_c=float(T_aging),
            precip_phase=precip_phase,
        )
    except Exception as exc:
        out["tc_error"] = repr(exc)
        return _fail("tc_driving_force_failed_surface", out)

    out["DF_bulk_tc"] = df_bulk
    out["DF_surf_tc"] = df_surf
    out["delta_DF_surf_minus_bulk"] = df_surf - df_bulk

    if mode == "surface_enhanced":
        if require_positive_surface_df and df_surf <= 0.0:
            return _fail("surface_df_nonpositive", out)

        if (df_surf - df_bulk) < delta_df_min:
            return _fail("surface_not_enhanced_tc", out)

    elif mode == "bulk_dominant":
        if require_positive_bulk_df and df_bulk <= 0.0:
            return _fail("bulk_df_nonpositive", out)

        if (df_bulk - df_surf) < delta_df_min:
            return _fail("bulk_not_dominant_tc", out)

    else:
        return _fail(f"invalid_mode_{mode}", out)

    # --------------------------------------------------------
    # Coarsening inputs for fitness
    # --------------------------------------------------------
    fitness_cfg = kwargs.get("fitness_config", {})
    coarsening_cfg = fitness_cfg.get("coarsening", {})

    coarsening_elements = coarsening_cfg.get("elements", [])
    matrix_phase = coarsening_cfg.get(
        "matrix_phase",
        phases.get("matrix_bcc", "BCC_A2"),
    )
    precipitate_phase = coarsening_cfg.get(
        "precipitate_phase",
        phases.get("laves", "C14_LAVES"),
    )

    if coarsening_elements:
        try:
            eq_result = tc_backend.calculate_equilibrium_result(
                composition_wt=bulk,
                temperature_c=float(T_aging),
            )

            out["coarsening_x_matrix"] = tc_backend.get_phase_mole_fractions(
                result=eq_result,
                phase_name=matrix_phase,
                elements=coarsening_elements,
            )

            out["coarsening_x_laves"] = tc_backend.get_phase_mole_fractions(
                result=eq_result,
                phase_name=precipitate_phase,
                elements=coarsening_elements,
            )

            out["coarsening_diffusivities"] = tc_backend.get_diffusivities(
                result=eq_result,
                phase_name=matrix_phase,
                elements=coarsening_elements,
                dependent_element="Fe",
            )

        except Exception as exc:
            out["tc_error"] = repr(exc)
            return _fail("tc_coarsening_inputs_failed", out)

    return _pass(out)