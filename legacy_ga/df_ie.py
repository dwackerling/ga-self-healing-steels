from tc_python import *

def calculate_driving_interfacial(
    calc_df,                # property model: "Driving force", MASS_PERCENT ya seteado
    calc_ie,                # property model: "Interfacial energy", MASS_PERCENT, fases ya seleccionadas
    *,
    elements,               # p.ej. ["C","N","Si","Ni","Ti","Mn","Mo","Cr","V","W"]
    comps_wt,               # wt% alineado con 'elements'
    T_celsius: float = 650.0,
    matrix_phase: str = "BCC_A2",
    precip_phase: str = "C14_LAVES",
    fe_symbol: str = "Fe",
    element_superset=("C","N","B","Si","Ni","Ti","Mn","Mo","Cr","V","W"),
):
    """
    Devuelve (DF, IE). Reasigna SIEMPRE todas las composiciones (incl. 0.0) y la temperatura.
    """
    # --- temperatura ---
    T_K = float(T_celsius) + 273.15
    calc_df.set_temperature(T_K)
    calc_ie.set_temperature(T_K)

    # --- construir mapa de wt% con casing canónico ---
    comps_map = {e: 0.0 for e in element_superset}
    for el, wt in zip(elements, comps_wt):
        if el in comps_map:
            comps_map[el] = float(wt)

    # --- balance Fe ---
    total = sum(comps_map.values())
    if total > 100.0 + 1e-9:
        raise ValueError(f"Sum of provided wt% exceeds 100: {total:.6f}")
    fe_wt = max(0.0, 100.0 - total)

    # --- fijar composiciones (pisar todo cada vez) ---
    calc_df.set_composition(fe_symbol, fe_wt)
    calc_ie.set_composition(fe_symbol, fe_wt)
    for el in element_superset:
        calc_df.set_composition(el, comps_map[el])
        calc_ie.set_composition(el, comps_map[el])

    # --- Driving force ---
    df_result = calc_df.set_argument("precipitate", precip_phase).calculate()
    DF = float(df_result.get_value_of("normalizedDrivingForce"))

    # --- Interfacial energy ---
    ie_result = (
        calc_ie
        .set_argument("matrix", matrix_phase)
        .set_argument("precipitate", precip_phase)
        .calculate()
    )
    IE = float(ie_result.get_value_of("interfacialEnergy"))

    return DF, IE
