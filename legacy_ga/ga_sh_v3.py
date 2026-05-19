# ga_nsga2_uint128.py
import csv, os, time, random
import numpy as np
from deap import base, creator, tools, algorithms
import copy

# --- Your modules ---
from binary_encoder import encode128 as enc, decode128 as dec, ORDER, RANGES
from binary_encoder import sanitize_code   # <--- NEW import
from binary_encoder import random_valid_code
from go_no_go import go_no_go_bulk
from df_ie import calculate_driving_interfacial
from mc_sim import compute_surface_composition_from_bulk

# =========================
# Config (elements from your encoder; exclude t_hom)
# =========================
SUBS_ORDER = tuple(k for k in ORDER if k in RANGES and k != "t_hom")
UINT_BITS, UINT_MASK = 128, (1 << 128) - 1

# mutation radii in ticks (use your quantization steps from RANGES)
MUT_RADIUS = {
    "C":1, "B":1, "N":1,
    "Si":2, "Ni":2, "Ti":2, "V":2, "Al":2, "Nb":2,
    "Mn":3, "Mo":3, "Cr":3, "W":3,
    "t_hom":1
}

# =========================
# GA operators (uint128)
# =========================
# def cx_two_point_uint128(a: int, b: int) -> tuple[int, int]:
#     i, j = sorted(random.sample(range(UINT_BITS), 2))
#     mask = ((1 << (j - i)) - 1) << i
#     c1 = ((a & ~mask) | (b & mask)) & UINT_MASK
#     c2 = ((b & ~mask) | (a & mask)) & UINT_MASK
#     # Sanear ambos hijos antes de devolver
#     return sanitize_code(c1), sanitize_code(c2)

def cx_uniform_fields(code_a: int, code_b: int, p: float = 0.5):
    A, B = dec(code_a), dec(code_b)             # dicts con valores reales por gen
    for k in RANGES.keys():                     # solo los genes cuantizados
        if random.random() < p:
            A[k], B[k] = B[k], A[k]
    return sanitize_code(enc(A)), sanitize_code(enc(B))


def mut_fieldwise_reencode(code: int, indpb: float = 0.20) -> int:
    vals = dec(code)
    changed = False
    for k, r in MUT_RADIUS.items():
        if k not in vals:
            continue
        if random.random() < indpb:
            step = RANGES[k][2] if k in RANGES else 0.0
            if step <= 0: 
                continue
            dq = random.randint(-r, r)
            if dq != 0:
                vals[k] = vals[k] + dq*step
                changed = True
    # Sanear SIEMPRE tras mutar
    return sanitize_code(enc(vals)) if changed else sanitize_code(code)

# =========================
# CSV logger
# =========================
class ResultLogger:
    def __init__(self, path: str, fieldnames=None, flush_every: int = 100):
        self.path, self.flush_every = path, flush_every
        self._fh = self._writer = None; self._n = 0
        self._fieldnames = list(fieldnames) if fieldnames else None

    def _open(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._fh = open(self.path, "a", newline="", encoding="utf-8")
        if self._fieldnames is None:
            raise RuntimeError("ResultLogger requires fixed fieldnames; pass build_csv_fieldnames()")
        self._writer = csv.DictWriter(self._fh, fieldnames=self._fieldnames)
        if self._fh.tell() == 0:
            self._writer.writeheader()

    def write(self, row: dict):
        if not self._writer:
            self._open()
        safe = {k: row.get(k, "") for k in self._fieldnames}
        self._writer.writerow(safe); self._n += 1
        if self._n % self.flush_every == 0:
            self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.flush(); self._fh.close(); self._fh = self._writer = None

# --- CSV header fijo ---
GNG_KEYS_BASE = [
    "reason","LIQ","BCC_A2@t_hom","BCC_A2","Cr_matrix_wt",
    "FCC_A1#2","M23C6_D84","D_Fe","D_Mo","D_W",
    "D_Mo_minus_D_Fe","D_W_minus_D_Fe"
]

def build_csv_fieldnames():
    base = [
        "ts","gen","phase","eval_id","code_int","code_hex",
        "fit_df_surface","fit_neg_df_bulk",
        "DF_bulk","IE_bulk","DF_surface","IE_surface",
        "ok_go_no_go","T_eval_C","elapsed_s","from_cache",
    ]
    vals  = [f"val_{k}" for k in ORDER]
    bulks = [f"bulk_{e}" for e in (k for k in ORDER if k in RANGES and k!="t_hom")]
    gng   = [f"gng_{k}" for k in GNG_KEYS_BASE]
    return base + vals + bulks + gng

# =========================
# Context (external handles)
# =========================
class Context:
    def __init__(self,
                 calc, calc_diff, calc_df, calc_ie,
                 masses: dict, db_path: str,
                 T_eval_C=650.0, precip_phase="C14_LAVES", matrix_phase="BCC_A2",
                 log_path: str | None = None):
        self.calc, self.calc_diff = calc, calc_diff
        self.calc_df, self.calc_ie = calc_df, calc_ie
        self.masses, self.db_path = masses, db_path
        self.T_eval_C, self.precip_phase, self.matrix_phase = T_eval_C, precip_phase, matrix_phase

        self.cache = {}                 # code -> (fitness_tuple, row_dict)
        self.current_gen = 0
        self.current_phase = "init"     # "init" or f"offspring_gen{gen}"
        self.eval_counter = 0

        
        fields = build_csv_fieldnames() if log_path else None
        self.logger = ResultLogger(log_path, fieldnames=fields, flush_every=100) if log_path else None

    def close(self):
        if self.logger: self.logger.close()

# =========================
# Fitness (with logging)
# Constraints:
#   - go/no-go True (difusión, BCC_A2@t_hom≥0.99, LIQ=0, M23C6 suprimida, FCC_A1#2≥5%, etc.)
#   - DF_bulk < 0.5
#   - IE_bulk > 0.25
#   - DF_surface > 0
# Objectives: (maximize DF_surface, minimize DF_bulk) => (DF_surface, -DF_bulk)
# =========================

def _build_bulk_wt(vals: dict) -> dict:
    return {e: float(vals.get(e, 0.0)) for e in SUBS_ORDER}  # Fe balance is handled inside the thermo functions

def _make_row(code, vals, ctx: Context, fit, *,
              DF_bulk, IE_bulk, DF_surf, IE_surf, gng, ok_gng, t0):
    flat_vals = {f"val_{k}": vals[k] for k in ORDER if k in vals}
    bulk = {f"bulk_{e}": float(vals.get(e, 0.0)) for e in SUBS_ORDER}
    gng_flat = {f"gng_{k}": v for k, v in (gng or {}).items()}

    row = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gen": ctx.current_gen,
        "phase": ctx.current_phase,
        "eval_id": ctx.eval_counter,
        "code_int": int(code),
        "code_hex": hex(int(code)),
        "fit_df_surface": fit[0],
        "fit_neg_df_bulk": fit[1],
        "DF_bulk": "" if DF_bulk is None else DF_bulk,
        "IE_bulk": "" if IE_bulk is None else IE_bulk,
        "DF_surface": "" if DF_surf is None else DF_surf,
        "IE_surface": "" if IE_surf is None else IE_surf,
        "ok_go_no_go": bool(ok_gng),
        "T_eval_C": ctx.T_eval_C,
        "elapsed_s": round(time.time() - t0, 6),
        "from_cache": False,
    }
    row.update(flat_vals); row.update(bulk); row.update(gng_flat)
    return row

def eval_individual_uint_with_row(code: int, ctx: Context):
    cached = ctx.cache.get(code)
    if cached is not None:
        fit, row = cached
        row = dict(row); row["from_cache"] = True
        return fit, row

    t0 = time.time()
    vals = dec(code)
    bulk_wt = _build_bulk_wt(vals)
    t_hom = float(vals["t_hom"])

    # 1) Go/No-Go
    ok_gng, rep_gng = go_no_go_bulk(
        ctx.calc, ctx.calc_diff,
        bulk_wt=bulk_wt, t_hom_C=t_hom, T_eval_C=ctx.T_eval_C,
        matrix_phase=ctx.matrix_phase, element_superset=SUBS_ORDER
    )
    if not ok_gng:
        fit = (-1e6, -1e6)
        row = _make_row(code, vals, ctx, fit, DF_bulk=None, IE_bulk=None,
                        DF_surf=None, IE_surf=None, gng=rep_gng, ok_gng=False, t0=t0)
        ctx.cache[code] = (fit, row)
        return fit, row

    # 2) DF/IE in bulk @ T_eval
    elems = list(SUBS_ORDER)
    comps = [bulk_wt[e] for e in elems]
    DF_bulk, IE_bulk = calculate_driving_interfacial(
        ctx.calc_df, ctx.calc_ie,
        elements=elems, comps_wt=comps,
        T_celsius=ctx.T_eval_C,
        matrix_phase=ctx.matrix_phase,
        precip_phase=ctx.precip_phase
    )

    if not (DF_bulk < 0.5 and IE_bulk > 0.25):
        fit = (-1e6, -1e6)
        row = _make_row(code, vals, ctx, fit, DF_bulk=DF_bulk, IE_bulk=IE_bulk,
                        DF_surf=None, IE_surf=None, gng=rep_gng, ok_gng=True, t0=t0)
        ctx.cache[code] = (fit, row)
        return fit, row

    # 3) MC → surface composition → DF_surface
    seed = code & 0xFFFFFFFF
    surf = compute_surface_composition_from_bulk(
        bulk_wt=bulk_wt,
        db_path=ctx.db_path,
        masses=ctx.masses,
        seed=seed,
        avg_top_layers=None  # average over ALL layers as requested
    )
    surf_elems, surf_wt = [], []
    for e in SUBS_ORDER:
        k = f"SURF_{e}"
        if k in surf:
            surf_elems.append(e); surf_wt.append(float(surf[k]))

    DF_surf, IE_surf = calculate_driving_interfacial(
        ctx.calc_df, ctx.calc_ie,
        elements=surf_elems, comps_wt=surf_wt,
        T_celsius=ctx.T_eval_C,
        matrix_phase=ctx.matrix_phase,
        precip_phase=ctx.precip_phase
    )
    if not (DF_surf > 0.0):
        fit = (-1e6, -1e6)
        row = _make_row(code, vals, ctx, fit, DF_bulk=DF_bulk, IE_bulk=IE_bulk,
                        DF_surf=DF_surf, IE_surf=IE_surf, gng=rep_gng, ok_gng=True, t0=t0)
        ctx.cache[code] = (fit, row)
        return fit, row

    fit = (float(DF_surf), float(-DF_bulk))
    row = _make_row(code, vals, ctx, fit, DF_bulk=DF_bulk, IE_bulk=IE_bulk,
                    DF_surf=DF_surf, IE_surf=IE_surf, gng=rep_gng, ok_gng=True, t0=t0)
    ctx.cache[code] = (fit, row)
    return fit, row

# =========================
# NSGA-II (DEAP) with logging
# =========================
def run_ga_uint128(ctx: Context, pop_size=120, ngen=220, cxpb=0.90, mutpb=0.15, seed=43):
    random.seed(seed)

    if not hasattr(creator, "FitnessMO"):
        creator.create("FitnessMO", base.Fitness, weights=(1.0, 1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", int, fitness=creator.FitnessMO)

    toolbox = base.Toolbox()
    
    toolbox.register("attr_code", lambda: sanitize_code(random_valid_code()))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_code)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # def mate(a, b):
    #     a2, b2 = cx_two_point_uint128(int(a), int(b))
    #     return creator.Individual(a2), creator.Individual(b2)

    def mate(a, b):
        c1, c2 = cx_uniform_fields(int(a), int(b), p=0.6)  # o p=0.4–0.6
        return creator.Individual(c1), creator.Individual(c2)
    
    def mutate(ind):
        new_code = mut_fieldwise_reencode(int(ind), indpb=0.20)
        return (creator.Individual(new_code),)


    toolbox.register("mate", mate)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selNSGA2)

    def evaluate_and_log(ind):
        ctx.eval_counter += 1
        code = int(ind)
        fit, row = eval_individual_uint_with_row(code, ctx)
        if ctx.logger:
            ctx.logger.write(row)
        return fit

    toolbox.register("evaluate", evaluate_and_log)

    # --- initial population
    pop = toolbox.population(n=pop_size)
    ctx.current_gen, ctx.current_phase = 0, "init"

    invalid = [ind for ind in pop if not ind.fitness.valid]
    fits = list(map(toolbox.evaluate, invalid))
    for ind, fit in zip(invalid, fits):
        ind.fitness.values = fit
    pop = toolbox.select(pop, len(pop))  # initialize crowding

    toolbox.register("clone", copy.deepcopy)  # requerido por algorithms.varAnd

    # --- generations
    for gen in range(1, ngen + 1):
        ctx.current_gen = gen
        
        offspring = algorithms.varAnd(pop, toolbox, cxpb=cxpb, mutpb=mutpb)

        ctx.current_phase = f"offspring_gen{gen}"
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fits = list(map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fits):
            ind.fitness.values = fit

        pop = toolbox.select(pop + offspring, pop_size)

        if gen % 10 == 0 and len(pop):
            f1 = np.array([ind.fitness.values[0] for ind in pop])
            f2 = np.array([ind.fitness.values[1] for ind in pop])
            feasible = float(np.mean(f1 > -1e5)) if len(f1) else 0.0
            df_bulk_min = (-f2).min()  # o bien: -f2.max()
            print(f"[Gen {gen}] DF_surf max={f1.max():.3f}  DF_bulk min={df_bulk_min:.3f}  ok≈{feasible:.2f}")

    pareto = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    pareto.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
    return pop, pareto

# =========================
# Example usage (uncomment when your handles are ready)
# =========================
if __name__ == "__main__":
    import os
    from tc_python import TCPython, CompositionUnit

    # Ruta al .tdb junto al script
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "steel_database_fix.tdb"))

    # Masas atómicas (mc_sim también tiene defaults; aquí las fijamos explícitas)
    masses = {
        "FE":55.845, "CR":52.00, "MN":54.938, "NI":58.693, "MO":95.95, "V":50.942, "W":183.84,
        "SI":28.085, "TI":47.867, "AL":26.982, "NB":92.906, "C":12.011, "B":10.81, "N":14.007
    }

    with TCPython() as session:
        elems = ["Fe"] + list(SUBS_ORDER)

        # --- Sistema base (TCFE12 + MOBFE7) ---
        base_builder = session.select_thermodynamic_and_kinetic_databases_with_elements("TCFE12", "MOBFE7", elems)
        base_sys = base_builder.get_system()

        # Equilibrio para fases (go/no-go, etc.)
        calc = base_sys.with_single_equilibrium_calculation()

        # Difusión (go/no-go)
        calc_diff = base_sys.with_single_equilibrium_calculation()

        # Driving force (bulk/superficie)
        calc_df = base_sys.with_property_model_calculation("Driving force").set_composition_unit(CompositionUnit.MASS_PERCENT)

        # Interfacial energy (matriz BCC_A2 vs precipitado C14_LAVES)
        ie_builder = session.select_thermodynamic_and_kinetic_databases_with_elements("TCFE12", "MOBFE7", elems)
        ie_sys = ie_builder.without_default_phases().select_phase("C14_LAVES").select_phase("BCC_A2").get_system()
        calc_ie = ie_sys.with_property_model_calculation("Interfacial energy").set_composition_unit(CompositionUnit.MASS_PERCENT)

        # Contexto GA
        ctx = Context(
            calc=calc, calc_diff=calc_diff, calc_df=calc_df, calc_ie=calc_ie,
            masses=masses, db_path=db_path,
            T_eval_C=650.0, precip_phase="C14_LAVES", matrix_phase="BCC_A2",
            log_path="runs/ga_results_v3.csv"    # se crea auto si no existe
        )

        # Ejecutar GA
        pop, pareto = run_ga_uint128(ctx, pop_size=250, ngen=150, cxpb=0.9, mutpb=0.3, seed=42)

        # --- guardar frente de Pareto en el CSV ---
        if ctx.logger:
            for i, ind in enumerate(pareto, 1):
                code = int(ind)
                vals = dec(code)
                row = {
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "gen": "pareto",
                    "phase": "final",
                    "eval_id": -1,
                    "code_int": code,
                    "code_hex": hex(code),
                    "fit_df_surface": ind.fitness.values[0],
                    "fit_neg_df_bulk": ind.fitness.values[1],
                    "DF_bulk": "",
                    "IE_bulk": "",
                    "DF_surface": "",
                    "IE_surface": "",
                    "ok_go_no_go": "",
                    "T_eval_C": ctx.T_eval_C,
                    "elapsed_s": 0.0,
                    "from_cache": False,
                }
                for k in ORDER:
                    if k in vals:
                        row[f"val_{k}"] = vals[k]
                ctx.logger.write(row)

        ctx.close()
