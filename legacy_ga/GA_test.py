# ============================================================
# GA NSGA-II (uint128) + encoder + STUBS (sin Thermo-Calc)
# ============================================================

import os, csv, time, random
import numpy as np
from deap import base, creator, tools

# ---------------------------
# Encoder (uint128)
# ---------------------------

ORDER = [
    'C','B','N',
    'Mn','Si','Cr','Mo','V','Ni','Ti','Al','Nb','W',
    't_hom',
    'version','flags','crc8','spare'
]

RANGES = {
    'C':(0.05,0.126,0.001),
    'B':(0.0,0.007,0.001),
    'N':(0.0,0.045,0.001),
    'Mn':(0.45,5.06,0.01),
    'Si':(0.0,0.18,0.01),
    'Cr':(8.0,18.2,0.01),
    'Mo':(0.46,6.64,0.01),
    'V':(0.0,0.18,0.01),
    'Ni':(0.0,0.15,0.01),
    'Ti':(0.0,0.21,0.01),
    'Al':(0.0004,2.92,0.01),
    'Nb':(0.018,2.92,0.01),
    'W':(0.52,6.0,0.01),
    't_hom':(900,1400,5),
}

BITS = {
    'C':7,'B':3,'N':6,
    'Mn':9,'Si':5,'Cr':10,'Mo':10,'V':5,'Ni':4,'Ti':5,'Al':9,'Nb':9,'W':10,
    't_hom':7,
    'version':8,'flags':8,'crc8':8,'spare':128-(99+8+8+8)
}

UINT_BITS, UINT_MASK = 128, (1<<128)-1

def _qparams(k):
    lo, hi, step = RANGES[k]; L = (1<<BITS[k]) - 1
    return lo, hi, step, L

def _clip(x, lo, hi): return max(lo, min(hi, x))

def encode128(values: dict) -> int:
    code = 0
    for k in ORDER:
        if k in RANGES:
            lo, hi, step, L = _qparams(k)
            v = float(values.get(k, lo))
            v = _clip(v, lo, hi)
            q = int(round((v - lo)/step))
            q = _clip(q, 0, L)
        elif k == 'version':
            q = int(values.get('version', 0)) & ((1<<BITS[k])-1)
        elif k == 'flags':
            q = int(values.get('flags', 0)) & ((1<<BITS[k])-1)
        elif k == 'crc8':
            q = 0
        else:
            q = 0
        code = (code << BITS[k]) | q
    return code

def decode128(code: int) -> dict:
    out, x = {}, int(code) & ((1<<128)-1)
    for k in reversed(ORDER):
        b = BITS[k]; m = (1<<b)-1
        q = x & m; x >>= b
        if k in RANGES:
            lo, hi, step, L = _qparams(k)
            out[k] = lo + q*step
        else:
            out[k] = q
    return out

import copy

def _varAnd(pop, toolbox, cxpb, mutpb):
    # clonar
    offspring = [copy.deepcopy(ind) for ind in pop]

    # cruce 2 a 2
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            c1, c2 = toolbox.mate(offspring[i-1], offspring[i])
            offspring[i-1], offspring[i] = c1, c2
            if hasattr(offspring[i-1].fitness, "values"):
                del offspring[i-1].fitness.values
            if hasattr(offspring[i].fitness, "values"):
                del offspring[i].fitness.values

    # mutación
    for i in range(len(offspring)):
        if random.random() < mutpb:
            (mutant,) = toolbox.mutate(offspring[i])
            offspring[i] = mutant
            if hasattr(offspring[i].fitness, "values"):
                del offspring[i].fitness.values

    return offspring


# ---------------------------
# GA helpers
# ---------------------------

SUBS_ORDER = tuple(k for k in ORDER if k in RANGES and k != "t_hom")

MUT_RADIUS = {
    "C":1, "B":1, "N":1,
    "Si":2, "Ni":2, "Ti":2, "V":2, "Al":2, "Nb":2,
    "Mn":3, "Mo":3, "Cr":3, "W":3,
    "t_hom":1
}

def cx_two_point_uint128(a: int, b: int) -> tuple[int, int]:
    i, j = sorted(random.sample(range(UINT_BITS), 2))
    mask = ((1 << (j - i)) - 1) << i
    return ((a & ~mask) | (b & mask)) & UINT_MASK, ((b & ~mask) | (a & mask)) & UINT_MASK

def mut_fieldwise_reencode(code: int, indpb: float = 0.20) -> int:
    vals = decode128(code); changed = False
    for k, r in MUT_RADIUS.items():
        if k not in vals: continue
        if random.random() < indpb:
            step = RANGES[k][2] if k in RANGES else 0.0
            if step <= 0: continue
            dq = random.randint(-r, r)
            if dq != 0:
                vals[k] = round(vals[k] + dq*step, 6)
                changed = True
    return encode128(vals) if changed else code

# --- CSV header helpers (fixed schema) ---
GNG_KEYS_BASE = [
    "reason", "LIQ", "BCC_A2@t_hom", "BCC_A2", "Cr_matrix_wt",
    "FCC_A1#2", "M23C6_D84", "D_Fe", "D_Mo", "D_W",
    "D_Mo_minus_D_Fe", "D_W_minus_D_Fe"
]

def build_csv_fieldnames():
    base = [
        "ts","gen","phase","eval_id","code_int","code_hex",
        "fit_df_surface","fit_neg_df_bulk",
        "DF_bulk","IE_bulk","DF_surface","IE_surface",
        "ok_go_no_go","T_eval_C","elapsed_s","from_cache",
    ]
    vals = [f"val_{k}" for k in ORDER]
    bulks = [f"bulk_{e}" for e in SUBS_ORDER]
    gng = [f"gng_{k}" for k in GNG_KEYS_BASE]
    return base + vals + bulks + gng

class ResultLogger:
    def __init__(self, path: str, fieldnames=None, flush_every: int = 100):
        self.path, self.flush_every = path, flush_every
        self._fh = self._writer = None; self._n = 0
        self._fieldnames = list(fieldnames) if fieldnames else None

    def _open(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._fh = open(self.path, "a", newline="", encoding="utf-8")
        # si no dieron fieldnames, usaremos las del primer row (no recomendado)
        if self._fieldnames is None:
            raise RuntimeError("ResultLogger requires fixed fieldnames; call with fieldnames=build_csv_fieldnames()")
        self._writer = csv.DictWriter(self._fh, fieldnames=self._fieldnames)
        if self._fh.tell() == 0:
            self._writer.writeheader()

    def write(self, row: dict):
        if not self._writer:
            self._open()
        # asegurar que sólo se escriben columnas conocidas; faltantes -> ""
        safe_row = {k: row.get(k, "") for k in self._fieldnames}
        self._writer.writerow(safe_row); self._n += 1
        if self._n % self.flush_every == 0:
            self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.flush(); self._fh.close(); self._fh = self._writer = None

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
        self.cache = {}
        self.current_gen = 0
        self.current_phase = "init"
        self.eval_counter = 0
        fields = build_csv_fieldnames() if log_path else None
        self.logger = ResultLogger(log_path, fieldnames=fields, flush_every=100) if log_path else None
    def close(self):
        if self.logger: self.logger.close()

def _build_bulk_wt(vals: dict) -> dict:
    return {e: float(vals.get(e, 0.0)) for e in SUBS_ORDER}

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

# ---------------------------
# STUBS (sin Thermo-Calc)
# ---------------------------

def _seed_from_comp(bulk_wt: dict, t_hom_C: float, T_eval_C: float) -> int:
    items = tuple(sorted((k.upper(), float(v)) for k, v in bulk_wt.items()))
    return abs(hash((items, round(t_hom_C, 3), round(T_eval_C, 3)))) & 0xFFFFFFFF

def go_no_go_bulk(
    calc, calc_diff,
    *,
    bulk_wt: dict,
    t_hom_C: float,
    T_eval_C: float,
    matrix_phase: str = "BCC_A2",
    element_superset=("C","N","B","Si","Ni","Ti","Mn","Mo","Cr","V","W","Al","Nb"),
    **kwargs
):
    wt = {k.upper(): float(bulk_wt.get(k, 0.0)) for k in element_superset}
    rng = np.random.default_rng(_seed_from_comp(wt, t_hom_C, T_eval_C))
    BCC_hom = 0.995
    if BCC_hom < 0.99:
        return False, {"reason": "BCC_A2 too low at t_hom", "BCC_A2@t_hom": BCC_hom}
    ferrite = 0.90 + 0.05*rng.random()
    if ferrite < 0.90:
        return False, {"reason": f"{matrix_phase} too low at T_eval", matrix_phase: ferrite}
    if wt.get("CR", 0.0) < 8.0:
        return False, {"reason": "Cr in matrix too low at T_eval", "Cr_matrix_wt": wt.get("CR", 0.0)/100.0}
    fcc2 = min(0.20, (wt.get("NI", 0.0) + wt.get("MN", 0.0)) / 100.0)
    if fcc2 < 0.05:
        return False, {"reason": "FCC_A1#2 below minimum at T_eval", "FCC_A1#2": fcc2}
    m23c6 = 1e-7 if not (wt.get("C", 0.0) > 0.11 and wt.get("CR", 0.0) > 16.0) else 1e-3
    if m23c6 > 1e-6:
        return False, {"reason": "M23C6 present at T_eval", "M23C6_D84": m23c6}
    D_Fe = 1.0
    D_Mo = 1.0 + 0.05*wt.get("MO", 0.0)
    D_W  = 0.8 + 0.03*wt.get("W",  0.0)
    ok_diff = (D_Mo - D_Fe) > 0.0 or (D_W - D_Fe) > 0.0
    if not ok_diff:
        return False, {
            "reason": "diffusion criterion failed",
            "D_Fe": D_Fe, "D_Mo": D_Mo, "D_W": D_W,
            "D_Mo_minus_D_Fe": D_Mo - D_Fe,
            "D_W_minus_D_Fe": D_W - D_Fe
        }
    return True, {"LIQ": 0.0, "BCC_A2@t_hom": BCC_hom, matrix_phase: ferrite,
                  "Cr_matrix_wt": wt.get("CR", 0.0)/100.0, "FCC_A1#2": fcc2, "M23C6_D84": m23c6}

def _seed_from_vec(elements, comps_wt, T_celsius):
    key = tuple(zip([e.upper() for e in elements], [float(x) for x in comps_wt])) + (round(T_celsius, 3),)
    return abs(hash(key)) & 0xFFFFFFFF

def calculate_driving_interfacial(
    calc_df, calc_ie,
    *, elements, comps_wt,
    T_celsius: float = 650.0,
    matrix_phase: str = "BCC_A2",
    precip_phase: str = "C14_LAVES",
):
    rng = np.random.default_rng(_seed_from_vec(elements, comps_wt, T_celsius))
    cr = 0.01 * sum(c for e, c in zip(elements, comps_wt) if e.upper() == "CR")
    mo = 0.01 * sum(c for e, c in zip(elements, comps_wt) if e.upper() == "MO")
    w  = 0.01 * sum(c for e, c in zip(elements, comps_wt) if e.upper() == "W")
    DF = float(min(0.6, 0.15 + 0.25*cr + 0.15*mo + 0.10*w + 0.05*rng.random()))
    IE = float(0.25 + 0.25*rng.random())
    return DF, IE

_INTERS = {"C","N","B"}

def _rng_from_comp(bulk_wt: dict, seed=None):
    if seed is not None:
        return np.random.default_rng(int(seed) & 0xFFFFFFFF)
    key = tuple(sorted((k.upper(), float(v)) for k, v in bulk_wt.items()))
    return np.random.default_rng(abs(hash(key)) & 0xFFFFFFFF)

def compute_surface_composition_from_bulk(
    *,
    bulk_wt: dict,
    db_path: str = "",
    masses: dict | None = None,
    seed: int | None = None,
    avg_top_layers=None,
    **kwargs
):
    wt = {k.upper(): float(v) for k, v in bulk_wt.items()}
    subs = {k: v for k, v in wt.items() if k not in _INTERS and v > 0.0}
    sum_subs = sum(subs.values())
    fe = max(0.0, 100.0 - sum_subs)
    subs["FE"] = subs.get("FE", 0.0) + fe
    elems = list(subs.keys())
    x = np.array([subs[e] for e in elems], dtype=float)
    total = x.sum()
    if total <= 0:
        return {f"SURF_{e}": 0.0 for e in wt if e not in _INTERS} | {"SURF_ENERGY": 0.0}
    rng = _rng_from_comp(wt, seed=seed)
    eps = 0.02
    deltas = rng.normal(0.0, 1.0, size=x.size)
    deltas -= deltas.mean()
    deltas *= eps * x
    y = x + deltas
    y = np.clip(y, 0.0, None)
    if y.sum() == 0:
        y = x.copy()
    y *= total / y.sum()
    res = {f"SURF_{e}": float(v) for e, v in zip(elems, y)}
    res["SURF_ENERGY"] = -float(np.var(y/total))
    for k in _INTERS:
        if k in wt:
            res[k] = wt[k]
    return res

# ---------------------------
# Fitness + NSGA-II
# ---------------------------

def eval_individual_uint_with_row(code: int, ctx: Context):
    cached = ctx.cache.get(code)
    if cached is not None:
        fit, row = cached
        row = dict(row); row["from_cache"] = True
        return fit, row

    t0 = time.time()
    vals = decode128(code)
    bulk_wt = _build_bulk_wt(vals)
    t_hom = float(vals["t_hom"])

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

    seed = code & 0xFFFFFFFF
    surf = compute_surface_composition_from_bulk(
        bulk_wt=bulk_wt, db_path=ctx.db_path, masses=ctx.masses, seed=seed, avg_top_layers=None
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

def run_ga_uint128(ctx: Context, pop_size=40, ngen=5, cxpb=0.9, mutpb=0.25, seed=42):
    random.seed(seed)
    if not hasattr(creator, "FitnessMO"):
        creator.create("FitnessMO", base.Fitness, weights=(1.0, 1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", int, fitness=creator.FitnessMO)
    toolbox = base.Toolbox()
    toolbox.register("attr_code", lambda: random.getrandbits(UINT_BITS))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_code)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def mate(a, b):
        a2, b2 = cx_two_point_uint128(int(a), int(b))
        return creator.Individual(a2), creator.Individual(b2)
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

    pop = toolbox.population(n=pop_size)
    ctx.current_gen, ctx.current_phase = 0, "init"
    invalid = [ind for ind in pop if not ind.fitness.valid]
    fits = list(map(toolbox.evaluate, invalid))
    for ind, fit in zip(invalid, fits):
        ind.fitness.values = fit
    pop = toolbox.select(pop, len(pop))

    for gen in range(1, ngen + 1):
        ctx.current_gen = gen
        offspring = _varAnd(pop, toolbox, cxpb=cxpb, mutpb=mutpb)
        ctx.current_phase = f"offspring_gen{gen}"
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fits = list(map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fits):
            ind.fitness.values = fit
        pop = toolbox.select(pop + offspring, pop_size)
        if gen % 1 == 0:
            f1 = np.array([ind.fitness.values[0] for ind in pop])
            f2 = np.array([ind.fitness.values[1] for ind in pop])
            feasible = np.mean(f1 > -1e5)
            print(f"[Gen {gen}] DF_surf max={f1.max():.3f}  DF_bulk min={-f2.min():.3f}  ok≈{feasible:.2f}")

    pareto = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    pareto.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
    return pop, pareto

# ---------------------------
# MAIN (sin Thermo-Calc)
# ---------------------------

if __name__ == "__main__":
    masses = {"FE":55.845,"CR":52.0,"MN":54.938,"NI":58.693,"MO":95.95,"V":50.942,"W":183.84,
              "SI":28.085,"TI":47.867,"AL":26.982,"NB":92.906,"C":12.011,"B":10.81,"N":14.007}
    db_path = "steel_database_fix.tdb"

    ctx = Context(calc=None, calc_diff=None, calc_df=None, calc_ie=None,
                  masses=masses, db_path=db_path,
                  T_eval_C=650.0, precip_phase="C14_LAVES", matrix_phase="BCC_A2",
                  log_path="runs/ga_stub_results.csv")

    pop, pareto = run_ga_uint128(ctx, pop_size=40, ngen=500, cxpb=0.9, mutpb=0.25, seed=42)

    # --- dump Pareto front to CSV ---
    if ctx.logger:
        for i, ind in enumerate(pareto, 1):
            code = int(ind)
            vals = decode128(code)
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
