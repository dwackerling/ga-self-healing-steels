# test_encoder.py
# Valida encode128/decode128: error <= paso/2, saturación y fuzzing.
import math, random
from collections import defaultdict

# Importa tus funciones/tablas desde binary_encoder.py
from binary_encoder import encode128, decode128, RANGES, BITS

EPS = 1e-12

def step_of(k): return RANGES[k][2]
def lo_of(k):   return RANGES[k][0]
def hi_of(k):   return RANGES[k][1]

def base_midpoint():
    """Vector base con el punto medio cuantizado de cada variable."""
    vals = {}
    for k,(lo,hi,step) in RANGES.items():
        mid = lo + 0.5*(hi-lo)
        # cuantiza al tick más cercano
        q = round((mid - lo)/step)
        vals[k] = lo + q*step
    return vals

def round_trip_error(vals):
    """Devuelve errores absolutos por campo tras encode->decode."""
    code = encode128(vals)
    dec  = decode128(code)
    err = {k: abs(vals[k] - dec[k]) for k in RANGES.keys()}
    return err, code, dec

def assert_within_half_step(errs):
    """Verifica |error| <= step/2 + EPS para todos los campos."""
    for k,e in errs.items():
        step = step_of(k)
        assert e <= (step/2 + EPS), f"{k}: error {e} > {step/2}"

def test_round_trip_min_mid_max():
    print(">> Round-trip: min / mid / max por campo")
    base = base_midpoint()
    maxima = defaultdict(float)
    for k in RANGES.keys():
        lo, hi, step = RANGES[k]
        for v in [lo, base[k], hi]:
            vals = dict(base); vals[k] = v
            errs, code, dec = round_trip_error(vals)
            assert_within_half_step(errs)
            for kk,e in errs.items(): maxima[kk] = max(maxima[kk], e)
    print("OK. Máx |error| observado por campo (min/mid/max):")
    for k in RANGES.keys():
        print(f"  {k:6s}: {maxima[k]:.6f}  (step/2={step_of(k)/2:.6f})")

def test_saturation():
    print("\n>> Saturación: valores fuera de rango → clip + cuantización")
    base = base_midpoint()
    for k in RANGES.keys():
        lo, hi, step = RANGES[k]
        for v in [lo - 10*step, hi + 10*step]:
            vals = dict(base); vals[k] = v
            errs, code, dec = round_trip_error(vals)

            # 1) El campo saturado k debe quedar a ≤ step/2 del límite clippeado
            target = lo if v < lo else hi
            e_bound = abs(dec[k] - target)
            assert e_bound <= (step/2 + EPS), f"{k}: saturación {e_bound} > {step/2}"

            # 2) Todos los demás campos deben cumplir ≤ step/2
            errs_others = {kk: e for kk, e in errs.items() if kk != k}
            assert_within_half_step(errs_others)
    print("OK. Saturación verificada (≤ step/2 en límite y sin efecto colateral).")


def test_fuzz(n=2000, seed=42):
    print(f"\n>> Fuzzing aleatorio: {n} muestras uniformes en rango")
    random.seed(seed)
    maxima = defaultdict(float)
    for _ in range(n):
        vals = {}
        for k,(lo,hi,step) in RANGES.items():
            v = random.uniform(lo, hi)
            vals[k] = v
        errs, code, dec = round_trip_error(vals)
        assert_within_half_step(errs)
        for k,e in errs.items(): maxima[k] = max(maxima[k], e)
        # idempotencia en la malla: decode -> encode -> decode debe fijar
        dec2 = decode128(encode128(dec))
        for k in RANGES.keys():
            assert abs(dec2[k] - dec[k]) <= EPS + step_of(k)*1e-9, f"{k}: no idempotente"
    print("OK. Fuzzing pasó. Máx |error| observado:")
    for k in RANGES.keys():
        print(f"  {k:6s}: {maxima[k]:.6f}  (step/2={step_of(k)/2:.6f})")

def main():
    test_round_trip_min_mid_max()
    test_saturation()
    test_fuzz()
    print("\n=== TODO OK: encoder/decoder cumplen tolerancias (≤ step/2) ===")

if __name__ == "__main__":
    main()
