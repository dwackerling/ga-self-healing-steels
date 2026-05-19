from math import ceil, log2
import numpy as np

ORDER = [
    'C','B','N',
    'Mn','Si','Cr','Mo','V','Ni','Ti','Al','Nb','W',
    't_hom',
    'version','flags','crc8','spare'
]

RANGES = {
    'C':   (0.002, 0.1, 0.001),
    'B':   (0.0,  0.007, 0.001),
    'N':   (0.0,  0.045, 0.001),
    'Mn':  (0.45, 5.06,  0.01),
    'Si':  (0.0,  0.18,  0.01),
    'Cr':  (8.0,  18.2,  0.01),
    'Mo':  (0.46, 6.64,  0.01),
    'V':   (0.0,  0.18,  0.01),
    'Ni':  (0.0,  0.15,  0.01),
    'Ti':  (0.0,  0.21,  0.01),
    'Al':  (0.0004, 2.92, 0.01),
    'Nb':  (0.018,  2.92, 0.01),
    'W':   (0.52,  6.0,  0.01),
    't_hom': (950, 1250, 5),   
}

# Bits fijos (pueden exceder Q_max; recortamos en encode/decode)
BITS = {
    'C':7,'B':3,'N':6,
    'Mn':9,'Si':5,'Cr':10,'Mo':10,'V':5,'Ni':4,'Ti':5,'Al':9,'Nb':9,'W':10,
    't_hom':7,
    'version':8,'flags':8,'crc8':8,'spare':128-(99+8+8+8)
}

def _clip(x, lo, hi):
    return max(lo, min(hi, x))

# ===== NEW: exact max index (no depende del número de bits, sino de (hi-lo)/step) =====
QMAX = {
    k: int(round((RANGES[k][1] - RANGES[k][0]) / RANGES[k][2]))
    for k in RANGES
}

def _qparams(k):
    lo, hi, step = RANGES[k]; Lbits = (1 << BITS[k]) - 1
    qmax = QMAX[k]
    return lo, hi, step, Lbits, qmax

def _clip(x, lo, hi): 
    return max(lo, min(hi, x))

def encode128(values: dict) -> int:
    code = 0
    for k in ORDER:
        if k in RANGES:
            lo, hi, step, Lbits, qmax = _qparams(k)
            v = float(values.get(k, lo))
            v = _clip(v, lo, hi)
            q = int(round((v - lo)/step))
            # ¡clamp a qmax REAL, no a todos los bits!
            q = max(0, min(q, qmax))
        elif k == 'version':
            q = int(values.get('version', 0)) & ((1<<BITS[k])-1)
        elif k == 'flags':
            q = int(values.get('flags', 0)) & ((1<<BITS[k])-1)
        elif k == 'crc8':
            q = 0
        else:  # spare
            q = 0
        code = (code << BITS[k]) | q
    return code

def decode128(code: int) -> dict:
    out = {}
    x = int(code) & ((1<<128)-1)
    for k in reversed(ORDER):
        b = BITS[k]; m = (1<<b)-1
        q = x & m; x >>= b
        if k in RANGES:
            lo, hi, step, Lbits, qmax = _qparams(k)
            # ¡clamp del índice decodificado!
            q = int(min(q, qmax))
            out[k] = lo + q*step
        else:
            out[k] = q
    return out

# ===== NEW: sanitize helpers =====
def sanitize_vals(vals: dict) -> dict:
    """Clip + cuantizar cada campo a su rejilla exacta."""
    fix = dict(vals)
    for k in RANGES:
        lo, hi, step, Lbits, qmax = _qparams(k)
        v = float(fix.get(k, lo))
        v = _clip(v, lo, hi)
        q = int(round((v - lo)/step))
        q = max(0, min(q, qmax))
        fix[k] = lo + q*step
    return fix

def sanitize_code(code: int) -> int:
    """Decode → clamp indices → re-encode (garantiza rangos válidos tras crossover bit a bit)."""
    return encode128(sanitize_vals(decode128(code)))

def random_valid_code(rng=None) -> int:
    """Útil para inicializar población con puntos 100% válidos en la rejilla."""
    import random as _r
    r = rng or _r
    vals = {}
    for k in RANGES:
        lo, hi, step, Lbits, qmax = _qparams(k)
        q = r.randint(0, qmax)
        vals[k] = lo + q*step
    # opcionales
    vals['version'] = 0
    vals['flags']   = 0
    vals['crc8']    = 0
    return encode128(vals)