from math import ceil, log2
from typing import Dict, Optional
import random

# ============================================================
# Genetic encoder for the self-healing ferritic steel project
# ============================================================
#
# Encoded genes:
#   Cr, Mn, Si, Mo, Nb, Ti, Ni, Cu, T_sol, T_aging
#
# Fixed:
#   C = 0.05
#
# Derived:
#   Fe = 100 - sum(all alloying elements including C)
#
# Notes:
# - Binary encoding is used only as the genotype representation
#   for GA operators (mutation / crossover / storage).
# - All physical evaluations must use decoded values in wt%.
# - Decoding always returns values quantised to the allowed grid.
# - All returned numeric values are rounded to 2 decimals.
# ============================================================

ORDER = [
    "Cr",
    "Mn",
    "Si",
    "W",
    "Mo",
    "Nb",
    "Ti",
    "Ni",
    "Cu",
    "T_sol",
    "T_aging",
]

RANGES = {
    "Cr":      (14.0, 20.0, 0.25),
    "Mn":      (0.0,  2.0,  0.1),
    "Si":      (0.0,  2.0,  0.1),
    "W":       (0.0,  5.0,  0.1),
    "Mo":      (0.0,  5.0,  0.1),
    "Nb":      (0.0,  3.0,  0.1),
    "Ti":      (0.0,  3.0,  0.1),
    "Ni":      (0.0,  6.0,  0.1),
    "Cu":      (0.0,  3.0,  0.1),
    "T_sol":   (900, 1300, 20),
    "T_aging": (550, 750, 20),
}

FIXED_VALUES = {
    "C": 0.05,
}

ROUND_DECIMALS = 2

# Exact maximum grid index for each variable
QMAX = {
    k: int(round((hi - lo) / step))
    for k, (lo, hi, step) in RANGES.items()
}

# Minimum number of bits required to represent each variable
BITS = {
    k: max(1, ceil(log2(QMAX[k] + 1)))
    for k in ORDER
}

TOTAL_BITS = sum(BITS[k] for k in ORDER)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _round_value(x: float) -> float:
    return round(float(x), ROUND_DECIMALS)


def _qparams(k: str):
    lo, hi, step = RANGES[k]
    qmax = QMAX[k]
    bits = BITS[k]
    return lo, hi, step, qmax, bits


def _value_to_index(k: str, value: float) -> int:
    lo, hi, step, qmax, _ = _qparams(k)
    v = _clip(float(value), lo, hi)
    q = int(round((v - lo) / step))
    return max(0, min(q, qmax))


def _index_to_value(k: str, q: int) -> float:
    lo, _, step, qmax, _ = _qparams(k)
    q = max(0, min(int(q), qmax))
    value = lo + q * step
    return _round_value(value)


def _compute_fe_balance(vals: Dict[str, float]) -> float:
    total = FIXED_VALUES["C"]
    for k in ["Cr", "Mn", "Si", "Mo", "Nb", "Ti", "W", "Ni", "Cu"]:
        total += float(vals.get(k, 0.0))
    return _round_value(100.0 - total)


def sanitize_values(vals: Dict[str, float]) -> Dict[str, float]:
    """
    Clip and quantise all encoded variables to the exact project grid.
    Add fixed C and computed Fe.
    All values are rounded to ROUND_DECIMALS.
    """
    out: Dict[str, float] = {}

    for k in ORDER:
        q = _value_to_index(k, vals.get(k, RANGES[k][0]))
        out[k] = _index_to_value(k, q)

    out.update({k: _round_value(v) for k, v in FIXED_VALUES.items()})
    out["Fe"] = _compute_fe_balance(out)
    return out


def validate_composition(vals: Dict[str, float], fe_min: Optional[float] = None) -> Dict[str, float]:
    """
    Basic validation helper for later AG use.
    Returns flags; does not modify the input.
    """
    fe = _compute_fe_balance(vals)
    valid_nonnegative_fe = fe >= 0.0
    valid_fe_min = True if fe_min is None else (fe >= fe_min)

    return {
        "valid_nonnegative_fe": valid_nonnegative_fe,
        "valid_fe_min": valid_fe_min,
        "Fe": fe,
    }


def encode(values: Dict[str, float]) -> int:
    """
    Encode a candidate into a packed integer using binary representation.
    """
    clean = sanitize_values(values)

    code = 0
    for k in ORDER:
        q = _value_to_index(k, clean[k])
        code = (code << BITS[k]) | q

    return code


def decode(code: int) -> Dict[str, float]:
    """
    Decode a packed integer into quantised project variables.
    Returns bulk composition in wt%, fixed C, and Fe balance.
    """
    x = int(code)
    out: Dict[str, float] = {}

    for k in reversed(ORDER):
        bits = BITS[k]
        mask = (1 << bits) - 1
        q = x & mask
        x >>= bits
        out[k] = _index_to_value(k, q)

    out.update({k: _round_value(v) for k, v in FIXED_VALUES.items()})
    out["Fe"] = _compute_fe_balance(out)
    return out


def sanitize_code(code: int) -> int:
    """
    Decode -> sanitise -> re-encode.
    Useful after crossover/mutation at bit level.
    """
    return encode(decode(code))


def to_bitstring(code: int, width: Optional[int] = None) -> str:
    """
    Convert encoded integer to binary string.
    """
    nbits = TOTAL_BITS if width is None else int(width)
    return format(int(code), f"0{nbits}b")


def from_bitstring(bitstring: str) -> int:
    """
    Convert binary string to encoded integer.
    """
    s = bitstring.strip().replace(" ", "")
    if not s:
        raise ValueError("Empty bitstring.")
    if any(c not in "01" for c in s):
        raise ValueError("Bitstring must contain only '0' and '1'.")
    return int(s, 2)


def random_valid_code(rng: Optional[random.Random] = None) -> int:
    """
    Generate a random encoded individual on the valid discretised grid.
    """
    r = rng or random
    vals = {}

    for k in ORDER:
        q = r.randint(0, QMAX[k])
        vals[k] = _index_to_value(k, q)

    return encode(vals)


def random_valid_values(rng: Optional[random.Random] = None) -> Dict[str, float]:
    """
    Generate random decoded values directly.
    """
    return decode(random_valid_code(rng=rng))


def describe_layout() -> Dict[str, Dict[str, float]]:
    """
    Useful for debugging and documentation.
    Returns bit allocation and discretisation metadata.
    """
    out = {}
    offset = TOTAL_BITS

    for k in ORDER:
        bits = BITS[k]
        lo, hi, step = RANGES[k]
        offset -= bits
        out[k] = {
            "bits": bits,
            "qmax": QMAX[k],
            "lo": lo,
            "hi": hi,
            "step": step,
            "bit_start": offset,
            "bit_end": offset + bits - 1,
        }

    return out