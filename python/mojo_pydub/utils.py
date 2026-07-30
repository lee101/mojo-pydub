from __future__ import annotations

import math


def db_to_float(db: float, using_amplitude: bool = True) -> float:
    return 10 ** (float(db) / (20 if using_amplitude else 10))


def ratio_to_db(
    ratio: float, val2: float | None = None, using_amplitude: bool = True
) -> float:
    ratio = float(ratio)
    if val2 is not None:
        ratio /= val2
    if ratio == 0:
        return -float("inf")
    return (20 if using_amplitude else 10) * math.log(ratio, 10)
