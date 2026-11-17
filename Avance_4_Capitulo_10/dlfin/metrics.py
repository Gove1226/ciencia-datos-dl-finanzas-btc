"""Métricas en la escala original del rendimiento. error = estimado - real."""
from __future__ import annotations

import numpy as np


def _chk(real, est):
    real, est = np.asarray(real, dtype=float), np.asarray(est, dtype=float)
    if real.shape != est.shape or real.ndim != 1:
        raise ValueError(f"Formas incompatibles: real={real.shape}, estimado={est.shape}.")
    if len(real) == 0:
        raise ValueError("No hay datos para calcular la métrica (conjunto vacío).")
    if not (np.isfinite(real).all() and np.isfinite(est).all()):
        raise ValueError("Se detectaron valores no finitos en reales o estimados.")
    return real, est


def mae(real, est) -> float:
    r, e = _chk(real, est)
    return float(np.mean(np.abs(e - r)))


def rmse(real, est) -> float:
    r, e = _chk(real, est)
    return float(np.sqrt(np.mean((e - r) ** 2)))


def precision_direccional(real, est) -> float:
    """Proporción de signos coincidentes. Tratamiento de ceros: se excluyen los ejemplos con real==0
    o estimado==0 (no tienen signo). Si no queda ninguno devuelve NaN (métrica no definida)."""
    r, e = _chk(real, est)
    m = (r != 0) & (e != 0)
    if not m.any():
        return float("nan")
    return float(np.mean(np.sign(r[m]) == np.sign(e[m])))


def resumen(real, est) -> dict:
    return {"mae": mae(real, est), "rmse": rmse(real, est), "dir_acc": precision_direccional(real, est), "n": len(real)}
