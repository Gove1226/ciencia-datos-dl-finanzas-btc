"""Simulación histórica long-only con costos, ejecución en t+1 y comparación con efectivo / comprar y mantener.

Reglas (protocolo, sección 9):
  * Capital inicial 100,000; posición comprada (100% del capital) o efectivo.
  * Señal emitida al cierre de t se ejecuta en la APERTURA de t+1 (si hay apertura).
  * Costo por compra y por venta sobre el monto negociado; solo se cobra al cambiar de posición.
  * Patrimonio valorado al cierre de cada periodo; al final se liquida al último cierre cobrando la venta.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .errors import DatosError


def max_drawdown(patrimonio) -> float:
    p = np.asarray(patrimonio, dtype=float)
    if len(p) == 0:
        raise ValueError("Serie de patrimonio vacía.")
    pico = np.maximum.accumulate(p)
    return float(((p - pico) / pico).min())


def simular(precios: pd.DataFrame, fecha_emision, senal, capital: float = 100_000.0, costo: float = 0.001,
            usar_apertura: bool = True) -> tuple[pd.DataFrame, dict]:
    """`precios` con columnas fecha, open, close. `fecha_emision` y `senal` (bool) son iguales en longitud;
    la ejecución es el día siguiente a la emisión (siguiente fila de `precios`)."""
    fechas = pd.DatetimeIndex(fecha_emision)
    senal = np.asarray(senal, dtype=bool)
    if len(fechas) != len(senal) or len(fechas) == 0:
        raise DatosError(f"Fechas ({len(fechas)}) y señales ({len(senal)}) deben ser no vacías y de igual tamaño.")
    p = precios.set_index("fecha")
    pos = {d: i for i, d in enumerate(p.index)}
    usar_apertura = usar_apertura and "open" in p.columns

    efectivo, acciones = float(capital), 0.0
    costos_tot, filas, trades = 0.0, [], []
    inversion = 0.0  # efectivo gastado en la compra abierta (incluye costo)
    for d_emi, s in zip(fechas, senal):
        i = pos.get(d_emi)
        if i is None or i + 1 >= len(p):
            raise DatosError(f"No hay precio de ejecución posterior a la emisión {d_emi.date()}.")
        d_ej = p.index[i + 1]
        px = float(p["open"].iloc[i + 1] if usar_apertura else p["close"].iloc[i + 1])
        accion = "mantener"
        if s and acciones == 0:
            monto = efectivo / (1 + costo)          # el costo sale del mismo efectivo
            c = monto * costo
            acciones, efectivo = monto / px, 0.0
            costos_tot += c
            inversion = monto + c
            accion = "compra"
        elif (not s) and acciones > 0:
            monto = acciones * px
            c = monto * costo
            efectivo, acciones = monto - c, 0.0
            costos_tot += c
            trades.append(efectivo - inversion)
            accion = "venta"
        cierre = float(p["close"].iloc[i + 1])
        filas.append({"fecha_emision": d_emi, "fecha_ejecucion": d_ej, "precio_ejecucion": px, "senal": int(s),
                      "accion": accion, "posicion": acciones, "efectivo": efectivo, "costos_acum": costos_tot,
                      "precio_cierre": cierre, "patrimonio": efectivo + acciones * cierre})
    # liquidación final al último cierre
    if acciones > 0:
        monto = acciones * filas[-1]["precio_cierre"]
        c = monto * costo
        efectivo, acciones = monto - c, 0.0
        costos_tot += c
        trades.append(efectivo - inversion)
        filas.append({**filas[-1], "accion": "liquidacion_final", "precio_ejecucion": filas[-1]["precio_cierre"],
                      "posicion": 0.0, "efectivo": efectivo, "costos_acum": costos_tot, "patrimonio": efectivo})
    sim = pd.DataFrame(filas)
    pat = sim["patrimonio"].to_numpy()
    res = {
        "patrimonio_final": float(pat[-1]), "rendimiento_neto": float(pat[-1] / capital - 1),
        "max_drawdown": max_drawdown(np.r_[capital, pat]), "operaciones": len(trades),
        "win_rate": (float(np.mean(np.array(trades) > 0)) if trades else float("nan")),
        "costos_totales": float(costos_tot),
    }
    return sim, res


def comprar_y_mantener(precios, fecha_emision, capital=100_000.0, costo=0.001, usar_apertura=True):
    """Compra en la primera fecha de ejecución y mantiene hasta el último cierre, con costos equivalentes."""
    n = len(fecha_emision)
    return simular(precios, fecha_emision, np.ones(n, dtype=bool), capital, costo, usar_apertura)


def efectivo(capital=100_000.0) -> dict:
    return {"patrimonio_final": capital, "rendimiento_neto": 0.0, "max_drawdown": 0.0, "operaciones": 0,
            "win_rate": float("nan"), "costos_totales": 0.0}


def elegir_umbral(precios, fecha_emision, pred, candidatos, filtro=None, capital=100_000.0, costo=0.001):
    """Elige δ (hasta 5 candidatos positivos) por rendimiento neto en VA; en empate, menor drawdown
    (es decir, el max_drawdown menos negativo). `filtro` es un vector bool opcional (confirmación técnica)."""
    cand = [float(c) for c in candidatos]
    if not (0 < len(cand) <= 5) or any(c <= 0 for c in cand):
        raise ValueError("Use entre 1 y 5 umbrales candidatos positivos.")
    pred = np.asarray(pred, dtype=float)
    filas = []
    for d in cand:
        s = pred > d
        if filtro is not None:
            s = s & np.asarray(filtro, dtype=bool)
        _, r = simular(precios, fecha_emision, s, capital, costo)
        filas.append({"delta": d, **r})
    tabla = pd.DataFrame(filas)
    orden = tabla.sort_values(["rendimiento_neto", "max_drawdown"], ascending=[False, False])
    return float(orden.iloc[0]["delta"]), tabla
