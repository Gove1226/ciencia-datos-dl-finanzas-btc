"""Validación temporal (ventana expansiva) y simulación de umbrales (Avance 3, capítulo 9).

Todo es NumPy/pandas: no depende de TensorFlow, de modo que se puede probar sin GPU.
El entrenamiento profundo (que sí exige la T4) se hace en el notebook.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .errors import DatosError, FugaError, VentanaError

# ---------------------------------------------------------------------------
# Ventana expansiva dentro de TR
# ---------------------------------------------------------------------------


@dataclass
class VentanaExpansiva:
    k: int
    idx_train: np.ndarray   # posiciones (en `ej`) de los ejemplos de entrenamiento
    idx_eval: np.ndarray    # posiciones (en `ej`) del bloque que se evalúa


def ventanas_expansivas(idx_tr: np.ndarray, n_ventanas: int = 3, frac_inicial: float = 0.40,
                        purga: int = 1) -> list[VentanaExpansiva]:
    """Divide TR (ordenado por fecha del objetivo) en `n_ventanas` evaluaciones con entrenamiento creciente.

    El entrenamiento de la ventana k usa los primeros `a_k` ejemplos; se omiten `purga` ejemplos
    y se evalúa el bloque siguiente. Los bloques de evaluación tienen igual tamaño y no se traslapan.
    """
    n = len(idx_tr)
    if n_ventanas < 1 or not (0.1 <= frac_inicial < 0.9) or purga < 0:
        raise VentanaError("Parámetros de ventana expansiva inválidos.")
    n0 = int(n * frac_inicial)
    bloque = (n - n0 - n_ventanas * purga) // n_ventanas
    if n0 < 30 or bloque < 30:
        raise VentanaError(f"TR ({n} ejemplos) es insuficiente para {n_ventanas} ventanas: entrenamiento inicial={n0}, bloque={bloque}.")
    out = []
    for k in range(n_ventanas):
        a = n0 + k * (bloque + purga)
        out.append(VentanaExpansiva(k + 1, idx_tr[:a], idx_tr[a + purga: a + purga + bloque]))
    return out


def verificar_ventanas(ej, ventanas: list[VentanaExpansiva], idx_tr: np.ndarray) -> None:
    """Cada entrenamiento termina antes del bloque que evalúa; todo permanece dentro de TR."""
    dentro_tr = set(map(int, idx_tr))
    previo_fin = None
    for v in ventanas:
        if not (set(map(int, v.idx_train)) <= dentro_tr and set(map(int, v.idx_eval)) <= dentro_tr):
            raise FugaError(f"Ventana {v.k}: hay ejemplos fuera de TR.")
        if set(map(int, v.idx_train)) & set(map(int, v.idx_eval)):
            raise FugaError(f"Ventana {v.k}: entrenamiento y evaluación comparten ejemplos.")
        ult_obj_train = ej.fecha_objetivo[v.idx_train].max()
        prim_emision_eval = ej.fecha_emision[v.idx_eval].min()
        if not ult_obj_train < prim_emision_eval:
            raise FugaError(f"Ventana {v.k}: el último objetivo del entrenamiento ({ult_obj_train.date()}) "
                            f"no es anterior a la primera emisión evaluada ({prim_emision_eval.date()}).")
        if previo_fin is not None and not ej.fecha_objetivo[v.idx_eval].min() > previo_fin:
            raise FugaError(f"Ventana {v.k}: los bloques de evaluación se traslapan.")
        previo_fin = ej.fecha_objetivo[v.idx_eval].max()


def tabla_ventanas(ej, ventanas: list[VentanaExpansiva]) -> pd.DataFrame:
    filas = []
    for v in ventanas:
        filas.append({
            "ventana": v.k,
            "n_entrenamiento": len(v.idx_train), "n_evaluacion": len(v.idx_eval),
            "entrenamiento_obj_inicial": ej.fecha_objetivo[v.idx_train].min().date(),
            "entrenamiento_obj_final": ej.fecha_objetivo[v.idx_train].max().date(),
            "evaluacion_emision_inicial": ej.fecha_emision[v.idx_eval].min().date(),
            "evaluacion_obj_inicial": ej.fecha_objetivo[v.idx_eval].min().date(),
            "evaluacion_obj_final": ej.fecha_objetivo[v.idx_eval].max().date(),
        })
    return pd.DataFrame(filas)


def division_interna(idx_train: np.ndarray, frac_val: float = 0.20, purga: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Parte cronológica del entrenamiento de una ventana: (ajuste, parada temprana). Sin tocar el bloque evaluado."""
    n = len(idx_train)
    corte = int(n * (1 - frac_val))
    if corte < 30 or n - corte - purga < 10:
        raise VentanaError(f"Entrenamiento de {n} ejemplos insuficiente para separar un tramo de parada temprana.")
    return idx_train[:corte], idx_train[corte + purga:]


# ---------------------------------------------------------------------------
# Simulación: posición comprada o efectivo, ejecución en la apertura de t+1
# ---------------------------------------------------------------------------


def max_drawdown(patrimonio) -> float:
    """Mayor caída porcentual desde un máximo previo (valor positivo; 0 si nunca cae)."""
    p = np.asarray(patrimonio, dtype=float)
    if p.size == 0:
        raise DatosError("Serie de patrimonio vacía.")
    pico = np.maximum.accumulate(p)
    return float(np.max((pico - p) / pico))


def simular(fecha_ejec, precio_open, precio_close, senal, capital: float = 100_000.0,
            costo: float = 0.001, liquidar_final: bool = True) -> tuple[pd.DataFrame, dict]:
    """Simula una estrategia largo/efectivo.

    La señal del ejemplo i se emitió al cierre de t y se EJECUTA en la apertura de t+1 (`precio_open[i]`).
    El patrimonio se valora al cierre de t+1 (`precio_close[i]`). El costo (`costo` = 0.10 %) se cobra sobre
    el monto negociado solo cuando la posición cambia; la compra usa todo el efectivo disponible incluyendo
    el costo. Sin ventas en corto ni apalancamiento. Si `liquidar_final`, se vende al último cierre
    (con costo) para poder comparar estrategias en la misma fecha.
    """
    fecha_ejec = pd.DatetimeIndex(fecha_ejec)
    op = np.asarray(precio_open, dtype=float)
    cl = np.asarray(precio_close, dtype=float)
    sg = np.asarray(senal).astype(bool)
    if not (len(fecha_ejec) == len(op) == len(cl) == len(sg)):
        raise DatosError(f"Longitudes distintas: fechas={len(fecha_ejec)}, open={len(op)}, close={len(cl)}, señal={len(sg)}.")
    if len(sg) == 0:
        raise DatosError("Simulación sin datos (conjunto vacío).")
    if not (np.isfinite(op).all() and np.isfinite(cl).all() and (op > 0).all() and (cl > 0).all()):
        raise DatosError("Precios no finitos o no positivos en la simulación.")
    if capital <= 0 or not (0 <= costo < 0.5):
        raise DatosError("Capital o costo inválidos.")

    efectivo, unidades = float(capital), 0.0
    costos_acum, n_compras, n_ventas = 0.0, 0, 0
    efectivo_antes_compra = None
    cierres = []              # beneficio neto de cada operación cerrada
    dias_pos = 0
    filas = []
    for i in range(len(sg)):
        costo_dia = 0.0
        if sg[i] and unidades == 0.0:                       # comprar con todo el efectivo, incluyendo el costo
            efectivo_antes_compra = efectivo
            unidades = efectivo / (op[i] * (1 + costo))
            costo_dia = unidades * op[i] * costo
            efectivo, n_compras = 0.0, n_compras + 1
        elif (not sg[i]) and unidades > 0.0:                # vender toda la posición
            monto = unidades * op[i]
            costo_dia = monto * costo
            efectivo = monto - costo_dia
            cierres.append(efectivo - efectivo_antes_compra)
            unidades, n_ventas = 0.0, n_ventas + 1
        costos_acum += costo_dia
        dias_pos += int(unidades > 0.0)
        filas.append({"fecha": fecha_ejec[i], "precio_ejecucion": op[i], "precio_cierre": cl[i], "senal": int(sg[i]),
                      "posicion_unidades": unidades, "efectivo": efectivo, "costo_dia": costo_dia,
                      "costos_acumulados": costos_acum, "patrimonio": efectivo + unidades * cl[i]})
    tabla = pd.DataFrame(filas)

    patrimonio_final = float(tabla["patrimonio"].iloc[-1])
    if liquidar_final and unidades > 0.0:                    # liquidación final al último cierre, con costo de venta
        monto = unidades * cl[-1]
        c = monto * costo
        efectivo = monto - c
        cierres.append(efectivo - efectivo_antes_compra)
        costos_acum += c
        n_ventas += 1
        patrimonio_final = float(efectivo)
        tabla.loc[tabla.index[-1], ["posicion_unidades", "efectivo", "patrimonio", "costos_acumulados"]] = [0.0, efectivo, patrimonio_final, costos_acum]
        tabla.loc[tabla.index[-1], "costo_dia"] += c

    serie = np.concatenate([[capital], tabla["patrimonio"].to_numpy()])
    res = {
        "patrimonio_final": patrimonio_final,
        "rendimiento_neto": patrimonio_final / capital - 1,
        "max_drawdown": max_drawdown(serie),
        "n_compras": n_compras, "n_ventas": n_ventas,
        "n_operaciones_cerradas": len(cierres),
        "win_rate": (float(np.mean(np.asarray(cierres) > 0)) if cierres else float("nan")),
        "costos_totales": float(costos_acum),
        "dias_en_posicion": dias_pos,
    }
    return tabla, res


def evaluar_umbrales(pred, fecha_ejec, precio_open, precio_close, umbrales, **kw) -> pd.DataFrame:
    """Para cada umbral δ: comprar si el rendimiento estimado supera δ; en otro caso, efectivo."""
    pred = np.asarray(pred, dtype=float)
    if not np.isfinite(pred).all():
        raise DatosError("Hay predicciones no finitas; no se puede definir la señal.")
    if len(umbrales) > 5 or any(u <= 0 for u in umbrales):
        raise DatosError("Se admiten hasta cinco umbrales y todos deben ser positivos.")
    filas = []
    for u in umbrales:
        _, r = simular(fecha_ejec, precio_open, precio_close, pred > u, **kw)
        filas.append({"umbral": u, "senales_compra": int((pred > u).sum()), **r})
    return pd.DataFrame(filas)


def elegir_umbral(tabla: pd.DataFrame) -> pd.Series:
    """Criterio fijado de antemano: mayor rendimiento neto en VA; en empate, menor drawdown;
    si persiste, el umbral más bajo (la regla menos restrictiva)."""
    t = tabla.assign(_r=tabla["rendimiento_neto"].round(10), _d=tabla["max_drawdown"].round(10))
    t = t.sort_values(["_r", "_d", "umbral"], ascending=[False, True, True])
    return t.iloc[0].drop(["_r", "_d"])
