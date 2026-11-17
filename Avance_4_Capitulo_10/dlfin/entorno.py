"""Entorno de trading (long-only, BTC) para el futuro agente DQN/DDQN. NO entrena nada: solo define estado, acciones,
restricciones, ejecución y recompensa, y permite verificar que el simulador no regala recompensas.

Línea de tiempo de un paso (t = día de emisión; todo lo del estado es <= cierre de t):
    cierre de t  : el agente observa s_t y elige a_t
    apertura t+1 : se ejecuta a_t (con costo)
    cierre de t+1: se valora el patrimonio; recompensa r = V(cierre t+1) - V(cierre t)  (ya neta de costos)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ACCIONES = {0: "mantener", 1: "comprar", 2: "vender"}
NOMBRES_ESTADO = ["ret_t-4", "ret_t-3", "ret_t-2", "ret_t-1", "ret_t", "pronostico_t", "vol20_t", "frac_efectivo", "frac_posicion"]


@dataclass
class Paso:
    estado: np.ndarray | None
    recompensa: float
    terminado: bool
    info: dict


class EntornoTrading:
    def __init__(self, fecha_emision, close_t, open_next, close_next, ret_hist, vol20, pron,
                 capital: float = 100_000.0, costo: float = 0.001, liquidar_al_final: bool = True):
        self.fecha = pd.DatetimeIndex(fecha_emision)
        self.close_t, self.open_next, self.close_next = (np.asarray(x, float) for x in (close_t, open_next, close_next))
        self.ret_hist, self.vol20, self.pron = np.asarray(ret_hist, float), np.asarray(vol20, float), np.asarray(pron, float)
        self.K = len(self.fecha)
        if not (self.K == len(self.close_t) == len(self.open_next) == len(self.close_next) == len(self.ret_hist) == len(self.vol20) == len(self.pron)):
            raise ValueError("Todas las series del entorno deben tener la misma longitud.")
        if self.K < 2 or self.ret_hist.ndim != 2:
            raise ValueError("Se requieren al menos 2 pasos y ret_hist con forma (pasos, n_retornos).")
        if not all(np.isfinite(a).all() for a in (self.close_t, self.open_next, self.close_next, self.ret_hist, self.vol20, self.pron)):
            raise ValueError("Hay valores no finitos en las series del entorno.")
        if (np.minimum.reduce([self.close_t, self.open_next, self.close_next]) <= 0).any():
            raise ValueError("Hay precios no positivos.")
        # días consecutivos: el cierre de t+1 de un paso es el cierre de t del siguiente (valoración coherente)
        if not np.allclose(self.close_next[:-1], self.close_t[1:]):
            raise ValueError("Los pasos no son consecutivos: close_next[k] debe ser close_t[k+1].")
        self.capital, self.costo, self.liquidar_al_final = float(capital), float(costo), liquidar_al_final
        self.reset()

    # ------------------------------------------------------------------ construcción desde los datos del proyecto
    @staticmethod
    def desde_datos(d: pd.DataFrame, ind: pd.DataFrame, idx_emision, pron, n_ret: int = 5, **kw) -> "EntornoTrading":
        """d: precios (fecha, open, close...); ind: dlfin.features.indicadores(d); idx_emision: posiciones t en d;
        pron: pronóstico para cada emisión (de un modelo ajustado SIN usar ese objetivo)."""
        t = np.asarray(idx_emision)
        ret = ind["ret"].to_numpy()
        hist = np.stack([ret[k - n_ret + 1: k + 1] for k in t])
        return EntornoTrading(d["fecha"].iloc[t].values, d["close"].to_numpy()[t], d["open"].to_numpy()[t + 1], d["close"].to_numpy()[t + 1],
                              hist, ind["vol20"].to_numpy()[t], pron, **kw)

    # ------------------------------------------------------------------ estado y reglas
    def reset(self) -> np.ndarray:
        self.k, self.efectivo, self.cantidad, self.costos, self.hecho = 0, self.capital, 0.0, 0.0, False
        self.recompensas: list[float] = []
        return self.estado()

    def patrimonio_cierre_t(self) -> float:
        return self.efectivo + self.cantidad * self.close_t[self.k]

    def estado(self, k: int | None = None) -> np.ndarray:
        """s_t: SOLO información disponible al cierre de t (rendimientos hasta t, pronóstico emitido en t, volatilidad hasta t,
        efectivo y posición actuales). No contiene precios de t+1."""
        k = self.k if k is None else k
        v = self.efectivo + self.cantidad * self.close_t[k]
        return np.r_[self.ret_hist[k], self.pron[k], self.vol20[k], self.efectivo / v, self.cantidad * self.close_t[k] / v]

    def en_posicion(self) -> bool:
        return self.cantidad > 0

    def acciones_validas(self) -> list[int]:
        """mantener siempre; comprar solo sin posición (y con efectivo); vender solo con posición (sin ventas en corto)."""
        v = [0]
        if not self.en_posicion() and self.efectivo > 0:
            v.append(1)
        if self.en_posicion():
            v.append(2)
        return v

    def descripcion_mantener(self) -> str:
        return "mantener posición (seguir comprado)" if self.en_posicion() else "permanecer en efectivo"

    # ------------------------------------------------------------------ transición
    def step(self, accion: int) -> Paso:
        if self.hecho:
            raise RuntimeError("El episodio terminó: llame a reset().")
        if accion not in ACCIONES:
            raise ValueError(f"Acción desconocida: {accion}. Válidas: {ACCIONES}.")
        k, c = self.k, self.costo
        info = {"fecha_emision": self.fecha[k], "accion_pedida": ACCIONES[accion], "invalida": accion not in self.acciones_validas()}
        if info["invalida"]:
            accion = 0                                    # acción inválida (corto, apalancamiento o recompra): se ignora y se registra
        v_t = self.patrimonio_cierre_t()
        px = self.open_next[k]                            # ejecución en la APERTURA de t+1 (no se conoce al decidir)
        costo_paso = 0.0
        if accion == 1:
            monto = self.efectivo / (1 + c)               # el costo sale del mismo efectivo: nunca se gasta más de lo disponible
            costo_paso = monto * c
            self.cantidad, self.efectivo = monto / px, 0.0
        elif accion == 2:
            monto = self.cantidad * px
            costo_paso = monto * c
            self.efectivo, self.cantidad = monto - costo_paso, 0.0
        terminal = k == self.K - 1
        if terminal and self.liquidar_al_final and self.cantidad > 0:   # misma regla que la simulación: liquidar al último cierre
            monto = self.cantidad * self.close_next[k]
            costo_fin = monto * c
            costo_paso += costo_fin
            self.efectivo, self.cantidad = monto - costo_fin, 0.0
            info["liquidacion_final"] = True
        self.costos += costo_paso
        v_next = self.efectivo + self.cantidad * self.close_next[k]
        r = v_next - v_t
        self.recompensas.append(r)
        info.update(accion_ejecutada=ACCIONES[accion], precio_ejecucion=px, costo=costo_paso, patrimonio_t=v_t, patrimonio_t1=v_next)
        self.k += 1
        self.hecho = terminal
        return Paso(None if terminal else self.estado(), r, terminal, info)

    def patrimonio_final(self) -> float:
        return self.efectivo + self.cantidad * (self.close_next[self.K - 1] if self.hecho else self.close_t[self.k])


def correr_politica(env: EntornoTrading, politica) -> dict:
    """Ejecuta un episodio completo con `politica(estado, env) -> accion` y resume el resultado."""
    s, filas = env.reset(), []
    while True:
        a = politica(s, env)
        p = env.step(a)
        filas.append({**p.info, "recompensa": p.recompensa, "efectivo": env.efectivo, "cantidad": env.cantidad})
        if p.terminado:
            break
        s = p.estado
    tr = pd.DataFrame(filas)
    pat = np.r_[env.capital, tr["patrimonio_t1"].to_numpy()]
    mdd = float(((pat - np.maximum.accumulate(pat)) / np.maximum.accumulate(pat)).min())
    return {"detalle": tr, "patrimonio_final": float(pat[-1]), "rendimiento_neto": float(pat[-1] / env.capital - 1), "max_drawdown": mdd,
            "costos_totales": env.costos, "suma_recompensas": float(tr["recompensa"].sum()),
            "compras": int((tr["accion_ejecutada"] == "comprar").sum()), "ventas": int((tr["accion_ejecutada"] == "vender").sum()
                                                                                      + tr.get("liquidacion_final", pd.Series(False, index=tr.index)).fillna(False).sum())}
