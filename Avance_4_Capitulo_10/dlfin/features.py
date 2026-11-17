"""Objetivo, indicadores técnicos, ventanas, partición TR/VA/TE y escalador.

Convención de fechas: el ejemplo se EMITE al cierre del día t (fecha_emision) y
pronostica y_t = P_{t+1}/P_t - 1 (fecha_objetivo = t+1). Todas las entradas usan
solo historia disponible hasta el cierre de t.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .errors import DatosError, FugaError, VentanaError

CANALES_BASE = ("ret", "dist_sma20", "vol20")
CATALOGO = {
    "ret": "Rendimiento simple del cierre P_t/P_{t-1}-1.",
    "dist_sma20": "Distancia a la SMA de 20 cierres: cierre/SMA20 - 1.",
    "vol20": "Desviación estándar muestral (ddof=1) de los últimos 20 rendimientos, sin anualizar.",
    "rsi14": "RSI de 14 periodos, suavizado de Wilder (EMA alpha=1/14). Sin pérdidas y con ganancias=100; sin ganancias ni pérdidas=50 (neutral).",
    "macd": "EMA12(cierre) - EMA26(cierre), normalizado por el cierre.",
    "macd_hist": "(MACD - señal EMA9 del MACD) normalizado por el cierre.",
    "bb_pos": "Posición dentro de bandas de Bollinger (SMA20 ± 2 desv. de cierres): (cierre-SMA)/(2*desv); 0 si desv=0.",
    "atr14_rel": "ATR de 14 (Wilder) del rango verdadero, dividido por el cierre. Requiere máximo y mínimo.",
}


def indicadores(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula indicadores solo con historia hasta t (rolling/ewm causales, sin rellenar hacia atrás)."""
    if len(df) == 0:
        raise DatosError("Conjunto vacío al calcular indicadores.")
    c = df["close"].astype(float)
    out = pd.DataFrame({"fecha": df["fecha"].values, "close": c.values})
    r = c.pct_change()
    out["ret"] = r.values
    sma20 = c.rolling(20).mean()
    out["sma20"] = sma20.values
    out["dist_sma20"] = (c / sma20 - 1).values
    out["vol20"] = r.rolling(20).std(ddof=1).values

    delta = c.diff()
    ag = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    al = (-delta).clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = ag / al.where(al != 0)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(~((al == 0) & (ag > 0)), 100.0)
    rsi = rsi.where(~((al == 0) & (ag == 0)), 50.0)
    rsi = rsi.where(ag.notna() & al.notna())
    out["rsi14"] = rsi.values

    ema12 = c.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = c.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd"] = (macd / c).values
    out["macd_hist"] = ((macd - sig) / c).values

    sd = c.rolling(20).std(ddof=1)
    out["bb_pos"] = ((c - sma20) / (2 * sd.where(sd != 0))).fillna(0.0).where(sma20.notna()).values

    if {"high", "low"} <= set(df.columns):
        h, l, cp = df["high"].astype(float), df["low"].astype(float), c.shift(1)
        tr = pd.concat([h - l, (h - cp).abs(), (l - cp).abs()], axis=1).max(axis=1)
        tr.iloc[0] = (h - l).iloc[0]
        atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        out["atr14_rel"] = (atr / c).values
    return out


def primera_fecha_valida(ind: pd.DataFrame, canales=tuple(CATALOGO)) -> pd.DataFrame:
    filas = []
    for ch in canales:
        if ch in ind.columns:
            v = ind[ch].notna() & np.isfinite(ind[ch].astype(float))
            i = int(v.values.argmax()) if v.any() else -1
            filas.append({"indicador": ch, "primer_indice_valido": i,
                          "primera_fecha_valida": str(ind["fecha"].iloc[i].date()) if i >= 0 else None,
                          "definicion": CATALOGO[ch]})
    return pd.DataFrame(filas)


@dataclass
class Ejemplos:
    X: np.ndarray                  # (n, T, F)
    y: np.ndarray                  # (n,)
    fecha_emision: pd.DatetimeIndex
    fecha_objetivo: pd.DatetimeIndex
    canales: tuple
    ventana: int
    idx_emision: np.ndarray        # posición de t dentro del DataFrame de precios
    futuro: dict | None = None     # ejemplo sin objetivo observable (pronóstico futuro)

    def __len__(self) -> int:
        return len(self.y)

    def sel(self, canales) -> np.ndarray:
        pos = [self.canales.index(c) for c in canales]
        return self.X[:, :, pos]

    def subconjunto(self, idx) -> "Ejemplos":
        return Ejemplos(self.X[idx], self.y[idx], self.fecha_emision[idx], self.fecha_objetivo[idx],
                        self.canales, self.ventana, self.idx_emision[idx], None)


def _primer_indice_comun(ind: pd.DataFrame, canales) -> int:
    i0 = 0
    for ch in canales:
        v = ind[ch].astype(float)
        ok = v.notna() & np.isfinite(v)
        if not ok.any():
            raise VentanaError(f"El indicador '{ch}' no tiene valores válidos; datos insuficientes.")
        i0 = max(i0, int(ok.values.argmax()))
    return i0


def construir_ejemplos(df: pd.DataFrame, ventana: int = 20, canales=CANALES_BASE,
                       canales_para_inicio=None) -> Ejemplos:
    """Construye ventanas (n, ventana, F). `canales_para_inicio` fuerza un periodo evaluable común
    entre versiones con y sin indicadores."""
    if ventana < 1:
        raise VentanaError(f"La ventana debe ser >= 1 (recibido {ventana}).")
    ind = indicadores(df)
    faltan = [c for c in canales if c not in ind.columns]
    if faltan:
        raise DatosError(f"Canales no disponibles: {faltan}. Disponibles: {list(ind.columns)}.")
    i0 = _primer_indice_comun(ind, canales_para_inicio or canales)
    t_min = i0 + ventana - 1
    N = len(ind)
    if N - 1 - t_min < 10:
        raise VentanaError(f"Ventanas insuficientes: {N} filas, ventana={ventana}, inicio={t_min}. Se requieren más datos.")
    M = ind[list(canales)].to_numpy(dtype=float)
    close = ind["close"].to_numpy(dtype=float)
    ts = np.arange(t_min, N - 1)  # t con objetivo observable
    X = np.stack([M[t - ventana + 1: t + 1] for t in ts])
    y = close[ts + 1] / close[ts] - 1
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise DatosError("Se generaron entradas u objetivos no finitos; revise indicadores y datos.")
    fechas = pd.DatetimeIndex(ind["fecha"])
    fut = {
        "fecha_emision": fechas[N - 1],
        "X": M[N - ventana: N][None, :, :],
        "close_t": float(close[N - 1]),
    }
    if not np.isfinite(fut["X"]).all():
        fut = None
    return Ejemplos(X, y, fechas[ts], fechas[ts + 1], tuple(canales), ventana, ts, fut)


def particionar(ej: Ejemplos, prop=(0.70, 0.15, 0.15)) -> dict[str, np.ndarray]:
    """Partición cronológica por fecha del objetivo (nunca aleatoria)."""
    n = len(ej)
    if n < 30:
        raise VentanaError(f"Ejemplos insuficientes para particionar: {n}.")
    orden = np.argsort(ej.fecha_objetivo.values, kind="stable")
    a, b = int(n * prop[0]), int(n * (prop[0] + prop[1]))
    idx = {"TR": orden[:a], "VA": orden[a:b], "TE": orden[b:]}
    verificar_particion(ej, idx)
    return idx


def verificar_particion(ej: Ejemplos, idx: dict) -> None:
    f = ej.fecha_objetivo
    if not (f[idx["TR"]].max() < f[idx["VA"]].min() and f[idx["VA"]].max() < f[idx["TE"]].min()):
        raise FugaError("Los objetivos de TR/VA/TE se traslapan en el tiempo: partición inválida.")
    if (ej.fecha_emision >= ej.fecha_objetivo).any():
        raise FugaError("Hay ejemplos con fecha de emisión >= fecha objetivo.")


def tabla_particiones(ej: Ejemplos, idx: dict) -> pd.DataFrame:
    filas = []
    for k, i in idx.items():
        filas.append({
            "conjunto": k, "n_ejemplos": len(i),
            "emision_inicial": ej.fecha_emision[i].min().date(), "emision_final": ej.fecha_emision[i].max().date(),
            "objetivo_inicial": ej.fecha_objetivo[i].min().date(), "objetivo_final": ej.fecha_objetivo[i].max().date(),
        })
    return pd.DataFrame(filas)


@dataclass
class EscaladorTR:
    """Escalador por canal para X y estandarización de y; se ajusta SOLO con TR."""
    canales: tuple
    sx: StandardScaler = field(default_factory=StandardScaler)
    y_mu: float = 0.0
    y_sd: float = 1.0
    n_ajuste: int = 0
    fecha_max_objetivo_ajuste: str | None = None

    def ajustar(self, X_tr: np.ndarray, y_tr: np.ndarray, fecha_obj_tr=None) -> "EscaladorTR":
        if X_tr.ndim != 3 or X_tr.shape[2] != len(self.canales):
            raise VentanaError(f"Forma esperada (n, T, {len(self.canales)}), recibida {X_tr.shape}.")
        self.sx.fit(X_tr.reshape(-1, X_tr.shape[2]))
        self.y_mu = float(np.mean(y_tr))
        self.y_sd = float(np.std(y_tr, ddof=0)) or 1.0
        self.n_ajuste = len(y_tr)
        if fecha_obj_tr is not None:
            self.fecha_max_objetivo_ajuste = str(pd.Timestamp(max(fecha_obj_tr)).date())
        return self

    def x(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 3 or X.shape[2] != len(self.canales):
            raise VentanaError(f"Forma esperada (n, T, {len(self.canales)}), recibida {X.shape}.")
        n, T, F = X.shape
        return self.sx.transform(X.reshape(-1, F)).reshape(n, T, F).astype(np.float32)

    def y(self, y: np.ndarray) -> np.ndarray:
        return ((y - self.y_mu) / self.y_sd).astype(np.float32)

    def y_inv(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=float) * self.y_sd + self.y_mu

    def guardar(self, ruta) -> None:
        joblib.dump(self, ruta)

    @staticmethod
    def cargar(ruta) -> "EscaladorTR":
        return joblib.load(ruta)


def verificar_ajuste_solo_TR(esc: EscaladorTR, ej: Ejemplos, idx: dict) -> None:
    if esc.n_ajuste != len(idx["TR"]):
        raise FugaError(f"El escalador se ajustó con {esc.n_ajuste} ejemplos y TR tiene {len(idx['TR'])}.")
    lim = ej.fecha_objetivo[idx["TR"]].max()
    if esc.fecha_max_objetivo_ajuste and pd.Timestamp(esc.fecha_max_objetivo_ajuste) > lim:
        raise FugaError("El escalador vio objetivos posteriores al final de TR.")
