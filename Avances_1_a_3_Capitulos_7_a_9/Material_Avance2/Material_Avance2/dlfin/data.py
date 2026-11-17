"""Carga, auditoría y limpieza de datos de precios (diario)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .errors import Bitacora, DatosError

COLUMNAS = ["fecha", "open", "high", "low", "close", "volume"]
PRECIOS = ["open", "high", "low", "close"]


def cargar_precios_db(db_path: str | Path, ticker: str, intervalo: str = "1d") -> pd.DataFrame:
    """Lee precios del activo en modo solo lectura. No modifica la base original."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise DatosError(f"No existe la base de datos '{db_path}'. Verifique la ruta de 'Base de datos/finanzas.db'.")
    try:
        con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
        try:
            df = pd.read_sql(
                "SELECT fecha_sesion AS fecha, open, high, low, close, adj_close, volume "
                "FROM precios WHERE ticker=? AND intervalo=? ORDER BY fecha",
                con,
                params=(ticker, intervalo),
            )
        finally:
            con.close()
    except (sqlite3.Error, pd.errors.DatabaseError) as e:
        raise DatosError(f"No se pudo leer '{ticker}' de {db_path.name}: {e}. Revise que exista la tabla 'precios'.") from e
    if df.empty:
        raise DatosError(
            f"Datos vacíos: fuente={db_path.name}, activo={ticker}, intervalo={intervalo}. "
            "Verifique el ticker o repita la descarga; no se continúa con datos inventados."
        )
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def cargar_descargas_db(db_path: str | Path, ticker: str, intervalo: str = "1d") -> pd.DataFrame:
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        return pd.read_sql(
            "SELECT * FROM descargas WHERE ticker=? AND intervalo=? ORDER BY ejecutado_utc",
            con,
            params=(ticker, intervalo),
        )
    finally:
        con.close()


def cargar_activo_db(db_path: str | Path, ticker: str) -> dict:
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        cur = con.execute("SELECT * FROM activos WHERE ticker=?", (ticker,))
        fila = cur.fetchone()
        cols = [c[0] for c in cur.description]
    finally:
        con.close()
    if fila is None:
        raise DatosError(f"El activo '{ticker}' no está en la tabla 'activos'.")
    return dict(zip(cols, fila))


def validar_columnas(df: pd.DataFrame) -> None:
    if df is None or len(df) == 0:
        raise DatosError("Conjunto vacío: no hay filas para procesar. Verifique la fuente y el activo.")
    faltan = [c for c in ["fecha", "close"] if c not in df.columns]
    if faltan:
        raise DatosError(f"Faltan columnas obligatorias {faltan}. Columnas recibidas: {list(df.columns)}.")


def auditar(df: pd.DataFrame) -> dict:
    """Auditoría sin modificar los datos."""
    validar_columnas(df)
    d = df.copy()
    d["fecha"] = pd.to_datetime(d["fecha"])
    num = d.select_dtypes(include=[np.number])
    precios = [c for c in PRECIOS if c in d.columns]
    ret = d["close"].pct_change()
    mad = (ret - ret.median()).abs().median()
    z_rob = 0.6745 * (ret - ret.median()) / mad if mad else ret * np.nan
    huecos = d["fecha"].sort_values().diff().dt.days.dropna()
    res = {
        "filas": len(d),
        "fecha_min": str(d["fecha"].min().date()),
        "fecha_max": str(d["fecha"].max().date()),
        "fechas_duplicadas": int(d["fecha"].duplicated().sum()),
        "fuera_de_orden": int((d["fecha"].diff().dt.days < 0).sum()),
        "faltantes_por_columna": {c: int(v) for c, v in d.isna().sum().items() if v},
        "infinitos": int(np.isinf(num.to_numpy(dtype=float)).sum()),
        "precios_no_positivos": int((d[precios] <= 0).sum().sum()) if precios else 0,
        "low_mayor_que_high": int((d["low"] > d["high"]).sum()) if {"low", "high"} <= set(d.columns) else 0,
        "volumen_negativo": int((d["volume"] < 0).sum()) if "volume" in d.columns else 0,
        "volumen_cero": int((d["volume"] == 0).sum()) if "volume" in d.columns else 0,
        "huecos_calendario(dias!=1)": int((huecos != 1).sum()),
        "rendimientos_extremos(|z_robusto|>6)": int((z_rob.abs() > 6).sum()),
        "rendimiento_min": float(ret.min()),
        "rendimiento_max": float(ret.max()),
    }
    if "adj_close" in d.columns:
        res["close_distinto_adj_close"] = int((d["close"] != d["adj_close"]).sum())
    return res


def extremos(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    d = df.sort_values("fecha").copy()
    d["rendimiento"] = d["close"].pct_change()
    return d.reindex(d["rendimiento"].abs().sort_values(ascending=False).index).head(n)[["fecha", "close", "rendimiento"]]


def limpiar(df: pd.DataFrame, bitacora: Bitacora | None = None) -> tuple[pd.DataFrame, dict]:
    """Aplica reglas explícitas y devuelve (df_limpio, conteo de filas afectadas).

    Reglas (ninguna usa información futura ni rellena huecos):
      1. Ordenar por fecha.
      2. Fechas duplicadas: conservar la última versión descargada.
      3. Filas con NaN/inf o precios <= 0: eliminar (se registran fechas y columnas).
      4. No se inventa volumen ni se rellenan días sin negociación.
    """
    validar_columnas(df)
    d = df.copy()
    d["fecha"] = pd.to_datetime(d["fecha"])
    rep = {"entrada": len(d)}

    ordenado = d["fecha"].is_monotonic_increasing
    rep["reordenadas"] = 0 if ordenado else int((d["fecha"].values != d["fecha"].sort_values().values).sum())
    d = d.sort_values("fecha", kind="stable")

    dup = int(d["fecha"].duplicated().sum())
    rep["duplicadas_eliminadas"] = dup
    d = d.drop_duplicates("fecha", keep="last")

    cols = [c for c in PRECIOS if c in d.columns]
    malas = ~np.isfinite(d[cols].to_numpy(dtype=float)).all(axis=1) | (d[cols] <= 0).any(axis=1).to_numpy()
    rep["invalidas_eliminadas"] = int(malas.sum())
    if malas.any() and bitacora is not None:
        bitacora.registrar(
            "limpieza",
            "valores no válidos",
            f"{int(malas.sum())} filas con NaN/inf/precio<=0 en {cols}: {d.loc[malas, 'fecha'].dt.date.astype(str).tolist()[:5]}",
            "eliminación sin rellenar (no se usa información futura)",
            "se verificó que no quedan NaN/inf/precios<=0",
        )
    d = d.loc[~malas]
    if dup and bitacora is not None:
        bitacora.registrar(
            "limpieza", "fecha repetida", f"{dup} filas duplicadas", "conservar última versión por fecha",
            "se verificó índice único y orden cronológico",
        )
    d = d.reset_index(drop=True)
    if d.empty:
        raise DatosError("Tras la limpieza no quedaron filas válidas; revise la fuente.")
    assert d["fecha"].is_unique and d["fecha"].is_monotonic_increasing
    rep["salida"] = len(d)
    return d, rep


def exigir_minimo(df: pd.DataFrame, minimo: int = 1500) -> None:
    if len(df) < minimo:
        raise DatosError(f"Observaciones válidas insuficientes: {len(df)} < {minimo} requeridas por el protocolo.")
