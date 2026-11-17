"""Cliente de DATOS PÚBLICOS de MEXC (solo GET de mercado). No existe ninguna ruta de órdenes reales.

Los endpoints usados (/api/v3/klines, /api/v3/exchangeInfo, /api/v3/time) no requieren clave.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from .errores import DatoAtrasadoError, DatosInvalidosError, MexcError

MS_DIA = 86_400_000
MS_INTERVALO = {"1d": MS_DIA, "4h": MS_DIA // 6, "1h": MS_DIA // 24}


class _Transitorio(Exception):
    pass


class MexcClient:
    def __init__(self, base_url: str = "https://api.mexc.com", reintentos: int = 3, espera_base: float = 1.0,
                 timeout: float = 10.0, sesion=None, dormir=time.sleep):
        self.base = base_url.rstrip("/")
        self.reintentos, self.espera_base, self.timeout = reintentos, espera_base, timeout
        self.s = sesion or requests.Session()
        self.dormir = dormir
        self.intentos_ultima_llamada = 0

    def _get(self, ruta: str, params: dict | None = None):
        """GET con reintentos limitados y espera creciente (espera_base * 2**intento)."""
        ultimo = None
        for i in range(self.reintentos + 1):
            self.intentos_ultima_llamada = i + 1
            try:
                r = self.s.get(self.base + ruta, params=params, timeout=self.timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    raise _Transitorio(f"HTTP {r.status_code} (límite de solicitudes o error del servidor)")
                r.raise_for_status()
                return r.json()
            except (requests.Timeout, requests.ConnectionError, _Transitorio) as e:
                ultimo = e
                if i < self.reintentos:
                    self.dormir(self.espera_base * (2 ** i))
            except (requests.RequestException, ValueError) as e:
                raise MexcError(f"MEXC devolvió una respuesta inválida en {ruta}: {e}.") from e
        raise MexcError(
            f"MEXC no respondió tras {self.reintentos + 1} intentos ({ultimo}). Verifique su conexión; "
            "se pausan las nuevas decisiones y se puede reanudar con 'Iniciar'.")

    def hora_servidor_ms(self) -> int:
        return int(self._get("/api/v3/time")["serverTime"])

    def velas_crudas(self, simbolo: str, intervalo: str = "1d", limite: int = 200) -> pd.DataFrame:
        filas = self._get("/api/v3/klines", {"symbol": simbolo, "interval": intervalo, "limit": limite})
        if not isinstance(filas, list) or not filas:
            raise DatosInvalidosError(f"MEXC devolvió velas vacías para {simbolo} {intervalo}.")
        df = pd.DataFrame([f[:7] for f in filas], columns=["abre_ms", "open", "high", "low", "close", "volume", "cierra_ms"])
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["abre_ms"] = df["abre_ms"].astype("int64")
        df["cierra_ms"] = df["abre_ms"] + MS_INTERVALO.get(intervalo, MS_DIA)
        df["fecha"] = pd.to_datetime(df["abre_ms"], unit="ms")
        return df

    def info_simbolo(self, simbolo: str) -> dict:
        s = self._get("/api/v3/exchangeInfo", {"symbol": simbolo})["symbols"][0]
        return {"paso_cantidad": float(s.get("baseSizePrecision", "0.000001")),
                "decimales_precio": int(s.get("quotePrecision", 2)),
                "tamano_minimo_usdt": float(s.get("quoteAmountPrecision", "1"))}


def validar_velas(df: pd.DataFrame, simbolo_esperado: str, simbolo: str, intervalo: str, ahora_ms: int,
                  antiguedad_max_h: float) -> pd.DataFrame:
    """Valida y devuelve SOLO velas cerradas. Una vela aún abierta nunca se usa como definitiva."""
    if simbolo != simbolo_esperado:
        raise DatosInvalidosError(f"Activo no permitido '{simbolo}'; solo se admite '{simbolo_esperado}'.")
    if df is None or df.empty:
        raise DatosInvalidosError("No hay velas para procesar.")
    if intervalo not in MS_INTERVALO or intervalo != "1d":
        raise DatosInvalidosError(f"Frecuencia '{intervalo}' no permitida; el modelo se entrenó con velas diarias (1d).")
    if not df["abre_ms"].is_monotonic_increasing or df["abre_ms"].duplicated().any():
        raise DatosInvalidosError("Las velas están desordenadas o duplicadas; se rechaza el lote.")
    num = df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    if not np.isfinite(num).all() or (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise DatosInvalidosError("Hay velas con valores no finitos o precios no positivos.")
    cerradas = df[df["cierra_ms"] <= ahora_ms].reset_index(drop=True)
    if cerradas.empty:
        raise DatosInvalidosError("No hay velas cerradas disponibles todavía.")
    edad_h = (ahora_ms - int(cerradas["cierra_ms"].iloc[-1])) / 3_600_000
    if edad_h > antiguedad_max_h:
        raise DatoAtrasadoError(
            f"Dato atrasado: la última vela cerrada tiene {edad_h:.1f} h (máximo {antiguedad_max_h:.1f} h). "
            "No se opera con datos viejos; revise la conexión o la fuente.")
    return cerradas
