"""Carga del modelo y escalador versionados y cálculo del pronóstico con la MISMA definición de variables del entrenamiento."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from dlfin import features as F
from dlfin import models as MO

from .errores import ConfiguracionError, DatosInvalidosError


class ModeloPredictivo:
    def __init__(self, contrato: dict, escalador, modelo, tipo: str, procedencia: dict):
        self.contrato, self.esc, self.modelo, self.tipo, self.procedencia = contrato, escalador, modelo, tipo, procedencia
        self.canales = tuple(contrato["canales_orden"])
        self.ventana = int(contrato["ventana"])
        self.delta = float(contrato["delta"])
        self.usar_filtro = bool(contrato.get("usar_filtro_sma20", False))

    @staticmethod
    def cargar(carpeta: str | Path) -> "ModeloPredictivo":
        carpeta = Path(carpeta)
        try:
            contrato = json.loads((carpeta / "contrato_variables.json").read_text(encoding="utf-8"))
            esc = F.EscaladorTR.cargar(carpeta / "escalador_final.joblib")
        except FileNotFoundError as e:
            raise ConfiguracionError(
                f"Falta el modelo aprobado en '{carpeta}' ({e.filename}). Ejecute el notebook y luego "
                "'python scripts/publicar_modelo.py' para versionar modelo, escalador y contrato.") from e
        proc_p = carpeta / "procedencia.json"
        proc = json.loads(proc_p.read_text(encoding="utf-8")) if proc_p.exists() else {"entrenado_en_T4": None}
        if contrato.get("tipo") in ("mlp", "lstm"):
            modelo, _ = MO.cargar_checkpoint(carpeta / "modelo_final.pt", "cpu")
        else:
            modelo = joblib.load(carpeta / "modelo_final.joblib")
        return ModeloPredictivo(contrato, esc, modelo, contrato["tipo"], proc)

    def version(self) -> dict:
        return {"modelo": self.contrato["modelo"], "tipo": self.tipo, "variables": list(self.canales), "ventana": self.ventana,
                "delta": self.delta, "usar_filtro_sma20": self.usar_filtro, "feature_version": self.contrato.get("feature_version"),
                "entrenado_hasta": self.contrato.get("entrenado_hasta_objetivo"), "validado_hasta": self.contrato.get("validado_hasta_objetivo"),
                "entrenado_en_T4": self.procedencia.get("entrenado_en_T4"), "dispositivo_entrenamiento": self.contrato.get("dispositivo")}

    def predecir(self, velas: pd.DataFrame) -> dict:
        """`velas`: velas CERRADAS con columnas fecha, open, high, low, close, volume. Pronostica el siguiente periodo."""
        d = velas[["fecha", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
        try:
            ej = F.construir_ejemplos(d, self.ventana, self.canales)
        except Exception as e:  # ventanas insuficientes / no finitas: se informa con contexto
            raise DatosInvalidosError(f"No se pudo construir la ventana del modelo: {e}") from e
        fut = ej.futuro
        if fut is None:
            raise DatosInvalidosError("La última ventana contiene valores no finitos; no se emite pronóstico.")
        X = self.esc.x(fut["X"])
        if self.tipo in ("mlp", "lstm"):
            z = MO.predecir(self.modelo, X, "cpu")
            pron = float(self.esc.y_inv(z)[0])
        else:
            pron = float(self.modelo.predict(X)[0])
        if not np.isfinite(pron):
            raise DatosInvalidosError("El pronóstico no es finito.")
        ind = F.indicadores(d).iloc[-1]
        sma_ok = bool(ind["close"] > ind["sma20"]) if np.isfinite(ind["sma20"]) else False
        senal = pron > self.delta and (sma_ok if self.usar_filtro else True)
        return {
            "fecha_corte": str(pd.Timestamp(fut["fecha_emision"]).date()), "pronostico": pron, "senal_modelo": bool(senal),
            "indicadores": {"close": float(ind["close"]), "sma20": _f(ind["sma20"]), "dist_sma20": _f(ind["dist_sma20"]),
                            "vol20": _f(ind["vol20"]), "rsi14": _f(ind["rsi14"]), "ret_ultimo": _f(ind["ret"])},
            "umbral_delta": self.delta,
        }


def _f(x) -> float | None:
    return float(x) if np.isfinite(x) else None
