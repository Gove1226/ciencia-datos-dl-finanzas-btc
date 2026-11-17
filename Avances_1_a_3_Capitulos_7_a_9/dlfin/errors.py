"""Errores específicos del proyecto y bitácora de incidentes.

No se usa `except: pass`; cada fallo conocido tiene su excepción y un mensaje que
explica cómo resolverlo.
"""
from __future__ import annotations

import csv
import datetime as _dt
from pathlib import Path


class DlFinError(Exception):
    """Base de los errores del proyecto."""


class DatosError(DlFinError):
    """Datos vacíos, corruptos o con formato inválido."""


class VentanaError(DlFinError):
    """Ventanas insuficientes o formas (shapes) incorrectas."""


class FugaError(DlFinError):
    """Se detectó (o podría existir) fuga de información."""


class GPUError(DlFinError):
    """T4 ausente o GPU no utilizada."""


class EntrenamientoError(DlFinError):
    """Pérdida/predicción no finita u otro fallo de entrenamiento."""


class Bitacora:
    """Registro: fecha | etapa | error | causa | acción correctiva | verificación."""

    CAMPOS = ["fecha", "etapa", "error", "causa", "accion_correctiva", "verificacion"]

    def __init__(self) -> None:
        self.filas: list[dict] = []

    def registrar(self, etapa: str, error: str, causa: str, accion: str, verificacion: str) -> None:
        self.filas.append(
            {
                "fecha": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "etapa": etapa,
                "error": error,
                "causa": causa,
                "accion_correctiva": accion,
                "verificacion": verificacion,
            }
        )

    def guardar(self, ruta: str | Path) -> Path:
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        with ruta.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.CAMPOS)
            w.writeheader()
            w.writerows(self.filas)
        return ruta

    def a_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.filas, columns=self.CAMPOS)
