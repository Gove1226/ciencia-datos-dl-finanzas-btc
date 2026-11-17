"""Fixtures: velas realistas (BTC histórico en formato MEXC), MEXC y LLM simulados. Sin red."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backend.app.agente import Agente
from backend.app.config import Ajustes, Riesgo
from backend.app.llm import PropuestaLLM
from backend.app.mexc import MS_DIA
from backend.app.modelo import ModeloPredictivo
from backend.app.store import Store

RAIZ = Path(__file__).resolve().parents[2]


def velas_mexc(n: int = 300, hasta: int | None = None) -> pd.DataFrame:
    """Últimas n velas del histórico original, con las columnas que produce MexcClient.velas_crudas."""
    d = pd.read_csv(Path(__file__).parent / "data" / "btc_velas.csv")
    d = d.iloc[: hasta or len(d)].tail(n).reset_index(drop=True)
    df = pd.DataFrame({"fecha": pd.to_datetime(d["fecha"]), "open": d["open"], "high": d["high"], "low": d["low"],
                       "close": d["close"], "volume": d["volume"].astype(float)})
    df["abre_ms"] = df["fecha"].map(lambda t: int(t.timestamp() * 1000))
    df["cierra_ms"] = df["abre_ms"] + MS_DIA
    return df[["abre_ms", "open", "high", "low", "close", "volume", "cierra_ms", "fecha"]]


class MexcFalso:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.fallo: Exception | None = None

    def velas_crudas(self, simbolo, intervalo="1d", limite=200):
        if self.fallo:
            raise self.fallo
        return self.df.tail(limite).reset_index(drop=True)


class LLMFalso:
    def __init__(self, accion="mantener"):
        self.accion, self.llamadas = accion, 0
        self.error: Exception | None = None

    def proponer(self, contexto, herramientas, simbolo, fecha_vela, max_iter_herramientas=3):
        self.llamadas += 1
        if self.error:
            raise self.error
        herramientas["consultar_riesgo"]()  # ejercita las herramientas de solo lectura
        return PropuestaLLM(accion=self.accion, simbolo=simbolo, fecha_vela=fecha_vela, justificacion="prueba"), {"latencia_ms": 5.0, "intentos": 1}

    def info(self):
        return {"modelo": "falso"}

    def disponible(self):
        return True


@pytest.fixture(scope="session")
def modelo():
    return ModeloPredictivo.cargar(RAIZ / "backend" / "modelo_aprobado")


@pytest.fixture
def riesgo():
    return Riesgo(capital_inicial=50_000, costo_operacion=0.0, deslizamiento_bps=5, exposicion_max=0.95, perdida_diaria_max=0.05)


@pytest.fixture
def velas():
    return velas_mexc(300)


@pytest.fixture
def ahora_ms(velas):
    return int(velas["cierra_ms"].iloc[-1]) + 3_600_000   # 1 h después del cierre: dato fresco


@pytest.fixture
def armar(tmp_path, modelo, riesgo, velas):
    def _armar(db=None, accion="comprar"):
        store = Store(db or tmp_path / "t.db")
        mexc, llm = MexcFalso(velas), LLMFalso(accion)
        ag = Agente(store, mexc, modelo, llm, riesgo, Ajustes(app_api_key="clave-app", llm_api_key="clave-llm"))
        return ag, mexc, llm, store
    return _armar
