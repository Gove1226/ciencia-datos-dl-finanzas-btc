"""Configuración. Las claves viven SOLO en el servidor (.env); nunca en React ni en logs."""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .errores import ConfiguracionError

RAIZ = Path(__file__).resolve().parents[2]
load_dotenv(RAIZ / ".env")
RUTA_RIESGO = RAIZ / "config" / "riesgo.json"


class Riesgo(BaseModel):
    """Límites deterministas. El LLM no tiene acceso de escritura a esta configuración."""
    capital_inicial: float = Field(50_000, gt=0)
    costo_operacion: float = Field(0.0, ge=0, le=0.01)          # 0.0 % en MEXC (protocolo)
    deslizamiento_bps: float = Field(5.0, ge=0, le=100)         # supuesto: 5 pb sobre el precio de ejecución
    exposicion_max: float = Field(0.95, gt=0, le=1.0)           # fracción máxima del patrimonio en BTC
    perdida_diaria_max: float = Field(0.05, gt=0, le=0.5)       # bloquea nuevas órdenes al alcanzarla
    antiguedad_max_horas: float = Field(30.0, gt=0)             # antigüedad máxima de la última vela cerrada
    tamano_minimo_usdt: float = Field(5.0, gt=0)
    paso_cantidad_btc: float = Field(0.000001, gt=0)
    decimales_precio: int = Field(2, ge=0, le=8)
    reintentos: int = Field(3, ge=0, le=6)
    espera_base_s: float = Field(1.0, ge=0)
    intervalo_consulta_s: float = Field(300.0, ge=1)


# Campos que la API de configuración permite cambiar (siempre con clave y dentro de límites)
CAMPOS_EDITABLES = {"exposicion_max", "perdida_diaria_max", "antiguedad_max_horas", "deslizamiento_bps", "intervalo_consulta_s"}


def cargar_riesgo(ruta: Path = RUTA_RIESGO) -> Riesgo:
    try:
        return Riesgo(**json.loads(Path(ruta).read_text(encoding="utf-8")))
    except FileNotFoundError as e:
        raise ConfiguracionError(f"Falta {ruta}. Restaure config/riesgo.json.") from e
    except ValueError as e:
        raise ConfiguracionError(f"config/riesgo.json inválido: {e}") from e


def guardar_riesgo(r: Riesgo, ruta: Path = RUTA_RIESGO) -> None:
    Path(ruta).write_text(r.model_dump_json(indent=2), encoding="utf-8")


class Ajustes(BaseModel):
    app_api_key: str = ""
    llm_api_key: str = ""
    # Opcionales y NO usadas en v0.1 (solo datos públicos). Si se usaran, viven solo en el servidor y jamás llegan al LLM.
    mexc_api_key: str = ""
    mexc_secret_key: str = ""

    def secretos(self) -> list[str]:
        return [self.app_api_key, self.llm_api_key, self.mexc_api_key, self.mexc_secret_key]
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:7b"
    mexc_base_url: str = "https://api.mexc.com"
    simbolo: str = "BTCUSDT"
    db_path: Path = RAIZ / "agente_datos" / "agente.db"
    modelo_dir: Path = RAIZ / "backend" / "modelo_aprobado"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]


def cargar_ajustes() -> Ajustes:
    e = os.environ.get

    def ruta(valor, defecto):
        p = Path(valor) if valor else defecto
        return p if p.is_absolute() else RAIZ / p

    a = Ajustes(
        app_api_key=e("APP_API_KEY", ""), llm_api_key=e("LLM_API_KEY", ""),
        mexc_api_key=e("MEXC_API_KEY", ""), mexc_secret_key=e("MEXC_SECRET_KEY", ""),
        ollama_url=e("OLLAMA_URL", "http://127.0.0.1:11434"), ollama_model=e("OLLAMA_MODEL", "qwen2.5:7b"),
        mexc_base_url=e("MEXC_BASE_URL", "https://api.mexc.com"), simbolo=e("SIMBOLO", "BTCUSDT"),
        db_path=ruta(e("DB_PATH"), RAIZ / "agente_datos" / "agente.db"),
        modelo_dir=ruta(e("MODELO_DIR"), RAIZ / "backend" / "modelo_aprobado"),
        cors_origins=[o.strip() for o in e("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()],
    )
    return a
