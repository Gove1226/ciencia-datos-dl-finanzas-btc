"""API (FastAPI) del agente. Documentación interactiva en /docs.

Toda ruta (salvo /api/salud) exige la cabecera X-API-Key. La API privada del LLM usa una clave distinta (LLM_API_KEY).
Las claves viven solo en el servidor (.env); la UI las pide en tiempo de ejecución y no las incluye en el bundle.
"""
from __future__ import annotations

import secrets
import threading
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .agente import Agente
from .config import CAMPOS_EDITABLES, Ajustes, Riesgo, cargar_ajustes, cargar_riesgo, guardar_riesgo
from .errores import AgenteError, OllamaNoDisponibleError
from .llm import OllamaClient
from .mexc import MexcClient
from .modelo import ModeloPredictivo
from .seguridad import GuardiaSecretos
from .store import Store
from .version import VERSION

MODOS_VALIDOS = {"live", "replay_llm", "replay_modelo", "replay_bh"}


def _comparar(a: str, b: str) -> bool:
    return bool(a) and secrets.compare_digest(a.encode(), b.encode())


def construir_agente(aj: Ajustes) -> Agente:
    riesgo = cargar_riesgo()
    store = Store(aj.db_path)
    mexc = MexcClient(aj.mexc_base_url, riesgo.reintentos, riesgo.espera_base_s)
    # LLM solo local (se valida la URL) y con guardia: ninguna clave, API o URL puede salir hacia el modelo
    llm = OllamaClient(aj.ollama_url, aj.ollama_model, reintentos=2, espera_base=riesgo.espera_base_s, guardia=GuardiaSecretos(aj.secretos()))
    return Agente(store, mexc, ModeloPredictivo.cargar(aj.modelo_dir), llm, riesgo, aj)


class ConsultaLLM(BaseModel):
    mensajes: list[dict] = Field(..., min_length=1, max_length=20)
    max_tokens: int = Field(256, ge=1, le=1024)


class ReplayReq(BaseModel):
    n: int = Field(20, ge=1, le=120)
    con_llm: bool = True


class CambioRiesgo(BaseModel):
    exposicion_max: float | None = Field(None, gt=0, le=1)
    perdida_diaria_max: float | None = Field(None, gt=0, le=0.5)
    antiguedad_max_horas: float | None = Field(None, gt=0)
    deslizamiento_bps: float | None = Field(None, ge=0, le=100)
    intervalo_consulta_s: float | None = Field(None, ge=1)


def crear_app(ajustes: Ajustes | None = None, agente: Agente | None = None) -> FastAPI:
    aj = ajustes or cargar_ajustes()

    @asynccontextmanager
    async def vida(app: FastAPI):
        app.state.agente = agente or construir_agente(aj)
        yield
        app.state.agente.apagar()

    app = FastAPI(title="Bot de trading MEXC (Paper) — BTC/USDT", version=VERSION, lifespan=vida,
                  description="Modelo predictivo + Ollama local + controles de riesgo + simulador Paper. Sin órdenes reales.")
    app.add_middleware(CORSMiddleware, allow_origins=aj.cors_origins, allow_methods=["GET", "POST", "PUT"], allow_headers=["X-API-Key", "Content-Type"])
    if agente is not None:
        app.state.agente = agente

    def ag() -> Agente:
        return app.state.agente

    def auth(x_api_key: str | None = Header(default=None)):
        if not aj.app_api_key:
            raise HTTPException(503, "Servidor sin APP_API_KEY configurada. Cree .env a partir de .env.example.")
        if not x_api_key or not _comparar(x_api_key, aj.app_api_key):
            raise HTTPException(401, "Clave de API ausente o incorrecta (cabecera X-API-Key).")

    def auth_llm(x_api_key: str | None = Header(default=None)):
        if not aj.llm_api_key:
            raise HTTPException(503, "Servidor sin LLM_API_KEY configurada.")
        if not x_api_key or not _comparar(x_api_key, aj.llm_api_key):
            raise HTTPException(401, "Clave del LLM ausente o incorrecta (cabecera X-API-Key).")

    @app.get("/api/salud")
    def salud():
        return {"ok": True, "version": VERSION, "ordenes_reales": "deshabilitadas"}

    @app.get("/api/estado", dependencies=[Depends(auth)])
    def estado():
        return ag().estado_publico()

    @app.post("/api/agente/iniciar", dependencies=[Depends(auth)])
    def iniciar():
        ag().iniciar()
        return {"estado": ag().estado}

    @app.post("/api/agente/pausar", dependencies=[Depends(auth)])
    def pausar():
        ag().pausar()
        return {"estado": ag().estado}

    @app.post("/api/agente/detener", dependencies=[Depends(auth)])
    def detener():
        ag().detener()
        return {"estado": ag().estado, "nota": "Nuevas órdenes bloqueadas. La posición NO se liquidó."}

    @app.post("/api/agente/liquidar", dependencies=[Depends(auth)])
    def liquidar():
        return ag().liquidar()

    @app.post("/api/agente/ciclo", dependencies=[Depends(auth)])
    def ciclo():
        return ag().ciclo_live()

    @app.post("/api/replay", dependencies=[Depends(auth)], status_code=202)
    def replay(req: ReplayReq):
        if ag().replay_estado.get("estado") == "ejecutando":
            raise HTTPException(409, "Ya hay un replay en ejecución.")
        threading.Thread(target=ag().replay, args=(req.n, req.con_llm), daemon=True, name="replay").start()
        return {"estado": "iniciado", "n": req.n, "nota": "REPLAY histórico: cuentas separadas de la conexión actual."}

    @app.get("/api/replay", dependencies=[Depends(auth)])
    def replay_estado():
        return ag().replay_estado

    @app.get("/api/decisiones", dependencies=[Depends(auth)])
    def decisiones(modo: str = "live", limite: int = Query(50, ge=1, le=500)):
        if modo not in MODOS_VALIDOS:
            raise HTTPException(422, f"modo inválido; use uno de {sorted(MODOS_VALIDOS)}")
        return ag().store.decisiones(modo, limite)

    @app.get("/api/ordenes", dependencies=[Depends(auth)])
    def ordenes(modo: str = "live", limite: int = Query(50, ge=1, le=500)):
        if modo not in MODOS_VALIDOS:
            raise HTTPException(422, f"modo inválido; use uno de {sorted(MODOS_VALIDOS)}")
        return ag().store.ordenes(modo, limite)

    @app.get("/api/patrimonio", dependencies=[Depends(auth)])
    def patrimonio(modo: str = "live", limite: int = Query(500, ge=1, le=5000)):
        if modo not in MODOS_VALIDOS:
            raise HTTPException(422, f"modo inválido; use uno de {sorted(MODOS_VALIDOS)}")
        return ag().store.patrimonio(modo, limite)

    @app.get("/api/errores", dependencies=[Depends(auth)])
    def errores(limite: int = Query(50, ge=1, le=500)):
        return ag().store.errores(limite)

    @app.get("/api/config/riesgo", dependencies=[Depends(auth)])
    def get_riesgo():
        return ag().cfg.model_dump()

    @app.put("/api/config/riesgo", dependencies=[Depends(auth)])
    def put_riesgo(cambio: CambioRiesgo):
        datos = {k: v for k, v in cambio.model_dump().items() if v is not None}
        if not datos or not set(datos) <= CAMPOS_EDITABLES:
            raise HTTPException(422, f"Solo se pueden editar {sorted(CAMPOS_EDITABLES)}.")
        nuevo = Riesgo(**{**ag().cfg.model_dump(), **datos})
        guardar_riesgo(nuevo)
        ag().cfg = nuevo
        for c in ag().cuentas.values():
            c.cfg = nuevo
        ag().store.registrar_error("config_cambiada", f"Configuración de riesgo cambiada por la API: {datos}", "live")
        return nuevo.model_dump()

    @app.post("/api/llm/consulta", dependencies=[Depends(auth_llm)])
    def llm_consulta(q: ConsultaLLM):
        try:
            return ag().llm.chat_libre(q.mensajes, q.max_tokens)
        except OllamaNoDisponibleError as e:
            raise HTTPException(503, str(e))

    @app.exception_handler(AgenteError)
    async def _manejador(request, exc):  # mensajes útiles, sin trazas internas
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=502, content={"error": type(exc).__name__, "mensaje": str(exc)})

    return app


# Instancia para uvicorn:  uvicorn backend.app.main:app
# (el agente, la BD y Ollama se crean en el arranque, no al importar)
app = crear_app()
