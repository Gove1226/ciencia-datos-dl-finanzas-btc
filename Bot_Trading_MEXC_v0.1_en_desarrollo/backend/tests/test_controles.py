"""Pruebas automatizadas de controles de riesgo, Paper, datos, LLM, idempotencia, recuperación y autenticación."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest
import requests
from fastapi.testclient import TestClient

from backend.app import riesgo as R
from backend.app.config import Ajustes
from backend.app.errores import DatoAtrasadoError, DatosInvalidosError, MexcError, OllamaNoDisponibleError, PropuestaInvalidaError
from backend.app.llm import OllamaClient, validar_propuesta
from backend.app.main import crear_app
from backend.app.mexc import MexcClient, validar_velas
from backend.app.paper import CuentaPaper
from backend.app.store import Store

SIM = "BTCUSDT"


# ----------------------------------------------------------------------------- riesgo determinista
def test_riesgo_sin_ventas_en_corto_ni_apalancamiento(riesgo):
    assert not R.evaluar("vender", 50_000, 0.0, 70_000, 70_000, riesgo).permitida          # sin posición
    assert not R.evaluar("comprar", 0, 1.0, 70_000, 70_000, riesgo).permitida              # ya hay posición
    v = R.evaluar("comprar", 50_000, 0.0, 70_000, 70_000, riesgo)
    assert v.permitida and v.monto_usdt == pytest.approx(0.95 * 50_000)                    # límite de exposición
    assert not R.evaluar("apalancar", 50_000, 0, 70_000, 70_000, riesgo).permitida          # acción no permitida


def test_riesgo_limite_perdida_diaria_bloquea(riesgo):
    v = R.evaluar("mantener", 0.0, 1.0, 60_000, 70_000, riesgo)                            # -14 % en el día
    assert not v.permitida and v.estado["bloqueado"] and "pérdida diaria" in v.motivos[0]
    assert not R.evaluar("vender", 0.0, 1.0, 60_000, 70_000, riesgo).permitida             # bloquea nuevas órdenes


def test_riesgo_sistema_detenido_bloquea(riesgo):
    v = R.evaluar("comprar", 50_000, 0, 70_000, 70_000, riesgo, sistema_activo=False)
    assert not v.permitida and v.accion_final == "mantener"


# ----------------------------------------------------------------------------- Paper
def test_paper_saldo_insuficiente_y_minimos(tmp_path, riesgo):
    s = Store(tmp_path / "p.db"); c = CuentaPaper(s, "live", riesgo, SIM)
    with s.tx() as con:
        o = c.ejecutar(con, "comprar", 70_000, 1, monto_usdt=1_000_000)                    # más que el efectivo
    assert o["estado"] == "rechazada" and "saldo insuficiente" in o["motivo"]
    assert c.estado()["efectivo"] == 50_000 and c.estado()["cantidad"] == 0                # nada cambió
    with s.tx() as con:
        o = c.ejecutar(con, "comprar", 70_000, 2, monto_usdt=1.0)                          # bajo el mínimo
    assert o["estado"] == "rechazada" and "tamaño mínimo" in o["motivo"]


def test_paper_precision_costos_y_deslizamiento(tmp_path, riesgo):
    r = riesgo.model_copy(update={"costo_operacion": 0.001})
    s = Store(tmp_path / "p.db"); c = CuentaPaper(s, "live", r, SIM)
    with s.tx() as con:
        o = c.ejecutar(con, "comprar", 70_000, 1, monto_usdt=10_000)
    assert o["estado"] == "ejecutada" and o["precio"] == pytest.approx(70_000 * 1.0005)   # deslizamiento en contra
    assert round(o["cantidad"] / r.paso_cantidad_btc) * r.paso_cantidad_btc == pytest.approx(o["cantidad"])  # precisión del par
    assert o["costo"] == pytest.approx(o["monto_usdt"] * 0.001)
    assert c.estado()["efectivo"] >= 0                                                     # nunca saldo negativo


# ----------------------------------------------------------------------------- datos y MEXC
def test_dato_atrasado_se_rechaza(velas, ahora_ms):
    viejo = ahora_ms + 40 * 3_600_000
    with pytest.raises(DatoAtrasadoError, match="atrasado"):
        validar_velas(velas, SIM, SIM, "1d", viejo, 30)


def test_no_usa_vela_abierta(velas, ahora_ms):
    abierta = velas.copy()
    nueva = abierta.iloc[[-1]].copy(); nueva["abre_ms"] += 86_400_000; nueva["cierra_ms"] += 86_400_000
    abierta = pd.concat([abierta, nueva], ignore_index=True)
    cerradas = validar_velas(abierta, SIM, SIM, "1d", ahora_ms, 30)
    assert len(cerradas) == len(velas) and cerradas["abre_ms"].iloc[-1] == velas["abre_ms"].iloc[-1]


def test_velas_invalidas_se_rechazan(velas, ahora_ms):
    with pytest.raises(DatosInvalidosError):
        validar_velas(velas.iloc[0:0], SIM, SIM, "1d", ahora_ms, 30)
    with pytest.raises(DatosInvalidosError, match="no permitido"):
        validar_velas(velas, SIM, "ETHUSDT", "1d", ahora_ms, 30)
    desord = velas.iloc[::-1].reset_index(drop=True)
    with pytest.raises(DatosInvalidosError, match="desordenadas"):
        validar_velas(desord, SIM, SIM, "1d", ahora_ms, 30)


class _Resp:
    def __init__(self, code=200, data=None):
        self.status_code, self._d = code, data if data is not None else []

    def json(self):
        return self._d

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class _Sesion:
    def __init__(self, respuestas):
        self.r, self.n = list(respuestas), 0

    def get(self, *a, **k):
        self.n += 1
        x = self.r.pop(0) if self.r else self.r_last
        if isinstance(x, Exception):
            raise x
        return x


def test_mexc_reintentos_con_espera_creciente_y_limite():
    esperas = []
    ok = _Resp(200, {"serverTime": 123})
    ses = _Sesion([_Resp(429), _Resp(429), ok])
    c = MexcClient(reintentos=3, espera_base=1.0, sesion=ses, dormir=esperas.append)
    assert c.hora_servidor_ms() == 123 and esperas == [1.0, 2.0]                          # espera creciente
    ses2 = _Sesion([requests.Timeout()] * 4)
    c2 = MexcClient(reintentos=3, espera_base=1.0, sesion=ses2, dormir=esperas.append)
    with pytest.raises(MexcError, match="no respondió tras 4 intentos"):
        c2.hora_servidor_ms()
    assert ses2.n == 4                                                                     # reintentos LIMITADOS


def test_sin_rutas_de_ordenes_reales():
    codigo = Path("backend/app/mexc.py").read_text(encoding="utf-8")
    assert ".post(" not in codigo and "/api/v3/order" not in codigo and "X-MEXC-APIKEY" not in codigo
    assert not [m for m in dir(MexcClient) if "orden" in m.lower() or "order" in m.lower()]


# ----------------------------------------------------------------------------- LLM
class _SesionLLM:
    def __init__(self, textos=None, error=None):
        self.textos, self.error, self.n = list(textos or []), error, 0

    def post(self, url, json=None, timeout=None):
        self.n += 1
        if self.error:
            raise self.error
        t = self.textos.pop(0) if self.textos else "{}"
        r = _Resp(200, {"message": {"role": "assistant", "content": t}})
        return r


def test_ollama_detenido():
    c = OllamaClient(sesion=_SesionLLM(error=requests.ConnectionError("rechazada")))
    with pytest.raises(OllamaNoDisponibleError, match="Ollama no responde"):
        c.proponer({}, {}, SIM, "2026-11-02")


def test_json_invalido_se_rechaza_tras_reintentos_limitados():
    ses = _SesionLLM(["esto no es json"] * 10)
    c = OllamaClient(sesion=ses, reintentos=2, dormir=lambda s: None)
    with pytest.raises(PropuestaInvalidaError, match="tras 3 intentos"):
        c.proponer({}, {}, SIM, "2026-11-02")
    assert ses.n == 1 + 3                                                                   # 1 de herramientas + 3 intentos


def test_validacion_rechaza_simbolo_y_accion_no_permitidos():
    ok = {"accion": "comprar", "simbolo": SIM, "fecha_vela": "2026-11-02", "justificacion": "datos"}
    assert validar_propuesta(ok, SIM, "2026-11-02").accion == "comprar"
    with pytest.raises(PropuestaInvalidaError, match="Símbolo"):
        validar_propuesta({**ok, "simbolo": "ETHUSDT"}, SIM, "2026-11-02")
    with pytest.raises(PropuestaInvalidaError, match="Acción no permitida"):
        validar_propuesta({**ok, "accion": "apalancar_x10"}, SIM, "2026-11-02")
    with pytest.raises(PropuestaInvalidaError, match="fecha"):
        validar_propuesta({**ok, "fecha_vela": "1999-01-01"}, SIM, "2026-11-02")
    with pytest.raises(PropuestaInvalidaError):
        validar_propuesta({"accion": 5}, SIM, "2026-11-02")


# ----------------------------------------------------------------------------- ciclo, idempotencia, recuperación, detención
def _ciclo(ag, ahora_ms, usar_llm=True):
    time_orig = time.time
    time.time = lambda: ahora_ms / 1000
    try:
        return ag.ciclo_live()
    finally:
        time.time = time_orig


def test_ciclo_completo_y_evento_duplicado(armar, ahora_ms):
    ag, mexc, llm, store = armar(accion="comprar")
    ag._set_estado("ejecutando")
    r1 = _ciclo(ag, ahora_ms)
    assert r1["estado"] == "procesada" and r1["accion_final"] in ("comprar", "mantener")
    r2 = _ciclo(ag, ahora_ms)                                                               # misma vela
    assert r2["estado"] == "duplicado"
    assert store.contar("decisiones", "live") == 1 and llm.llamadas == 1                    # no se procesó dos veces
    assert any(e["tipo"] == "evento_duplicado" for e in store.errores())
    ag.apagar()


def test_reinicio_recupera_estado_sin_duplicar(armar, tmp_path, ahora_ms):
    db = tmp_path / "reinicio.db"
    ag, _, _, store = armar(db=db, accion="comprar")
    ag._set_estado("ejecutando")
    r = _ciclo(ag, ahora_ms)
    efectivo, btc = ag.cuenta("live").estado()["efectivo"], ag.cuenta("live").estado()["cantidad"]
    ordenes = store.contar("ordenes_paper", "live")
    store.cerrar()
    ag2, _, llm2, store2 = armar(db=db, accion="comprar")                                   # "reinicio del backend"
    assert ag2.estado == "pausado"                                                          # vuelve seguro
    assert ag2.cuenta("live").estado()["efectivo"] == pytest.approx(efectivo) and ag2.cuenta("live").estado()["cantidad"] == pytest.approx(btc)
    ag2._set_estado("ejecutando")
    assert _ciclo(ag2, ahora_ms)["estado"] == "duplicado"                                   # la vela ya estaba procesada
    assert store2.contar("ordenes_paper", "live") == ordenes and llm2.llamadas == 0
    ag2.apagar()


def test_detener_bloquea_ordenes_y_no_liquida(armar, ahora_ms):
    ag, _, llm, store = armar(accion="comprar")
    ag._set_estado("ejecutando")
    _ciclo(ag, ahora_ms)
    btc = ag.cuenta("live").estado()["cantidad"]
    ag.detener()
    assert ag.estado == "detenido"
    r = ag.ciclo_live()
    assert r["estado"] == "bloqueado" and store.contar("ordenes_paper", "live") == (1 if btc > 0 else 0)
    assert ag.cuenta("live").estado()["cantidad"] == btc                                    # detener NO liquida


def test_dato_atrasado_pausa_y_no_opera(armar, ahora_ms):
    ag, _, llm, store = armar(accion="comprar")
    ag._set_estado("ejecutando")
    r = _ciclo(ag, ahora_ms + 72 * 3_600_000)
    assert r["estado"] == "error" and "atrasado" in r["mensaje"].lower()
    assert ag.estado == "pausado" and store.contar("ordenes_paper", "live") == 0 and llm.llamadas == 0


def test_mexc_sin_respuesta_pausa(armar, ahora_ms):
    ag, mexc, _, store = armar()
    mexc.fallo = MexcError("sin respuesta tras 4 intentos")
    ag._set_estado("ejecutando")
    r = _ciclo(ag, ahora_ms)
    assert r["estado"] == "error" and ag.estado == "pausado" and store.errores()[0]["tipo"] == "MexcError"


def test_ollama_detenido_y_json_invalido_pausan(armar, ahora_ms):
    ag, _, llm, store = armar()
    llm.error = OllamaNoDisponibleError("Ollama no responde")
    ag._set_estado("ejecutando")
    assert _ciclo(ag, ahora_ms)["estado"] == "error" and ag.estado == "pausado"
    llm.error = PropuestaInvalidaError("JSON inválido")
    ag._set_estado("ejecutando")
    assert _ciclo(ag, ahora_ms)["estado"] == "error" and store.contar("ordenes_paper", "live") == 0


def test_llm_no_puede_saltarse_el_riesgo(armar, ahora_ms):
    ag, _, llm, store = armar(accion="vender")                                              # sin posición: venta en corto
    ag._set_estado("ejecutando")
    r = _ciclo(ag, ahora_ms)
    assert r["accion_final"] == "mantener" and not r["riesgo_ok"] and store.contar("ordenes_paper", "live") == 0


# ----------------------------------------------------------------------------- autenticación y secretos
@pytest.fixture
def cliente(armar):
    ag, *_ = armar()
    return TestClient(crear_app(Ajustes(app_api_key="clave-app", llm_api_key="clave-llm"), ag))


@pytest.mark.parametrize("ruta,metodo", [("/api/estado", "get"), ("/api/agente/iniciar", "post"), ("/api/agente/detener", "post"),
                                           ("/api/config/riesgo", "get"), ("/api/decisiones", "get"), ("/api/agente/liquidar", "post")])
def test_rutas_protegidas_rechazan_clave_ausente_o_incorrecta(cliente, ruta, metodo):
    assert getattr(cliente, metodo)(ruta).status_code == 401
    assert getattr(cliente, metodo)(ruta, headers={"X-API-Key": "incorrecta"}).status_code == 401


def test_clave_correcta_y_sin_secretos_en_respuestas(cliente):
    r = cliente.get("/api/estado", headers={"X-API-Key": "clave-app"})
    assert r.status_code == 200 and "clave-app" not in r.text and "clave-llm" not in r.text
    assert r.json()["ordenes_reales"] == "deshabilitadas"
    assert cliente.get("/api/salud").status_code == 200


def test_api_llm_exige_su_propia_clave(cliente):
    cuerpo = {"mensajes": [{"role": "user", "content": "hola"}]}
    assert cliente.post("/api/llm/consulta", json=cuerpo).status_code == 401
    assert cliente.post("/api/llm/consulta", json=cuerpo, headers={"X-API-Key": "clave-app"}).status_code == 401   # la clave de la UI no sirve


def test_config_solo_campos_editables_y_con_limites(cliente):
    h = {"X-API-Key": "clave-app"}
    assert cliente.put("/api/config/riesgo", json={"exposicion_max": 5}, headers=h).status_code == 422
    assert cliente.put("/api/config/riesgo", json={"capital_inicial": 1}, headers=h).status_code in (200, 422)
    assert cliente.get("/api/config/riesgo", headers=h).json()["capital_inicial"] == 50_000


def test_clave_no_esta_en_el_frontend():
    """Ningún valor de clave ni variable VITE_* en el código de React (la clave se escribe en tiempo de ejecución)."""
    import os
    from backend.app.config import cargar_ajustes
    aj = cargar_ajustes()
    secretos = [s for s in (aj.app_api_key, aj.llm_api_key, os.environ.get("APP_API_KEY", ""), os.environ.get("LLM_API_KEY", "")) if s]
    if not Path("frontend").exists():
        pytest.skip("La interfaz (Vue) aún no está en el repositorio (v0.1 en desarrollo).")
    fuentes = [p for p in Path("frontend").rglob("*") if p.is_file() and "node_modules" not in p.parts and "dist" not in p.parts
               and p.suffix in {".ts", ".tsx", ".js", ".json", ".html", ".css", ".env", ".vue"}]
    assert fuentes and not [p for p in Path("frontend").glob(".env*")]
    for p in fuentes:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        assert "import.meta.env" not in txt and "VITE_" not in txt, p
        assert not any(s in txt for s in secretos), p
