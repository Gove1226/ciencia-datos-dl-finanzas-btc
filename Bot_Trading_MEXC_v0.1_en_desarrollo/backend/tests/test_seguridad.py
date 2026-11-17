"""v0.1: el LLM es solo local y NUNCA recibe APIs, claves ni credenciales."""
from __future__ import annotations

import json

import pytest

from backend.app.errores import ConfiguracionError
from backend.app.llm import OllamaClient
from backend.app.seguridad import GuardiaSecretos, SecretoEnPromptError, validar_url_local

CLAVE_APP = "app-clave-super-secreta-123456"
CLAVE_MEXC = "mexc-api-key-ABCDEF123456"
SECRETO_MEXC = "mexc-secret-ZYXWV987654321"


@pytest.mark.parametrize("url", ["http://127.0.0.1:11434", "http://localhost:11434", "http://[::1]:11434", "http://127.0.0.5:11434"])
def test_llm_local_aceptado(url):
    assert validar_url_local(url) == url


@pytest.mark.parametrize("url", ["https://api.openai.com", "http://192.168.1.50:11434", "http://ollama.miempresa.com:11434", "ftp://127.0.0.1", "127.0.0.1:11434"])
def test_llm_no_local_rechazado(url):
    with pytest.raises(ConfiguracionError):
        validar_url_local(url)
    with pytest.raises(ConfiguracionError):
        OllamaClient(url=url)


class _Sesion:
    def __init__(self):
        self.enviado = []

    def post(self, url, json=None, timeout=None):
        self.enviado.append(json)

        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"message": {"role": "assistant", "content": "{}"}}
        return R()


def _cliente():
    s = _Sesion()
    return OllamaClient(sesion=s, guardia=GuardiaSecretos([CLAVE_APP, CLAVE_MEXC, SECRETO_MEXC])), s


@pytest.mark.parametrize("texto", [CLAVE_APP, f"usa esto {CLAVE_MEXC}", SECRETO_MEXC, "X-MEXC-APIKEY: abc", "Authorization: Bearer abcdef", "api_key=123",
                                    "https://api.mexc.com/api/v3/klines", "mi password es 1234", "lee el archivo .env"])
def test_guardia_bloquea_credenciales_apis_y_urls(texto):
    cli, ses = _cliente()
    with pytest.raises(SecretoEnPromptError):
        cli.chat_libre([{"role": "user", "content": texto}])
    assert ses.enviado == []                    # nada salió hacia el LLM


def test_datos_de_mercado_normales_si_pasan():
    cli, ses = _cliente()
    ctx = {"simbolo": "BTCUSDT", "fecha_vela": "2026-11-04", "precio_cierre": 80498.63, "pronostico_modelo_siguiente_dia": -0.0006, "posicion": {"efectivo_usdt": 50000}}
    cli.chat_libre([{"role": "user", "content": "DATOS: " + json.dumps(ctx)}])
    assert len(ses.enviado) == 1


def test_el_mensaje_de_sistema_del_agente_no_dispara_la_guardia():
    from backend.app.llm import SISTEMA
    GuardiaSecretos([CLAVE_APP]).revisar(SISTEMA)


def test_ciclo_completo_no_envia_claves_al_llm(armar, ahora_ms):
    """Ejecuta un ciclo real del agente con un LLM que captura todo lo que recibe y verifica que no hay claves."""
    import time
    from backend.app.config import Ajustes
    ag, mexc, llm, store = armar()
    capturado = []
    orig = llm.proponer

    def espia(contexto, herramientas, simbolo, fecha_vela, max_iter_herramientas=3):
        capturado.append(json.dumps(contexto))
        for f in herramientas.values():
            capturado.append(json.dumps(f(), default=str))
        return orig(contexto, herramientas, simbolo, fecha_vela)
    llm.proponer = espia
    ag.aj = Ajustes(app_api_key=CLAVE_APP, llm_api_key="llm-clave-xyz-98765432", mexc_api_key=CLAVE_MEXC, mexc_secret_key=SECRETO_MEXC)
    ag._set_estado("ejecutando")
    t0 = time.time
    time.time = lambda: ahora_ms / 1000
    try:
        r = ag.ciclo_live()
    finally:
        time.time = t0
    assert r["estado"] == "procesada" and capturado
    GuardiaSecretos(ag.aj.secretos()).revisar("\n".join(capturado))
