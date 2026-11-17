"""Barreras para que el LLM sea SOLO local y NUNCA reciba APIs, claves ni credenciales.

1) `validar_url_local`: el LLM debe ser un Ollama en loopback (127.0.0.0/8, ::1 o localhost). Cualquier otro host se rechaza al arrancar.
2) `GuardiaSecretos`: revisa TODO texto que saldría hacia el LLM (mensajes de sistema, de usuario y de herramientas) y lo bloquea si contiene
   el valor de una clave configurada, cabeceras/palabras de credenciales, direcciones de APIs o URLs.
El LLM solo recibe datos de mercado numéricos ya procesados; no tiene herramientas de red ni acceso a variables de entorno.
"""
from __future__ import annotations

import ipaddress
import re
from typing import Iterable
from urllib.parse import urlparse

from .errores import AgenteError, ConfiguracionError


class SecretoEnPromptError(AgenteError):
    """Se detuvo una llamada al LLM porque su contenido incluía una credencial, una API o una URL."""


def validar_url_local(url: str) -> str:
    """Acepta solo http(s) hacia loopback. Garantiza que la inferencia ocurre en este equipo (sin nube)."""
    p = urlparse(url)
    host = (p.hostname or "").lower()
    if p.scheme not in ("http", "https") or not host:
        raise ConfiguracionError(f"URL del LLM inválida: '{url}'.")
    local = host == "localhost"
    if not local:
        try:
            local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = False
    if not local:
        raise ConfiguracionError(
            f"El LLM debe ser local (loopback): '{host}' no lo es. v0.1 solo consume un modelo local de Ollama en 127.0.0.1.")
    return url


# Palabras/estructuras que delatan credenciales o APIs (sin distinguir mayúsculas)
_PATRONES = [
    r"api[_\- ]?key", r"api[_\- ]?secret", r"secret[_\- ]?key", r"x-mexc-apikey", r"authorization", r"bearer\s+\S+",
    r"password|passwd|contraseña", r"private[_\- ]?key", r"access[_\- ]?token", r"\bsignature=", r"https?://", r"api\.mexc\.com", r"\.env\b",
]
_RE = re.compile("|".join(_PATRONES), re.IGNORECASE)


class GuardiaSecretos:
    def __init__(self, valores_secretos: Iterable[str] = ()):
        # Se ignoran valores demasiado cortos (evita falsos positivos) pero toda clave real es larga
        self.valores = [v for v in {str(v).strip() for v in valores_secretos if v} if len(v) >= 8]

    def revisar(self, texto: str) -> None:
        for v in self.valores:
            if v in texto:
                raise SecretoEnPromptError("Bloqueado: el texto enviado al LLM contiene el valor de una clave/credencial. No se realizó la llamada.")
        m = _RE.search(texto)
        if m:
            raise SecretoEnPromptError(
                f"Bloqueado: el texto enviado al LLM menciona una credencial, API o URL ('{m.group(0)[:20]}'). El LLM solo recibe datos de mercado.")

    def revisar_mensajes(self, mensajes: list[dict]) -> None:
        """Revisa lo que saldría hacia el LLM. Los mensajes del propio asistente (eco) no se revisan."""
        for m in mensajes:
            if m.get("role") == "assistant":
                continue
            self.revisar(str(m.get("content", "")))
