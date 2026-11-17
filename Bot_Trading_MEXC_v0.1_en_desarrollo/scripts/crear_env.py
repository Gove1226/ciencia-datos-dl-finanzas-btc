"""Crea .env con claves aleatorias a partir de .env.example (no sobrescribe uno existente)."""
import secrets
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
destino = RAIZ / ".env"
if destino.exists():
    raise SystemExit(".env ya existe; no se sobrescribe.")
txt = (RAIZ / ".env.example").read_text(encoding="utf-8")
txt = txt.replace("APP_API_KEY=\n", f"APP_API_KEY={secrets.token_urlsafe(32)}\n").replace("LLM_API_KEY=\n", f"LLM_API_KEY={secrets.token_urlsafe(32)}\n")
destino.write_text(txt, encoding="utf-8")
print("Creado .env (las claves están solo en ese archivo).")
