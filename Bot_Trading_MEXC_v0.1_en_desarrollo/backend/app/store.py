"""Persistencia SQLite: velas procesadas, decisiones, órdenes Paper, costos, patrimonio y errores.

La unicidad (modo, simbolo, vela_abre_ms) garantiza idempotencia incluso tras un reinicio.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ESQUEMA = """
CREATE TABLE IF NOT EXISTS velas_procesadas(
  modo TEXT NOT NULL, simbolo TEXT NOT NULL, vela_abre_ms INTEGER NOT NULL, procesada_utc TEXT NOT NULL,
  PRIMARY KEY(modo, simbolo, vela_abre_ms));
CREATE TABLE IF NOT EXISTS decisiones(
  id INTEGER PRIMARY KEY AUTOINCREMENT, modo TEXT NOT NULL, simbolo TEXT NOT NULL, vela_abre_ms INTEGER NOT NULL,
  fecha_vela TEXT NOT NULL, precio_ref REAL, pronostico REAL, senal_modelo INTEGER, accion_propuesta TEXT,
  justificacion TEXT, accion_final TEXT, riesgo_ok INTEGER, riesgo_motivos TEXT, latencia_llm_ms REAL,
  fuente TEXT, creado_utc TEXT NOT NULL, UNIQUE(modo, simbolo, vela_abre_ms));
CREATE TABLE IF NOT EXISTS ordenes_paper(
  id INTEGER PRIMARY KEY AUTOINCREMENT, modo TEXT NOT NULL, simbolo TEXT NOT NULL, vela_abre_ms INTEGER NOT NULL,
  lado TEXT NOT NULL, estado TEXT NOT NULL, precio REAL, cantidad REAL, monto_usdt REAL, costo REAL,
  deslizamiento REAL, motivo TEXT, creado_utc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS patrimonio(
  id INTEGER PRIMARY KEY AUTOINCREMENT, modo TEXT NOT NULL, vela_abre_ms INTEGER NOT NULL, fecha TEXT NOT NULL,
  efectivo REAL, cantidad REAL, precio_valoracion REAL, patrimonio REAL, costos_acum REAL, creado_utc TEXT NOT NULL,
  UNIQUE(modo, vela_abre_ms));
CREATE TABLE IF NOT EXISTS errores(
  id INTEGER PRIMARY KEY AUTOINCREMENT, modo TEXT, tipo TEXT NOT NULL, mensaje TEXT NOT NULL, contexto TEXT, creado_utc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS cuenta(
  modo TEXT PRIMARY KEY, efectivo REAL NOT NULL, cantidad REAL NOT NULL, capital_inicial REAL NOT NULL,
  costos_acum REAL NOT NULL DEFAULT 0, costo_base REAL NOT NULL DEFAULT 0, actualizado_utc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS kv(clave TEXT PRIMARY KEY, valor TEXT NOT NULL);
"""


def ahora() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Store:
    def __init__(self, ruta: str | Path):
        self.ruta = str(ruta)
        if self.ruta != ":memory:":
            Path(self.ruta).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._con = sqlite3.connect(self.ruta, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        with self._lock:
            self._con.executescript(ESQUEMA)
            self._con.commit()

    @contextmanager
    def tx(self):
        with self._lock:
            try:
                yield self._con
                self._con.commit()
            except Exception:
                self._con.rollback()
                raise

    def cerrar(self):
        with self._lock:
            self._con.close()

    # ---- consultas
    def _all(self, sql, args=()):
        with self._lock:
            return [dict(r) for r in self._con.execute(sql, args).fetchall()]

    def vela_procesada(self, modo, simbolo, abre_ms) -> bool:
        return bool(self._all("SELECT 1 FROM velas_procesadas WHERE modo=? AND simbolo=? AND vela_abre_ms=?", (modo, simbolo, abre_ms)))

    def marcar_vela(self, con, modo, simbolo, abre_ms) -> bool:
        """True si se insertó; False si ya existía (evento duplicado)."""
        cur = con.execute("INSERT OR IGNORE INTO velas_procesadas VALUES(?,?,?,?)", (modo, simbolo, abre_ms, ahora()))
        return cur.rowcount == 1

    def cuenta(self, modo):
        r = self._all("SELECT * FROM cuenta WHERE modo=?", (modo,))
        return r[0] if r else None

    def crear_cuenta(self, modo, capital):
        with self.tx() as c:
            c.execute("INSERT OR IGNORE INTO cuenta(modo,efectivo,cantidad,capital_inicial,costos_acum,costo_base,actualizado_utc) VALUES(?,?,?,?,0,0,?)",
                      (modo, capital, 0.0, capital, ahora()))

    def reiniciar_cuenta(self, modo, capital):
        with self.tx() as c:
            for t in ("velas_procesadas", "decisiones", "ordenes_paper", "patrimonio"):
                c.execute(f"DELETE FROM {t} WHERE modo=?", (modo,))
            c.execute("DELETE FROM cuenta WHERE modo=?", (modo,))
        self.crear_cuenta(modo, capital)

    def decisiones(self, modo, limite=50):
        return self._all("SELECT * FROM decisiones WHERE modo=? ORDER BY vela_abre_ms DESC LIMIT ?", (modo, limite))

    def ordenes(self, modo, limite=50):
        return self._all("SELECT * FROM ordenes_paper WHERE modo=? ORDER BY id DESC LIMIT ?", (modo, limite))

    def patrimonio(self, modo, limite=500):
        return self._all("SELECT * FROM (SELECT * FROM patrimonio WHERE modo=? ORDER BY vela_abre_ms DESC LIMIT ?) ORDER BY vela_abre_ms", (modo, limite))

    def errores(self, limite=50):
        return self._all("SELECT * FROM errores ORDER BY id DESC LIMIT ?", (limite,))

    def contar(self, tabla, modo=None):
        if modo:
            return self._all(f"SELECT COUNT(*) n FROM {tabla} WHERE modo=?", (modo,))[0]["n"]
        return self._all(f"SELECT COUNT(*) n FROM {tabla}")[0]["n"]

    # ---- escritura
    def registrar_error(self, tipo: str, mensaje: str, modo: str | None = None, contexto: dict | None = None):
        with self.tx() as c:
            c.execute("INSERT INTO errores(modo,tipo,mensaje,contexto,creado_utc) VALUES(?,?,?,?,?)",
                      (modo, tipo, mensaje, json.dumps(contexto or {}, default=str), ahora()))

    def kv_get(self, clave, defecto=None):
        r = self._all("SELECT valor FROM kv WHERE clave=?", (clave,))
        return json.loads(r[0]["valor"]) if r else defecto

    def kv_set(self, clave, valor):
        with self.tx() as c:
            c.execute("INSERT INTO kv(clave,valor) VALUES(?,?) ON CONFLICT(clave) DO UPDATE SET valor=excluded.valor", (clave, json.dumps(valor, default=str)))
