"""Ciclo del agente: observar -> modelo -> LLM propone -> riesgo -> Paper -> registrar -> esperar.

Estados: detenido | ejecutando | pausado. Detener/pausar bloquean nuevas órdenes; NO liquidan (la liquidación es una acción explícita).
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd

from . import riesgo as R
from .config import Ajustes, Riesgo
from .errores import (AgenteError, DatoAtrasadoError, DatosInvalidosError, MexcError, OllamaNoDisponibleError,
                      PropuestaInvalidaError)
from .llm import ACCIONES, OllamaClient
from .mexc import MexcClient, validar_velas
from .modelo import ModeloPredictivo
from .paper import CuentaPaper, tiene_posicion
from .store import Store, ahora

from .seguridad import SecretoEnPromptError
from .version import VERSION

ERRORES_PAUSA = (MexcError, DatoAtrasadoError, DatosInvalidosError, OllamaNoDisponibleError, PropuestaInvalidaError, SecretoEnPromptError)
MODOS_REPLAY = {"replay_llm": "llm", "replay_modelo": "modelo", "replay_bh": "bh"}


class Agente:
    def __init__(self, store: Store, mexc: MexcClient, modelo: ModeloPredictivo, llm: OllamaClient, riesgo: Riesgo, ajustes: Ajustes):
        self.store, self.mexc, self.modelo, self.llm, self.cfg, self.aj = store, mexc, modelo, llm, riesgo, ajustes
        self.simbolo = ajustes.simbolo
        self._lock = threading.RLock()
        self._despertar, self._parar = threading.Event(), threading.Event()
        self._hilo: threading.Thread | None = None
        self.cuentas: dict[str, CuentaPaper] = {}
        # Recuperación tras reinicio: el estado financiero vive en SQLite; el agente vuelve 'pausado' (seguro).
        previo = store.kv_get("estado_agente", "detenido")
        self.estado = "pausado" if previo == "ejecutando" else previo
        if previo == "ejecutando":
            store.registrar_error("reinicio", "El backend se reinició con el agente en ejecución; se recuperó el estado desde SQLite "
                                  "y el agente quedó en pausa. Pulse 'Iniciar' para reanudar (las velas ya procesadas no se repiten).", "live")
        self.ultimo_error = store.kv_get("ultimo_error")
        self.replay_estado = store.kv_get("replay_estado", {"estado": "sin_ejecutar"})
        self.cuenta("live")

    # ------------------------------------------------------------------ utilidades
    def cuenta(self, modo: str) -> CuentaPaper:
        if modo not in self.cuentas:
            self.cuentas[modo] = CuentaPaper(self.store, modo, self.cfg, self.simbolo)
        return self.cuentas[modo]

    def _set_estado(self, e: str):
        self.estado = e
        self.store.kv_set("estado_agente", e)

    def _fallo(self, e: Exception, modo: str):
        """Registra, informa y PAUSA nuevas decisiones."""
        msg = f"{type(e).__name__}: {e}"
        self.store.registrar_error(type(e).__name__, str(e), modo)
        self.ultimo_error = {"tipo": type(e).__name__, "mensaje": str(e), "utc": ahora()}
        self.store.kv_set("ultimo_error", self.ultimo_error)
        if modo == "live":
            self._set_estado("pausado")
        return msg

    # ------------------------------------------------------------------ núcleo de decisión
    def procesar_vela(self, modo: str, cerradas: pd.DataFrame, px_ejec: float, px_val: float, usar_llm: bool = True,
                      activo: bool = True, regla: str = "modelo") -> dict:
        """Pasos 2-7 sobre la ÚLTIMA vela cerrada de `cerradas` (ya validada). Idempotente por vela."""
        fila = cerradas.iloc[-1]
        abre_ms, fecha_vela = int(fila["abre_ms"]), str(pd.Timestamp(fila["fecha"]).date())
        cuenta = self.cuenta(modo)
        if self.store.vela_procesada(modo, self.simbolo, abre_ms):
            self.store.registrar_error("evento_duplicado", f"La vela {fecha_vela} ya fue procesada en modo '{modo}'; se ignora sin operar.", modo)
            return {"estado": "duplicado", "fecha_vela": fecha_vela}
        pred = self.modelo.predecir(cerradas)                                   # paso 2
        close_t, close_prev = float(fila["close"]), float(cerradas["close"].iloc[-2])
        cu = cuenta.estado()
        pos = tiene_posicion(cu["cantidad"], self.cfg.paso_cantidad_btc)
        met = {}
        if regla == "bh":
            accion_prop, just, fuente = ("mantener" if pos else "comprar"), "comprar y mantener (referencia pasiva)", "bh"
        elif usar_llm:                                                           # pasos 3-4
            ctx = {"simbolo": self.simbolo, "fecha_vela": fecha_vela, "precio_cierre": close_t,
                   "pronostico_modelo_siguiente_dia": round(pred["pronostico"], 6), "umbral_modelo": pred["umbral_delta"],
                   "senal_modelo": pred["senal_modelo"], "indicadores": pred["indicadores"],
                   "posicion": {"efectivo_usdt": round(cu["efectivo"], 2), "btc": cu["cantidad"], "tiene_posicion": pos},
                   "acciones_permitidas": list(ACCIONES)}
            herr = {
                "consultar_indicadores": lambda: pred["indicadores"],
                "consultar_posicion": lambda: {"efectivo_usdt": round(cu["efectivo"], 2), "btc": cu["cantidad"],
                                               "valor_posicion": round(cu["cantidad"] * close_t, 2), "patrimonio": round(cu["efectivo"] + cu["cantidad"] * close_t, 2)},
                "consultar_riesgo": lambda: {**R.estado_riesgo(cu["efectivo"], cu["cantidad"], close_t, close_prev, self.cfg),
                                             "tamano_minimo_usdt": self.cfg.tamano_minimo_usdt},
            }
            prop, met = self.llm.proponer(ctx, herr, self.simbolo, fecha_vela)
            accion_prop, just, fuente = prop.accion, prop.justificacion, "llm"
        else:
            accion_prop = "comprar" if (pred["senal_modelo"] and not pos) else "vender" if (not pred["senal_modelo"] and pos) else "mantener"
            just, fuente = "regla del modelo (sin LLM): pronóstico vs umbral", "modelo"
        ver = R.evaluar(accion_prop, cu["efectivo"], cu["cantidad"], close_t, close_prev, self.cfg, activo)   # paso 5
        with self.store.tx() as con:                                              # paso 6 (atómico)
            if not self.store.marcar_vela(con, modo, self.simbolo, abre_ms):
                return {"estado": "duplicado", "fecha_vela": fecha_vela}
            orden = None
            if ver.permitida and ver.accion_final in ("comprar", "vender"):
                orden = cuenta.ejecutar(con, ver.accion_final, px_ejec, abre_ms, ver.monto_usdt)
            pat = cuenta.valorar(con, abre_ms, fecha_vela, px_val)
            con.execute(
                "INSERT INTO decisiones(modo,simbolo,vela_abre_ms,fecha_vela,precio_ref,pronostico,senal_modelo,accion_propuesta,justificacion,"
                "accion_final,riesgo_ok,riesgo_motivos,latencia_llm_ms,fuente,creado_utc) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (modo, self.simbolo, abre_ms, fecha_vela, px_ejec, pred["pronostico"], int(pred["senal_modelo"]), accion_prop, just,
                 ver.accion_final, int(ver.permitida), "; ".join(ver.motivos), met.get("latencia_ms"), fuente, ahora()))
        if orden and orden["estado"] == "rechazada":
            self.store.registrar_error("orden_rechazada", orden["motivo"], modo, {"vela": fecha_vela})
        res = {"estado": "procesada", "modo": modo, "fecha_vela": fecha_vela, "pronostico": pred["pronostico"], "senal_modelo": pred["senal_modelo"],
               "indicadores": pred["indicadores"], "accion_propuesta": accion_prop, "justificacion": just, "accion_final": ver.accion_final,
               "riesgo_ok": ver.permitida, "riesgo_motivos": ver.motivos, "riesgo": ver.estado, "orden": orden, "patrimonio": pat,
               "latencia_llm_ms": met.get("latencia_ms"), "precio_ejecucion": px_ejec, "fuente_decision": fuente}
        return res

    # ------------------------------------------------------------------ ciclo en vivo
    def ciclo_live(self) -> dict:
        with self._lock:
            if self.estado != "ejecutando":
                self.store.registrar_error("bloqueado", f"Ciclo no ejecutado: el agente está '{self.estado}'. Nuevas órdenes bloqueadas.", "live")
                return {"estado": "bloqueado", "mensaje": f"El agente está '{self.estado}'; no se emiten nuevas decisiones."}
            try:
                ahora_ms = int(time.time() * 1000)
                crudas = self.mexc.velas_crudas(self.simbolo, "1d", 200)                    # paso 1
                cerradas = validar_velas(crudas, self.simbolo, self.simbolo, "1d", ahora_ms, self.cfg.antiguedad_max_horas)
                abierta = crudas[crudas["cierra_ms"] > ahora_ms]
                px = float(abierta["open"].iloc[0]) if len(abierta) else float(cerradas["close"].iloc[-1])
                res = self.procesar_vela("live", cerradas, px, px, usar_llm=True, activo=True)
                if res["estado"] == "procesada":
                    self.store.kv_set("ultimo_ciclo_live", res)
                    self.ultimo_error = None
                    self.store.kv_set("ultimo_error", None)
                return res
            except ERRORES_PAUSA as e:
                return {"estado": "error", "mensaje": self._fallo(e, "live")}

    def _bucle(self):
        while not self._parar.is_set():
            if self.estado == "ejecutando":
                try:
                    self.ciclo_live()
                except Exception as e:  # último recurso: nunca morir en silencio; se registra y se pausa
                    self._fallo(e, "live")
            self._despertar.wait(self.cfg.intervalo_consulta_s)
            self._despertar.clear()

    def iniciar(self):
        with self._lock:
            self._parar.clear()
            self._set_estado("ejecutando")
            if not self._hilo or not self._hilo.is_alive():
                self._hilo = threading.Thread(target=self._bucle, name="agente-live", daemon=True)
                self._hilo.start()
            self._despertar.set()

    def pausar(self):
        with self._lock:
            self._set_estado("pausado")

    def detener(self):
        """Bloquea nuevas órdenes y conserva la posición (NO liquida)."""
        with self._lock:
            self._set_estado("detenido")

    def apagar(self):
        self._parar.set()
        self._despertar.set()

    def liquidar(self) -> dict:
        """Acción EXPLÍCITA del usuario: vende toda la posición Paper al último precio disponible."""
        with self._lock:
            c = self.cuenta("live")
            if not c.posicion_abierta():
                return {"estado": "sin_posicion", "mensaje": "No hay posición abierta que liquidar."}
            try:
                crudas = self.mexc.velas_crudas(self.simbolo, "1d", 5)
            except MexcError as e:
                return {"estado": "error", "mensaje": self._fallo(e, "live")}
            px = float(crudas["close"].iloc[-1])
            abre = int(crudas["abre_ms"].iloc[-1])
            with self.store.tx() as con:
                o = c.ejecutar(con, "liquidar", px, abre)
                c.valorar(con, abre, str(pd.Timestamp(crudas["fecha"].iloc[-1]).date()), px)
            return {"estado": o["estado"], "orden": o}

    # ------------------------------------------------------------------ replay histórico (separado de la conexión actual)
    def replay(self, n: int = 20, con_llm: bool = True) -> dict:
        """Repite `n` decisiones diarias con datos históricos de MEXC. Se identifica como REPLAY y usa cuentas separadas."""
        with self._lock:
            self.replay_estado = {"estado": "ejecutando", "n": n, "hecho": 0, "inicio_utc": ahora()}
            self.store.kv_set("replay_estado", self.replay_estado)
            try:
                crudas = self.mexc.velas_crudas(self.simbolo, "1d", min(1000, max(300, n + 260)))
                cerradas = validar_velas(crudas, self.simbolo, self.simbolo, "1d", int(time.time() * 1000), self.cfg.antiguedad_max_horas)
                N = len(cerradas)
                if N < n + self.modelo.ventana + 60:
                    raise DatosInvalidosError(f"Velas insuficientes para el replay: {N}.")
                for modo in MODOS_REPLAY:
                    self.store.reiniciar_cuenta(modo, self.cfg.capital_inicial)
                    self.cuentas.pop(modo, None)
                for k, i in enumerate(range(N - 1 - n, N - 1)):
                    ventana = cerradas.iloc[: i + 1].tail(260).reset_index(drop=True)
                    px_ej, px_val = float(cerradas["open"].iloc[i + 1]), float(cerradas["close"].iloc[i + 1])
                    for modo, tipo in MODOS_REPLAY.items():
                        if tipo == "llm" and not con_llm:
                            continue
                        self.procesar_vela(modo, ventana, px_ej, px_val, usar_llm=(tipo == "llm"), activo=True,
                                           regla="bh" if tipo == "bh" else "modelo")
                    self.replay_estado = {**self.replay_estado, "hecho": k + 1}
                    self.store.kv_set("replay_estado", self.replay_estado)
                # liquidación final (mismo tratamiento para las tres estrategias)
                abre_f, fecha_f, px_f = int(cerradas["abre_ms"].iloc[-1]), str(pd.Timestamp(cerradas["fecha"].iloc[-1]).date()), float(cerradas["close"].iloc[-1])
                for modo in MODOS_REPLAY:
                    c = self.cuenta(modo)
                    if c.posicion_abierta():
                        with self.store.tx() as con:
                            c.ejecutar(con, "liquidar", px_f, abre_f)
                            c.valorar(con, abre_f, fecha_f, px_f)
                self.replay_estado = {"estado": "terminado", "n": n, "hecho": n, "fin_utc": ahora(), "metricas": self.comparacion_replay()}
            except (AgenteError, Exception) as e:   # se informa y se registra; el replay no toca la cuenta live
                self.replay_estado = {"estado": "error", "mensaje": f"{type(e).__name__}: {e}"}
                self.store.registrar_error(type(e).__name__, str(e), "replay")
            self.store.kv_set("replay_estado", self.replay_estado)
            return self.replay_estado

    def metricas_modo(self, modo: str) -> dict:
        pat = self.store.patrimonio(modo, 5000)
        cu = self.store.cuenta(modo)
        if not pat or not cu:
            return {"modo": modo, "ciclos": 0}
        eq = np.array([p["patrimonio"] for p in pat], dtype=float)
        cap = cu["capital_inicial"]
        serie = np.r_[cap, eq]
        mdd = float(((serie - np.maximum.accumulate(serie)) / np.maximum.accumulate(serie)).min())
        ords = [o for o in reversed(self.store.ordenes(modo, 5000)) if o["estado"] == "ejecutada"]
        trades, gasto = [], None
        for o in ords:
            if o["lado"] == "comprar":
                gasto = o["monto_usdt"] + o["costo"]
            elif gasto is not None:
                trades.append((o["monto_usdt"] - o["costo"]) - gasto)
                gasto = None
        dec = self.store.decisiones(modo, 5000)
        lat = [d["latencia_llm_ms"] for d in dec if d["latencia_llm_ms"]]
        rech = sum(1 for d in dec if not d["riesgo_ok"] and d["accion_propuesta"] in ("comprar", "vender")) + \
            sum(1 for o in self.store.ordenes(modo, 5000) if o["estado"] == "rechazada")
        return {"modo": modo, "ciclos": len(dec), "patrimonio_final": float(eq[-1]), "rendimiento_neto": float(eq[-1] / cap - 1),
                "max_drawdown": mdd, "operaciones_cerradas": len(trades), "win_rate": (float(np.mean(np.array(trades) > 0)) if trades else None),
                "ordenes_ejecutadas": len(ords), "costos_totales": float(cu["costos_acum"]),
                "latencia_media_ms": (float(np.mean(lat)) if lat else None), "acciones_rechazadas": int(rech)}

    def comparacion_replay(self) -> dict:
        return {m: self.metricas_modo(m) for m in MODOS_REPLAY}

    # ------------------------------------------------------------------ estado para la UI
    def estado_publico(self) -> dict:
        cu = self.cuenta("live").estado()
        ult = self.store.kv_get("ultimo_ciclo_live")
        pat = self.store.patrimonio("live", 1)
        px = pat[-1]["precio_valoracion"] if pat else None
        patrimonio = (cu["efectivo"] + cu["cantidad"] * px) if px else cu["efectivo"] + cu["cantidad"] * 0
        return {
            "activo": self.simbolo, "fuente_datos": "MEXC (datos públicos, sin clave de trading)", "modo": "paper (dinero virtual)",
            "version": VERSION, "estado_agente": self.estado, "ordenes_reales": "deshabilitadas",
            "modelo": self.modelo.version(), "llm": {**self.llm.info(), "disponible": self.llm.disponible()},
            "ultimo_ciclo": ult, "ultimo_error": self.ultimo_error,
            "cuenta": {"efectivo": cu["efectivo"], "btc": cu["cantidad"], "capital_inicial": cu["capital_inicial"], "costos_acum": cu["costos_acum"],
                       "patrimonio": patrimonio, "rendimiento": patrimonio / cu["capital_inicial"] - 1, "precio_valoracion": px},
            "riesgo_config": self.cfg.model_dump(),
            "ciclos_live": self.store.contar("decisiones", "live"),
            "replay": self.replay_estado,
        }
