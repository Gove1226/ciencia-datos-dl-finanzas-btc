"""Controles de riesgo DETERMINISTAS. La propuesta del LLM nunca los sustituye."""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Riesgo
from .paper import tiene_posicion

ACCIONES = ("comprar", "mantener", "vender")


@dataclass
class Veredicto:
    permitida: bool
    accion_final: str
    motivos: list[str] = field(default_factory=list)
    monto_usdt: float | None = None
    estado: dict = field(default_factory=dict)


def estado_riesgo(efectivo: float, cantidad: float, precio: float, precio_previo: float, cfg: Riesgo) -> dict:
    """Exposición actual y pérdida diaria (cierre previo -> cierre actual con las tenencias actuales)."""
    eq = efectivo + cantidad * precio
    eq_prev = efectivo + cantidad * precio_previo
    perdida = max(0.0, 1 - eq / eq_prev) if eq_prev > 0 else 0.0
    expo = (cantidad * precio / eq) if eq > 0 else 0.0
    bloqueos = []
    if perdida >= cfg.perdida_diaria_max:
        bloqueos.append(f"pérdida diaria {perdida:.2%} >= límite {cfg.perdida_diaria_max:.2%}")
    return {"patrimonio": eq, "exposicion_actual": expo, "exposicion_max": cfg.exposicion_max,
            "perdida_diaria": perdida, "perdida_diaria_max": cfg.perdida_diaria_max,
            "bloqueado": bool(bloqueos), "motivo_bloqueo": "; ".join(bloqueos), "nivel": "BLOQUEADO" if bloqueos else "OK"}


def evaluar(accion: str, efectivo: float, cantidad: float, precio: float, precio_previo: float,
            cfg: Riesgo, sistema_activo: bool = True) -> Veredicto:
    est = estado_riesgo(efectivo, cantidad, precio, precio_previo, cfg)
    pos = tiene_posicion(cantidad, cfg.paso_cantidad_btc)
    if accion not in ACCIONES:
        return Veredicto(False, "mantener", [f"acción no permitida: '{accion}'"], None, est)
    if not sistema_activo:
        return Veredicto(False, "mantener", ["sistema pausado o detenido: nuevas órdenes bloqueadas"], None, est)
    if est["bloqueado"]:
        return Veredicto(False, "mantener", [f"límite alcanzado: {est['motivo_bloqueo']}; nuevas órdenes bloqueadas"], None, est)
    if accion == "mantener":
        return Veredicto(True, "mantener", [], None, est)
    if accion == "vender":
        if not pos:
            return Veredicto(False, "mantener", ["no hay posición que vender (sin ventas en corto)"], None, est)
        return Veredicto(True, "vender", [], None, est)
    # comprar
    if pos:
        return Veredicto(False, "mantener", ["ya existe una posición (sin apalancamiento ni aumentos)"], None, est)
    monto = min(cfg.exposicion_max * est["patrimonio"], efectivo)
    if monto < cfg.tamano_minimo_usdt:
        return Veredicto(False, "mantener", [f"monto {monto:.2f} USDT inferior al mínimo {cfg.tamano_minimo_usdt:.2f}"], None, est)
    return Veredicto(True, "comprar", [], monto, est)
