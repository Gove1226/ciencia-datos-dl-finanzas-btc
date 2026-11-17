"""Simulador PAPER (dinero virtual). No existe código que envíe órdenes reales a ningún exchange.

Respeta saldo, tamaño mínimo, precisión del par, costos y momento de ejecución.
Supuesto de deslizamiento: el precio de ejecución es el precio de referencia (apertura de la vela siguiente)
ajustado `deslizamiento_bps` pb en contra del operador (compra más cara, venta más barata).
"""
from __future__ import annotations

import math

from .config import Riesgo
from .store import Store, ahora


def _piso(x: float, paso: float) -> float:
    return math.floor(x / paso + 1e-9) * paso


def tiene_posicion(cantidad: float, paso: float) -> bool:
    return cantidad >= paso


class CuentaPaper:
    def __init__(self, store: Store, modo: str, riesgo: Riesgo, simbolo: str):
        self.store, self.modo, self.cfg, self.simbolo = store, modo, riesgo, simbolo
        store.crear_cuenta(modo, riesgo.capital_inicial)   # recupera el estado si ya existe (reinicio)

    def estado(self) -> dict:
        return self.store.cuenta(self.modo)

    def posicion_abierta(self) -> bool:
        return tiene_posicion(self.estado()["cantidad"], self.cfg.paso_cantidad_btc)

    def patrimonio(self, precio: float) -> float:
        e = self.estado()
        return e["efectivo"] + e["cantidad"] * precio

    # ------------------------------------------------------------------ ejecución (dentro de una transacción)
    def ejecutar(self, con, lado: str, precio_ref: float, abre_ms: int, monto_usdt: float | None = None) -> dict:
        c = self.cfg
        cu = dict(con.execute("SELECT * FROM cuenta WHERE modo=?", (self.modo,)).fetchone())
        slip = c.deslizamiento_bps / 10_000
        res = {"lado": lado, "estado": "rechazada", "precio": None, "cantidad": 0.0, "monto_usdt": 0.0, "costo": 0.0, "motivo": ""}
        if lado == "comprar":
            px = round(precio_ref * (1 + slip), c.decimales_precio)
            monto = float(monto_usdt or 0.0)
            if monto <= 0 or monto > cu["efectivo"] + 1e-9:
                res["motivo"] = f"saldo insuficiente: se requieren {monto:.2f} USDT y hay {cu['efectivo']:.2f}"
            else:
                invertido = monto / (1 + c.costo_operacion)
                qty = _piso(invertido / px, c.paso_cantidad_btc)
                nocional = qty * px
                if qty <= 0 or nocional < c.tamano_minimo_usdt:
                    res["motivo"] = f"tamaño mínimo no alcanzado: {nocional:.2f} < {c.tamano_minimo_usdt:.2f} USDT (paso {c.paso_cantidad_btc})"
                else:
                    fee = nocional * c.costo_operacion
                    cu["efectivo"] -= nocional + fee
                    cu["cantidad"] += qty
                    cu["costos_acum"] += fee
                    cu["costo_base"] += nocional + fee
                    res.update(estado="ejecutada", precio=px, cantidad=qty, monto_usdt=nocional, costo=fee)
        elif lado in ("vender", "liquidar"):
            px = round(precio_ref * (1 - slip), c.decimales_precio)
            qty = _piso(cu["cantidad"], c.paso_cantidad_btc)
            nocional = qty * px
            if qty <= 0:
                res["motivo"] = "no hay posición que vender (sin ventas en corto)"
            elif nocional < c.tamano_minimo_usdt:
                res["motivo"] = f"tamaño mínimo no alcanzado: {nocional:.2f} < {c.tamano_minimo_usdt:.2f} USDT"
            else:
                fee = nocional * c.costo_operacion
                cu["efectivo"] += nocional - fee
                cu["cantidad"] -= qty
                cu["costos_acum"] += fee
                if not tiene_posicion(cu["cantidad"], c.paso_cantidad_btc):
                    cu["costo_base"] = 0.0
                res.update(estado="ejecutada", precio=px, cantidad=qty, monto_usdt=nocional, costo=fee)
        else:
            res["motivo"] = f"lado no permitido: {lado}"
        con.execute("INSERT INTO ordenes_paper(modo,simbolo,vela_abre_ms,lado,estado,precio,cantidad,monto_usdt,costo,deslizamiento,motivo,creado_utc) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (self.modo, self.simbolo, abre_ms, lado, res["estado"], res["precio"], res["cantidad"], res["monto_usdt"],
                     res["costo"], slip, res["motivo"], ahora()))
        if res["estado"] == "ejecutada":
            con.execute("UPDATE cuenta SET efectivo=?,cantidad=?,costos_acum=?,costo_base=?,actualizado_utc=? WHERE modo=?",
                        (cu["efectivo"], cu["cantidad"], cu["costos_acum"], cu["costo_base"], ahora(), self.modo))
        return res

    def valorar(self, con, abre_ms: int, fecha: str, precio_valoracion: float) -> float:
        cu = dict(con.execute("SELECT * FROM cuenta WHERE modo=?", (self.modo,)).fetchone())
        pat = cu["efectivo"] + cu["cantidad"] * precio_valoracion
        con.execute("INSERT OR REPLACE INTO patrimonio(modo,vela_abre_ms,fecha,efectivo,cantidad,precio_valoracion,patrimonio,costos_acum,creado_utc) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (self.modo, abre_ms, fecha, cu["efectivo"], cu["cantidad"], precio_valoracion, pat, cu["costos_acum"], ahora()))
        return pat
