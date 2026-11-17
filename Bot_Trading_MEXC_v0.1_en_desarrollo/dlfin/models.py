"""Modelos de referencia (cero, regresión lineal), MLP y LSTM en PyTorch, y verificación de GPU/T4."""
from __future__ import annotations

import copy
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .errors import Bitacora, EntrenamientoError, GPUError, VentanaError


# ----------------------------------------------------------------------------- referencias
class PronosticoCero:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class RegresionLineal:
    """OLS sobre los mismos rezagos (rendimientos), aplanados."""

    def __init__(self):
        self.m = LinearRegression()

    def fit(self, X, y):
        self.m.fit(X.reshape(len(X), -1), y)
        return self

    def predict(self, X):
        return self.m.predict(X.reshape(len(X), -1))


# ----------------------------------------------------------------------------- GPU / T4
def nvidia_smi() -> str:
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30, check=True).stdout
    except (FileNotFoundError, subprocess.SubprocessError) as e:
        return f"nvidia-smi no disponible: {e}"


def info_entorno() -> dict:
    import sklearn
    import torch

    info = {
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "pandas": pd.__version__, "sklearn": sklearn.__version__, "cuda_disponible": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "so": platform.platform(),
    }
    return info


def verificar_gpu(exigir_t4: bool = True, bitacora: Bitacora | None = None) -> str:
    """Devuelve 'cuda' si la GPU es utilizable. Con exigir_t4=True detiene el flujo si no hay una T4.

    Además de reconocer la GPU, ejecuta una operación en ella (el menú de Colab no basta).
    """
    import torch

    if not torch.cuda.is_available():
        msg = "PyTorch no reconoce ninguna GPU. En Colab: Entorno de ejecución > Cambiar tipo > GPU T4, y reinicie."
        if bitacora:
            bitacora.registrar("T4", "GPU no disponible", msg, "se detiene el entrenamiento profundo", "pendiente reprogramar")
        if exigir_t4:
            raise GPUError(msg)
        return "cpu"
    nombre = torch.cuda.get_device_name(0)
    if exigir_t4 and "T4" not in nombre:
        msg = f"Se requiere NVIDIA T4 y se detectó '{nombre}'. Reprograme la ejecución hasta obtener una T4."
        if bitacora:
            bitacora.registrar("T4", "GPU distinta de T4", msg, "se detiene el entrenamiento profundo", "pendiente reprogramar")
        raise GPUError(msg)
    a = torch.randn(256, 256, device="cuda")
    s = float((a @ a).sum().item())
    if not np.isfinite(s):
        raise GPUError("La operación de prueba en GPU devolvió un valor no finito.")
    return "cuda"


# ----------------------------------------------------------------------------- redes
def _torch():
    import torch
    import torch.nn as nn

    return torch, nn


def crear_red(tipo: str, T: int, F: int, neuronas: int = 32, capas: int = 2, dropout: float = 0.1):
    torch, nn = _torch()
    if tipo == "mlp":
        capas_l, d = [], T * F
        for _ in range(capas):
            capas_l += [nn.Linear(d, neuronas), nn.ReLU(), nn.Dropout(dropout)]
            d = neuronas
        capas_l.append(nn.Linear(d, 1))

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(*capas_l)

            def forward(self, x):
                return self.net(x.reshape(x.shape[0], -1)).squeeze(-1)

        return MLP()
    if tipo == "lstm":
        class LSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.LSTM(F, neuronas, num_layers=1, batch_first=True)
                self.drop = nn.Dropout(dropout)
                self.out = nn.Linear(neuronas, 1)

            def forward(self, x):
                if x.ndim != 3:
                    raise VentanaError(f"LSTM espera (ejemplos, pasos, variables); recibió {tuple(x.shape)}.")
                h, _ = self.rnn(x)
                return self.out(self.drop(h[:, -1, :])).squeeze(-1)

        return LSTM()
    raise ValueError(f"Tipo de red desconocido: {tipo}")


def fijar_semilla(s: int) -> None:
    torch, _ = _torch()
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def contar_parametros(m) -> int:
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad))


CFG_DEFECTO = dict(neuronas=32, capas=2, dropout=0.1, lr=1e-3, batch=64, max_epocas=200, paciencia=20, weight_decay=0.0)


def entrenar(tipo: str, Xtr, ytr, Xva, yva, cfg: dict | None = None, semilla: int = 0,
             device: str = "cpu", ckpt: str | Path | None = None, bitacora: Bitacora | None = None):
    """Entrena con parada temprana sobre la pérdida de VA y restaura los mejores pesos.

    Xtr/Xva ya escalados, forma (n, T, F). Devuelve (modelo, historial, info).
    """
    torch, nn = _torch()
    c = {**CFG_DEFECTO, **(cfg or {})}
    for nombre, X in (("TR", Xtr), ("VA", Xva)):
        if X.ndim != 3:
            raise VentanaError(f"{nombre}: forma esperada (n, T, F), recibida {X.shape}.")
    if Xtr.shape[1:] != Xva.shape[1:]:
        raise VentanaError(f"Formas TR {Xtr.shape[1:]} y VA {Xva.shape[1:]} no coinciden.")
    if len(Xtr) < c["batch"] or len(Xva) == 0:
        raise VentanaError(f"Ejemplos insuficientes: TR={len(Xtr)}, VA={len(Xva)}, lote={c['batch']}.")
    if not (np.isfinite(Xtr).all() and np.isfinite(ytr).all() and np.isfinite(Xva).all() and np.isfinite(yva).all()):
        raise EntrenamientoError("Hay valores no finitos en las entradas/objetivos; revise escalas y datos.")

    batch = c["batch"]
    while True:
        try:
            fijar_semilla(semilla)
            T, F = Xtr.shape[1], Xtr.shape[2]
            modelo = crear_red(tipo, T, F, c["neuronas"], c["capas"], c["dropout"]).to(device)
            opt = torch.optim.Adam(modelo.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
            perdida = nn.MSELoss()
            xt, yt = torch.tensor(Xtr, device=device), torch.tensor(ytr, device=device)
            xv, yv = torch.tensor(Xva, device=device), torch.tensor(yva, device=device)
            g = torch.Generator().manual_seed(semilla)
            hist, mejor, mejor_ep, sin_mejora = [], float("inf"), 0, 0
            mejores_pesos = None
            t0 = time.time()
            for ep in range(1, c["max_epocas"] + 1):
                modelo.train()
                perm = torch.randperm(len(xt), generator=g).to(device)
                tot = 0.0
                for i in range(0, len(xt), batch):
                    b = perm[i:i + batch]
                    opt.zero_grad()
                    l = perdida(modelo(xt[b]), yt[b])
                    if not torch.isfinite(l):
                        raise EntrenamientoError(
                            f"Pérdida no finita en época {ep}. Revise escalas, datos y la tasa de aprendizaje (lr={c['lr']}).")
                    l.backward()
                    opt.step()
                    tot += l.item() * len(b)
                modelo.eval()
                with torch.no_grad():
                    lv = float(perdida(modelo(xv), yv))
                if not np.isfinite(lv):
                    raise EntrenamientoError(f"Pérdida de VA no finita en época {ep}.")
                hist.append({"epoca": ep, "perdida_tr": tot / len(xt), "perdida_va": lv})
                if lv < mejor - 1e-7:
                    mejor, mejor_ep, sin_mejora = lv, ep, 0
                    mejores_pesos = copy.deepcopy(modelo.state_dict())
                    if ckpt:
                        Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
                        torch.save({"state_dict": mejores_pesos, "tipo": tipo, "cfg": c, "T": T, "F": F,
                                    "epoca": ep, "perdida_va": lv}, ckpt)
                else:
                    sin_mejora += 1
                    if sin_mejora >= c["paciencia"]:
                        break
            modelo.load_state_dict(mejores_pesos)
            modelo.eval()
            info = {"tipo": tipo, "cfg": {**c, "batch": batch}, "semilla": semilla, "dispositivo": device,
                    "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                    "epoca_mejor": mejor_ep, "epocas_ejecutadas": len(hist), "mejor_perdida_va": mejor,
                    "parametros_entrenables": contar_parametros(modelo), "tiempo_s": time.time() - t0,
                    "optimizador": "Adam", "perdida": "MSE (y estandarizado con TR)",
                    "forma_entrada": tuple(Xtr.shape)}
            return modelo, pd.DataFrame(hist), info
        except torch.cuda.OutOfMemoryError:
            if bitacora:
                bitacora.registrar("entrenamiento", "memoria insuficiente (CUDA OOM)", f"lote={batch}",
                                   f"reducir lote a {max(batch // 2, 1)} y reintentar", "pendiente")
            if batch <= 1:
                raise
            batch //= 2
            torch.cuda.empty_cache()


def predecir(modelo, X, device: str = "cpu") -> np.ndarray:
    torch, _ = _torch()
    modelo.eval()
    with torch.no_grad():
        z = modelo(torch.tensor(X, device=device)).cpu().numpy()
    if not np.isfinite(z).all():
        raise EntrenamientoError("El modelo produjo predicciones no finitas; revise escalas y datos de entrada.")
    return z


def cargar_checkpoint(ruta, device: str = "cpu"):
    torch, _ = _torch()
    ck = torch.load(ruta, map_location=device, weights_only=False)
    c = ck["cfg"]
    m = crear_red(ck["tipo"], ck["T"], ck["F"], c["neuronas"], c["capas"], c["dropout"])
    m.load_state_dict(ck["state_dict"])
    m.to(device).eval()
    return m, ck
