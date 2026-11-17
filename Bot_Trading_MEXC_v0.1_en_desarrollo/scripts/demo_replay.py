"""Replay histórico (MEXC) de N ciclos diarios: agente con LLM vs modelo sin LLM vs comprar y mantener.

Es un REPLAY (no la conexión actual): usa cuentas Paper separadas. Resultados en resultados_agente/.
Uso: python scripts/demo_replay.py [N=20]
"""
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd  # noqa: E402

from backend.app.config import cargar_ajustes  # noqa: E402
from backend.app.main import construir_agente  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
ag = construir_agente(cargar_ajustes())
print(f"Replay de {n} ciclos con LLM {ag.llm.modelo} ...", flush=True)
res = ag.replay(n, con_llm=True)
print("estado:", res.get("estado"), res.get("mensaje", ""))
out = RAIZ / "resultados_agente"
out.mkdir(exist_ok=True)
if res.get("estado") == "terminado":
    tabla = pd.DataFrame(res["metricas"]).T
    tabla.to_csv(out / "comparacion_replay.csv")
    print(tabla.to_string())
    for modo in ("replay_llm", "replay_modelo", "replay_bh"):
        pd.DataFrame(ag.store.decisiones(modo, 1000)).to_csv(out / f"decisiones_{modo}.csv", index=False)
        pd.DataFrame(ag.store.ordenes(modo, 1000)).to_csv(out / f"ordenes_{modo}.csv", index=False)
        pd.DataFrame(ag.store.patrimonio(modo, 5000)).to_csv(out / f"patrimonio_{modo}.csv", index=False)
    (out / "replay_estado.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
