"""Versiona modelo, escalador y contrato de variables en backend/modelo_aprobado/ para que el agente los cargue.

Uso:
  python scripts/publicar_modelo.py                 # usa resultados/ (corrida final en Colab T4)
  python scripts/publicar_modelo.py resultados_dev  # corrida local de desarrollo (PROVISIONAL, no es evidencia de T4)
"""
import json
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
origen = RAIZ / (sys.argv[1] if len(sys.argv) > 1 else "resultados")
destino = RAIZ / "backend" / "modelo_aprobado"

if not (origen / "modelo").exists():
    sys.exit(f"No existe {origen / 'modelo'}. Ejecute primero el notebook (en Colab T4 para la entrega final).")

destino.mkdir(parents=True, exist_ok=True)
for f in (origen / "modelo").iterdir():
    shutil.copy(f, destino / f.name)
for extra in ("configuracion_final.json", "evidencia_T4.txt"):
    if (origen / extra).exists():
        shutil.copy(origen / extra, destino / extra)

contrato = json.loads((destino / "contrato_variables.json").read_text(encoding="utf-8"))
gpu = (contrato.get("entorno") or {}).get("gpu") or ""
proc = {
    "origen": origen.name,
    "entrenado_en_T4": "T4" in gpu,
    "gpu_entrenamiento": gpu,
    "provisional": "T4" not in gpu,
    "nota": "Modelo provisional (no entrenado en T4): reemplácelo con la corrida final de Colab." if "T4" not in gpu
            else "Entrenado en NVIDIA T4 (evidencia en evidencia_T4.txt).",
}
(destino / "procedencia.json").write_text(json.dumps(proc, indent=2, ensure_ascii=False), encoding="utf-8")
print("Modelo publicado en", destino, "->", proc["nota"])
