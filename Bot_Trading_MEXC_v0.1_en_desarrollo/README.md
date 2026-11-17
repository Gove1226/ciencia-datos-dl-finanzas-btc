# Bot de trading MEXC (BTC/USDT) — v0.1.0 · en desarrollo

**Versión del proyecto:** `0.1.0` (trabajo en progreso). Bot de simulación en papel que decide con un **LLM local** y controles de riesgo deterministas.

> **Estado: avance parcial.** El backend funciona y está probado; la interfaz de usuario (Vue) y la documentación final están pendientes. Ver «Pendiente».

## Principios de seguridad (indispensables)
- **El LLM es solo local:** únicamente se consume un modelo de [Ollama](https://ollama.com) en `127.0.0.1`. Cualquier otro host se **rechaza al arrancar**.
- **Nunca se pasan APIs, claves ni credenciales a la IA.** El LLM solo recibe datos de mercado ya procesados (precio, indicadores, pronóstico, posición). Una *guardia*
  (`backend/app/seguridad.py`) revisa todo texto que saldría hacia el modelo y **bloquea** el valor de cualquier clave configurada, cabeceras/palabras de credenciales y direcciones/URLs de APIs.
- **Las claves viven solo en el servidor** (`.env`, fuera del repositorio). No hay secretos en el frontend ni en el código.
- **MEXC v0.1:** solo **datos públicos** (velas diarias) — no requiere claves — y simulación en papel. **Las órdenes reales están deshabilitadas**; no existe código que las envíe.

## Tecnologías
| Capa | Tecnología |
|---|---|
| Backend | Python 3.12, FastAPI, Uvicorn, Pydantic, SQLite |
| Modelo predictivo | PyTorch (MLP entrenado en Colab con GPU NVIDIA T4), scikit-learn, pandas, NumPy |
| LLM local | Ollama con `qwen2.5:7b` (cuantizado Q4_K_M) |
| Integración de mercado | MEXC API pública (`/api/v3/klines`) |
| Interfaz (en desarrollo) | TypeScript + Vite + Vue |
| Pruebas | pytest |

## Cómo funciona un ciclo
1. Consulta velas **cerradas** de MEXC y las valida (activo, frecuencia, orden, antigüedad).
2. Calcula las variables con la misma definición del entrenamiento y obtiene el **pronóstico** del modelo.
3. Pide al **LLM local** una propuesta (`comprar` / `mantener` / `vender`) en JSON, con herramientas de **solo lectura**.
4. Valida la respuesta (tipos, acción, símbolo, fecha) y aplica **controles de riesgo deterministas** (exposición, pérdida diaria, sin apalancamiento ni cortos). El LLM no puede saltárselos.
5. Ejecuta en **Paper** (dinero virtual), guarda todo en SQLite (idempotente por vela) y espera la siguiente vela.

## Ejecutar (local)
```powershell
py -3.12 -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
ollama pull qwen2.5:7b
python scripts/crear_env.py                      # crea .env con claves aleatorias (nunca se sube)
uvicorn backend.app.main:app --port 8000         # API en http://127.0.0.1:8000/docs
python -m pytest -q                              # pruebas
```

## Hecho
- Cliente MEXC de datos públicos con reintentos y espera creciente; validación de velas (nunca usa una vela abierta; rechaza datos atrasados).
- Modelo versionado (`backend/modelo_aprobado/`, entrenado en T4) con su escalador y contrato de variables.
- Agente con LLM local, JSON validado, riesgo determinista, simulador Paper, SQLite, reinicio sin duplicar operaciones, replay histórico separado de la conexión actual.
- API autenticada con clave (`X-API-Key`) y API privada del LLM con otra clave.
- **50 pruebas automatizadas** (riesgo, Paper, datos, LLM, idempotencia, autenticación y seguridad del LLM).

## Pendiente (v0.1.0 → v0.1 final)
- Interfaz **Vue + TypeScript + Vite** (hay un prototipo en React que será migrado).
- README final con capturas, y reproducir el replay con el modelo de T4 incluido aquí.
- Opcional: integración con Polymarket (no incluida).

> Proyecto académico. No es asesoría financiera.
