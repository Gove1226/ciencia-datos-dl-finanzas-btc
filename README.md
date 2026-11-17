# Deep Learning para Finanzas — BTC-USD diario

Práctica de la materia **Ciencia de Datos** (Prof. Rubén Miguel Rodríguez Rangel): del pronóstico de rendimientos a las bases de un agente de trading.
Este repositorio muestra el **progreso por avances** (un avance por capítulo del libro *Deep Learning for Finance*, capítulos 7 a 11).

**Estado:** Avances 1 a 4 entregados · Avance 5 (capítulo 11) en preparación · Bot de trading MEXC v0.1.0 en desarrollo (ver tabla).
**Activo:** Bitcoin (BTC-USD), frecuencia diaria, 1,853 observaciones (2021-10-07 → 2026-11-02), fuente Yahoo Finance vía `yfinance`.
**Equipo:** Alexander Góngora y Diego Muñoz.

## Qué se predice
Al cierre del día *t* se estima el rendimiento del siguiente día, `y_t = P_{t+1} / P_t − 1`, con los **20 rendimientos más recientes**.
La partición es **cronológica** por fecha del objetivo: TR 70 % (1,269 ejemplos) · VA 15 % (272) · TE 15 % (272). **TE no se ha usado en los avances 1 a 4**; se evaluará una sola vez en el Avance 5.

## Avances

| Avance | Capítulo | Tema | Dónde está |
|---|---|---|---|
| 1 | 7 | Datos auditados, partición TR–VA–TE, pronóstico cero y regresión lineal, hipótesis técnica | [`Avances_1_a_3_Capitulos_7_a_9/`](Avances_1_a_3_Capitulos_7_a_9/) (`Avance1_Capitulo7_BTC.ipynb`) |
| 2 | 8 | MLP y LSTM entrenados en **Google Colab con GPU NVIDIA T4** | [`Avance2_Capitulo8_BTC_MLP_LSTM.ipynb`](Avances_1_a_3_Capitulos_7_a_9/Avance2_Capitulo8_BTC_MLP_LSTM.ipynb) |
| 3 | 9 | Validación temporal (3 ventanas expansivas), umbrales y regla de señal | [`Avance3_Capitulo9_BTC_Validacion_Umbrales.ipynb`](Avances_1_a_3_Capitulos_7_a_9/Avance3_Capitulo9_BTC_Validacion_Umbrales.ipynb) |
| 4 | 10 | Diseño del entorno del agente (estado, acciones, recompensa) y verificación del simulador | [`Avance_4_Capitulo_10/`](Avance_4_Capitulo_10/) |
| — | — | **Bot de trading MEXC `v0.1.0` (en desarrollo)**: LLM local + controles de riesgo + simulador en papel; nunca se pasan claves ni APIs a la IA | [`Bot_Trading_MEXC_v0.1_en_desarrollo/`](Bot_Trading_MEXC_v0.1_en_desarrollo/) |

Cada carpeta incluye un `LEEME.txt` con cómo ejecutarla, además de `datos/`, `dlfin/` (código de apoyo) y `salidas*/` (resultados).

## Resultados hasta ahora (solo TR y VA)

| Modelo | MAE VA | RMSE VA |
|---|---|---|
| Pronóstico cero (referencia) | 0.01421 | **0.01926** |
| Regresión lineal (20 rezagos) | 0.01455 | 0.01968 |
| LSTM (32), media de 3 semillas | 0.01424 | 0.01931 |
| MLP (64), media de 3 semillas | 0.01476 | 0.01993 |

- **Ningún modelo supera de forma material al pronóstico cero en VA**: los rendimientos diarios de BTC son muy ruidosos.
- **Avance 3:** la regla «comprar si el estimado del LSTM supera 0.20 %» se eligió **solo con VA** (costo 0.10 % por operación): +1.00 % neto con 3 operaciones y drawdown 2.4 %, frente a −22.2 % de comprar y mantener en el mismo periodo.
  Es un resultado en un solo periodo de validación; **no es evidencia de rentabilidad futura**.
- **Avance 4:** el entorno del agente reproduce al centavo el simulador del Avance 3, no regala recompensas (la suma de recompensas = patrimonio final − capital) y no usa información futura. Detalle y transición calculada a mano en el notebook.

## Tecnologías
- **Lenguaje y análisis:** Python, NumPy, pandas, scikit-learn, Matplotlib.
- **Deep learning:** TensorFlow / Keras (MLP y LSTM), entrenamiento en **Google Colab (GPU NVIDIA T4)**.
- **Datos:** `yfinance` (Yahoo Finance) y SQLite (base original del curso); datos originales y procesados incluidos en cada avance.
- **Entorno de ejecución:** Jupyter / Google Colab; el Avance 4 corre en CPU.

## Cómo ejecutar
1. Abre la carpeta del avance y lee su `LEEME.txt`.
2. Avances 1 y 4: Jupyter/VS Code o Colab → *Ejecutar todo* (CPU).
3. Avances 2 y 3: Colab con GPU **T4** (*Entorno de ejecución → Cambiar tipo → T4 GPU*); el notebook se detiene si no hay T4.

## Notas
- Los notebooks se guardan **ya ejecutados** (con salidas) para que se pueda revisar sin volver a correrlos.
- No hay claves ni credenciales en este repositorio; los datos son públicos.
- Material académico: no constituye asesoría financiera.
