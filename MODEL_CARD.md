# Model Card — Forecasting de demanda eléctrica (PJM East)

Ficha del modelo que pronostica la demanda eléctrica horaria de PJM East. Una página para dejar
claro qué hace, con qué datos, cómo se evaluó, y sobre todo **cuándo NO confiarse**.

## Resumen

- **Tarea:** pronóstico (regresión sobre el tiempo) de la demanda horaria `PJME_MW`. Horizonte
  por defecto: **30 días** (720 h).
- **Modelo servido:** **Prophet** con estacionalidad diaria + semanal + anual. Es el que gana el
  backtesting, así que es el que se persiste y sirve.
- **Modelos comparados:** un baseline **seasonal-naive** (misma hora de la semana pasada),
  **SARIMAX + Fourier** (estacionalidad múltiple como regresores) y **Prophet**.
- **Versión:** 1.1.0 · **Artefacto:** `src/forecast_model.json` (Prophet serializado con
  `model_to_json`). El CLI (`src/forecast.py`) y la API (`src/api.py`) lo cargan y proyectan a
  futuro **sin reentrenar**.

## Para qué sirve (y para qué no)

- **Uso previsto:** demostración de portfolio — pronosticar la curva de demanda a corto/medio
  plazo y comparar modelos de forecasting de forma honesta contra un baseline.
- **Fuera de alcance:** no es un sistema de operación de red ni de despacho energético. No modela
  los **drivers físicos reales** de la demanda (temperatura, actividad económica, festivos); solo
  aprende el patrón histórico de la propia serie.

## Datos

- **Hourly Energy Consumption** (Kaggle), serie `PJME_hourly.csv`: **145,366 horas** de 2002 a
  2018 (~16.6 años). El dataset **no se versiona** en el repo.
- **Ventana de entrenamiento:** las últimas **26,280 h (~3 años)**. La serie completa son 16 años;
  entrenar SARIMA sobre todo eso es inviable y, además, el pasado lejano aporta poco al patrón
  actual. Tres años bastan para ver el ciclo anual completo.
- **Limpieza:** 4 timestamps duplicados y huecos de frecuencia → interpolados para tener un índice
  horario continuo.
- **Split cronológico** (nunca aleatorio): test = últimas **336 h (14 días)** de la ventana.

## Cómo funciona

- **Baseline seasonal-naive:** copia el valor de hace 168 h (misma hora de la semana pasada). Es
  la vara de medir obligatoria — a corto plazo es sorprendentemente difícil de batir.
- **SARIMAX + Fourier:** orden no estacional `(2,1,1)`; la estacionalidad (diaria 24 h, semanal
  168 h, anual 8766 h) entra como **términos de Fourier** (regresores exógenos), porque el
  `seasonal_order` clásico solo maneja una estacionalidad y se ahoga con la anual.
- **Prophet:** estacionalidad diaria + semanal + anual nativa. El modelo que se sirve se
  **reentrena con toda la ventana reciente** (train + test) para que el pronóstico arranque en la
  última fecha real, no a mitad del test.

## Evaluación

Una sola ventana engaña; lo que manda es el backtesting. Reportamos ambos.

**Ventana única (test = 14 días, 336 h)** — `reports/metrics.json`:

| Modelo | MAE | RMSE | MAPE |
|---|---|---|---|
| Seasonal-Naive | 3,277 | 4,245 | 9.2 % |
| SARIMAX + Fourier | 4,286 | 5,331 | 11.1 % |
| Prophet | 3,532 | 4,238 | 10.9 % |

**Backtesting walk-forward (4 ventanas, media ± desviación)** — `reports/backtest_metrics.json`:

| Modelo | MAPE medio | desviación |
|---|---|---|
| Seasonal-Naive | 13.9 % | 3.6 |
| SARIMAX + Fourier | 13.7 % | 3.6 |
| **Prophet** | **13.0 %** | **2.3** |

> En la ventana única el naive parecía el mejor; el backtesting lo desmiente. **Prophet gana de
> forma robusta** — menor MAPE medio y, sobre todo, **menor varianza** (más consistente entre
> ventanas). Por eso es el modelo que se sirve.

## Límites y consideraciones

- **Sin variables exógenas:** el mayor driver de la demanda eléctrica es la **temperatura**
  (calefacción/aire), seguida de festivos y actividad económica. El modelo no las usa, así que un
  invierno atípico o una ola de calor lo despistarán. Incorporarlas es la mejora con más techo.
- **No anticipa shocks:** apagones, eventos extremos o cambios estructurales de consumo no están
  en el patrón histórico; el forecast no los verá venir.
- **Degradación con el horizonte:** el MAPE crece cuanto más lejos se pronostica. A 30 días es una
  guía de tendencia, no un valor hora a hora fiable.
- **Ventana de ~3 años:** ignora deliberadamente el pasado lejano. Si la demanda cambió de régimen
  hace poco, conviene revisar el tamaño de ventana.
- **Datos hasta 2018:** la serie termina en 2018; el modelo no conoce nada posterior.

## Monitoreo y mantenimiento

- **Seguir el error en el tiempo:** comparar el pronóstico contra el real conforme llega y vigilar
  el MAPE. Si se dispara de forma sostenida, el modelo se quedó viejo (cambió el patrón de demanda).
- **Reentrenamiento periódico:** como la ventana es móvil (~3 años), reentrenar cada cierto tiempo
  mantiene el modelo al día. Es un `python -m src.pipeline` — automatizable con cron o un workflow
  programado de GitHub Actions (`schedule:`), que regeneraría `src/forecast_model.json`.
- `reports/experiments.csv` guarda cada corrida (modelo, métricas) para comparar en el tiempo.

## Cómo usarlo

- Entrenar/evaluar todo: `python -m src.pipeline`.
- Pronóstico: `python -m src.forecast --days 30` (CLI) o `uvicorn src.api:app` y `GET /forecast?days=N`.

---
*Autor: Omar Mora Flores · Última actualización: 2026-06-16*
