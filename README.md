# Time Series Forecasting — Demanda Eléctrica (PJM)

> **Pronosticar la demanda eléctrica horaria con un MAPE de 7.9% (Prophet)**
> *EDA temporal, descomposición STL y forecasting con baseline, SARIMA y Prophet*

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![statsmodels](https://img.shields.io/badge/statsmodels-SARIMA-3776ab)](https://www.statsmodels.org)
[![Prophet](https://img.shields.io/badge/Prophet-forecasting-209cee)](https://facebook.github.io/prophet/)

---

## Objetivo

Modelar y pronosticar la demanda eléctrica de **PJM East (PJME)** — **145,366 horas de
2002 a 2018** (~16.6 años) — caracterizando su estacionalidad y comparando modelos contra un
baseline honesto.

---

## Resultados (horizonte de test: 90 días)

| Modelo | MAE (MW) | RMSE (MW) | MAPE |
|---|---|---|---|
| Seasonal-Naive (baseline) | 4,028 | 5,178 | 12.5 % |
| SARIMA | 7,569 | 9,297 | 21.2 % |
| **Prophet** | **2,613** | **3,368** | **7.9 %** |

> **Hallazgo clave:** solo **Prophet supera al baseline**. SARIMA (estacionalidad semanal sobre
> 2 años) **queda por debajo del seasonal-naive** porque no captura el ciclo **anual** en un
> horizonte de 90 días — la mejor demostración de por qué un baseline es obligatorio.

---

## Metodología

1. **EDA temporal** (`01_EDA.ipynb`) — serie completa, estacionalidad (hora/día/mes), tendencia, ADF.
2. **Descomposición** (`02_decomposition.ipynb`) — STL (tendencia+estacional+residuo), ADF +
   diferenciación, ACF/PACF, **split cronológico** (nunca aleatorio).
3. **Modelado** (`03_modeling.ipynb`) — **baseline estacional-naive**, SARIMA y Prophet;
   comparación MAE/RMSE/MAPE y **pronóstico a 30 días** del mejor modelo.

> La Fase 4 (LSTM) del roadmap es opcional y se omite: Prophet + baseline ya constituyen un
> pipeline de forecasting sólido y reproducible.

---

## Estacionalidad y pronóstico

La demanda tiene **triple estacionalidad**: diaria (pico de tarde), semanal (menor en fin de
semana) y anual (picos de verano e invierno por climatización).

![Pronóstico 30 días](reports/10_forecast_30days.png)

---

## Estructura

```
time-series-forecasting/
├── data/                         # CSVs PJM (no versionado) + splits.pkl
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_decomposition.ipynb
│   └── 03_modeling.ipynb
├── reports/                      # 11 visualizaciones + metrics.json
├── HALLAZGOS.md
├── README.md
└── ROADMAP.md
```

---

## Cómo ejecutar

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/01_EDA.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_decomposition.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_modeling.ipynb
```

> Dataset: [Hourly Energy Consumption — Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
> (serie `PJME_hourly.csv`; no se versiona). En Windows, Prophet requiere `pip install prophet`.

> Detalle de detecciones y aprendizajes en [`HALLAZGOS.md`](HALLAZGOS.md).

---

## Autor

**Omar Mora Flores** · Data Analyst & ML Engineer
 omar13mor@gmail.com · [linkedin.com/in/omar-mora-flores](https://linkedin.com/in/omar-mora-flores)
