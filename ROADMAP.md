# Roadmap — Time Series Forecasting

**Proyecto de portfolio:** Omar Mora Flores  
**Objetivo:** Predecir tendencias futuras usando datos históricos, demostrando habilidades en análisis de series de tiempo, detección de estacionalidad y modelos de forecasting.

**Caso de uso sugerido:** Pronóstico de demanda de energía eléctrica (dataset público disponible en Kaggle)  
**Dataset recomendado:** [Hourly Energy Consumption — Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) *(o ventas, tráfico web, según preferencia)*

---

## Estado general

| Fase | Componente | Estado |
|---|---|---|
| 0 | Setup del entorno | ✅ Completado |
| 1 | EDA & Análisis temporal | ✅ Completado |
| 2 | Descomposición de la serie | ✅ Completado |
| 3 | Modelado — ARIMA & Prophet | ✅ Completado (+ baseline seasonal-naive) |
| 4 | (Opcional) LSTM — modelo avanzado | ⏭️ Opcional — omitida |
| 5 | Cierre de portfolio | ✅ Completado |

Leyenda: ⬜ Pendiente · 🔄 En progreso · ✅ Completado

---

## Fase 0 — Setup del entorno

- [ ] Descargar dataset y colocar en `data/`
- [ ] Crear `requirements.txt` con: pandas, numpy, matplotlib, seaborn, statsmodels, prophet, scikit-learn, plotly
- [ ] Crear `.gitignore` (excluir `data/`, `*.csv`, `__pycache__/`)
- [ ] Cargar serie y confirmar índice de tiempo correcto (`DatetimeIndex`)

### Entregables
- `requirements.txt` funcional
- Serie cargada con índice temporal

---

## Fase 1 — EDA & Análisis temporal

**Archivo:** `notebooks/01_EDA.ipynb`  
**Pregunta central:** ¿Qué patrones, tendencias y estacionalidades tiene la serie?

### Secciones

#### 1.1 Carga y descripción
- [ ] Shape, rango de fechas, frecuencia de muestreo, valores nulos
- [ ] Resamplear si es necesario (e.g., horario → diario)

#### 1.2 Visualización de la serie completa
- [ ] Gráfico de línea de la serie completa
- [ ] Zoom en períodos específicos (una semana, un mes)
- [ ] Guardar → `reports/01_full_series.png`

#### 1.3 Patrones de estacionalidad
- [ ] Promedio por hora del día, día de la semana, mes del año
- [ ] Heatmap: hora × día de semana con intensidad de consumo
- [ ] Guardar → `reports/02_seasonality_patterns.png`

#### 1.4 Análisis de tendencia
- [ ] Media móvil de 7 y 30 días superpuesta a la serie
- [ ] Guardar → `reports/03_trend.png`

#### 1.5 Detección de outliers
- [ ] IQR o Z-score para identificar valores atípicos en la serie
- [ ] Visualizar outliers sobre la serie original

#### 1.6 Conclusiones
- [ ] ¿La serie es estacionaria? (test ADF — Augmented Dickey-Fuller)
- [ ] ¿Qué tipo de estacionalidad tiene? (diaria, semanal, anual)

### Entregables
- `notebooks/01_EDA.ipynb` sin errores
- 3 imágenes en `reports/`

---

## Fase 2 — Descomposición de la serie

**Archivo:** `notebooks/02_decomposition.ipynb`

### Tareas

#### 2.1 Descomposición STL / Clásica
- [ ] Aplicar descomposición aditiva o multiplicativa con `statsmodels`
- [ ] Visualizar: Tendencia + Estacionalidad + Residuo
- [ ] Guardar → `reports/04_decomposition.png`

#### 2.2 Test de estacionariedad
- [ ] Test ADF (Augmented Dickey-Fuller)
- [ ] Si no es estacionaria: aplicar diferenciación y re-testear

#### 2.3 ACF y PACF
- [ ] Gráficos de autocorrelación y autocorrelación parcial
- [ ] Usar para identificar parámetros p, d, q del ARIMA
- [ ] Guardar → `reports/05_acf_pacf.png`

#### 2.4 Split train/test cronológico
- [ ] Train: histórico hasta N meses antes del final
- [ ] Test: últimos N meses (nunca aleatorio en series de tiempo)
- [ ] Guardar splits en `data/splits.pkl`

### Entregables
- `notebooks/02_decomposition.ipynb` sin errores
- `data/splits.pkl`
- 2 imágenes en `reports/`

---

## Fase 3 — Modelado — ARIMA & Prophet

**Archivo:** `notebooks/03_modeling.ipynb`

### Métricas de evaluación
- MAE — error absoluto promedio en unidades reales
- RMSE — penaliza errores grandes
- MAPE (%) — error porcentual medio (interpretable para negocio)

### Secciones

#### 3.1 Modelo ARIMA
- [ ] Seleccionar parámetros (p, d, q) con AIC/BIC o ACF/PACF
- [ ] Entrenar sobre train set
- [ ] Predecir sobre test set y calcular métricas
- [ ] Visualizar predicción vs. real → `reports/06_arima_forecast.png`

#### 3.2 Modelo Prophet (Facebook/Meta)
- [ ] Preparar datos en formato `ds` / `y` que requiere Prophet
- [ ] Entrenar con estacionalidades activadas (diaria, semanal, anual según corresponda)
- [ ] Predecir sobre test set y calcular métricas
- [ ] Visualizar componentes (tendencia + estacionalidades) → `reports/07_prophet_components.png`
- [ ] Visualizar predicción vs. real → `reports/08_prophet_forecast.png`

#### 3.3 Comparación de modelos
- [ ] Tabla de MAE / RMSE / MAPE para ARIMA y Prophet
- [ ] Gráfico comparativo de predicciones → `reports/09_model_comparison.png`

#### 3.4 Pronóstico a 30 días con el mejor modelo
- [ ] Generar predicción de los próximos 30 días
- [ ] Visualizar con intervalo de confianza
- [ ] Guardar → `reports/10_forecast_30days.png`

### Entregables
- `notebooks/03_modeling.ipynb` sin errores
- Pronóstico a 30 días generado
- 5 imágenes en `reports/`

---

## Fase 4 — (Opcional) LSTM — Modelo Avanzado

**Archivo:** `notebooks/04_lstm.ipynb`  
**Requerimientos adicionales:** `tensorflow` o `pytorch`

### Tareas
- [ ] Crear secuencias de entrada (ventana deslizante de N pasos)
- [ ] Arquitectura LSTM básica (1-2 capas)
- [ ] Entrenar y evaluar sobre test set
- [ ] Comparar métricas contra ARIMA y Prophet
- [ ] Guardar → `reports/11_lstm_forecast.png`

> **Nota:** Esta fase es opcional. Incluirla si quieres mostrar conocimiento de deep learning. ARIMA + Prophet ya es un portfolio sólido.

---

## Fase 5 — Cierre de portfolio

- [ ] Ejecutar todos los notebooks sin errores en entorno limpio
- [ ] Escribir `README.md` con: contexto del problema, metodología, resultados (con métricas reales), instrucciones de ejecución
- [ ] Incluir gráfico de pronóstico a 30 días en el README
- [ ] `.gitignore` excluye el dataset

### Checklist de calidad
- [ ] Narrativa en Markdown entre celdas
- [ ] Gráficos con títulos, ejes etiquetados y leyendas
- [ ] Split siempre cronológico — nunca aleatorio
- [ ] `random_state=42` donde aplique

---

## Orden de desarrollo

```
Fase 0 → Fase 1 → Fase 2 → Fase 3 → (Fase 4) → Fase 5
  Setup    EDA    Decomp   Modeling    LSTM      Cierre
```

---

## Notas técnicas

- En series de tiempo el split **nunca es aleatorio** — siempre cronológico para simular predicción real
- Prophet maneja mejor múltiples estacionalidades y días festivos que ARIMA
- ARIMA requiere que la serie sea estacionaria; aplicar diferenciación si el test ADF falla
- El MAPE (%) es la métrica más comunicable a stakeholders no técnicos
