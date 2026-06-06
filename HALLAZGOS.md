# 🔎 Hallazgos y Aprendizajes — Time Series Forecasting (PJM)

> Detecciones del análisis temporal y del forecasting, más los aprendizajes del proyecto.

**Autor:** Omar Mora Flores · **Última actualización:** 2026-06-06

---

## 🧭 Resumen ejecutivo

Sobre **16.6 años** de demanda eléctrica horaria de PJM East, **Prophet** pronostica a 90 días
con **MAPE 7.9%**, superando claramente al baseline estacional-naive (12.5%). El aprendizaje
metodológico central: **sin un baseline no se puede juzgar un modelo** — SARIMA, pese a su
complejidad, quedó **por debajo** del baseline en este horizonte.

---

## 📊 Detecciones del EDA y descomposición (Fases 1-2)

| # | Detección | Evidencia |
|---|---|---|
| 1 | **Triple estacionalidad:** diaria (pico de tarde), semanal (baja en finde) y anual (verano/invierno) | `02_seasonality_patterns.png`, `02b_heatmap.png` |
| 2 | **Tendencia plana** a largo plazo (media móvil 365d estable) | `03_trend.png` |
| 3 | La serie diaria **no es estacionaria**, pero **1 diferenciación la estabiliza** (ADF p<0.05) | `02_decomposition` |
| 4 | 4 timestamps duplicados y huecos de frecuencia → interpolados | `01_EDA` |

---

## 🔮 Detecciones del modelado (Fase 3) — horizonte 90 días

| Modelo | MAE | RMSE | MAPE |
|---|---|---|---|
| Seasonal-Naive (baseline) | 4,028 | 5,178 | 12.5 % |
| SARIMA | 7,569 | 9,297 | 21.2 % |
| **Prophet** ✅ | **2,613** | **3,368** | **7.9 %** |

1. **Prophet gana** porque modela de forma nativa la estacionalidad **anual + semanal**.
2. **SARIMA queda por debajo del baseline:** con estacionalidad solo semanal (m=7) sobre 2 años,
   no captura el ciclo anual y se desvía en 90 días. *Más complejidad ≠ mejor modelo.*
3. El **baseline estacional-naive** (demanda del mismo día del año anterior) es sorprendentemente
   competitivo — la vara de medir correcta.

---

## 🎓 Aprendizajes

**Técnicos**
1. **Siempre un baseline primero.** El seasonal-naive define el umbral que cualquier modelo debe
   superar para justificar su complejidad.
2. **Split cronológico, nunca aleatorio:** en series de tiempo, mezclar fechas filtra el futuro
   en el entrenamiento.
3. **La estacionalidad correcta importa más que el algoritmo:** SARIMA con m=7 ignoró el ciclo
   anual; Prophet con `yearly_seasonality=True` lo capturó.
4. **Entrenar SARIMA en una ventana reciente** (2 años) acelera el ajuste sin perder relevancia.
5. **MAPE para negocio, RMSE para selección:** el MAPE (%) es interpretable por stakeholders.

**De proceso**
6. **Prophet en Windows** requirió instalación dedicada (`pip install prophet`); se verificó el
   *import* antes de depender de él (plan B: Holt-Winters de statsmodels).
7. **Validar el roadmap contra los datos:** se fijó `PJME_hourly.csv` como serie canónica entre
   los 12 CSVs regionales disponibles.

---

## ⚠️ Limitaciones y próximos pasos

- [ ] **Backtesting / validación walk-forward** en varias ventanas (no un único split de 90 días).
- [ ] SARIMA con estacionalidad anual (vía variables de Fourier o SARIMAX con exógenas) para una
      comparación más justa frente a Prophet.
- [ ] Incorporar **variables exógenas** (temperatura, festivos) — principal driver de la demanda.
- [ ] Probar la **Fase 4 opcional (LSTM)** y comparar contra Prophet.

---

*Documento vivo — se actualiza conforme evoluciona el proyecto.*
