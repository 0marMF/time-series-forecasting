# 🔎 Hallazgos y Aprendizajes — Time Series Forecasting (PJM)

> Detecciones del análisis temporal y del forecasting, más los aprendizajes del proyecto.

**Autor:** Omar Mora Flores · **Última actualización:** 2026-06-16

---

## 🧭 Resumen ejecutivo

Sobre **16.6 años** de demanda eléctrica horaria de PJM East, el modelo ganador es **Prophet**:
en backtesting walk-forward (4 ventanas) logra **MAPE 13.0% ± 2.3**, batiendo al baseline
seasonal-naive (13.9%) y al SARIMAX+Fourier (13.7%) sobre todo por su **menor varianza**. Dos
aprendizajes centrales: **(1) sin baseline no se puede juzgar un modelo** —el naive semanal es
durísimo a corto plazo— y **(2) un solo split engaña**: en la ventana única el naive parecía el
mejor; el backtesting lo desmiente. El SARIMA del v1.0.0 quedaba por debajo del baseline, pero era
una limitación del montaje (le faltaba la estacionalidad anual): con términos de Fourier pasa a
ser competitivo.

---

## 📊 Detecciones del EDA y descomposición (Fases 1-2)

| # | Detección | Evidencia |
|---|---|---|
| 1 | **Triple estacionalidad:** diaria (pico de tarde), semanal (baja en finde) y anual (verano/invierno) | `02_seasonality_patterns.png`, `02b_heatmap.png` |
| 2 | **Tendencia plana** a largo plazo (media móvil 365d estable) | `03_trend.png` |
| 3 | La serie diaria **no es estacionaria**, pero **1 diferenciación la estabiliza** (ADF p<0.05) | `02_decomposition` |
| 4 | 4 timestamps duplicados y huecos de frecuencia → interpolados | `01_EDA` |

---

## 🔮 Detecciones del modelado (Fase 3) — forecasting horario, test 336 h

| Modelo | MAE | RMSE | MAPE |
|---|---|---|---|
| Seasonal-Naive (baseline) | 3,277 | 4,245 | 9.2 % |
| SARIMAX + Fourier | 4,286 | 5,331 | 11.1 % |
| Prophet | 3,532 | 4,238 | 10.9 % |

1. **El naive semanal es un baseline durísimo** en horario: copiar la misma hora de la semana
   pasada acierta mucho a corto plazo. La vara de medir correcta.
2. **SARIMAX + Fourier ya es competitivo** (~11% MAPE), no el desastre del v1.0.0 (21% en diario):
   meter la estacionalidad múltiple (diaria/semanal/anual) como **regresores de Fourier** le da a
   SARIMA lo que el `seasonal_order` semanal no podía. Comparación por fin justa.
3. **Prophet queda a la par** (mejor RMSE, 4,238) gracias a su estacionalidad nativa.
4. Una sola ventana de 14 días es **ruidosa**: el ranking real exige **backtesting walk-forward**
   (varias ventanas). Ver abajo.

---

## 📊 Backtesting walk-forward (Fase 3b)

Repitiendo la evaluación en **4 ventanas** y promediando, el ranking cambia respecto a la ventana
única:

| Modelo | MAPE medio | desviación |
|---|---|---|
| Seasonal-Naive | 13.9 % | 3.6 |
| SARIMAX + Fourier | 13.7 % | 3.6 |
| **Prophet** | **13.0 %** | **2.3** |

- **Prophet gana de forma robusta**: menor MAPE medio y, sobre todo, **menor varianza** (más
  consistente). En la ventana única (CP2) el naive parecía el mejor — el backtesting lo desmiente.
- Moraleja: **un solo split puede señalar al modelo equivocado**. Por eso se backtestea.

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

- [x] **Backtesting / validación walk-forward** en varias ventanas (hecho: 4 ventanas, Prophet robusto).
- [x] **SARIMA con estacionalidad anual** vía términos de Fourier — hecho: pasa de ~21% a ~11% en
      la ventana única y queda a la par del baseline en backtest. Comparación por fin justa.
- [ ] Incorporar **variables exógenas** (temperatura, festivos) — el principal driver real de la
      demanda y la mejora con más techo (ver `MODEL_CARD.md`).
- [ ] Probar la **Fase 4 opcional (LSTM)** y comparar contra Prophet.

---

*Documento vivo — se actualiza conforme evoluciona el proyecto.*
