"""Términos de Fourier para modelar estacionalidad múltiple (diaria, semanal, anual).

SARIMA con `seasonal_order` solo maneja UNA estacionalidad y se ahoga con periodos largos
(anual = 8766 horas). El truco estándar: meter la estacionalidad como regresores de Fourier
(senos y cosenos) y dejar que SARIMAX (o cualquier modelo lineal) los use. Prophet hace esto por
dentro; aquí lo hacemos explícito para darle a SARIMA una comparación justa (CP2).
"""
import numpy as np
import pandas as pd


def fourier_terms(index: pd.DatetimeIndex, periods: dict, origin: pd.Timestamp) -> pd.DataFrame:
    """Construye las columnas seno/coseno para cada estacionalidad.

    `t` se mide en horas desde un `origin` fijo, así la fase es consistente entre el train y el
    futuro (si no, el pronóstico quedaría desfasado).
    """
    t = (index - origin) / pd.Timedelta("1h")
    cols = {}
    for nombre, p in periods.items():
        for k in range(1, p["order"] + 1):
            ang = 2 * np.pi * k * t / p["period"]
            cols[f"{nombre}_sin{k}"] = np.sin(ang)
            cols[f"{nombre}_cos{k}"] = np.cos(ang)
    return pd.DataFrame(cols, index=index)
