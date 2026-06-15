"""Modelos de forecasting y métricas."""
import numpy as np
import pandas as pd


def metrics(y, yhat) -> dict:
    """MAE, RMSE y MAPE. El MAPE (%) es la métrica más comunicable a negocio."""
    y, yhat = np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)
    return {
        "mae": float(np.mean(np.abs(y - yhat))),
        "rmse": float(np.sqrt(np.mean((y - yhat) ** 2))),
        "mape": float(np.mean(np.abs((y - yhat) / y)) * 100),
    }


def seasonal_naive(s: pd.Series, test_index: pd.DatetimeIndex, period_hours: int) -> pd.Series:
    """Baseline: el valor de hace `period_hours` (p. ej. la misma hora de la semana pasada).

    Es la vara de medir obligatoria: si un modelo no le saca ventaja, no aporta nada.
    """
    return s.shift(period_hours).loc[test_index]


def fit_prophet(train: pd.Series, daily=True, weekly=True, yearly=True):
    """Ajusta Prophet con estacionalidad diaria + semanal + anual."""
    from prophet import Prophet
    df = train.reset_index()
    df.columns = ["ds", "y"]
    m = Prophet(daily_seasonality=daily, weekly_seasonality=weekly, yearly_seasonality=yearly)
    m.fit(df)
    return m


def prophet_predict(model, index: pd.DatetimeIndex) -> pd.Series:
    """Pronóstico de Prophet sobre un índice de fechas concreto (test o futuro)."""
    fc = model.predict(pd.DataFrame({"ds": index}))
    return pd.Series(fc["yhat"].values, index=index)
