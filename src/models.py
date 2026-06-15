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


def fit_sarimax_fourier(train: pd.Series, cfg: dict):
    """SARIMAX con la estacionalidad metida como términos de Fourier (diaria+semanal+anual).

    Así SARIMA por fin compite de forma justa: el `seasonal_order` clásico solo maneja una
    estacionalidad y se ahoga con la anual (8766h); los Fourier la capturan como regresores.
    Devuelve (modelo, origin) — origin fija la fase de los Fourier para que el forecast no se
    desfase.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    from .features import fourier_terms
    origin = train.index[0]
    exog = fourier_terms(train.index, cfg["fourier"], origin)
    order = tuple(cfg["sarimax"]["order"])
    model = SARIMAX(train.values, exog=exog.values, order=order,
                    enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False, maxiter=cfg["sarimax"].get("maxiter", 50))
    return fit, origin


def sarimax_predict(fit, origin, index: pd.DatetimeIndex, cfg: dict) -> pd.Series:
    """Pronóstico de SARIMAX sobre `index`, con los mismos Fourier (fase consistente vía origin)."""
    from .features import fourier_terms
    exog = fourier_terms(index, cfg["fourier"], origin)
    fc = fit.get_forecast(steps=len(index), exog=exog.values)
    return pd.Series(fc.predicted_mean, index=index)
