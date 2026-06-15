"""Backtesting walk-forward.

Un único test de 14 días es ruidoso: el ranking de modelos puede cambiar de una ventana a otra.
El walk-forward repite la evaluación en varias ventanas consecutivas hacia atrás y promedia, así
sabemos qué modelo es mejor de forma robusta (media ± desviación), no por suerte.
"""
import numpy as np

from . import models

NAMES = ["Seasonal-Naive", "SARIMAX-Fourier", "Prophet"]


def walk_forward(s, cfg, n_folds: int | None = None):
    """Evalúa los 3 modelos en `n_folds` ventanas. Cada fold: entrena en `train_hours` y mide en
    el bloque de `test_hours` siguiente; el siguiente fold retrocede un bloque."""
    n_folds = n_folds or cfg["backtest"]["n_folds"]
    tw, th = cfg["window"]["train_hours"], cfg["window"]["test_hours"]
    naive_p = cfg["baseline"]["naive_period_hours"]

    per_fold = {m: [] for m in NAMES}
    windows = []
    for i in range(n_folds):
        end = len(s) - i * th
        test = s.iloc[end - th:end]
        train = s.iloc[end - th - tw:end - th]

        per_fold["Seasonal-Naive"].append(
            models.metrics(test, models.seasonal_naive(s, test.index, naive_p)))
        sx, origin = models.fit_sarimax_fourier(train, cfg)
        per_fold["SARIMAX-Fourier"].append(
            models.metrics(test, models.sarimax_predict(sx, origin, test.index, cfg)))
        pr = models.fit_prophet(train)
        per_fold["Prophet"].append(
            models.metrics(test, models.prophet_predict(pr, test.index)))
        windows.append((str(test.index[0]), str(test.index[-1])))
    return per_fold, windows


def summarize(per_fold: dict) -> dict:
    """Media y desviación de cada métrica por modelo a lo largo de los folds."""
    out = {}
    for m, folds in per_fold.items():
        out[m] = {}
        for k in ("mae", "rmse", "mape"):
            vals = [f[k] for f in folds]
            out[m][f"{k}_mean"] = float(np.mean(vals))
            out[m][f"{k}_std"] = float(np.std(vals))
    return out
