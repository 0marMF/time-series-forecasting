"""Test del backtesting walk-forward.

Stubeamos los ajustes caros (SARIMAX/Prophet) para probar la LÓGICA del walk-forward (ventanas,
agregación) sin entrenar modelos de verdad — rápido y sin dependencias pesadas.
"""
import pandas as pd

from src import backtest, models


def test_walk_forward_y_summarize(synth_series, cfg, monkeypatch):
    monkeypatch.setattr(models, "fit_sarimax_fourier", lambda tr, c: (None, None))
    monkeypatch.setattr(models, "sarimax_predict", lambda f, o, idx, c: pd.Series(30000.0, index=idx))
    monkeypatch.setattr(models, "fit_prophet", lambda tr, **k: None)
    monkeypatch.setattr(models, "prophet_predict", lambda m, idx: pd.Series(30000.0, index=idx))

    per_fold, windows = backtest.walk_forward(synth_series, cfg)
    assert len(windows) == cfg["backtest"]["n_folds"]
    for folds in per_fold.values():
        assert len(folds) == cfg["backtest"]["n_folds"]

    resumen = backtest.summarize(per_fold)
    assert set(resumen) == {"Seasonal-Naive", "SARIMAX-Fourier", "Prophet"}
    for v in resumen.values():
        assert "mape_mean" in v and "mape_std" in v
