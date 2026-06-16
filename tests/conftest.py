"""Fixtures compartidas. Los tests NO usan la serie real (no se versiona): usan una serie horaria
sintética con ciclo diario + semanal, así corren en cualquier sitio (incluido CI)."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def cfg():
    """Config pequeña (ventanas y órdenes reducidos) para que los tests vuelen."""
    return {
        "seed": 42,
        "data": {"raw_csv": "data/PJME_hourly.csv", "datetime_col": "Datetime",
                 "value_col": "PJME_MW", "model": "src/forecast_model.json"},
        "paths": {"reports": "reports", "metrics": "reports/metrics.json",
                  "experiments": "reports/experiments.csv"},
        "window": {"train_hours": 480, "test_hours": 48},
        "forecast": {"horizon_hours": 48},
        "fourier": {"daily": {"period": 24, "order": 3}, "weekly": {"period": 168, "order": 2}},
        "sarimax": {"order": [1, 0, 1], "maxiter": 20},
        "backtest": {"n_folds": 2},
        "baseline": {"naive_period_hours": 168},
    }


@pytest.fixture
def synth_series():
    """Serie horaria sintética (~1000 h) con estacionalidad diaria + semanal y algo de ruido."""
    n = 1000
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    t = np.arange(n)
    val = (30000
           + 5000 * np.sin(2 * np.pi * t / 24)
           + 2000 * np.sin(2 * np.pi * t / 168)
           + np.random.RandomState(0).normal(0, 500, n))
    return pd.Series(val, index=idx, name="PJME_MW")
