"""Tests de métricas, baseline y SARIMAX (statsmodels)."""
from src import data, models


def test_metrics_pronostico_perfecto():
    m = models.metrics([10.0, 20.0, 30.0], [10.0, 20.0, 30.0])
    assert m["mae"] == 0 and m["rmse"] == 0 and m["mape"] == 0


def test_seasonal_naive(synth_series, cfg):
    test_idx = synth_series.index[-cfg["window"]["test_hours"]:]
    sn = models.seasonal_naive(synth_series, test_idx, cfg["baseline"]["naive_period_hours"])
    assert len(sn) == cfg["window"]["test_hours"]
    assert sn.notna().all()


def test_sarimax_fourier(synth_series, cfg):
    train, test = data.train_test(synth_series, cfg)
    fit, origin = models.fit_sarimax_fourier(train, cfg)
    fc = models.sarimax_predict(fit, origin, test.index, cfg)
    assert len(fc) == len(test)
    assert fc.notna().all()
