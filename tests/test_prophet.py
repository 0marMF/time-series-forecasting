"""Test de Prophet. Se salta si prophet no está instalado (p. ej. en la CI ligera)."""
import pytest

from src import data, models


def test_prophet_fit_predict(synth_series, cfg):
    pytest.importorskip("prophet")
    train, test = data.train_test(synth_series, cfg)
    # yearly=False: la serie sintética es muy corta para estacionalidad anual.
    m = models.fit_prophet(train, yearly=False)
    fc = models.prophet_predict(m, test.index)
    assert len(fc) == len(test)
    assert fc.notna().all()
