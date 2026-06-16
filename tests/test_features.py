"""Tests de los términos de Fourier."""
import numpy as np

from src.features import fourier_terms


def test_fourier_shape_e_indice(synth_series, cfg):
    F = fourier_terms(synth_series.index, cfg["fourier"], synth_series.index[0])
    esperado = 2 * sum(p["order"] for p in cfg["fourier"].values())   # sin + cos por armónico
    assert F.shape == (len(synth_series), esperado)
    assert (F.index == synth_series.index).all()
    assert np.isfinite(F.to_numpy()).all()


def test_fourier_fase_consistente(synth_series, cfg):
    # Con el mismo origin, los Fourier de un tramo coinciden con los del tramo completo
    # (fase continua). Si no, el pronóstico quedaría desfasado.
    origin = synth_series.index[0]
    full = fourier_terms(synth_series.index, cfg["fourier"], origin)
    tramo = fourier_terms(synth_series.index[-48:], cfg["fourier"], origin)
    assert np.allclose(tramo.to_numpy(), full.to_numpy()[-48:])
