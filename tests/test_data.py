"""Tests del split cronológico."""
from src import data


def test_train_test_cronologico(synth_series, cfg):
    train, test = data.train_test(synth_series, cfg)
    assert len(train) == cfg["window"]["train_hours"]
    assert len(test) == cfg["window"]["test_hours"]
    # En series el split NUNCA es aleatorio: el test va estrictamente después del train.
    assert train.index.max() < test.index.min()


def test_recent_window(synth_series, cfg):
    w = data.recent_window(synth_series, cfg)
    assert len(w) == cfg["window"]["train_hours"] + cfg["window"]["test_hours"]
    assert w.index[-1] == synth_series.index[-1]   # es la cola de la serie
