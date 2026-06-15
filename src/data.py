"""Carga de la serie horaria y split cronológico."""
import pandas as pd

from .config import load_config, path


def load_series(cfg: dict | None = None) -> pd.Series:
    """Devuelve la serie horaria limpia (frecuencia horaria explícita, sin huecos)."""
    cfg = cfg or load_config()
    dt, val = cfg["data"]["datetime_col"], cfg["data"]["value_col"]
    df = pd.read_csv(path(cfg["data"]["raw_csv"]), parse_dates=[dt])
    s = (df.drop_duplicates(dt).sort_values(dt).set_index(dt)[val])
    # Frecuencia horaria explícita; los pocos huecos los interpolamos (si no, ARIMA se queja).
    return s.asfreq("h").interpolate()


def recent_window(s: pd.Series, cfg: dict) -> pd.Series:
    """Recorta a la ventana reciente (train + test). 16 años de horario es demasiado para SARIMA."""
    n = cfg["window"]["train_hours"] + cfg["window"]["test_hours"]
    return s.iloc[-n:]


def train_test(s: pd.Series, cfg: dict) -> tuple[pd.Series, pd.Series]:
    """Split cronológico: el test son las últimas `test_hours`. En series NUNCA es aleatorio."""
    th = cfg["window"]["test_hours"]
    ventana = recent_window(s, cfg)
    return ventana.iloc[:-th], ventana.iloc[-th:]
