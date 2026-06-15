"""Pronóstico a futuro cargando el modelo Prophet ya entrenado.

No reentrena: carga `src/forecast_model.json` (lo deja el pipeline) y proyecta N días hacia
adelante con su intervalo. Lo usan el CLI y la API.

Uso por terminal:  python -m src.forecast --days 30
"""
import pandas as pd

from .config import load_config, path


def load_model(cfg: dict | None = None):
    from prophet.serialize import model_from_json
    cfg = cfg or load_config()
    with open(path(cfg["data"]["model"]), encoding="utf-8") as f:
        return model_from_json(f.read())


def forecast(days: int = 30, cfg: dict | None = None, model=None) -> pd.DataFrame:
    """Pronóstico horario de los próximos `days` días (Prophet usa la última fecha del modelo)."""
    cfg = cfg or load_config()
    m = model or load_model(cfg)
    future = m.make_future_dataframe(periods=int(days) * 24, freq="h", include_history=False)
    fc = m.predict(future)
    return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def _main():
    import argparse
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Pronóstico de demanda eléctrica horaria.")
    ap.add_argument("--days", type=int, default=30, help="días a pronosticar")
    args = ap.parse_args()

    fc = forecast(args.days)
    print(f"Pronóstico de {args.days} días ({len(fc)} horas): "
          f"{fc['ds'].iloc[0]} -> {fc['ds'].iloc[-1]}")
    print(f"Demanda media prevista: {fc['yhat'].mean():,.0f} MW "
          f"(min {fc['yhat'].min():,.0f}, max {fc['yhat'].max():,.0f})")
    print("\nPrimeras horas:")
    print(fc.head(5).to_string(index=False))


if __name__ == "__main__":
    _main()
