"""Pipeline de forecasting de punta a punta.

Correr con:  python -m src.pipeline

Carga la serie horaria, entrena baseline + Prophet sobre la ventana reciente, los evalúa en el
test held-out, persiste el modelo Prophet (para la API) y guarda métricas + experimento.
SARIMAX con Fourier llega en el CP2.
"""
import csv
import json
from datetime import datetime

from . import data, models
from .config import load_config, path


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()

    # 1) Serie + split cronológico
    s = data.load_series(cfg)
    train, test = data.train_test(s, cfg)
    print(f"Serie: {len(s):,} horas | train {len(train):,}h | test {len(test):,}h "
          f"({train.index.min().date()} -> {test.index.max().date()})")

    res = {}

    # 2) Baseline: naive estacional (misma hora de la semana pasada)
    snaive = models.seasonal_naive(s, test.index, cfg["baseline"]["naive_period_hours"])
    res["Seasonal-Naive"] = models.metrics(test, snaive)

    # 3) Prophet (estacionalidad diaria + semanal + anual)
    prophet = models.fit_prophet(train)
    fc = models.prophet_predict(prophet, test.index)
    res["Prophet"] = models.metrics(test, fc)

    for nombre, m in res.items():
        print(f"  {nombre:<16} MAE {m['mae']:.0f}  RMSE {m['rmse']:.0f}  MAPE {m['mape']:.2f}%")

    best = min(res, key=lambda k: res[k]["rmse"])
    print(f"Mejor modelo (menor RMSE): {best}")

    # 4) Persistir Prophet + métricas + experimento
    _save_prophet(prophet, cfg)
    _write_metrics(res, best, cfg)
    _log_experiment(res, cfg)
    return {"results": res, "best": best}


def _save_prophet(model, cfg):
    from prophet.serialize import model_to_json
    with open(path(cfg["data"]["model"]), "w", encoding="utf-8") as f:
        f.write(model_to_json(model))


def _write_metrics(res, best, cfg):
    metrics = {
        "best_model": best,
        "test_hours": cfg["window"]["test_hours"],
        "train_hours": cfg["window"]["train_hours"],
        "comparison": {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in res.items()},
        "best": {k: round(v, 3) for k, v in res[best].items()},
    }
    with open(path(cfg["paths"]["metrics"]), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def _log_experiment(res, cfg):
    fp = path(cfg["paths"]["experiments"])
    nuevo = not fp.exists()
    with open(fp, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "model", "mae", "rmse", "mape"])
        if nuevo:
            w.writeheader()
        ts = datetime.now().isoformat(timespec="seconds")
        for nombre, m in res.items():
            w.writerow({"timestamp": ts, "model": nombre,
                        "mae": round(m["mae"], 2), "rmse": round(m["rmse"], 2),
                        "mape": round(m["mape"], 3)})


if __name__ == "__main__":
    run()
