"""API de pronóstico de demanda eléctrica (FastAPI).

Levantar en local:  uvicorn src.api:app --reload
Docs interactivas en  http://localhost:8000/docs

Sirve el modelo Prophet ya entrenado (no reentrena): pides N días y devuelve el pronóstico
horario con su intervalo. El modelo se carga una sola vez.
"""
from fastapi import FastAPI, Query

from .forecast import forecast as make_forecast
from .forecast import load_model

app = FastAPI(
    title="Energy Demand Forecast API",
    version="1.1.0",
    description="Pronóstico horario de demanda eléctrica (PJM) con Prophet.",
)

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = load_model()       # carga perezosa del modelo persistido
    return _model


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/forecast")
def forecast_endpoint(days: int = Query(7, ge=1, le=90, description="Días a pronosticar")) -> dict:
    """Devuelve el pronóstico horario de los próximos `days` días con intervalo."""
    fc = make_forecast(days, model=_get_model())
    return {
        "days": days,
        "horizon_hours": len(fc),
        "forecast": [
            {"ds": str(r.ds), "yhat": round(r.yhat, 1),
             "lower": round(r.yhat_lower, 1), "upper": round(r.yhat_upper, 1)}
            for r in fc.itertuples()
        ],
    }
