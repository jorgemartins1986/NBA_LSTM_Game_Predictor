from pathlib import Path
from typing import Literal
from datetime import datetime
import os

import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.web.predict_service import run_todays_predictions
from src.paths import PREDICTION_HISTORY_FILE
from main import get_profitable_filters


class PredictRequest(BaseModel):
    model: Literal["ensemble", "xgboost", "random_forest", "rf", "logistic", "lstm"] = "ensemble"


app = FastAPI(
    title="NBA Predictor API",
    version="1.0.0",
    description="FastAPI wrapper for the NBA ensemble predictor",
)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _record_str(correct: int, total: int) -> str:
    return f"{correct}/{total}"


def _compute_stats_payload() -> dict:
    if not os.path.exists(PREDICTION_HISTORY_FILE):
        return {"has_data": False, "message": "No prediction history file found yet."}

    df = pd.read_csv(PREDICTION_HISTORY_FILE)
    completed = df[df["correct"].notna() & (df["correct"] != "")].copy()

    if len(completed) == 0:
        return {"has_data": False, "message": "No completed predictions yet."}

    completed["correct"] = completed["correct"].astype(int)
    completed["date"] = pd.to_datetime(completed["date"])

    if "confidence" in completed.columns:
        completed["confidence"] = pd.to_numeric(completed["confidence"], errors="coerce")
    if "home_win_prob" in completed.columns:
        completed["home_win_prob"] = pd.to_numeric(completed["home_win_prob"], errors="coerce")
    if "model_agreement" in completed.columns:
        completed["model_agreement"] = pd.to_numeric(completed["model_agreement"], errors="coerce")

    total = len(completed)
    correct_count = int(completed["correct"].sum())
    pending_count = len(df[df["correct"].isna() | (df["correct"] == "")])
    accuracy = (correct_count / total) * 100 if total else 0.0

    tier_rows = []
    tier_order = ["EXCELLENT", "STRONG", "GOOD", "MODERATE", "RISKY", "SKIP"]
    for tier in tier_order:
        tier_games = completed[completed["tier"] == tier]
        if len(tier_games) == 0:
            continue
        tier_correct = int(tier_games["correct"].sum())
        tier_total = len(tier_games)
        tier_acc = tier_correct / tier_total * 100
        losses = tier_total - tier_correct
        roi = (tier_correct * 0.41 - losses) / tier_total * 100
        tier_rows.append(
            {
                "label": tier,
                "record": _record_str(tier_correct, tier_total),
                "accuracy": round(tier_acc, 1),
                "roi": round(roi, 1),
            }
        )

    confidence_rows = []
    if "confidence" in completed.columns and "home_win_prob" in completed.columns:
        confidence_bins = [
            (0.45, 1.0, "45%+"),
            (0.40, 0.45, "40-45%"),
            (0.35, 0.40, "35-40%"),
            (0.30, 0.35, "30-35%"),
            (0.25, 0.30, "25-30%"),
            (0.20, 0.25, "20-25%"),
            (0.15, 0.20, "15-20%"),
            (0.10, 0.15, "10-15%"),
            (0.05, 0.10, "5-10%"),
            (0.0, 0.05, "<5%"),
        ]
        for low, high, label in confidence_bins:
            subset = completed[(completed["confidence"] >= low) & (completed["confidence"] < high)]
            if len(subset) == 0:
                continue
            sub_correct = int(subset["correct"].sum())
            sub_total = len(subset)
            confidence_rows.append(
                {
                    "label": label,
                    "record": _record_str(sub_correct, sub_total),
                    "accuracy": round(sub_correct / sub_total * 100, 1),
                    "avg_prob": round(float(subset["home_win_prob"].mean() * 100), 1),
                }
            )

    agreement_rows = []
    if "model_agreement" in completed.columns:
        agree_bins = [
            (0.98, 1.01, "98%+"),
            (0.95, 0.98, "95-98%"),
            (0.92, 0.95, "92-95%"),
            (0.90, 0.92, "90-92%"),
            (0.85, 0.90, "85-90%"),
            (0.0, 0.85, "<85%"),
        ]
        for low, high, label in agree_bins:
            subset = completed[(completed["model_agreement"] >= low) & (completed["model_agreement"] < high)]
            if len(subset) == 0:
                continue
            sub_correct = int(subset["correct"].sum())
            sub_total = len(subset)
            agreement_rows.append(
                {
                    "label": label,
                    "record": _record_str(sub_correct, sub_total),
                    "accuracy": round(sub_correct / sub_total * 100, 1),
                }
            )

    home_prob_rows = []
    if "home_win_prob" in completed.columns:
        completed["home_won"] = (completed["winner"] == completed["home_team"]).astype(int)
        prob_bins = [
            (0.75, 1.0, "75%+ home"),
            (0.65, 0.75, "65-75% home"),
            (0.55, 0.65, "55-65% home"),
            (0.45, 0.55, "45-55% toss"),
            (0.35, 0.45, "55-65% away"),
            (0.25, 0.35, "65-75% away"),
            (0.0, 0.25, "75%+ away"),
        ]
        for low, high, label in prob_bins:
            subset = completed[(completed["home_win_prob"] >= low) & (completed["home_win_prob"] < high)]
            if len(subset) == 0:
                continue
            sub_correct = int(subset["correct"].sum())
            sub_total = len(subset)
            home_prob_rows.append(
                {
                    "label": label,
                    "record": _record_str(sub_correct, sub_total),
                    "home_win_rate": round(float(subset["home_won"].mean() * 100), 1),
                    "accuracy": round(sub_correct / sub_total * 100, 1),
                }
            )

    time_rows = []
    now = datetime.now()
    for days, label in [(3, "Last 3 days"), (7, "Last 7 days"), (14, "Last 14 days"), (30, "Last 30 days")]:
        recent = completed[completed["date"] >= (now - pd.Timedelta(days=days))]
        if len(recent) == 0:
            continue
        recent_correct = int(recent["correct"].sum())
        recent_total = len(recent)
        time_rows.append(
            {
                "label": label,
                "record": _record_str(recent_correct, recent_total),
                "accuracy": round(recent_correct / recent_total * 100, 1),
            }
        )

    day_rows = []
    completed["day_of_week"] = completed["date"].dt.day_name()
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        subset = completed[completed["day_of_week"] == day]
        if len(subset) == 0:
            continue
        sub_correct = int(subset["correct"].sum())
        sub_total = len(subset)
        day_rows.append(
            {
                "label": day,
                "record": _record_str(sub_correct, sub_total),
                "accuracy": round(sub_correct / sub_total * 100, 1),
            }
        )

    filters = get_profitable_filters()
    filter_rows = [
        {
            "name": f["name"],
            "games": f["n_games"],
            "accuracy": round(float(f["accuracy"]), 1),
            "roi": round(float(f["roi"]), 1),
        }
        for f in filters[:10]
    ]

    return {
        "has_data": True,
        "overall": {
            "total": total,
            "pending": pending_count,
            "correct": correct_count,
            "accuracy": round(accuracy, 1),
            "date_min": completed["date"].min().strftime("%Y-%m-%d"),
            "date_max": completed["date"].max().strftime("%Y-%m-%d"),
        },
        "tier_rows": tier_rows,
        "confidence_rows": confidence_rows,
        "agreement_rows": agreement_rows,
        "home_prob_rows": home_prob_rows,
        "time_rows": time_rows,
        "day_rows": day_rows,
        "filter_rows": filter_rows,
        "best_filter": filter_rows[0] if filter_rows else None,
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/stats", response_class=HTMLResponse)
def stats_page(request: Request):
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stats = _compute_stats_payload()
    return templates.TemplateResponse(
        request,
        "stats.html",
        {
            "stats": stats,
            "generated_at": generated_at,
        },
    )


@app.post("/api/predict")
def api_predict(payload: PredictRequest) -> dict:
    try:
        return run_todays_predictions(single_model=payload.model)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
