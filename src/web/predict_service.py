"""Prediction service used by FastAPI and web UI."""

from functools import lru_cache
from typing import Dict, List, Optional

from nba_api.stats.static import teams

from src.prediction import (
    ModelLoader,
    PredictionPipeline,
    fetch_season_games,
    get_default_standings,
    get_live_odds,
    get_live_standings,
    get_todays_games,
    get_recent_team_stats,
    compute_head_to_head,
    match_game_to_odds,
    get_confidence_tier,
)


def _filter_regular_games(todays_games: List[Dict], all_teams: List[Dict]) -> List[Dict]:
    all_team_ids = {team["id"] for team in all_teams}
    return [
        game
        for game in todays_games
        if game.get("home_team_id") in all_team_ids and game.get("away_team_id") in all_team_ids
    ]


@lru_cache(maxsize=4)
def _load_pipeline(single_model: Optional[str]) -> PredictionPipeline:
    loader = ModelLoader()
    ensemble = loader.load_ensemble(verbose=False)

    if single_model and single_model != "ensemble":
        model_name = "random_forest" if single_model == "rf" else single_model
        indices = [i for i, mt in enumerate(ensemble.model_types) if mt == model_name]
        if not indices:
            raise ValueError(
                f"Model '{single_model}' not found. Available: {', '.join(ensemble.model_types)}"
            )
        idx = indices[0]

        from src.prediction.loader import LoadedEnsemble

        ensemble = LoadedEnsemble(
            models=[ensemble.models[idx]],
            scalers=[ensemble.scalers[idx]],
            feature_cols=ensemble.feature_cols,
            model_types=[ensemble.model_types[idx]],
            meta_clf=None,
            platt=None,
            ensemble_weights=None,
            ensemble_threshold=None,
        )

    return PredictionPipeline(ensemble)


def run_todays_predictions(single_model: str = "ensemble") -> Dict:
    pipeline = _load_pipeline(single_model)
    all_teams = teams.get_teams()
    team_id_to_info = {team["id"]: team for team in all_teams}

    games_df = fetch_season_games(verbose=False)
    todays_games, game_date, all_finished = get_todays_games(verbose=False)

    if not todays_games:
        return {
            "game_date": game_date,
            "all_finished": all_finished,
            "predictions": [],
            "skipped_reason": "No games found for today.",
        }

    live_standings = get_live_standings(verbose=False)
    live_odds = get_live_odds(verbose=False)

    predictions = []
    regular_games = _filter_regular_games(todays_games, all_teams)

    for game in regular_games:
        home_team = team_id_to_info.get(game["home_team_id"])
        away_team = team_id_to_info.get(game["away_team_id"])
        if not home_team or not away_team:
            continue

        home_features = get_recent_team_stats(game["home_team_id"], games_df)
        away_features = get_recent_team_stats(game["away_team_id"], games_df)
        if home_features is None or away_features is None:
            continue

        h2h_home = compute_head_to_head(games_df, game["home_team_id"], game["away_team_id"])
        h2h_away = compute_head_to_head(games_df, game["away_team_id"], game["home_team_id"])

        home_standings = live_standings.get(game["home_team_id"], get_default_standings())
        away_standings = live_standings.get(game["away_team_id"], get_default_standings())

        game_odds = match_game_to_odds(home_team["full_name"], away_team["full_name"], live_odds)

        features = pipeline.feature_computer.build_feature_vector(
            home_features=home_features,
            away_features=away_features,
            feature_cols=pipeline.ensemble.feature_cols,
            h2h_home=h2h_home,
            h2h_away=h2h_away,
            home_standings=home_standings,
            away_standings=away_standings,
            home_odds=game_odds,
            away_odds=game_odds,
        )

        result = pipeline.predict_from_features(features)
        winner_name = (
            home_team["full_name"] if result.predicted_winner == "HOME" else away_team["full_name"]
        )

        predictions.append(
            {
                "away_team": away_team["full_name"],
                "home_team": home_team["full_name"],
                "predicted_winner": winner_name,
                "confidence": round(float(result.confidence), 4),
                "tier": get_confidence_tier(float(result.confidence)),
                "home_win_prob": round(float(result.home_win_probability), 4),
                "away_win_prob": round(float(result.away_win_probability), 4),
                "model_agreement": round(float(result.model_agreement), 4),
                "game_status": game.get("game_status", "Scheduled"),
            }
        )

    predictions.sort(key=lambda item: item["confidence"], reverse=True)

    return {
        "game_date": game_date,
        "all_finished": all_finished,
        "single_model": single_model,
        "num_games": len(predictions),
        "predictions": predictions,
    }
