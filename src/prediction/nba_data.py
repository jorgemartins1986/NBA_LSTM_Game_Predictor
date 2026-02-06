"""
NBA Data Module
===============
Functions for fetching live NBA data from the NBA API.
Separated from prediction logic for testability.
"""

import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from backports.zoneinfo import ZoneInfo


def get_eastern_date() -> str:
    """Get current date in Eastern Time (NBA's timezone)"""
    eastern = ZoneInfo('America/New_York')
    return datetime.now(eastern).strftime('%Y-%m-%d')


def get_current_season() -> str:
    """Get current NBA season string (e.g., '2024-25')"""
    eastern = ZoneInfo('America/New_York')
    now = datetime.now(eastern)
    # NBA season starts in October
    if now.month >= 10:
        return f"{now.year}-{str(now.year + 1)[-2:]}"
    else:
        return f"{now.year - 1}-{str(now.year)[-2:]}"


def get_live_standings(verbose: bool = True) -> Dict[int, Dict]:
    """Fetch current standings from NBA API.
    
    Returns:
        dict: {team_id: {WINS, LOSSES, WIN_PCT, CONF_RANK, LEAGUE_RANK, GAMES_BACK, STREAK}}
    """
    import time
    from nba_api.stats.endpoints import leaguestandings
    
    season = get_current_season()
    if verbose:
        print(f"📊 Fetching live standings for {season}...")
    
    try:
        time.sleep(0.6)  # Rate limiting
        ls = leaguestandings.LeagueStandings(season=season)
        df = ls.get_data_frames()[0]
        
        standings = {}
        for _, row in df.iterrows():
            team_id = row['TeamID']
            
            # Parse streak (format like "W 5" or "L 3")
            streak_str = row.get('strCurrentStreak', 'W 0')
            try:
                streak_parts = streak_str.split()
                streak_val = int(streak_parts[1]) if len(streak_parts) > 1 else 0
                streak = streak_val if streak_parts[0] == 'W' else -streak_val
            except:
                streak = 0
            
            standings[team_id] = {
                'WINS': int(row['WINS']),
                'LOSSES': int(row['LOSSES']),
                'WIN_PCT': float(row['WinPCT']),
                'CONF_RANK': int(row['PlayoffRank']),
                'LEAGUE_RANK': int(row['LeagueRank']) if pd.notna(row.get('LeagueRank')) else int(row['PlayoffRank']),
                'GAMES_BACK': float(row['ConferenceGamesBack']) if pd.notna(row.get('ConferenceGamesBack')) else 0.0,
                'STREAK': streak
            }
        
        if verbose:
            print(f"   ✓ Got standings for {len(standings)} teams")
        return standings
        
    except Exception as e:
        if verbose:
            print(f"   ⚠️ Could not fetch standings: {e}")
        return {}


def get_todays_games(verbose: bool = True) -> Tuple[List[Dict], str, bool]:
    """Fetch today's NBA games
    
    Returns:
        tuple: (list of games, game_date string in YYYY-MM-DD format, all_finished bool)
        
    Uses ScoreboardV2 (stats API) which shows scheduled games for today,
    with live scoreboard as fallback for game status.
    """
    from nba_api.live.nba.endpoints import scoreboard
    from nba_api.stats.endpoints import scoreboardv2
    from nba_api.stats.static import teams
    
    game_date_str = get_eastern_date()
    all_teams_list = teams.get_teams()
    team_id_to_info = {t['id']: t for t in all_teams_list}
    
    try:
        # Use ScoreboardV2 which has scheduled games (not just live/recent)
        sb = scoreboardv2.ScoreboardV2(game_date=game_date_str)
        games_df = sb.get_data_frames()[0]  # GameHeader dataframe
        
        if len(games_df) == 0:
            if verbose:
                print(f"   No games scheduled for {game_date_str}")
            return [], game_date_str, False
        
        # IMPORTANT: NBA API sometimes returns duplicate rows - deduplicate by GAME_ID
        games_df = games_df.drop_duplicates(subset=['GAME_ID'], keep='first')
        
        # Try to get live status from live scoreboard
        live_status = {}
        try:
            board = scoreboard.ScoreBoard()
            live_data = board.get_dict()
            live_date = live_data.get('scoreboard', {}).get('gameDate', '')
            # Only use live status if it matches today's date
            if live_date == game_date_str:
                for game in board.games.get_dict():
                    live_status[game['gameId']] = game['gameStatusText']
        except:
            pass
        
        today_games = []
        finished_count = 0
        
        for _, row in games_df.iterrows():
            game_id = row['GAME_ID']
            home_team_id = row['HOME_TEAM_ID']
            visitor_team_id = row['VISITOR_TEAM_ID']
            
            home_info = team_id_to_info.get(home_team_id, {})
            visitor_info = team_id_to_info.get(visitor_team_id, {})
            
            # Get status from live board if available, else from schedule
            status = live_status.get(game_id, row.get('GAME_STATUS_TEXT', 'Scheduled'))
            
            today_games.append({
                'game_id': game_id,
                'home_team': home_info.get('nickname', row.get('HOME_TEAM_ID', 'Unknown')),
                'away_team': visitor_info.get('nickname', row.get('VISITOR_TEAM_ID', 'Unknown')),
                'home_team_id': home_team_id,
                'away_team_id': visitor_team_id,
                'game_status': status
            })
            
            if status == 'Final':
                finished_count += 1
        
        all_finished = len(today_games) > 0 and finished_count == len(today_games)
        return today_games, game_date_str, all_finished
        
    except Exception as e:
        if verbose:
            print(f"   Error fetching games: {e}")
        return [], game_date_str, False


def fetch_season_games(season: str = None, verbose: bool = True) -> pd.DataFrame:
    """Fetch all games for a season.
    
    Args:
        season: Season string (e.g., '2025-26'). Defaults to current season.
        verbose: Whether to print progress.
        
    Returns:
        DataFrame with game data sorted by date.
    """
    from nba_api.stats.endpoints import leaguegamefinder
    
    if season is None:
        season = get_current_season()
    
    if verbose:
        print(f"📊 Fetching games for {season} season...")
    
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable='00'
    )
    games_df = gamefinder.get_data_frames()[0]
    games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
    games_df = games_df.sort_values('GAME_DATE')
    
    if verbose:
        print(f"✓ Loaded {len(games_df)} games from {season} season")
    
    return games_df


def get_recent_team_stats(team_id: int, games_df: pd.DataFrame, window_size: int = 20) -> Optional[Dict]:
    """Get recent statistics for a team including fatigue features.
    
    Args:
        team_id: NBA team ID
        games_df: DataFrame of games
        window_size: Rolling window size for features
        
    Returns:
        Dict of feature values, or None if insufficient data
    """
    from ..nba_predictor import FeatureEngineering
    
    team_games = games_df[games_df['TEAM_ID'] == team_id].sort_values('GAME_DATE')
    
    if len(team_games) == 0:
        return None
    
    fe = FeatureEngineering(window_size=window_size)
    
    # CRITICAL: Calculate four factors on the FULL games_df before creating rolling features
    games_df_with_factors = fe.calculate_four_factors(games_df.copy())
    
    recent_with_features = fe.create_rolling_features(games_df_with_factors, team_id)
    
    if len(recent_with_features) == 0:
        return None
    
    latest = recent_with_features.iloc[-1]
    # Include fatigue features
    roll_cols = [col for col in recent_with_features.columns 
                 if 'ROLL' in col or col in ['WIN_STREAK', 'RECENT_FORM', 'WIN_PATTERN_3GAME', 
                                              'MOMENTUM_TREND', 'WIN_CONSISTENCY', 'IS_HOME',
                                              'DAYS_REST', 'IS_BACK_TO_BACK', 'IS_3_IN_4']]
    features = {col: latest[col] for col in roll_cols if col in latest.index}
    
    return features


def compute_head_to_head(games_df: pd.DataFrame, team_id: int, opponent_id: int, window: int = 10) -> Dict[str, float]:
    """Compute head-to-head record against specific opponent (last N meetings).
    
    Args:
        games_df: DataFrame of games
        team_id: Team ID
        opponent_id: Opponent team ID
        window: Number of recent games to consider
        
    Returns:
        Dict with H2H_WIN_RATE, H2H_GAMES, H2H_PTS_DIFF
    """
    # Get all completed games for this team
    team_games = games_df[games_df['TEAM_ID'] == team_id].copy()
    
    # Get opponent abbreviation
    opp_games = games_df[games_df['TEAM_ID'] == opponent_id]
    if len(opp_games) == 0:
        return {'H2H_WIN_RATE': 0.5, 'H2H_GAMES': 0, 'H2H_PTS_DIFF': 0}
    
    opp_abbrev = opp_games.iloc[0].get('TEAM_ABBREVIATION', '')
    
    # Find games against this opponent
    h2h_games = team_games[team_games['MATCHUP'].str.contains(opp_abbrev, na=False)].tail(window)
    
    if len(h2h_games) == 0:
        return {'H2H_WIN_RATE': 0.5, 'H2H_GAMES': 0, 'H2H_PTS_DIFF': 0}
    
    wins = (h2h_games['WL'] == 'W').sum()
    games_played = len(h2h_games)
    pts_diff = h2h_games['PLUS_MINUS'].mean() if 'PLUS_MINUS' in h2h_games.columns else 0
    
    return {
        'H2H_WIN_RATE': wins / games_played,
        'H2H_GAMES': min(games_played, window),
        'H2H_PTS_DIFF': pts_diff if not pd.isna(pts_diff) else 0
    }


def get_default_standings() -> Dict[str, Any]:
    """Get default standings dict for teams without standings data."""
    return {
        'WINS': 0, 
        'LOSSES': 0, 
        'WIN_PCT': 0.5,
        'CONF_RANK': 8, 
        'LEAGUE_RANK': 15, 
        'GAMES_BACK': 0, 
        'STREAK': 0
    }
