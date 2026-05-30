"""
NBA Data Module
===============
Functions for fetching live NBA data from the NBA API.
Separated from prediction logic for testability.
"""

import pandas as pd
from datetime import datetime, timedelta
import os
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import urllib.error
import json
import gzip
import ssl

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from backports.zoneinfo import ZoneInfo

from ..paths import GAMES_CACHE_FILE


# Headers that stats.nba.com requires (Akamai CDN fingerprinting)
_NBA_STATS_HEADERS = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true',
    'Connection': 'keep-alive',
    'Referer': 'https://stats.nba.com/',
    'Pragma': 'no-cache',
    'Cache-Control': 'no-cache',
    'sec-ch-ua': '"Chromium";v="133", "Not(A:Brand";v="99", "Google Chrome";v="133"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
}

_NBA_CDN_HEADERS = {
    'User-Agent': _NBA_STATS_HEADERS['User-Agent'],
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'identity',
    'Connection': 'keep-alive',
}

# Custom SSL context with browser-like cipher suite
_SSL_CONTEXT = ssl.create_default_context()
_SSL_CONTEXT.set_ciphers('DEFAULT:@SECLEVEL=1')
_SSL_CONTEXT.check_hostname = True
_SSL_CONTEXT.verify_mode = ssl.CERT_REQUIRED
_HTTPS_HANDLER = urllib.request.HTTPSHandler(context=_SSL_CONTEXT)
_OPENER = urllib.request.build_opener(_HTTPS_HANDLER)


def _fetch_nba_stats(endpoint: str, params: dict, timeout: int = 30, retries: int = 3) -> dict:
    """Fetch data from stats.nba.com using urllib (bypasses CDN blocking of requests/urllib3).
    
    Args:
        endpoint: API endpoint name (e.g., 'leaguegamefinder')
        params: Query parameters dict
        timeout: Request timeout in seconds
        retries: Number of allowed attempts before giving up
        
    Returns:
        Parsed JSON response dict
    """
    import time
    from urllib.parse import quote_plus
    
    sorted_params = sorted(params.items(), key=lambda kv: kv[0])
    param_string = '&'.join(f'{k}={quote_plus(str(v))}' for k, v in sorted_params)
    url = f'https://stats.nba.com/stats/{endpoint}?{param_string}'
    
    last_err = None
    for attempt in range(retries):
        try:
            if attempt > 0:
                time.sleep(1)
            req = urllib.request.Request(url, headers=_NBA_CDN_HEADERS)
            resp = urllib.request.urlopen(req, timeout=timeout)
            data = resp.read()
            
            encoding = resp.headers.get('Content-Encoding', '')
            if encoding == 'gzip':
                data = gzip.decompress(data)
            
            return json.loads(data)
        except (TimeoutError, urllib.error.URLError, ConnectionError, OSError) as e:
            last_err = e
    
    raise last_err


def _result_set_to_df(result: dict, index: int = 0) -> pd.DataFrame:
    """Convert an NBA stats API resultSets entry to a DataFrame."""
    rs = result['resultSets'][index]
    return pd.DataFrame(rs['rowSet'], columns=rs['headers'])


def _fetch_cdn_scoreboard(date_str: str, timeout: int = 10) -> list:
    """Try to fetch the NBA CDN scoreboard JSON for a given date.

    Returns a list of simple game dicts with keys: game_id, home_team_id, away_team_id, game_status_text
    """
    date_key = date_str.replace('-', '')
    candidates = [
        f"https://cdn.nba.com/static/json/liveData/scoreboard/{date_key}/scoreboard_00.json",
        f"https://cdn.nba.com/static/json/liveData/scoreboard/{date_key}/scoreboard.json",
        "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json",
        "https://nba-prod-us-east-1-mediaops-stats.s3.amazonaws.com/NBA/liveData/scoreboard/todaysScoreboard_00.json",
    ]
    for url in candidates:
        try:
            req = urllib.request.Request(url, headers=_NBA_CDN_HEADERS)
            resp = urllib.request.urlopen(req, timeout=timeout)
            data = resp.read()
            encoding = resp.headers.get('Content-Encoding', '')
            if encoding == 'gzip':
                data = gzip.decompress(data)
            j = json.loads(data)
        except Exception:
            continue

        sb = j.get('scoreboard') if isinstance(j, dict) else None
        if not isinstance(sb, dict):
            continue

        sb_date = sb.get('gameDate')
        if sb_date and sb_date != date_str:
            continue

        raw_games = sb.get('games', [])
        games = []
        for g in raw_games:
            gid = g.get('gameId') or g.get('gameIdStr') or g.get('GAME_ID')
            home_team = g.get('homeTeam') if isinstance(g.get('homeTeam'), dict) else {}
            away_team = g.get('awayTeam') if isinstance(g.get('awayTeam'), dict) else {}

            home = home_team.get('teamId') or g.get('homeTeamId')
            away = away_team.get('teamId') or g.get('awayTeamId') or g.get('visitorTeamId')
            status = g.get('gameStatusText') or g.get('status') or g.get('gameStatus')

            try:
                home_id = int(home) if home is not None else None
            except Exception:
                home_id = None
            try:
                away_id = int(away) if away is not None else None
            except Exception:
                away_id = None

            games.append({
                'game_id': gid,
                'home_team_id': home_id,
                'away_team_id': away_id,
                'game_status': status or 'Scheduled'
            })

        if games:
            return games
            
    # Try the static schedule JSON if live scoreboard has not rolled over
    try:
        url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"
        req = urllib.request.Request(url, headers=_NBA_CDN_HEADERS)
        resp = urllib.request.urlopen(req, timeout=timeout)
        data = resp.read()
        encoding = resp.headers.get('Content-Encoding', '')
        if encoding == 'gzip':
            data = gzip.decompress(data)
        j = json.loads(data)
        
        schedule_games = []
        league_schedule = j.get('leagueSchedule', {})
        for date_block in league_schedule.get('gameDates', []):
            for g in date_block.get('games', []):
                # g_date is like "2026-05-09T00:00:00Z"
                g_date = g.get('gameDateEst', '')
                if g_date.startswith(date_str):
                    gid = g.get('gameId')
                    home = g.get('homeTeam', {}).get('teamId')
                    away = g.get('awayTeam', {}).get('teamId')
                    status = g.get('gameStatusText') or g.get('gameStatus', 'Scheduled')
                    
                    try:
                        home_id = int(home) if home is not None else None
                    except Exception:
                        home_id = None
                    try:
                        away_id = int(away) if away is not None else None
                    except Exception:
                        away_id = None
                    
                    schedule_games.append({
                        'game_id': gid,
                        'home_team_id': home_id,
                        'away_team_id': away_id,
                        'game_status': status or 'Scheduled'
                    })
        
        if schedule_games:
            return schedule_games
    except Exception:
        pass

    return []


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
    
    Uses urllib directly to bypass CDN blocking of requests/urllib3.
    
    Returns:
        dict: {team_id: {WINS, LOSSES, WIN_PCT, CONF_RANK, LEAGUE_RANK, GAMES_BACK, STREAK}}
    """
    import time
    
    season = get_current_season()
    if verbose:
        print(f"📊 Fetching live standings for {season}...")
    
    try:
        time.sleep(0.6)  # Rate limiting
        
        # Primary: urllib-based fetch
        try:
            params = {
                'LeagueID': '00',
                'Season': season,
                'SeasonType': 'Regular Season',
            }
            result = _fetch_nba_stats('leaguestandingsv3', params, timeout=10, retries=1)
            rs = result['resultSets'][0]
            df = pd.DataFrame(rs['rowSet'], columns=rs['headers'])
        except Exception as urllib_err:
            if verbose:
                print(f"   ⚠ Direct fetch failed ({urllib_err}), trying nba_api...")
            from nba_api.stats.endpoints import leaguestandings
            ls = leaguestandings.LeagueStandings(season=season, timeout=10)
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


def get_todays_games(verbose: bool = True, date_str: str | None = None) -> Tuple[List[Dict], str, bool]:
    """Fetch NBA games for a date (defaults to today)
    
    Returns:
        tuple: (list of games, game_date string in YYYY-MM-DD format, all_finished bool)
        
    Uses ScoreboardV2 (stats API) which shows scheduled games for today,
    with live scoreboard as fallback for game status.
    """
    import time
    from nba_api.live.nba.endpoints import scoreboard
    from nba_api.stats.static import teams
    
    game_date_str = date_str or get_eastern_date()
    all_teams_list = teams.get_teams()
    team_id_to_info = {t['id']: t for t in all_teams_list}

    # Fast path: CDN scoreboard JSON (often available even when stats endpoints lag)
    try:
        cdn_games = _fetch_cdn_scoreboard(game_date_str)
        if cdn_games:
            today_games = []
            for g in cdn_games:
                home_id = g.get('home_team_id')
                away_id = g.get('away_team_id')
                home_info = team_id_to_info.get(home_id, {})
                away_info = team_id_to_info.get(away_id, {})
                status = g.get('game_status', 'Scheduled')
                today_games.append({
                    'game_id': g.get('game_id'),
                    'home_team': home_info.get('nickname', str(home_id)),
                    'away_team': away_info.get('nickname', str(away_id)),
                    'home_team_id': home_id,
                    'away_team_id': away_id,
                    'game_status': status,
                })
            if verbose:
                print("   Result source: NBA CDN JSON")
            return today_games, game_date_str, False
    except Exception:
        pass
    
    try:
        # Primary: urllib-based fetch of ScoreboardV2 (bypasses CDN blocking)
        # Add retry/backoff to be resilient to transient 503s from stats.nba.com
        time.sleep(0.6)
        games_df = None
        params = {
            'DayOffset': '0',
            'GameDate': game_date_str,
            'LeagueID': '00',
        }
        max_attempts = 1
        for attempt in range(max_attempts):
            try:
                from nba_api.stats.endpoints import scoreboardv2
                sb = scoreboardv2.ScoreboardV2(game_date=game_date_str, timeout=10)
                games_df = sb.get_data_frames()[0]
                break
            except Exception as fallback_err:
                if verbose:
                    print(f"   ⚠ nba_api scoreboardv2 failed: {fallback_err}")
                games_df = None
        
        if games_df is None or len(games_df) == 0:
            # No results from scoreboardv2 — try leaguegamefinder DateFrom/DateTo as fallback
            if verbose:
                print(f"   No games found via scoreboardv2 for {game_date_str}; trying leaguegamefinder fallback...")
            try:
                # Try season types in order for scheduled games
                season = get_current_season()
                season_types = ['Regular Season', 'PlayIn', 'Playoffs']
                lg_games = None
                date_us = datetime.strptime(game_date_str, '%Y-%m-%d').strftime('%m/%d/%Y')
                for st in season_types:
                    try:
                        from nba_api.stats.endpoints import leaguegamefinder
                        lg = leaguegamefinder.LeagueGameFinder(
                            date_from_nullable=date_us,
                            date_to_nullable=date_us,
                            season_nullable=season,
                            season_type_nullable=st,
                            timeout=10
                        )
                        lg_df = lg.get_data_frames()[0]
                        if len(lg_df) > 0:
                            lg_games = lg_df
                            break
                    except Exception:
                        continue

                if lg_games is None or len(lg_games) == 0:
                    if verbose:
                        print(f"   leaguegamefinder fallback also returned no rows for {game_date_str}.")
                    # Try a slightly wider window (date-1 to date+1)
                    try:
                        alt_from = (datetime.strptime(game_date_str, '%Y-%m-%d') - timedelta(days=1)).strftime('%m/%d/%Y')
                        alt_to = (datetime.strptime(game_date_str, '%Y-%m-%d') + timedelta(days=1)).strftime('%m/%d/%Y')
                        for st in season_types:
                            try:
                                from nba_api.stats.endpoints import leaguegamefinder
                                lg = leaguegamefinder.LeagueGameFinder(
                                    date_from_nullable=alt_from,
                                    date_to_nullable=alt_to,
                                    season_nullable=season,
                                    season_type_nullable=st,
                                    timeout=10
                                )
                                lg_df2 = lg.get_data_frames()[0]
                                if len(lg_df2) > 0:
                                    # filter to exact date
                                    lg_df2['GAME_DATE'] = pd.to_datetime(lg_df2['GAME_DATE'])
                                    exact = lg_df2[lg_df2['GAME_DATE'] == pd.to_datetime(game_date_str)]
                                    if len(exact) > 0:
                                        lg_games = exact
                                        break
                            except Exception:
                                continue
                    except Exception:
                        pass

                    # If still empty, try nba_api's LeagueGameFinder as a last resort
                    if lg_games is None or len(lg_games) == 0:
                        try:
                            if verbose:
                                print("   Trying nba_api LeagueGameFinder fallback (may be slower)...")
                            from nba_api.stats.endpoints import leaguegamefinder as lgf
                            time.sleep(0.5)
                            finder = lgf.LeagueGameFinder(season_nullable=season, league_id_nullable='00')
                            df_all = finder.get_data_frames()[0]
                            df_all['GAME_DATE'] = pd.to_datetime(df_all['GAME_DATE'])
                            exact = df_all[df_all['GAME_DATE'] == pd.to_datetime(game_date_str)]
                            if len(exact) > 0:
                                lg_games = exact
                        except Exception:
                            if verbose:
                                print("   nba_api LeagueGameFinder fallback failed or returned no rows.")
                            lg_games = None

                    if lg_games is None or len(lg_games) == 0:
                        # Try CDN JSON before giving up
                        try:
                            cdn_games = _fetch_cdn_scoreboard(game_date_str)
                            if cdn_games:
                                today_games = []
                                team_id_to_info = {t['id']: t for t in teams.get_teams()}
                                for g in cdn_games:
                                    home_id = g.get('home_team_id')
                                    away_id = g.get('away_team_id')
                                    home_info = team_id_to_info.get(home_id, {})
                                    away_info = team_id_to_info.get(away_id, {})
                                    status = g.get('game_status', 'Scheduled')
                                    today_games.append({
                                        'game_id': g.get('game_id'),
                                        'home_team': home_info.get('nickname', str(home_id)),
                                        'away_team': away_info.get('nickname', str(away_id)),
                                        'home_team_id': home_id,
                                        'away_team_id': away_id,
                                        'game_status': status,
                                    })
                                if verbose:
                                    print(f"   Result source: NBA CDN JSON")
                                return today_games, game_date_str, False
                        except Exception:
                            pass

                        return [], game_date_str, False

                # leaguegamefinder returns one row per team per game; group by GAME_ID
                today_games = []
                for gid, group in lg_games.groupby('GAME_ID'):
                    home_matches = group[group['MATCHUP'].str.contains(' vs. ', na=False)]
                    away_matches = group[group['MATCHUP'].str.contains(' @ ', na=False)]
                    
                    if len(home_matches) > 0 and len(away_matches) > 0:
                        h_row = home_matches.iloc[0]
                        a_row = away_matches.iloc[0]
                    else:
                        h_row = group.iloc[0]
                        a_row = group.iloc[-1]
                        
                    home_id = h_row.get('TEAM_ID')
                    away_id = a_row.get('TEAM_ID')
                    home_team = h_row.get('TEAM_NAME', 'Unknown')
                    away_team = a_row.get('TEAM_NAME', 'Unknown')

                    today_games.append({
                        'game_id': gid,
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_team_id': int(home_id) if home_id is not None and str(home_id).isdigit() else None,
                        'away_team_id': int(away_id) if away_id is not None and str(away_id).isdigit() else None,
                        'game_status': 'Scheduled'
                    })

                # If we built games from leaguegamefinder, return them
                if len(today_games) > 0:
                    return today_games, game_date_str, False
            except Exception:
                if verbose:
                    print("   leaguegamefinder fallback raised an exception")
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
        if len(today_games) > 0:
            if verbose:
                print(f"   Result source: scoreboardv2")
            return today_games, game_date_str, all_finished

        # If scoreboardv2 produced nothing, earlier we tried leaguegamefinder and nba_api fallbacks.
        # As a final attempt, try the NBA CDN JSON
        try:
            cdn_games = _fetch_cdn_scoreboard(game_date_str)
            if cdn_games:
                today_games = []
                team_id_to_info = {t['id']: t for t in teams.get_teams()}
                for g in cdn_games:
                    home_id = g.get('home_team_id')
                    away_id = g.get('away_team_id')
                    home_info = team_id_to_info.get(home_id, {})
                    away_info = team_id_to_info.get(away_id, {})
                    status = g.get('game_status', 'Scheduled')
                    today_games.append({
                        'game_id': g.get('game_id'),
                        'home_team': home_info.get('nickname', str(home_id)),
                        'away_team': away_info.get('nickname', str(away_id)),
                        'home_team_id': home_id,
                        'away_team_id': away_id,
                        'game_status': status,
                    })
                if verbose:
                    print(f"   Result source: NBA CDN JSON")
                return today_games, game_date_str, False
        except Exception:
            pass

        return today_games, game_date_str, all_finished
        
    except Exception as e:
        if verbose:
            print(f"   Error fetching games: {e}")
        return [], game_date_str, False


def fetch_season_games(season: str = None, verbose: bool = True) -> pd.DataFrame:
    """Fetch all games for a season.
    
    Uses urllib directly to bypass CDN blocking of requests/urllib3.
    Falls back to nba_api if urllib fails.
    
    Args:
        season: Season string (e.g., '2025-26'). Defaults to current season.
        verbose: Whether to print progress.
        
    Returns:
        DataFrame with game data sorted by date.
    """
    import time
    
    if season is None:
        season = get_current_season()
    
    if verbose:
        print(f"📊 Fetching games for {season} season...")

    # Prefer cached games to avoid long API hangs. Use NBA_FORCE_FETCH=1 to bypass cache.
    if os.getenv('NBA_FORCE_FETCH') != '1' and os.path.exists(GAMES_CACHE_FILE):
        try:
            cached_df = pd.read_csv(GAMES_CACHE_FILE)
            if 'GAME_DATE' in cached_df.columns:
                cached_df['GAME_DATE'] = pd.to_datetime(cached_df['GAME_DATE'])
            if 'SEASON' in cached_df.columns:
                cached_df = cached_df[cached_df['SEASON'] == season]
            cached_df = cached_df.sort_values('GAME_DATE')
            if len(cached_df) > 0:
                if verbose:
                    print(f"📂 Loaded {len(cached_df)} cached games from {GAMES_CACHE_FILE}")
                return cached_df
        except Exception as cache_err:
            if verbose:
                print(f"⚠ Cache load failed: {cache_err}; falling back to API.")
    
    try:
        time.sleep(0.6)  # Rate limiting
        params = {
            'Conference': '',
            'DateFrom': '',
            'DateTo': '',
            'Division': '',
            'DraftNumber': '',
            'DraftRound': '',
            'DraftTeamID': '',
            'DraftYear': '',
            'GameID': '',
            'LeagueID': '00',
            'Location': '',
            'Outcome': '',
            'PORound': '',
            'PlayerID': '',
            'PlayerOrTeam': 'T',
            'RookieYear': '',
            'Season': season,
            'SeasonSegment': '',
            'SeasonType': 'Regular Season',
            'StarterBench': '',
            'TeamID': '',
            'VsConference': '',
            'VsDivision': '',
            'VsTeamID': '',
            'YearsExperience': '',
        }
        result = _fetch_nba_stats('leaguegamefinder', params, timeout=60)
        games_df = _result_set_to_df(result, index=0)
        games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
        games_df = games_df.sort_values('GAME_DATE')
        
        if verbose:
            print(f"✓ Loaded {len(games_df)} games from {season} season")
        
        return games_df
    except Exception as e:
        if verbose:
            print(f"⚠ Direct fetch failed: {e}")
            print("  Trying nba_api fallback...")
        # Fallback: try nba_api (uses requests)
        try:
            from nba_api.stats.endpoints import leaguegamefinder
            time.sleep(1)
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                league_id_nullable='00',
                timeout=120
            )
            games_df = gamefinder.get_data_frames()[0]
            games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
            games_df = games_df.sort_values('GAME_DATE')
            if verbose:
                print(f"✓ Loaded {len(games_df)} games from {season} season (via fallback)")
            return games_df
        except Exception as fallback_err:
            raise RuntimeError(
                f"Failed to fetch season games. "
                f"NBA API may be temporarily unavailable. "
                f"urllib error: {e} | nba_api error: {fallback_err}"
            ) from fallback_err


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
