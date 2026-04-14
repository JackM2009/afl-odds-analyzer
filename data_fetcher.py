# data_fetcher.py
# ═══════════════════════════════════════════════════════════
# Downloads AFL results AND advanced performance stats from
# api.squiggle.com.au — completely free, no API key needed.
#
# Stats pulled per game (where available):
#   inside50s, clearances, turnovers, kicks, handballs,
#   marks, tackles, hitouts, goals, behinds
#
# Also downloads historical bookmaker odds from
# aussportsbetting.com.au for real backtest validation.
# ═══════════════════════════════════════════════════════════

import requests
import pandas as pd
import numpy as np
import time
import os
import io

HEADERS = {"User-Agent": "AFL-Odds-Analyzer - personal learning project"}
SQUIGGLE_URL = "https://api.squiggle.com.au/"
DATA_FILE       = "afl_results.csv"
STATS_FILE      = "afl_stats.csv"
HIST_ODDS_FILE  = "afl_historical_odds.csv"

# aussportsbetting.com.au direct download URL for AFL historical odds
HIST_ODDS_URL = "https://www.aussportsbetting.com/historical_data/afl.xlsx"


# ─────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────

def fetch_season_results(year):
    """Download all completed games for one year."""
    params = {"q": "games", "year": year, "complete": 100}
    try:
        r = requests.get(SQUIGGLE_URL, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.json().get("games", [])
    except Exception as e:
        print(f"  Warning: could not fetch results for {year}: {e}")
        return []


def download_results(start_year=2015, end_year=2025):
    """Download multi-season results and save as CSV."""
    all_games = []
    print(f"Downloading AFL results {start_year}–{end_year}...")
    for year in range(start_year, end_year + 1):
        print(f"  Fetching {year}...", end=" ")
        games = fetch_season_results(year)
        all_games.extend(games)
        print(f"{len(games)} games")
        time.sleep(0.8)

    if not all_games:
        print("No data downloaded.")
        return None

    df = pd.DataFrame(all_games)
    keep = ['id','year','round','roundname','hteam','ateam',
            'hscore','ascore','hgoals','agoals','venue','date','complete']
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    df = df.dropna(subset=['hscore','ascore'])
    df['hscore'] = pd.to_numeric(df['hscore'], errors='coerce')
    df['ascore'] = pd.to_numeric(df['ascore'], errors='coerce')
    df = df.dropna(subset=['hscore','ascore'])
    df['home_win'] = (df['hscore'] > df['ascore']).astype(int)
    df['margin']   = df['hscore'] - df['ascore']
    df = df.sort_values(['year','round']).reset_index(drop=True)
    df.to_csv(DATA_FILE, index=False)
    print(f"Saved {len(df)} games to {DATA_FILE}")
    return df


# ─────────────────────────────────────────────────────────
# ADVANCED STATS
# ─────────────────────────────────────────────────────────

def fetch_season_stats(year):
    """
    Download per-game team stats for one year from Squiggle.
    Squiggle returns one row per team per game (two rows per game).
    """
    params = {"q": "stats", "year": year}
    try:
        r = requests.get(SQUIGGLE_URL, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return r.json().get("stats", [])
    except Exception as e:
        print(f"  Warning: could not fetch stats for {year}: {e}")
        return []


def download_stats(results_df, start_year=2015, end_year=2025):
    """
    Download stats for all seasons and join to results.

    Returns a DataFrame with one row per game containing
    opponent-context-aware stats for both home and away teams.
    """
    all_stats = []
    print(f"Downloading AFL advanced stats {start_year}–{end_year}...")
    for year in range(start_year, end_year + 1):
        print(f"  Fetching stats {year}...", end=" ")
        stats = fetch_season_stats(year)
        all_stats.extend(stats)
        print(f"{len(stats)} records")
        time.sleep(1.0)

    if not all_stats:
        print("  No stats available — proceeding without them.")
        return None

    sdf = pd.DataFrame(all_stats)

    # Squiggle stats columns (names vary by season):
    # gameid, year, round, team, abb, goals, behinds, score,
    # inside50s, clearances, kicks, handballs, marks, tackles, hitouts
    stat_cols = ['gameid','year','round','team',
                 'goals','behinds','inside50s','clearances',
                 'kicks','handballs','marks','tackles','hitouts']
    stat_cols = [c for c in stat_cols if c in sdf.columns]
    sdf = sdf[stat_cols].copy()

    # Numeric conversion
    for col in stat_cols:
        if col not in ('gameid','team'):
            sdf[col] = pd.to_numeric(sdf[col], errors='coerce')

    # Join to results: match on gameid (squiggle 'id' == stats 'gameid')
    if 'id' in results_df.columns and 'gameid' in sdf.columns:
        # Pivot to wide format: one row per game with home/away stats
        home_stats = sdf.copy()
        away_stats = sdf.copy()

        # We'll merge twice — once for home team, once for away
        results_df['gameid'] = results_df['id']

        # Build home stat rows
        home_merged = results_df.merge(
            home_stats.add_prefix('h_'),
            left_on=['gameid','hteam'],
            right_on=['h_gameid','h_team'],
            how='left'
        )
        # Build away stat rows
        full_merged = home_merged.merge(
            away_stats.add_prefix('a_'),
            left_on=['gameid','ateam'],
            right_on=['a_gameid','a_team'],
            how='left'
        )

        # Derived features: differential stats
        for stat in ['inside50s','clearances','kicks','handballs','marks','tackles','hitouts']:
            hcol = f'h_{stat}'
            acol = f'a_{stat}'
            if hcol in full_merged.columns and acol in full_merged.columns:
                full_merged[f'{stat}_diff'] = full_merged[hcol] - full_merged[acol]

        # Opponent-adjusted scoring: normalise each team's scoring
        # by opponent's defensive average (requires rolling calculation)
        full_merged = _add_opponent_adjusted_scoring(full_merged)

        full_merged.to_csv(STATS_FILE, index=False)
        print(f"Saved stats for {len(full_merged)} games to {STATS_FILE}")
        return full_merged

    return None


def _add_opponent_adjusted_scoring(df):
    """
    Adds opponent-adjusted scoring metric.

    For each game, a team's 'quality' score is:
      team_score / opponent_defensive_avg_allowed

    This rewards teams who score big against good defences.
    """
    df = df.reset_index(drop=True)
    # Rolling defensive average per team (how many pts they allow on avg)
    team_defense = {}  # team -> list of scores allowed

    opp_adj_h = []
    opp_adj_a = []

    for idx, row in df.iterrows():
        home = row['hteam']
        away = row['ateam']
        hs   = row['hscore']
        as_  = row['ascore']

        # Get opponent defensive avg before this game
        h_opp_def = np.mean(team_defense.get(away, [85.0])[-10:])
        a_opp_def = np.mean(team_defense.get(home, [85.0])[-10:])

        opp_adj_h.append(round(hs / max(h_opp_def, 1.0), 4) if pd.notna(hs) else 1.0)
        opp_adj_a.append(round(as_ / max(a_opp_def, 1.0), 4) if pd.notna(as_) else 1.0)

        # Update defensive history
        if pd.notna(as_):
            team_defense.setdefault(home, []).append(as_)
        if pd.notna(hs):
            team_defense.setdefault(away, []).append(hs)

    df['h_opp_adj_score'] = opp_adj_h
    df['a_opp_adj_score'] = opp_adj_a
    df['opp_adj_score_diff'] = df['h_opp_adj_score'] - df['a_opp_adj_score']
    return df


# ─────────────────────────────────────────────────────────
# HISTORICAL ODDS (for real backtesting)
# ─────────────────────────────────────────────────────────

def download_historical_odds():
    """
    Downloads historical AFL bookmaker odds from aussportsbetting.com.au.
    This gives us REAL historical prices to backtest against.

    The file contains: Date, Home Team, Away Team, Home Odds, Away Odds,
    Home Score, Away Score, Venue (columns vary by year).
    """
    print("Downloading historical AFL odds from aussportsbetting.com.au...")
    try:
        r = requests.get(HIST_ODDS_URL, headers=HEADERS, timeout=30)
        r.raise_for_status()

        # It's an Excel file
        xls = pd.ExcelFile(io.BytesIO(r.content))
        sheet = xls.sheet_names[0]
        df = pd.read_excel(io.BytesIO(r.content), sheet_name=sheet, header=0)

        print(f"  Raw columns: {list(df.columns)}")

        # Normalise column names (the file format is somewhat inconsistent)
        df.columns = [str(c).strip().lower().replace(' ','_') for c in df.columns]

        # Try to identify the key columns
        col_map = {}
        for col in df.columns:
            cl = col.lower()
            if 'date' in cl:               col_map['date'] = col
            elif 'home' in cl and 'team' in cl: col_map['home_team'] = col
            elif 'away' in cl and 'team' in cl: col_map['away_team'] = col
            elif 'home' in cl and ('odd' in cl or 'price' in cl or cl.endswith('_h')): col_map['home_odds'] = col
            elif 'away' in cl and ('odd' in cl or 'price' in cl or cl.endswith('_a')): col_map['away_odds'] = col
            elif 'home' in cl and 'score' in cl: col_map['home_score'] = col
            elif 'away' in cl and 'score' in cl: col_map['away_score'] = col

        if not col_map.get('home_team') or not col_map.get('home_odds'):
            # Fallback: assume standard column order
            # Date, Home, Away, HomeOdds, AwayOdds, HomeScore, AwayScore
            cols = list(df.columns)
            if len(cols) >= 5:
                col_map = {
                    'date':       cols[0],
                    'home_team':  cols[1],
                    'away_team':  cols[2],
                    'home_odds':  cols[3],
                    'away_odds':  cols[4],
                }
                if len(cols) > 5: col_map['home_score'] = cols[5]
                if len(cols) > 6: col_map['away_score'] = cols[6]

        # Build clean DataFrame
        clean = pd.DataFrame()
        for target, src in col_map.items():
            if src in df.columns:
                clean[target] = df[src]

        # Parse date and extract year
        clean['date'] = pd.to_datetime(clean['date'], errors='coerce', dayfirst=True)
        clean = clean.dropna(subset=['date'])
        clean['year'] = clean['date'].dt.year

        # Numeric odds
        for col in ['home_odds','away_odds','home_score','away_score']:
            if col in clean.columns:
                clean[col] = pd.to_numeric(clean[col], errors='coerce')

        # Filter: only valid odds
        clean = clean[clean['home_odds'].between(1.01, 30.0)]
        clean = clean[clean['away_odds'].between(1.01, 30.0)]

        # Normalise team names to match our model
        clean['home_team'] = clean['home_team'].apply(_normalise_team_name)
        clean['away_team'] = clean['away_team'].apply(_normalise_team_name)

        # Remove overround → true market probability
        hi = 1.0 / clean['home_odds']
        ai = 1.0 / clean['away_odds']
        total = hi + ai
        clean['market_home_prob'] = hi / total
        clean['market_away_prob'] = ai / total
        clean['overround'] = (total - 1.0) * 100

        clean = clean.sort_values('date').reset_index(drop=True)
        clean.to_csv(HIST_ODDS_FILE, index=False)
        print(f"Saved {len(clean)} historical odds records to {HIST_ODDS_FILE}")
        return clean

    except Exception as e:
        print(f"  Could not download historical odds: {e}")
        print("  Backtesting will use simulated 5% overround instead.")
        return None


def _normalise_team_name(name):
    """Map aussportsbetting team names to our model's names."""
    if not isinstance(name, str):
        return str(name)
    name = name.strip()
    MAP = {
        "Adelaide Crows":    "Adelaide",
        "Adelaide":          "Adelaide",
        "Brisbane Lions":    "Brisbane Lions",
        "Brisbane":          "Brisbane Lions",
        "Carlton Blues":     "Carlton",
        "Carlton":           "Carlton",
        "Collingwood":       "Collingwood",
        "Collingwood Magpies": "Collingwood",
        "Essendon":          "Essendon",
        "Essendon Bombers":  "Essendon",
        "Fremantle":         "Fremantle",
        "Fremantle Dockers": "Fremantle",
        "Geelong":           "Geelong",
        "Geelong Cats":      "Geelong",
        "Gold Coast":        "Gold Coast",
        "Gold Coast Suns":   "Gold Coast",
        "Greater Western Sydney": "GWS Giants",
        "GWS Giants":        "GWS Giants",
        "GWS":               "GWS Giants",
        "Hawthorn":          "Hawthorn",
        "Hawthorn Hawks":    "Hawthorn",
        "Melbourne":         "Melbourne",
        "Melbourne Demons":  "Melbourne",
        "North Melbourne":   "North Melbourne",
        "North Melbourne Kangaroos": "North Melbourne",
        "Kangaroos":         "North Melbourne",
        "Port Adelaide":     "Port Adelaide",
        "Port Adelaide Power": "Port Adelaide",
        "Richmond":          "Richmond",
        "Richmond Tigers":   "Richmond",
        "St Kilda":          "St Kilda",
        "St Kilda Saints":   "St Kilda",
        "Sydney":            "Sydney",
        "Sydney Swans":      "Sydney",
        "West Coast":        "West Coast",
        "West Coast Eagles": "West Coast",
        "Western Bulldogs":  "Western Bulldogs",
        "Bulldogs":          "Western Bulldogs",
        "Footscray":         "Western Bulldogs",
    }
    return MAP.get(name, name)


# ─────────────────────────────────────────────────────────
# MASTER DOWNLOAD + LOAD
# ─────────────────────────────────────────────────────────

def download_all_data(start_year=2015, end_year=2025):
    """Download results, stats, and historical odds."""
    results_df   = download_results(start_year, end_year)
    stats_df     = download_stats(results_df, start_year, end_year) if results_df is not None else None
    hist_odds_df = download_historical_odds()
    return results_df, stats_df, hist_odds_df


def load_data():
    """
    Load all data from disk. Download if not present.
    Returns: (results_df, stats_df, hist_odds_df)
    stats_df and hist_odds_df may be None if unavailable.
    """
    results_df   = None
    stats_df     = None
    hist_odds_df = None

    if os.path.exists(DATA_FILE):
        results_df = pd.read_csv(DATA_FILE)
        print(f"Loaded {len(results_df)} results ({results_df['year'].min()}–{results_df['year'].max()})")

    if os.path.exists(STATS_FILE):
        stats_df = pd.read_csv(STATS_FILE)
        print(f"Loaded {len(stats_df)} stat rows")

    if os.path.exists(HIST_ODDS_FILE):
        hist_odds_df = pd.read_csv(HIST_ODDS_FILE)
        hist_odds_df['date'] = pd.to_datetime(hist_odds_df['date'], errors='coerce')
        print(f"Loaded {len(hist_odds_df)} historical odds records")

    if results_df is None:
        print("No data found — downloading now...")
        results_df, stats_df, hist_odds_df = download_all_data()

    return results_df, stats_df, hist_odds_df
