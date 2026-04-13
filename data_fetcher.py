# data_fetcher.py
# Downloads real AFL results from api.squiggle.com.au
# No API key needed — completely free and public.
# Results are saved so we don't re-download every time.

import requests
import pandas as pd
import time
import os

HEADERS = {"User-Agent": "AFL-Odds-Analyzer - personal learning project"}
SQUIGGLE_URL = "https://api.squiggle.com.au/"
DATA_FILE = "afl_results.csv"


def fetch_season(year):
    """Download all completed games for one year."""
    params = {"q": "games", "year": year, "complete": 100}
    try:
        r = requests.get(SQUIGGLE_URL, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        games = r.json().get("games", [])
        return games
    except Exception as e:
        print(f"  Warning: could not fetch {year}: {e}")
        return []


def download_all_data(start_year=2015, end_year=2025):
    """Download multiple seasons and save as CSV."""
    all_games = []
    print(f"Downloading AFL data {start_year}–{end_year}...")

    for year in range(start_year, end_year + 1):
        print(f"  Fetching {year}...", end=" ")
        games = fetch_season(year)
        all_games.extend(games)
        print(f"{len(games)} games")
        time.sleep(0.8)  # Be polite to the server

    if not all_games:
        print("No data downloaded.")
        return None

    df = pd.DataFrame(all_games)

    # Keep only the columns we need
    keep = ['id','year','round','roundname','hteam','ateam',
            'hscore','ascore','hgoals','agoals','venue','date','complete']
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    # Remove games with missing scores
    df = df.dropna(subset=['hscore','ascore'])
    df['hscore'] = pd.to_numeric(df['hscore'], errors='coerce')
    df['ascore'] = pd.to_numeric(df['ascore'], errors='coerce')
    df = df.dropna(subset=['hscore','ascore'])

    # Add useful columns
    df['home_win'] = (df['hscore'] > df['ascore']).astype(int)
    df['margin']   = df['hscore'] - df['ascore']
    df = df.sort_values(['year','round']).reset_index(drop=True)

    df.to_csv(DATA_FILE, index=False)
    print(f"Saved {len(df)} games to {DATA_FILE}")
    return df


def load_data():
    """Load saved data, or download if not available."""
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        print(f"Loaded {len(df)} games from file ({df['year'].min()}–{df['year'].max()})")
        return df
    return download_all_data()
