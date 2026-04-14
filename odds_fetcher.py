# odds_fetcher.py
# Fetches REAL bookmaker odds from The Odds API.
# Free tier: 500 requests/month. One call = 1 request.
# API docs: https://the-odds-api.com/liveapi/guides/v4/

import requests
import streamlit as st

AFL_SPORT_KEY = "aussierules_afl"

TEAM_NAME_MAP = {
    "Adelaide Crows":            "Adelaide",
    "Brisbane Lions":            "Brisbane Lions",
    "Carlton Blues":             "Carlton",
    "Collingwood Magpies":       "Collingwood",
    "Essendon Bombers":          "Essendon",
    "Fremantle Dockers":         "Fremantle",
    "Geelong Cats":              "Geelong",
    "Gold Coast Suns":           "Gold Coast",
    "Greater Western Sydney":    "GWS Giants",
    "GWS Giants":                "GWS Giants",
    "Hawthorn Hawks":            "Hawthorn",
    "Melbourne Demons":          "Melbourne",
    "North Melbourne Kangaroos": "North Melbourne",
    "North Melbourne":           "North Melbourne",
    "Port Adelaide Power":       "Port Adelaide",
    "Richmond Tigers":           "Richmond",
    "St Kilda Saints":           "St Kilda",
    "Sydney Swans":              "Sydney",
    "West Coast Eagles":         "West Coast",
    "Western Bulldogs":          "Western Bulldogs",
    "Adelaide":                  "Adelaide",
    "Brisbane":                  "Brisbane Lions",
    "Carlton":                   "Carlton",
    "Collingwood":               "Collingwood",
    "Essendon":                  "Essendon",
    "Fremantle":                 "Fremantle",
    "Geelong":                   "Geelong",
    "Gold Coast":                "Gold Coast",
    "Hawthorn":                  "Hawthorn",
    "Melbourne":                 "Melbourne",
    "Port Adelaide":             "Port Adelaide",
    "Richmond":                  "Richmond",
    "St Kilda":                  "St Kilda",
    "Sydney":                    "Sydney",
    "West Coast":                "West Coast",
}


def normalize_team(api_name):
    return TEAM_NAME_MAP.get(api_name, api_name)


def get_api_key():
    try:
        return st.secrets["ODDS_API_KEY"]
    except Exception:
        return None


def fetch_afl_odds():
    """
    Fetch all upcoming AFL games with bookmaker odds from The Odds API.
    Returns (games_list, status_message).
    """
    api_key = get_api_key()
    if not api_key:
        return None, "No API key. Add ODDS_API_KEY to Streamlit secrets."

    url = f"https://api.the-odds-api.com/v4/sports/{AFL_SPORT_KEY}/odds/"
    params = {
        "apiKey":     api_key,
        "regions":    "au",
        "markets":    "h2h",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        requests_remaining = response.headers.get("x-requests-remaining", "unknown")
        requests_used      = response.headers.get("x-requests-used",      "unknown")

        if response.status_code == 401:
            return None, "Invalid API key."
        if response.status_code == 429:
            return None, "Monthly limit reached (500/month free tier)."
        if response.status_code != 200:
            return None, f"API error: {response.status_code}"

        games_raw = response.json()
        if not games_raw:
            return [], f"No AFL games listed. Between rounds? Requests remaining: {requests_remaining}"

        games = []
        for game in games_raw:
            home_team = normalize_team(game.get("home_team", ""))
            away_team = normalize_team(game.get("away_team", ""))

            bookmaker_data = []
            home_odds_list = []
            away_odds_list = []

            for bm in game.get("bookmakers", []):
                bm_name  = bm.get("title", bm.get("key", "Unknown"))
                markets  = bm.get("markets", [])
                h2h      = next((m for m in markets if m.get("key") == "h2h"), None)
                if not h2h:
                    continue

                home_odds = None
                away_odds = None
                for outcome in h2h.get("outcomes", []):
                    oname = normalize_team(outcome.get("name", ""))
                    price = outcome.get("price")
                    if oname == home_team:
                        home_odds = price
                    elif oname == away_team:
                        away_odds = price

                if home_odds and away_odds:
                    bookmaker_data.append({
                        "bookmaker": bm_name,
                        "home_odds": home_odds,
                        "away_odds": away_odds,
                    })
                    home_odds_list.append(home_odds)
                    away_odds_list.append(away_odds)

            if not bookmaker_data:
                continue

            best_home = max(home_odds_list)
            best_away = max(away_odds_list)
            avg_home  = round(sum(home_odds_list) / len(home_odds_list), 3)
            avg_away  = round(sum(away_odds_list) / len(away_odds_list), 3)

            hi = 1.0 / avg_home
            ai = 1.0 / avg_away
            total = hi + ai

            games.append({
                "home_team":       home_team,
                "away_team":       away_team,
                "commence_time":   game.get("commence_time", ""),
                "bookmakers":      bookmaker_data,
                "bookmaker_count": len(bookmaker_data),
                "best_home_odds":  best_home,
                "best_away_odds":  best_away,
                "avg_home_odds":   avg_home,
                "avg_away_odds":   avg_away,
                "market_home_prob":round(hi / total, 4),
                "market_away_prob":round(ai / total, 4),
                "bookmaker_margin":round((total - 1) * 100, 2),
                "requests_remaining": requests_remaining,
            })

        status = (
            f"✅ Loaded {len(games)} games. "
            f"API requests used: {requests_used} | Remaining: {requests_remaining}"
        )
        return games, status

    except requests.Timeout:
        return None, "Request timed out."
    except Exception as e:
        return None, f"Error: {str(e)}"


def get_best_bookmaker_for_team(api_game, team, is_home):
    """Returns (bookmaker_name, best_odds) for a team."""
    key  = "home_odds" if is_home else "away_odds"
    best = ("None", 0)
    for bm in api_game.get("bookmakers", []):
        o = bm.get(key, 0)
        if o > best[1]:
            best = (bm["bookmaker"], o)
    return best
