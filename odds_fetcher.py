# odds_fetcher.py
# ─────────────────────────────────────────────────────────
# Fetches REAL bookmaker odds from The Odds API.
#
# What this does:
#   - Calls The Odds API once to get all upcoming AFL games
#   - Returns the odds from every Australian bookmaker
#     (Sportsbet, TAB, Neds, Ladbrokes, Betfair, Unibet)
#   - Matches those games to the teams in our model
#   - Returns a clean dictionary we can use in the app
#
# Free tier: 500 requests/month. One call here = 1 request.
# Checking once or twice a week = ~8 requests/month. Very safe.
#
# API documentation: https://the-odds-api.com/liveapi/guides/v4/
# ─────────────────────────────────────────────────────────

import requests
import streamlit as st

# The sport key The Odds API uses for AFL
AFL_SPORT_KEY = "aussierules_afl"

# Australian bookmakers we want odds from
# These are the ones most Australians actually use
AU_BOOKMAKERS = [
    "sportsbet",
    "tab",
    "neds",
    "ladbrokes",
    "betfair",
    "unibet",
    "pointsbetau",
]

# ── Team name mapping ──────────────────────────────────
# The Odds API uses slightly different team names to our model.
# This dictionary translates between them.
# Left side = what The Odds API returns
# Right side = what our Elo model uses
TEAM_NAME_MAP = {
    "Adelaide Crows":          "Adelaide",
    "Brisbane Lions":          "Brisbane Lions",
    "Carlton Blues":           "Carlton",
    "Collingwood Magpies":     "Collingwood",
    "Essendon Bombers":        "Essendon",
    "Fremantle Dockers":       "Fremantle",
    "Geelong Cats":            "Geelong",
    "Gold Coast Suns":         "Gold Coast",
    "Greater Western Sydney":  "GWS Giants",
    "GWS Giants":              "GWS Giants",
    "Hawthorn Hawks":          "Hawthorn",
    "Melbourne Demons":        "Melbourne",
    "North Melbourne Kangaroos": "North Melbourne",
    "North Melbourne":         "North Melbourne",
    "Port Adelaide Power":     "Port Adelaide",
    "Richmond Tigers":         "Richmond",
    "St Kilda Saints":         "St Kilda",
    "Sydney Swans":            "Sydney",
    "West Coast Eagles":       "West Coast",
    "Western Bulldogs":        "Western Bulldogs",
    # Sometimes the API returns shortened names too
    "Adelaide":                "Adelaide",
    "Brisbane":                "Brisbane Lions",
    "Carlton":                 "Carlton",
    "Collingwood":             "Collingwood",
    "Essendon":                "Essendon",
    "Fremantle":               "Fremantle",
    "Geelong":                 "Geelong",
    "Gold Coast":              "Gold Coast",
    "Hawthorn":                "Hawthorn",
    "Melbourne":               "Melbourne",
    "Port Adelaide":           "Port Adelaide",
    "Richmond":                "Richmond",
    "St Kilda":                "St Kilda",
    "Sydney":                  "Sydney",
    "West Coast":              "West Coast",
}


def normalize_team(api_name):
    """
    Converts a team name from The Odds API format
    to the format our Elo model uses.
    Falls back to the original name if not in the map.
    """
    return TEAM_NAME_MAP.get(api_name, api_name)


def get_api_key():
    """
    Reads the API key from Streamlit Secrets.
    If not found (e.g. running locally), returns None.
    """
    try:
        return st.secrets["ODDS_API_KEY"]
    except Exception:
        return None


def fetch_afl_odds():
    """
    Main function. Fetches all upcoming AFL games with bookmaker odds.

    Returns a list of game dictionaries, each containing:
      - home_team: home team name (normalized to our format)
      - away_team: away team name (normalized)
      - commence_time: when the game starts (ISO string)
      - bookmakers: list of bookmaker odds dicts
      - best_home_odds: the highest home odds across all bookmakers
      - best_away_odds: the highest away odds across all bookmakers
      - avg_home_odds: average home odds across bookmakers
      - avg_away_odds: average away odds across bookmakers
      - bookmaker_count: how many bookmakers had this game listed
      - requests_remaining: how many API calls you have left this month

    Returns None if the API call fails or key is missing.
    """
    api_key = get_api_key()

    if not api_key:
        return None, "No API key found. Add ODDS_API_KEY to your Streamlit secrets."

    url = f"https://api.the-odds-api.com/v4/sports/{AFL_SPORT_KEY}/odds/"

    params = {
        "apiKey":      api_key,
        "regions":     "au",          # Australian bookmakers only
        "markets":     "h2h",         # Head-to-head (win/loss) market
        "oddsFormat":  "decimal",     # e.g. 1.85 not +85
        "dateFormat":  "iso",
    }

    try:
        response = requests.get(url, params=params, timeout=15)

        # Check how many requests we have left (shown in response headers)
        requests_remaining = response.headers.get("x-requests-remaining", "unknown")
        requests_used      = response.headers.get("x-requests-used",      "unknown")

        if response.status_code == 401:
            return None, "Invalid API key. Check your key in Streamlit Secrets."

        if response.status_code == 429:
            return None, "Monthly request limit reached (500/month on free tier)."

        if response.status_code != 200:
            return None, f"API error: {response.status_code} — {response.text[:200]}"

        games_raw = response.json()

        if not games_raw:
            return [], f"No AFL games currently listed. Season may be between rounds. Requests remaining: {requests_remaining}"

        # ── Process each game ──────────────────────────
        games = []
        for game in games_raw:
            home_raw = game.get("home_team", "")
            away_raw = game.get("away_team", "")

            home_team = normalize_team(home_raw)
            away_team = normalize_team(away_raw)

            # Collect odds from each bookmaker
            bookmaker_data = []
            home_odds_list = []
            away_odds_list = []

            for bm in game.get("bookmakers", []):
                bm_name  = bm.get("title", bm.get("key", "Unknown"))
                bm_key   = bm.get("key", "")

                # Only include Australian bookmakers we care about
                # (filter out any irrelevant ones the API might include)
                markets = bm.get("markets", [])
                h2h     = next((m for m in markets if m.get("key") == "h2h"), None)

                if not h2h:
                    continue

                outcomes = h2h.get("outcomes", [])
                home_odds = None
                away_odds = None

                for outcome in outcomes:
                    oname = normalize_team(outcome.get("name", ""))
                    price = outcome.get("price", None)
                    if oname == home_team:
                        home_odds = price
                    elif oname == away_team:
                        away_odds = price

                if home_odds and away_odds:
                    bookmaker_data.append({
                        "bookmaker": bm_name,
                        "home_odds": home_odds,
                        "away_odds": away_odds,
                        "last_update": bm.get("last_update", ""),
                    })
                    home_odds_list.append(home_odds)
                    away_odds_list.append(away_odds)

            if not bookmaker_data:
                continue

            # Best odds = highest payout available across all bookmakers
            best_home = max(home_odds_list)
            best_away = max(away_odds_list)

            # Average odds = the "market consensus"
            avg_home  = round(sum(home_odds_list) / len(home_odds_list), 3)
            avg_away  = round(sum(away_odds_list) / len(away_odds_list), 3)

            # Market implied probability (removes bookmaker margin)
            # Raw implied probs add up to >100% because of the margin
            raw_home_prob = 1 / avg_home
            raw_away_prob = 1 / avg_away
            total         = raw_home_prob + raw_away_prob
            market_home_prob = round(raw_home_prob / total, 4)  # Normalize to 100%
            market_away_prob = round(raw_away_prob / total, 4)

            # Bookmaker margin = how much extra they're taking
            # e.g. 5% margin means for every $100 bet, they keep $5 on average
            margin = round((total - 1) * 100, 2)

            games.append({
                "home_team":          home_team,
                "away_team":          away_team,
                "home_team_raw":      home_raw,
                "away_team_raw":      away_raw,
                "commence_time":      game.get("commence_time", ""),
                "bookmakers":         bookmaker_data,
                "bookmaker_count":    len(bookmaker_data),
                "best_home_odds":     best_home,
                "best_away_odds":     best_away,
                "avg_home_odds":      avg_home,
                "avg_away_odds":      avg_away,
                "market_home_prob":   market_home_prob,
                "market_away_prob":   market_away_prob,
                "bookmaker_margin":   margin,
                "requests_remaining": requests_remaining,
                "requests_used":      requests_used,
            })

        status_msg = (
            f"✅ Loaded {len(games)} games from {len(AU_BOOKMAKERS)} bookmakers. "
            f"API requests used: {requests_used} | Remaining this month: {requests_remaining}"
        )
        return games, status_msg

    except requests.Timeout:
        return None, "Request timed out. Check your internet connection."
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


def match_game_to_model(api_game, model_teams):
    """
    Checks whether both teams in an API game exist in our Elo model.
    Returns True if we can analyse this game, False if we can't.
    """
    home = api_game.get("home_team", "")
    away = api_game.get("away_team", "")
    return home in model_teams and away in model_teams


def get_best_bookmaker_for_team(api_game, team, is_home):
    """
    Finds which bookmaker is offering the best (highest) odds for a team.
    Returns: (bookmaker_name, odds)
    """
    key   = "home_odds" if is_home else "away_odds"
    best  = ("None", 0)
    for bm in api_game.get("bookmakers", []):
        o = bm.get(key, 0)
        if o > best[1]:
            best = (bm["bookmaker"], o)
    return best
