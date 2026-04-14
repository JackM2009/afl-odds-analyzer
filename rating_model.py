# rating_model.py
# ═══════════════════════════════════════════════════════════
# MODULE: Rating Model (Elo)
#
# Margin-of-victory adjusted Elo system.
# Replays every AFL game since 2015 to build current ratings.
#
# Key design choices:
#   K = 20 (team sports standard, more stable than 32)
#   Scale = 400 (standard logistic)
#   Home advantage = +50 Elo pts (validated for AFL)
#   Season reversion = 25% toward 1500 each off-season
#   Margin multiplier: FiveThirtyEight NFL formula adapted for AFL
# ═══════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
from feature_engineering import margin_of_victory_multiplier

INITIAL_RATING   = 1500.0
K_FACTOR         = 20.0
ELO_SCALE        = 400.0
HOME_ADVANTAGE   = 50.0
SEASON_REVERSION = 0.25


def win_probability_from_elo(rating_a, rating_b, home_advantage=HOME_ADVANTAGE):
    """
    P(Team A wins) using logistic Elo formula with home advantage.

    P = 1 / (1 + 10^((B - (A + home_adv)) / scale))

    Reference values:
      Equal teams (1500 v 1500):  56.7% (home wins more often)
      +100 Elo advantage:         64.0%
      +200 Elo advantage:         76.0%
      +300 Elo advantage:         84.9%
    """
    effective_home = rating_a + home_advantage
    exponent       = (rating_b - effective_home) / ELO_SCALE
    return 1.0 / (1.0 + 10.0 ** exponent)


def update_elo(home_rating, away_rating, home_won, margin):
    """
    Updates both Elo ratings after a game (zero-sum).

    new_rating = old + K × G × (actual - expected)
    G = margin_of_victory_multiplier (log-scaled, capped 1.0–2.5)
    """
    expected = win_probability_from_elo(home_rating, away_rating)
    actual   = float(home_won)

    if home_won:
        winner_elo, loser_elo = home_rating, away_rating
    else:
        winner_elo, loser_elo = away_rating, home_rating

    G = margin_of_victory_multiplier(abs(margin), winner_elo, loser_elo)

    change   = K_FACTOR * G * (actual - expected)
    new_home = round(home_rating + change, 2)
    new_away = round(away_rating - change, 2)
    return new_home, new_away


def calculate_elo_ratings(results_df):
    """
    Replays all AFL games in order to compute current Elo ratings.

    Returns:
      ratings:     dict {team: current_elo}
      history_df:  DataFrame with pre/post Elo for every game
                   (used for feature engineering and backtesting)
    """
    ratings     = {}
    history     = []
    prev_season = None

    results_df = results_df.reset_index(drop=True)

    for idx, game in results_df.iterrows():
        home     = game['hteam']
        away     = game['ateam']
        year     = int(game['year'])
        margin   = game['margin']
        home_won = int(game['home_win'])

        # Off-season reversion
        if year != prev_season and prev_season is not None:
            for team in list(ratings.keys()):
                r = ratings[team]
                ratings[team] = round(r + SEASON_REVERSION * (INITIAL_RATING - r), 2)
        prev_season = year

        if home not in ratings: ratings[home] = INITIAL_RATING
        if away not in ratings: ratings[away] = INITIAL_RATING

        pre_home = ratings[home]
        pre_away = ratings[away]

        new_home, new_away = update_elo(pre_home, pre_away, home_won, abs(margin))
        ratings[home] = new_home
        ratings[away] = new_away

        history.append({
            'game_index':    idx,
            'year':          year,
            'round':         game.get('round', 0),
            'home':          home,
            'away':          away,
            'home_won':      home_won,
            'margin':        margin,
            'pre_home_elo':  pre_home,
            'pre_away_elo':  pre_away,
            'post_home_elo': new_home,
            'post_away_elo': new_away,
            'elo_diff_pre':  pre_home - pre_away,
        })

    return ratings, pd.DataFrame(history)


def get_elo_probability(home_team, away_team, ratings, adjustments=None):
    """Win probability using current ratings, with optional Elo adjustments."""
    hr = ratings.get(home_team, INITIAL_RATING)
    ar = ratings.get(away_team, INITIAL_RATING)
    if adjustments:
        hr += adjustments.get(home_team, 0.0)
        ar += adjustments.get(away_team, 0.0)
    return win_probability_from_elo(hr, ar)
