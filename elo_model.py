# elo_model.py
# Calculates Elo ratings by replaying real match history.
# Every team starts at 1500. Wins add points, losses subtract.
# The bigger the upset, the more points change hands.

import pandas as pd
import numpy as np

HOME_ADVANTAGE = 50    # Home teams get a 50-point rating boost
K_FACTOR       = 32   # How much ratings shift after each game
INITIAL_RATING = 1500
SEASON_REVERSION = 0.3  # Each off-season, ratings drift 30% back to 1500


def expected_prob(rating_a, rating_b):
    """Standard Elo probability formula. Returns P(A beats B)."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_ratings(home_r, away_r, home_won, k=K_FACTOR):
    """Update both teams' ratings after a game."""
    exp_home = expected_prob(home_r + HOME_ADVANTAGE, away_r)
    change   = k * (home_won - exp_home)
    return round(home_r + change, 1), round(away_r - change, 1)


def calculate_elo_ratings(df):
    """
    Replay every game in order to build up Elo ratings.
    Returns: (ratings dict, history dataframe)
    """
    ratings = {}
    history = []
    current_season = None

    for _, game in df.iterrows():
        home, away, year = game['hteam'], game['ateam'], game['year']

        # Off-season: pull ratings back toward 1500
        if year != current_season:
            if current_season is not None:
                for team in ratings:
                    ratings[team] += SEASON_REVERSION * (INITIAL_RATING - ratings[team])
            current_season = year

        if home not in ratings: ratings[home] = INITIAL_RATING
        if away not in ratings: ratings[away] = INITIAL_RATING

        pre_h, pre_a = ratings[home], ratings[away]
        new_h, new_a = update_ratings(ratings[home], ratings[away], game['home_win'])
        ratings[home], ratings[away] = new_h, new_a

        history.append({
            'year': year, 'round': game['round'],
            'home': home, 'away': away,
            'pre_home_elo': pre_h, 'pre_away_elo': pre_a,
            'post_home_elo': new_h, 'post_away_elo': new_a,
            'home_won': game['home_win']
        })

    return ratings, pd.DataFrame(history)


def get_win_probability(home_team, away_team, ratings, injury_adj=None):
    """
    Get win probability for home team, with optional injury adjustments.
    injury_adj: dict like {"Collingwood": -30, "Carlton": -15}
    """
    hr = ratings.get(home_team, INITIAL_RATING)
    ar = ratings.get(away_team, INITIAL_RATING)
    if injury_adj:
        hr += injury_adj.get(home_team, 0)
        ar += injury_adj.get(away_team, 0)
    return round(expected_prob(hr + HOME_ADVANTAGE, ar), 4)
