# feature_engineering.py
# ═══════════════════════════════════════════════════════════
# MODULE: Feature Engineering
#
# Responsibility: Extract ALL predictive features from raw
# match data. This module is the only place features are
# calculated — no feature logic lives anywhere else.
#
# Features produced:
#   TIER 1 — Primary (highest predictive power)
#     • Elo rating difference (pre-game)
#     • Margin of victory multiplier (for Elo updates)
#
#   TIER 2 — Form-based
#     • Weighted recent form (last 5 games, exponential decay)
#     • Scoring differential (offence - defence)
#     • Consistency (std dev of recent margins)
#
#   TIER 3 — Situational
#     • Home/away indicator
#     • Days rest (short turnaround penalty)
#     • Interstate travel flag
#
# IMPORTANT: All features are calculated BEFORE the game
# being predicted. No data leakage from future games.
# ═══════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
from datetime import datetime

# ── Venue → State mapping for travel fatigue calculation ──
VENUE_STATE = {
    "MCG": "VIC", "Melbourne Cricket Ground": "VIC",
    "Marvel Stadium": "VIC", "Docklands": "VIC",
    "GMHBA Stadium": "VIC", "Kardinia Park": "VIC",
    "Mars Stadium": "VIC", "Ikon Park": "VIC",
    "Gabba": "QLD", "The Gabba": "QLD",
    "People First Stadium": "QLD",
    "Adelaide Oval": "SA",
    "Optus Stadium": "WA", "Perth Stadium": "WA",
    "SCG": "NSW", "Sydney Cricket Ground": "NSW",
    "Giants Stadium": "NSW", "ENGIE Stadium": "NSW",
    "Blundstone Arena": "TAS", "York Park": "TAS",
    "Manuka Oval": "ACT",
}

# Team → home state mapping
TEAM_STATE = {
    "Collingwood": "VIC", "Carlton": "VIC", "Melbourne": "VIC",
    "Richmond": "VIC", "Hawthorn": "VIC", "Essendon": "VIC",
    "Western Bulldogs": "VIC", "St Kilda": "VIC", "North Melbourne": "VIC",
    "Geelong": "VIC",
    "Brisbane Lions": "QLD", "Gold Coast": "QLD",
    "Adelaide": "SA", "Port Adelaide": "SA",
    "Fremantle": "WA", "West Coast": "WA",
    "Sydney": "NSW", "GWS Giants": "NSW",
}


def margin_of_victory_multiplier(margin, winner_elo, loser_elo):
    """
    Calculates how much to scale the Elo update based on margin.

    Formula adapted from FiveThirtyEight's NFL model:
      G = log(|margin| + 1) × 2.2 / (elo_diff × 0.001 + 2.2)

    This does two things:
      1. log(margin + 1) gives diminishing returns:
         winning by 10 is worth more than winning by 1,
         but winning by 50 is NOT proportionally more than winning by 40.
      2. The denominator suppresses the multiplier when the favourite wins
         (less informative) and boosts it when the underdog wins (more informative).

    Result is capped between 1.0 and 2.5 to prevent extreme swings.

    Why this matters: a team winning by 80 points should update ratings MORE
    than a team squeaking through by 1. Without this, all wins look the same.
    """
    if margin <= 0:
        margin = 1  # Safety — should never be called with 0 or negative

    # Elo difference from the winner's perspective (positive = favourite won)
    elo_diff = max(winner_elo - loser_elo, -400)

    # FiveThirtyEight-derived formula (using natural log)
    numerator   = np.log(abs(margin) + 1) * 2.2
    denominator = (elo_diff * 0.001 + 2.2)
    multiplier  = numerator / denominator

    # Clamp to reasonable range
    return float(np.clip(multiplier, 1.0, 2.5))


def weighted_recent_form(team, results_df, before_index, n=5):
    """
    Calculates exponentially-weighted recent form for a team.

    'Exponentially weighted' means the most recent game counts most,
    with older games contributing less. Specifically:
      Game 1 (most recent) weight: 1.0
      Game 2: 0.8
      Game 3: 0.64
      Game 4: 0.51
      Game 5: 0.41

    Returns dict with:
      win_rate:      weighted fraction of games won (0–1)
      avg_score:     weighted average points scored
      avg_against:   weighted average points conceded
      avg_margin:    weighted average point margin (can be negative)
      consistency:   1 / (1 + std_dev_of_margins) — higher = more consistent
    """
    decay = 0.8  # Each game back is worth 80% of the previous one

    # Get all games involving this team before this index
    home_games = results_df[
        (results_df['hteam'] == team) &
        (results_df.index < before_index)
    ].tail(n).copy()

    away_games = results_df[
        (results_df['ateam'] == team) &
        (results_df.index < before_index)
    ].tail(n).copy()

    # Build unified game records
    records = []
    for _, g in home_games.iterrows():
        records.append({
            'won':    int(g['hscore'] > g['ascore']),
            'score':  g['hscore'],
            'against':g['ascore'],
            'margin': g['hscore'] - g['ascore'],
            'idx':    g.name,
        })
    for _, g in away_games.iterrows():
        records.append({
            'won':    int(g['ascore'] > g['hscore']),
            'score':  g['ascore'],
            'against':g['hscore'],
            'margin': g['ascore'] - g['hscore'],
            'idx':    g.name,
        })

    # Sort by index (chronological), take last N
    records.sort(key=lambda x: x['idx'])
    records = records[-n:]

    if not records:
        return {
            'win_rate': 0.5, 'avg_score': 85.0,
            'avg_against': 85.0, 'avg_margin': 0.0,
            'consistency': 0.5, 'n_games': 0
        }

    # Apply exponential decay weights (most recent = highest weight)
    # records[-1] is most recent, records[0] is oldest
    n_rec = len(records)
    weights = [decay ** (n_rec - 1 - i) for i in range(n_rec)]
    total_w = sum(weights)

    w_wins    = sum(records[i]['won']     * weights[i] for i in range(n_rec)) / total_w
    w_score   = sum(records[i]['score']   * weights[i] for i in range(n_rec)) / total_w
    w_against = sum(records[i]['against'] * weights[i] for i in range(n_rec)) / total_w
    w_margin  = sum(records[i]['margin']  * weights[i] for i in range(n_rec)) / total_w

    # Consistency: inverse of standard deviation — higher = more predictable
    margins   = [r['margin'] for r in records]
    std_margin = np.std(margins) if len(margins) > 1 else 20.0
    consistency = 1.0 / (1.0 + std_margin / 30.0)  # Normalised to 0–1 range

    return {
        'win_rate':    round(w_wins, 4),
        'avg_score':   round(w_score, 2),
        'avg_against': round(w_against, 2),
        'avg_margin':  round(w_margin, 2),
        'consistency': round(consistency, 4),
        'n_games':     n_rec,
    }


def days_rest(team, results_df, before_date, before_index):
    """
    Calculates how many days of rest a team has had since their last game.

    Short rest (< 7 days) indicates a fatigue penalty.
    Normal rest (7–14 days) is neutral.
    Long rest (> 14 days) may indicate rust (minor negative).

    Returns:
      days: integer days since last game (999 if no prior game found)
      penalty: Elo adjustment to apply (-ve = penalty, 0 = neutral)
    """
    prior = results_df[
        ((results_df['hteam'] == team) | (results_df['ateam'] == team)) &
        (results_df.index < before_index)
    ].tail(1)

    if prior.empty or 'date' not in prior.columns:
        return 999, 0.0

    try:
        last_date = pd.to_datetime(prior['date'].values[0])
        current   = pd.to_datetime(before_date)
        diff      = (current - last_date).days
    except Exception:
        return 999, 0.0

    if diff < 6:
        penalty = -12.0    # Very short rest: big fatigue hit
    elif diff < 8:
        penalty = -5.0     # Short rest: minor hit
    elif diff > 21:
        penalty = -3.0     # Rust from long break
    else:
        penalty = 0.0      # Normal rest

    return diff, penalty


def travel_penalty(team, venue_name):
    """
    Estimates Elo penalty for interstate travel.

    Flying interstate disrupts preparation and sleep.
    Cross-country travel (e.g. VIC → WA) is penalised more.

    Returns Elo penalty (negative number or 0).
    """
    team_state  = TEAM_STATE.get(team, "VIC")
    venue_state = VENUE_STATE.get(venue_name, "VIC")

    if team_state == venue_state:
        return 0.0  # No travel

    # Cross-country: WA ↔ east coast
    cross_country = {
        ("WA", "VIC"), ("WA", "NSW"), ("WA", "QLD"), ("WA", "SA"), ("WA", "ACT"),
        ("VIC", "WA"), ("NSW", "WA"), ("QLD", "WA"), ("SA", "WA"), ("ACT", "WA"),
    }

    if (team_state, venue_state) in cross_country:
        return -10.0   # Cross-country flight: significant penalty

    return -5.0        # Interstate but not cross-country


def build_feature_matrix(results_df, elo_history_df):
    """
    Builds the complete feature matrix for ML training.

    Each row = one game (after the first 50, so teams have history).
    Each column = one feature.
    Last column = target (home_win: 1 or 0).

    This is called once during training and cached.
    """
    results_df = results_df.reset_index(drop=True)
    rows = []

    for idx, game in results_df.iterrows():
        if idx < 50:
            continue  # Skip early games — not enough history

        home, away = game['hteam'], game['ateam']

        # Pre-game Elo from history
        pre_h = elo_history_df['pre_home_elo'].iloc[idx] if idx < len(elo_history_df) else 1500.0
        pre_a = elo_history_df['pre_away_elo'].iloc[idx] if idx < len(elo_history_df) else 1500.0
        elo_diff = pre_h - pre_a

        # Recent form (weighted)
        hf = weighted_recent_form(home, results_df, idx)
        af = weighted_recent_form(away, results_df, idx)

        # Scoring differential: how well each team attacks vs defends
        # Positive = scores more than they concede on average
        h_scoring_diff = hf['avg_score'] - hf['avg_against']
        a_scoring_diff = af['avg_score'] - af['avg_against']

        rows.append({
            # Primary Elo feature
            'elo_diff':          elo_diff,
            # Form features (differential = home minus away)
            'form_win_diff':     hf['win_rate']    - af['win_rate'],
            'form_margin_diff':  hf['avg_margin']  - af['avg_margin'],
            'scoring_diff':      h_scoring_diff    - a_scoring_diff,
            'h_consistency':     hf['consistency'],
            'a_consistency':     af['consistency'],
            # Raw form values
            'h_win_rate':        hf['win_rate'],
            'a_win_rate':        af['win_rate'],
            'h_avg_margin':      hf['avg_margin'],
            'a_avg_margin':      af['avg_margin'],
            # Target
            'home_win':          game['home_win'],
        })

    return pd.DataFrame(rows)
