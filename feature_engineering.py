# feature_engineering.py
# ═══════════════════════════════════════════════════════════
# MODULE: Feature Engineering
#
# Builds ALL predictive features. Single source of truth —
# no feature logic lives anywhere else.
#
# TIER 1 — Primary (highest signal)
#   • Elo rating difference (pre-game, margin-adjusted)
#   • Opponent-adjusted scoring (not just raw score)
#
# TIER 2 — Form-based (last 5 games, exponential decay)
#   • Weighted win rate
#   • Weighted average margin
#   • Scoring differential (offence - defence)
#   • Score consistency
#
# TIER 3 — Advanced stats (where available from Squiggle)
#   • Inside 50 differential
#   • Clearance differential
#   • Disposal differential (kicks + handballs)
#   • Contested possession proxy (tackles + hitouts)
#
# TIER 4 — Situational
#   • Days rest (fatigue/rust)
#   • Interstate travel
#   • Home/away indicator
#
# DATA INTEGRITY: All features are computed using only data
# that was available BEFORE the game being predicted.
# No future information leaks into training.
# ═══════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
from datetime import datetime

# ── Venue → State mapping ─────────────────────────────────
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

# Advanced stat columns available from Squiggle
ADV_STAT_COLS = ['inside50s', 'clearances', 'kicks', 'handballs', 'marks', 'tackles', 'hitouts']


def margin_of_victory_multiplier(margin, winner_elo, loser_elo):
    """
    Scale Elo update by margin — adapted from FiveThirtyEight NFL model.

    G = log(|margin| + 1) × 2.2 / (elo_diff × 0.001 + 2.2)

    Gives diminishing returns (blowing out by 80 ≠ 2× the signal of 40).
    Suppresses update when favourite wins easily (less informative).
    Capped [1.0, 2.5] to prevent extreme swings.
    """
    if margin <= 0:
        margin = 1
    elo_diff   = max(winner_elo - loser_elo, -400)
    numerator  = np.log(abs(margin) + 1) * 2.2
    denominator = (elo_diff * 0.001 + 2.2)
    return float(np.clip(numerator / denominator, 1.0, 2.5))


def weighted_recent_form(team, results_df, before_index, n=5):
    """
    Exponentially-weighted form over last N games.

    Decay = 0.8: most recent game weight = 1.0, prev = 0.8, etc.
    Returns win_rate, avg_score, avg_against, avg_margin, consistency.
    """
    decay = 0.8

    home_games = results_df[
        (results_df['hteam'] == team) & (results_df.index < before_index)
    ].tail(n).copy()
    away_games = results_df[
        (results_df['ateam'] == team) & (results_df.index < before_index)
    ].tail(n).copy()

    records = []
    for _, g in home_games.iterrows():
        records.append({
            'won':    int(g['hscore'] > g['ascore']),
            'score':  g['hscore'], 'against': g['ascore'],
            'margin': g['hscore'] - g['ascore'], 'idx': g.name,
        })
    for _, g in away_games.iterrows():
        records.append({
            'won':    int(g['ascore'] > g['hscore']),
            'score':  g['ascore'], 'against': g['hscore'],
            'margin': g['ascore'] - g['hscore'], 'idx': g.name,
        })

    records.sort(key=lambda x: x['idx'])
    records = records[-n:]

    if not records:
        return {
            'win_rate': 0.5, 'avg_score': 85.0,
            'avg_against': 85.0, 'avg_margin': 0.0,
            'consistency': 0.5, 'n_games': 0
        }

    n_rec   = len(records)
    weights = [decay ** (n_rec - 1 - i) for i in range(n_rec)]
    total_w = sum(weights)

    w_wins    = sum(records[i]['won']     * weights[i] for i in range(n_rec)) / total_w
    w_score   = sum(records[i]['score']   * weights[i] for i in range(n_rec)) / total_w
    w_against = sum(records[i]['against'] * weights[i] for i in range(n_rec)) / total_w
    w_margin  = sum(records[i]['margin']  * weights[i] for i in range(n_rec)) / total_w

    margins    = [r['margin'] for r in records]
    std_margin = np.std(margins) if len(margins) > 1 else 20.0
    consistency = 1.0 / (1.0 + std_margin / 30.0)

    return {
        'win_rate':    round(w_wins, 4),
        'avg_score':   round(w_score, 2),
        'avg_against': round(w_against, 2),
        'avg_margin':  round(w_margin, 2),
        'consistency': round(consistency, 4),
        'n_games':     n_rec,
    }


def weighted_recent_stats(team, stats_df, results_df, before_index, n=5):
    """
    Exponentially-weighted average of advanced stats over last N games.
    Falls back gracefully if stats_df is None.

    Returns dict with per-stat averages for the team.
    """
    if stats_df is None:
        return {}

    # Find games involving this team before the cut-off
    home_stat_rows = stats_df[
        (stats_df['hteam'] == team) &
        (stats_df.index < before_index)
    ].tail(n)

    away_stat_rows = stats_df[
        (stats_df['ateam'] == team) &
        (stats_df.index < before_index)
    ].tail(n)

    records = []
    for _, row in home_stat_rows.iterrows():
        entry = {'idx': row.name}
        for stat in ADV_STAT_COLS:
            col = f'h_{stat}'
            if col in row:
                entry[stat] = row[col]
        records.append(entry)

    for _, row in away_stat_rows.iterrows():
        entry = {'idx': row.name}
        for stat in ADV_STAT_COLS:
            col = f'a_{stat}'
            if col in row:
                entry[stat] = row[col]
        records.append(entry)

    records.sort(key=lambda x: x['idx'])
    records = records[-n:]

    if not records:
        return {}

    decay   = 0.8
    n_rec   = len(records)
    weights = [decay ** (n_rec - 1 - i) for i in range(n_rec)]
    total_w = sum(weights)

    result = {}
    for stat in ADV_STAT_COLS:
        vals = [r.get(stat, np.nan) for r in records]
        valid_vals = [(v, weights[i]) for i, v in enumerate(vals) if pd.notna(v)]
        if valid_vals:
            total_stat_w = sum(w for _, w in valid_vals)
            result[stat] = round(sum(v * w for v, w in valid_vals) / total_stat_w, 2)

    return result


def opponent_adjusted_scoring_avg(team, results_df, elo_ratings, before_index, n=8):
    """
    Calculates opponent-adjusted scoring average.

    Raw scoring average is misleading — a team scoring 110 against
    weak defences isn't as impressive as 95 against top defences.

    Method:
      For each of the last N games:
        adjusted_score = team_score / opponent_defensive_elo_factor
      where defensive_elo_factor = opponent_elo / 1500

    This rewards high scoring against strong defences.
    """
    home_games = results_df[
        (results_df['hteam'] == team) & (results_df.index < before_index)
    ].tail(n).copy()
    away_games = results_df[
        (results_df['ateam'] == team) & (results_df.index < before_index)
    ].tail(n).copy()

    records = []
    for _, g in home_games.iterrows():
        opp = g['ateam']
        opp_elo = elo_ratings.get(opp, 1500.0)
        factor  = opp_elo / 1500.0
        records.append({'score': g['hscore'], 'factor': factor, 'idx': g.name})

    for _, g in away_games.iterrows():
        opp = g['hteam']
        opp_elo = elo_ratings.get(opp, 1500.0)
        factor  = opp_elo / 1500.0
        records.append({'score': g['ascore'], 'factor': factor, 'idx': g.name})

    records.sort(key=lambda x: x['idx'])
    records = records[-n:]

    if not records:
        return 1.0  # Neutral

    adj_scores = [r['score'] / max(r['factor'], 0.5) for r in records]
    return round(np.mean(adj_scores), 2)


def days_rest(team, results_df, before_date, before_index):
    """
    Days since last game → rest/fatigue penalty.

    < 6 days: heavy fatigue (-12 Elo)
    < 8 days: light fatigue (-5 Elo)
    > 21 days: rust (-3 Elo)
    else: neutral (0)
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

    if diff < 6:   penalty = -12.0
    elif diff < 8: penalty = -5.0
    elif diff > 21:penalty = -3.0
    else:          penalty = 0.0

    return diff, penalty


def travel_penalty(team, venue_name):
    """
    Elo penalty for interstate travel.
    Cross-country (e.g. VIC → WA): -10 Elo
    Interstate (not cross-country):  -5 Elo
    Home state: 0
    """
    team_state  = TEAM_STATE.get(team, "VIC")
    venue_state = VENUE_STATE.get(venue_name, "VIC")

    if team_state == venue_state:
        return 0.0

    cross_country = {
        ("WA","VIC"),("WA","NSW"),("WA","QLD"),("WA","SA"),("WA","ACT"),
        ("VIC","WA"),("NSW","WA"),("QLD","WA"),("SA","WA"),("ACT","WA"),
    }
    if (team_state, venue_state) in cross_country:
        return -10.0
    return -5.0


def build_feature_matrix(results_df, elo_history_df, stats_df=None):
    """
    Builds the complete feature matrix for ML training.

    Each row = one game (after first 50 for sufficient history).
    Includes Elo, form, advanced stats (if available), and interactions.

    CRITICAL: All features computed from data BEFORE the game.
    """
    results_df = results_df.reset_index(drop=True)

    # Build a running Elo dict for opponent-adjusted scoring
    # (use pre-game Elo from history, indexed by game)
    elo_by_game = {}
    if elo_history_df is not None:
        for idx, row in elo_history_df.iterrows():
            elo_by_game[idx] = {
                row['home']: row['pre_home_elo'],
                row['away']: row['pre_away_elo'],
            }

    rows = []
    running_elo = {}  # Track ratings as we go for opponent adjustment

    for idx, game in results_df.iterrows():
        if idx < 50:
            # Still warm up the running Elo
            home, away = game['hteam'], game['ateam']
            if home not in running_elo: running_elo[home] = 1500.0
            if away not in running_elo: running_elo[away] = 1500.0
            # Don't update running_elo here — let rating_model handle truth
            continue

        home, away = game['hteam'], game['ateam']

        # Pre-game Elo from history
        pre_h = elo_history_df['pre_home_elo'].iloc[idx] if idx < len(elo_history_df) else 1500.0
        pre_a = elo_history_df['pre_away_elo'].iloc[idx] if idx < len(elo_history_df) else 1500.0
        elo_diff = pre_h - pre_a

        # Running elo for opponent-adjustment (use pre-game snapshot)
        snap_elo = {}
        if idx in elo_by_game:
            snap_elo = elo_by_game[idx]
        else:
            snap_elo = {home: pre_h, away: pre_a}

        # Form features
        hf = weighted_recent_form(home, results_df, idx)
        af = weighted_recent_form(away, results_df, idx)

        # Advanced stats
        hs = weighted_recent_stats(home, stats_df, results_df, idx) if stats_df is not None else {}
        as_ = weighted_recent_stats(away, stats_df, results_df, idx) if stats_df is not None else {}

        # Opponent-adjusted scoring
        h_opp_adj = opponent_adjusted_scoring_avg(home, results_df, snap_elo, idx)
        a_opp_adj = opponent_adjusted_scoring_avg(away, results_df, snap_elo, idx)

        # Scoring differential: offence minus defence
        h_scoring_diff = hf['avg_score'] - hf['avg_against']
        a_scoring_diff = af['avg_score'] - af['avg_against']

        row = {
            # Tier 1: Elo
            'elo_diff':            elo_diff,
            # Tier 2: Form
            'form_win_diff':       hf['win_rate']   - af['win_rate'],
            'form_margin_diff':    hf['avg_margin']  - af['avg_margin'],
            'scoring_diff':        h_scoring_diff    - a_scoring_diff,
            'h_consistency':       hf['consistency'],
            'a_consistency':       af['consistency'],
            'h_win_rate':          hf['win_rate'],
            'a_win_rate':          af['win_rate'],
            'h_avg_margin':        hf['avg_margin'],
            'a_avg_margin':        af['avg_margin'],
            # Opponent-adjusted scoring differential
            'opp_adj_score_diff':  h_opp_adj - a_opp_adj,
            'h_opp_adj_score':     h_opp_adj,
            'a_opp_adj_score':     a_opp_adj,
            # Target
            'home_win':            game['home_win'],
        }

        # Tier 3: Advanced stats differential (if available)
        for stat in ADV_STAT_COLS:
            hv = hs.get(stat, np.nan)
            av = as_.get(stat, np.nan)
            if pd.notna(hv) and pd.notna(av):
                row[f'{stat}_diff'] = hv - av
                row[f'h_{stat}']    = hv
                row[f'a_{stat}']    = av

        rows.append(row)

    return pd.DataFrame(rows)
