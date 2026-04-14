# odds_comparison.py
# ═══════════════════════════════════════════════════════════
# MODULE: Odds Comparison
#
# Removes bookmaker overround to get true market probabilities,
# calculates edge, EV, Kelly sizing, and detects bias patterns.
#
# THE OVERROUND:
#   Raw implied probs sum to >100%. We normalise:
#     true_home = (1/home_odds) / ((1/home_odds) + (1/away_odds))
#   This strips the bookmaker's guaranteed margin.
#
# EDGE:
#   edge = (model_prob × bookmaker_odds) - 1
#   Positive = bookmaker paying more than fair value = bet.
#   This is the SAME formula used in backtesting for consistency.
# ═══════════════════════════════════════════════════════════

import numpy as np


def remove_overround(home_odds, away_odds):
    """
    Removes bookmaker margin from a two-way market.
    Returns (true_home_prob, true_away_prob, overround_pct).

    Example:
      Home 1.55 → implied 64.5%
      Away 2.50 → implied 40.0%
      Total: 104.5% → overround = 4.5%
      True home: 64.5/104.5 = 61.7%
      True away: 40.0/104.5 = 38.3%
    """
    if home_odds <= 1.0 or away_odds <= 1.0:
        return 0.5, 0.5, 0.0

    home_implied = 1.0 / home_odds
    away_implied = 1.0 / away_odds
    total        = home_implied + away_implied
    overround    = round((total - 1.0) * 100, 3)
    return round(home_implied / total, 6), round(away_implied / total, 6), overround


def calculate_edge(model_prob, bookmaker_odds):
    """
    edge = (model_probability × bookmaker_odds) - 1

    > 0: bookmaker paying more than fair → value bet
    = 0: break-even
    < 0: bookmaker has the edge
    """
    return round((model_prob * bookmaker_odds) - 1.0, 6)


def expected_value(model_prob, bookmaker_odds, stake):
    """
    EV in dollars for a given stake.
    EV = (P_win × profit) - (P_lose × stake)
    """
    profit_if_win = (bookmaker_odds - 1.0) * stake
    ev = (model_prob * profit_if_win) - ((1 - model_prob) * stake)
    return round(ev, 4)


def kelly_fraction(model_prob, bookmaker_odds, fraction=0.25):
    """
    Fractional Kelly criterion (25% of full Kelly).

    Full Kelly: f* = (b×p - q) / b
    where b = net odds, p = win prob, q = lose prob

    25% Kelly is standard practice — full Kelly requires
    perfect model calibration and causes large psychological swings.
    Returns 0 if no edge (negative Kelly = don't bet).
    """
    b = bookmaker_odds - 1.0
    p = model_prob
    q = 1.0 - p
    if b <= 0:
        return 0.0
    full_kelly = (b * p - q) / b
    return round(max(full_kelly, 0.0) * fraction, 4)


def detect_bookmaker_bias(home_team, away_team, model_prob, true_market_prob):
    """
    Flags patterns of systematic bookmaker mispricing.

    Known biases:
    1. Favourite bias: bettors overback favourites → short odds
    2. Recency bias: market overreacts to recent results
    3. Popular team bias: big clubs get more recreational money
    """
    biases = []
    diff = model_prob - true_market_prob

    if true_market_prob > 0.70 and model_prob < 0.62:
        biases.append(
            f"⚠️ Market heavy fav ({true_market_prob*100:.0f}%) but model disagrees "
            f"({model_prob*100:.0f}%) — possible favourite bias"
        )
    if true_market_prob < 0.35 and model_prob > 0.42:
        biases.append(
            f"🔍 Away heavily favoured but model sees value in home — check underdog bias"
        )
    if abs(diff) > 0.10:
        direction = "home" if diff > 0 else "away"
        biases.append(
            f"📊 Large model/market gap ({abs(diff)*100:.1f}%) on {direction} — "
            f"high conviction signal"
        )
    if 0.45 < true_market_prob < 0.55 and abs(model_prob - 0.5) > 0.10:
        biases.append(
            f"💡 Market sees coin flip; model has strong view ({model_prob*100:.1f}%) — "
            f"potential inefficiency"
        )
    return biases


def full_odds_analysis(home_team, away_team, model_prob, bookmaker_odds_list, stake=10.0):
    """
    Complete analysis across all bookmakers for one match.

    bookmaker_odds_list: [{'bookmaker': str, 'home_odds': float, 'away_odds': float}, ...]

    Returns comprehensive dict for both summary table and deep-dive UI.
    """
    if not bookmaker_odds_list:
        return None

    away_prob = 1.0 - model_prob

    best_home_odds = max(bm['home_odds'] for bm in bookmaker_odds_list)
    best_away_odds = max(bm['away_odds'] for bm in bookmaker_odds_list)
    best_home_bm   = next(bm['bookmaker'] for bm in bookmaker_odds_list if bm['home_odds'] == best_home_odds)
    best_away_bm   = next(bm['bookmaker'] for bm in bookmaker_odds_list if bm['away_odds'] == best_away_odds)

    avg_home_odds  = np.mean([bm['home_odds'] for bm in bookmaker_odds_list])
    avg_away_odds  = np.mean([bm['away_odds'] for bm in bookmaker_odds_list])

    true_home_prob, true_away_prob, avg_overround = remove_overround(avg_home_odds, avg_away_odds)

    fair_home_odds = round(1.0 / model_prob, 3)
    fair_away_odds = round(1.0 / away_prob, 3)

    home_edge  = calculate_edge(model_prob, best_home_odds)
    away_edge  = calculate_edge(away_prob,  best_away_odds)
    home_ev    = expected_value(model_prob, best_home_odds, stake)
    away_ev    = expected_value(away_prob,  best_away_odds, stake)
    home_kelly = kelly_fraction(model_prob, best_home_odds)
    away_kelly = kelly_fraction(away_prob,  best_away_odds)

    biases = detect_bookmaker_bias(home_team, away_team, model_prob, true_home_prob)

    per_bm = []
    for bm in bookmaker_odds_list:
        tm_h, tm_a, ovr = remove_overround(bm['home_odds'], bm['away_odds'])
        per_bm.append({
            'bookmaker':        bm['bookmaker'],
            'home_odds':        bm['home_odds'],
            'away_odds':        bm['away_odds'],
            'home_edge':        calculate_edge(model_prob, bm['home_odds']),
            'away_edge':        calculate_edge(away_prob,  bm['away_odds']),
            'home_ev':          expected_value(model_prob, bm['home_odds'], stake),
            'away_ev':          expected_value(away_prob,  bm['away_odds'], stake),
            'market_home_prob': round(tm_h, 4),
            'market_away_prob': round(tm_a, 4),
            'overround':        ovr,
        })

    return {
        'model_home_prob':   round(model_prob, 4),
        'model_away_prob':   round(away_prob, 4),
        'market_home_prob':  round(true_home_prob, 4),
        'market_away_prob':  round(true_away_prob, 4),
        'fair_home_odds':    fair_home_odds,
        'fair_away_odds':    fair_away_odds,
        'best_home_odds':    best_home_odds,
        'best_away_odds':    best_away_odds,
        'best_home_bm':      best_home_bm,
        'best_away_bm':      best_away_bm,
        'avg_home_odds':     round(avg_home_odds, 3),
        'avg_away_odds':     round(avg_away_odds, 3),
        'avg_overround':     avg_overround,
        'home_edge':         home_edge,
        'away_edge':         away_edge,
        'home_ev':           home_ev,
        'away_ev':           away_ev,
        'home_kelly':        home_kelly,
        'away_kelly':        away_kelly,
        'bias_flags':        biases,
        'per_bookmaker':     per_bm,
        'n_bookmakers':      len(bookmaker_odds_list),
    }
