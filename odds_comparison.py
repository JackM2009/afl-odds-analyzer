# odds_comparison.py
# ═══════════════════════════════════════════════════════════
# MODULE: Odds Comparison
#
# Responsibility: Process raw bookmaker odds and compare them
# against the model's true probability.
#
# THE OVERROUND PROBLEM (why this module exists):
#
#   Bookmakers don't offer fair odds. If the true probability
#   of a home win is 60%, a fair odds would be 1/0.6 = 1.667.
#   But the bookmaker might offer 1.55 — implying 64.5%.
#
#   If you add up the implied probabilities of both sides:
#     Home: 1/1.55 = 64.5%
#     Away: 1/2.50 = 40.0%
#     Total: 104.5%
#
#   That extra 4.5% is the OVERROUND (or "vig" / "juice").
#   It's the bookmaker's guaranteed profit.
#
#   To get the TRUE MARKET PROBABILITY (what the market
#   actually believes), we must remove this margin:
#
#     true_home = (1/home_odds) / ((1/home_odds) + (1/away_odds))
#     true_away = (1/away_odds) / ((1/home_odds) + (1/away_odds))
#
#   Now they sum to 100% exactly.
#
# BOOKMAKER BIAS DETECTION:
#   Bookmakers systematically bias odds toward:
#   • Popular teams (more money bet on them = tighter odds)
#   • Recent form over-reaction
#   • Home favourites
#   This module detects these patterns and flags them.
# ═══════════════════════════════════════════════════════════

import numpy as np


def remove_overround(home_odds, away_odds):
    """
    Removes bookmaker margin from a two-way market.

    Returns the 'true' market-implied probabilities that sum to 1.0.

    Step-by-step:
      1. Convert odds to raw implied probabilities
         home_implied = 1 / home_odds
         away_implied = 1 / away_odds
      2. Sum them (this will be > 1.0 — that's the overround)
      3. Normalise by dividing each by the sum

    Example:
      Home odds: 1.55 → implied 64.5%
      Away odds: 2.50 → implied 40.0%
      Total: 104.5% → overround = 4.5%
      True home prob: 64.5% / 104.5% = 61.7%
      True away prob: 40.0% / 104.5% = 38.3%
      Check: 61.7% + 38.3% = 100% ✓
    """
    if home_odds <= 1.0 or away_odds <= 1.0:
        return 0.5, 0.5, 0.0  # Safety fallback

    home_implied = 1.0 / home_odds
    away_implied = 1.0 / away_odds
    total        = home_implied + away_implied

    overround = round((total - 1.0) * 100, 3)  # As a percentage

    true_home = home_implied / total
    true_away = away_implied / total

    return round(true_home, 6), round(true_away, 6), overround


def calculate_edge(model_prob, bookmaker_odds):
    """
    Calculates the betting edge for one team.

    Formula:
      edge = (model_probability × bookmaker_odds) - 1

    Interpretation:
      edge > 0: Bookmaker is paying MORE than fair value.
                You have a mathematical advantage.
      edge = 0: Exactly fair odds. No advantage either way.
      edge < 0: Bookmaker is paying LESS than fair value.
                The bookmaker has the advantage.

    Example:
      model_prob = 0.62 (model says 62% chance)
      bookmaker_odds = 1.85
      edge = (0.62 × 1.85) - 1 = 1.147 - 1 = +0.147 = +14.7%

      This means for every $100 bet, you expect $14.70 PROFIT
      on average over many identical bets.

    NOTE: Edge is NOT guaranteed profit on any single bet.
    It's the LONG-RUN average profit per dollar risked.
    """
    return round((model_prob * bookmaker_odds) - 1.0, 6)


def expected_value(model_prob, bookmaker_odds, stake):
    """
    Calculates Expected Value in dollars for a given stake.

    Formula:
      EV = (P_win × profit) - (P_lose × stake)
         = (P_win × (odds - 1) × stake) - (P_lose × stake)

    Example:
      model_prob = 0.62, odds = 1.85, stake = $50
      Profit if win:  (1.85 - 1) × $50 = $42.50
      EV = (0.62 × $42.50) - (0.38 × $50)
         = $26.35 - $19.00
         = +$7.35

      So on average, this $50 bet earns $7.35 over the long run.
    """
    profit_if_win  = (bookmaker_odds - 1.0) * stake
    loss_if_lose   = stake
    ev             = (model_prob * profit_if_win) - ((1 - model_prob) * loss_if_lose)
    return round(ev, 4)


def kelly_fraction(model_prob, bookmaker_odds, fraction=0.25):
    """
    Kelly Criterion: optimal bet size as fraction of bankroll.

    Full Kelly formula:
      f* = (b × p - q) / b
    where:
      b = net odds (bookmaker_odds - 1)
      p = model probability of winning
      q = probability of losing = 1 - p

    We use FRACTIONAL Kelly (default 25% of full Kelly).
    Why? Full Kelly maximises long-run growth but:
    - Assumes your model is perfectly calibrated (it isn't)
    - Leads to very large bets that cause psychological issues
    - 25% Kelly is the practical standard for sports betting

    Returns: fraction of bankroll to bet (e.g. 0.04 = 4% of bankroll)
    Returns 0 if no edge (Kelly becomes negative — don't bet).
    """
    b = bookmaker_odds - 1.0
    p = model_prob
    q = 1.0 - p

    if b <= 0:
        return 0.0

    full_kelly = (b * p - q) / b

    if full_kelly <= 0:
        return 0.0  # No edge — don't bet

    return round(full_kelly * fraction, 4)  # Fractional Kelly


def detect_bookmaker_bias(home_team, away_team, model_prob, true_market_prob):
    """
    Detects patterns of systematic bookmaker bias.

    Bookmakers are not perfectly efficient. Known biases:
    1. FAVOURITE BIAS: Bettors love favourites → bookmakers shade
       favourite odds short → value often in underdogs
    2. RECENCY BIAS: After a big win/loss, market overreacts
    3. POPULAR TEAM BIAS: Big clubs (Collingwood, Carlton etc.)
       attract recreational money → bookmakers can afford to
       shade their odds shorter

    Returns a list of detected bias flags (strings).
    """
    biases = []

    diff = model_prob - true_market_prob  # Positive = we're more bullish on home

    # Market has home as heavy favourite but we disagree significantly
    if true_market_prob > 0.70 and model_prob < 0.62:
        biases.append(f"⚠️ Market heavy favourite ({true_market_prob*100:.0f}%) but model disagrees ({model_prob*100:.0f}%) — possible favourite bias")

    # Market has away as heavy favourite but we're closer to 50/50
    if true_market_prob < 0.35 and model_prob > 0.42:
        biases.append(f"🔍 Away team heavily favoured by market but model sees value in home team — check underdog bias")

    # We see significantly more value than the market
    if abs(diff) > 0.10:
        direction = "home" if diff > 0 else "away"
        biases.append(f"📊 Large model/market disagreement ({abs(diff)*100:.1f}%) on {direction} — high conviction signal")

    # Nearly coinflip for market but we have a strong view
    if 0.45 < true_market_prob < 0.55 and abs(model_prob - 0.5) > 0.10:
        biases.append(f"💡 Market sees a 50/50 — model has stronger view ({model_prob*100:.1f}%) — potential inefficiency")

    return biases


def full_odds_analysis(
    home_team, away_team, model_prob,
    bookmaker_odds_list, stake=10.0
):
    """
    Complete analysis of a single match across all bookmakers.

    bookmaker_odds_list: list of dicts like:
      [{'bookmaker': 'Sportsbet', 'home_odds': 1.85, 'away_odds': 2.10}, ...]

    Returns a comprehensive dict with everything needed for
    both the table and the deep-dive explanation.
    """
    if not bookmaker_odds_list:
        return None

    away_prob = 1.0 - model_prob

    # Best odds available across all bookmakers
    best_home_odds = max(bm['home_odds'] for bm in bookmaker_odds_list)
    best_away_odds = max(bm['away_odds'] for bm in bookmaker_odds_list)
    best_home_bm   = next(bm['bookmaker'] for bm in bookmaker_odds_list if bm['home_odds'] == best_home_odds)
    best_away_bm   = next(bm['bookmaker'] for bm in bookmaker_odds_list if bm['away_odds'] == best_away_odds)

    # Average odds (market consensus)
    avg_home_odds  = np.mean([bm['home_odds'] for bm in bookmaker_odds_list])
    avg_away_odds  = np.mean([bm['away_odds'] for bm in bookmaker_odds_list])

    # Remove overround from average odds → true market probability
    true_home_prob, true_away_prob, avg_overround = remove_overround(avg_home_odds, avg_away_odds)

    # Overround for best odds
    _, _, best_overround = remove_overround(best_home_odds, best_away_odds)

    # Fair odds (what the odds SHOULD be given our model)
    fair_home_odds = round(1.0 / model_prob, 3)
    fair_away_odds = round(1.0 / away_prob, 3)

    # Edge against best available odds
    home_edge = calculate_edge(model_prob, best_home_odds)
    away_edge = calculate_edge(away_prob,  best_away_odds)

    # EV
    home_ev = expected_value(model_prob, best_home_odds, stake)
    away_ev  = expected_value(away_prob,  best_away_odds, stake)

    # Kelly sizing
    home_kelly = kelly_fraction(model_prob, best_home_odds)
    away_kelly  = kelly_fraction(away_prob,  best_away_odds)

    # Bias detection
    biases = detect_bookmaker_bias(home_team, away_team, model_prob, true_home_prob)

    # Per-bookmaker analysis
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
        # Probabilities
        'model_home_prob':    round(model_prob, 4),
        'model_away_prob':    round(away_prob, 4),
        'market_home_prob':   round(true_home_prob, 4),
        'market_away_prob':   round(true_away_prob, 4),
        # Fair odds
        'fair_home_odds':     fair_home_odds,
        'fair_away_odds':     fair_away_odds,
        # Best available
        'best_home_odds':     best_home_odds,
        'best_away_odds':     best_away_odds,
        'best_home_bm':       best_home_bm,
        'best_away_bm':       best_away_bm,
        # Averages
        'avg_home_odds':      round(avg_home_odds, 3),
        'avg_away_odds':      round(avg_away_odds, 3),
        'avg_overround':      avg_overround,
        # Edge
        'home_edge':          home_edge,
        'away_edge':          away_edge,
        # EV
        'home_ev':            home_ev,
        'away_ev':            away_ev,
        # Kelly
        'home_kelly':         home_kelly,
        'away_kelly':         away_kelly,
        # Bias flags
        'bias_flags':         biases,
        # Per-bookmaker detail
        'per_bookmaker':      per_bm,
        # Bookmaker count
        'n_bookmakers':       len(bookmaker_odds_list),
    }
