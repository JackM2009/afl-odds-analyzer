# decision_engine.py
# ═══════════════════════════════════════════════════════════
# MODULE: Decision Engine + Explanation Output
#
# Responsibility: Takes ALL model outputs and produces:
#   1. A clear BET / NO BET decision
#   2. A complete, data-driven written explanation
#   3. Backtesting against historical data
#
# BET CLASSIFICATION:
#   STRONG BET:  edge > 8%  AND model_prob > 0.57
#   BET:         edge > 4%  AND model_prob > 0.54
#   SMALL BET:   edge > 2%  AND model_prob > 0.52
#   NO BET:      any other case
#   FADE:        edge < -10% (bookmaker strongly disagrees)
#
# WHY THESE THRESHOLDS?
#   - Average bookmaker overround is 4-6% for AFL h2h
#   - You need edge > overround just to break even
#   - Minimum 2% edge provides a buffer for model error
#   - Probability filter (>0.52) avoids near-coinflip games
#     where variance dominates and edge is unreliable
# ═══════════════════════════════════════════════════════════

import numpy as np
import pandas as pd

# Bet classification thresholds
STRONG_BET_EDGE   = 0.08   # 8%+ edge
STRONG_BET_PROB   = 0.57   # 57%+ model probability
BET_EDGE          = 0.04   # 4%+ edge
BET_PROB          = 0.54   # 54%+ model probability
SMALL_BET_EDGE    = 0.02   # 2%+ edge
SMALL_BET_PROB    = 0.52   # 52%+ model probability


def classify_bet(model_prob, edge):
    """
    Returns (classification, label, colour_code) for a given prob/edge.

    Colour codes (for Streamlit):
      'green-strong' = dark green = strong bet
      'green'        = green      = bet
      'yellow'       = yellow     = small bet
      'red'          = red        = no bet
      'grey'         = grey       = fade (bet the other side)
    """
    if edge >= STRONG_BET_EDGE and model_prob >= STRONG_BET_PROB:
        return "STRONG_BET", "🔥 STRONG BET", "green-strong"
    elif edge >= BET_EDGE and model_prob >= BET_PROB:
        return "BET",        "✅ BET",         "green"
    elif edge >= SMALL_BET_EDGE and model_prob >= SMALL_BET_PROB:
        return "SMALL_BET",  "🟡 SMALL BET",   "yellow"
    elif edge < -0.10:
        return "FADE",       "🔄 FADE (other side)", "grey"
    else:
        return "NO_BET",     "❌ NO BET",       "red"


def generate_explanation(
    home_team, away_team,
    prob_breakdown,   # from probability_engine.calculate_true_probability()
    odds_analysis,    # from odds_comparison.full_odds_analysis()
    injury_notes,     # list of strings describing injuries
    weather_data,     # dict from weather.py
    stake,
):
    """
    Generates a full, structured explanation of the prediction.

    This function converts raw numbers into a human-readable
    breakdown that shows EXACTLY what drove the prediction.

    Every number in the explanation comes from real model
    components — nothing is fabricated or estimated.
    """
    p = prob_breakdown
    o = odds_analysis

    home_prob = p['final_prob']
    away_prob = 1.0 - home_prob

    # Pick which team we're potentially betting on
    if o['home_edge'] > o['away_edge'] and o['home_edge'] > 0:
        bet_team  = home_team
        bet_edge  = o['home_edge']
        bet_prob  = home_prob
        bet_odds  = o['best_home_odds']
        bet_bm    = o['best_home_bm']
        bet_kelly = o['home_kelly']
        bet_ev    = o['home_ev']
    elif o['away_edge'] > 0:
        bet_team  = away_team
        bet_edge  = o['away_edge']
        bet_prob  = away_prob
        bet_odds  = o['best_away_odds']
        bet_bm    = o['best_away_bm']
        bet_kelly = o['away_kelly']
        bet_ev    = o['away_ev']
    else:
        bet_team  = None
        bet_edge  = max(o['home_edge'], o['away_edge'])
        bet_prob  = home_prob
        bet_odds  = o['best_home_odds']
        bet_bm    = o['best_home_bm']
        bet_kelly = 0.0
        bet_ev    = min(o['home_ev'], o['away_ev'])

    classification, label, colour = classify_bet(
        bet_prob if bet_team else 0.5,
        bet_edge if bet_team else -1
    )

    lines = []

    lines.append(f"{'═'*52}")
    lines.append(f"  {home_team.upper()} vs {away_team.upper()}")
    lines.append(f"{'═'*52}")
    lines.append("")

    # ── Model Probabilities ───────────────────────────────
    lines.append("📊 MODEL PROBABILITIES")
    lines.append(f"  {home_team:22s}: {home_prob*100:.1f}%")
    lines.append(f"  {away_team:22s}: {away_prob*100:.1f}%")
    lines.append("")

    # ── Probability Breakdown ─────────────────────────────
    lines.append("🔬 PROBABILITY BREAKDOWN (home team)")
    lines.append(f"  Base (50/50 starting point):   50.0%")
    elo_contrib = p['elo_component']
    lines.append(f"  Elo rating advantage:          {elo_contrib:+.1f}pp")
    lines.append(f"    Home Elo: {p['home_elo']:.0f} | Away Elo: {p['away_elo']:.0f}")
    lines.append(f"    Elo diff: {p['home_elo']-p['away_elo']:+.0f} pts → {p['elo_prob']*100:.1f}% via Elo")

    form_contrib = p['form_component']
    if abs(form_contrib) > 0.5:
        lines.append(f"  Recent form adjustment:        {form_contrib:+.1f}pp")
        hf = p['home_form']
        af = p['away_form']
        lines.append(f"    {home_team} (last {hf['n_games']}): {hf['win_rate']*100:.0f}% wins, avg margin {hf['avg_margin']:+.1f}")
        lines.append(f"    {away_team} (last {af['n_games']}): {af['win_rate']*100:.0f}% wins, avg margin {af['avg_margin']:+.1f}")
    else:
        lines.append(f"  Recent form:                   {form_contrib:+.1f}pp (negligible)")

    if abs(p['injury_component']) > 0.2:
        lines.append(f"  Injury adjustment:             {p['injury_component']:+.1f}pp")
        for note in (injury_notes or []):
            lines.append(f"    • {note}")

    if abs(p['rest_component']) > 0.2:
        lines.append(f"  Rest/fatigue:                  {p['rest_component']:+.1f}pp")

    if abs(p['travel_component']) > 0.2:
        lines.append(f"  Travel penalty:                {p['travel_component']:+.1f}pp")

    # Weather
    if weather_data and not weather_data.get('is_indoor', False):
        cond = weather_data.get('condition', 'Fine')
        impact = weather_data.get('elo_adjustment', 0)
        if impact >= 5:
            lines.append(f"  Weather ({cond}): suppression applied")
    lines.append("")

    # ── Market Section ────────────────────────────────────
    lines.append("💰 MARKET ODDS")
    lines.append(f"  True market prob (overround removed):")
    lines.append(f"    {home_team}: {o['market_home_prob']*100:.1f}%")
    lines.append(f"    {away_team}: {o['market_away_prob']*100:.1f}%")
    lines.append(f"  Avg bookmaker overround: {o['avg_overround']:.1f}%")
    lines.append("")
    lines.append(f"  Fair odds (our model):")
    lines.append(f"    {home_team}: {o['fair_home_odds']}")
    lines.append(f"    {away_team}: {o['fair_away_odds']}")
    lines.append("")
    lines.append(f"  Best available odds:")
    lines.append(f"    {home_team}: {o['best_home_odds']} at {o['best_home_bm']}")
    lines.append(f"    {away_team}: {o['best_away_odds']} at {o['best_away_bm']}")
    lines.append("")

    # ── Edge Section ──────────────────────────────────────
    lines.append("📈 VALUE ANALYSIS")
    h_edge_pct = o['home_edge'] * 100
    a_edge_pct = o['away_edge'] * 100
    lines.append(f"  {home_team} edge: {h_edge_pct:+.2f}%")
    lines.append(f"  {away_team} edge: {a_edge_pct:+.2f}%")

    model_mkt_diff = (p['final_prob'] - o['market_home_prob']) * 100
    lines.append(f"  Model vs market (home): {model_mkt_diff:+.1f}pp")
    if abs(model_mkt_diff) > 5:
        direction = "more confident in home" if model_mkt_diff > 0 else "more confident in away"
        lines.append(f"    → We are significantly {direction} than the market")
    lines.append("")

    # Bias flags
    if o['bias_flags']:
        lines.append("🔍 MARKET ANALYSIS FLAGS")
        for flag in o['bias_flags']:
            lines.append(f"  {flag}")
        lines.append("")

    # ── Decision ──────────────────────────────────────────
    lines.append("⚡ DECISION")
    lines.append(f"  {label}")

    if bet_team and classification != "NO_BET":
        lines.append(f"  Team to bet: {bet_team}")
        lines.append(f"  Best odds: {bet_odds} at {bet_bm}")
        lines.append(f"  Edge: {bet_edge*100:+.2f}%")
        lines.append(f"  EV on ${stake:.0f}: ${bet_ev:+.2f}")
        bankroll_pct = bet_kelly * 100
        lines.append(f"  Kelly sizing: {bankroll_pct:.1f}% of bankroll")
        lines.append(f"  (= ${stake:.0f} stake if bankroll = ${stake/max(bet_kelly,0.001):.0f})")
    else:
        lines.append(f"  Best edge: {bet_edge*100:+.2f}% — below threshold")
        lines.append(f"  Pass on this game.")

    lines.append(f"{'═'*52}")

    return "\n".join(lines), classification, label, colour


def backtest_model(results_df, elo_history_df, min_edge=0.02, min_prob=0.52):
    """
    Backtests the Elo model against historical AFL results.

    This replays past seasons, making "bets" wherever the model
    had sufficient edge over the historical bookmaker-implied odds,
    then measures actual returns.

    Since we don't have historical bookmaker odds in the Squiggle data,
    we use the market-implied probability from Elo ratings as a proxy:
    specifically, we simulate betting when our Elo model disagrees
    significantly with a naive fair-odds calculation.

    Returns a dict with:
      total_bets:    how many bets we would have placed
      wins:          how many won
      win_rate:      wins / total_bets
      total_staked:  total money risked (at $10/bet)
      total_return:  total money returned
      roi:           (return - staked) / staked × 100
      by_year:       breakdown per season
      brier_score:   model calibration score (lower = better)
    """
    STAKE = 10.0
    results_df    = results_df.reset_index(drop=True)
    elo_history_df = elo_history_df.reset_index(drop=True)

    bets = []
    brier_terms = []

    for idx, game in results_df.iterrows():
        if idx < 100:
            continue  # Skip early season — not enough data

        if idx >= len(elo_history_df):
            break

        pre_home = elo_history_df.loc[idx, 'pre_home_elo']
        pre_away = elo_history_df.loc[idx, 'pre_away_elo']

        from rating_model import win_probability_from_elo
        model_prob = win_probability_from_elo(pre_home, pre_away)

        # Brier score: measures calibration (are 60% bets winning ~60%?)
        # Lower Brier score = better calibrated model
        outcome = float(game['home_win'])
        brier_terms.append((model_prob - outcome) ** 2)

        # Simulate: if we had bet at "fair" implied odds
        # We bet on home team when model says significantly >50%
        edge = model_prob - 0.5  # Simple proxy for historical edge

        if abs(model_prob - 0.5) >= (min_prob - 0.5):
            # Decide which team to bet
            if model_prob >= min_prob:
                bet_home = True
                bet_prob = model_prob
                # Simulated odds: fair odds with a typical 5% overround
                sim_odds = round((1.0 / model_prob) * 0.95, 3)
                sim_odds = max(1.05, sim_odds)  # Floor
                actual_win = int(game['home_win'])
            else:
                bet_home = False
                bet_prob = 1 - model_prob
                sim_odds = round((1.0 / (1 - model_prob)) * 0.95, 3)
                sim_odds = max(1.05, sim_odds)
                actual_win = int(not game['home_win'])

            sim_edge = calculate_backtest_edge(bet_prob, sim_odds)
            if sim_edge >= min_edge:
                profit = (sim_odds - 1) * STAKE if actual_win else -STAKE
                bets.append({
                    'year':       game['year'],
                    'home':       game['hteam'],
                    'away':       game['ateam'],
                    'bet_team':   game['hteam'] if bet_home else game['ateam'],
                    'model_prob': round(bet_prob, 4),
                    'sim_odds':   sim_odds,
                    'edge':       round(sim_edge, 4),
                    'staked':     STAKE,
                    'profit':     round(profit, 2),
                    'won':        int(actual_win),
                })

    if not bets:
        return {
            'total_bets': 0, 'wins': 0, 'win_rate': 0,
            'total_staked': 0, 'total_return': 0, 'roi': 0,
            'by_year': {}, 'brier_score': 0.25,
            'bets_df': pd.DataFrame(),
        }

    bets_df      = pd.DataFrame(bets)
    total_bets   = len(bets_df)
    wins         = bets_df['won'].sum()
    total_staked = bets_df['staked'].sum()
    total_profit = bets_df['profit'].sum()
    total_return = total_staked + total_profit
    roi          = (total_profit / total_staked) * 100

    by_year = {}
    for year, grp in bets_df.groupby('year'):
        by_year[int(year)] = {
            'bets':   len(grp),
            'wins':   int(grp['won'].sum()),
            'profit': round(grp['profit'].sum(), 2),
            'roi':    round((grp['profit'].sum() / grp['staked'].sum()) * 100, 2),
        }

    brier = round(np.mean(brier_terms), 4) if brier_terms else 0.25

    return {
        'total_bets':   total_bets,
        'wins':         int(wins),
        'win_rate':     round(wins / total_bets, 4),
        'total_staked': round(total_staked, 2),
        'total_return': round(total_return, 2),
        'roi':          round(roi, 2),
        'by_year':      by_year,
        'brier_score':  brier,
        'bets_df':      bets_df,
    }


def calculate_backtest_edge(model_prob, sim_odds):
    """Helper for backtesting edge calculation."""
    return (model_prob * sim_odds) - 1.0
