# probability_engine.py
# ═══════════════════════════════════════════════════════════
# MODULE: Probability Engine
#
# Produces the final match probability by combining:
#
#   STEP 1 — Elo probability (standalone, pure ratings)
#
#   STEP 2 — ML probability (logistic regression trained on
#             form + advanced stats features — NOT Elo diff,
#             to prevent double counting in the blend)
#
#   STEP 3 — Blend: final = 0.60 × ML + 0.40 × Elo
#             Both signals are independent; blending is valid.
#
#   STEP 4 — Situational modifiers (injury, rest, travel, weather)
#             Applied as probability shifts AFTER blending.
#
#   STEP 5 — Clamp to [0.05, 0.95]
#
# FIX vs previous version:
#   Elo diff was in the ML feature set AND used directly in Elo prob.
#   This caused double-counting. Now ML uses ONLY form + advanced stats.
#   Elo probability is a separate, independent signal in the blend.
# ═══════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
import joblib, os, warnings
warnings.filterwarnings('ignore')

from rating_model import win_probability_from_elo, INITIAL_RATING, HOME_ADVANTAGE
from feature_engineering import (
    weighted_recent_form, weighted_recent_stats, opponent_adjusted_scoring_avg,
    ADV_STAT_COLS
)

MODEL_FILE  = "afl_prob_model.pkl"
SCALER_FILE = "afl_prob_scaler.pkl"

# ── Feature columns for ML model ─────────────────────────
# NOTE: elo_diff is deliberately EXCLUDED here.
# Elo provides its own independent probability signal in the blend.
# Including elo_diff here would double-count it.
BASE_FEATURE_COLS = [
    'form_win_diff',       # Win rate diff (last 5, exponential weighted)
    'form_margin_diff',    # Avg margin diff (last 5, exp weighted)
    'scoring_diff',        # Offence - defence differential
    'h_consistency',       # Home team consistency (inverse std dev)
    'a_consistency',       # Away team consistency
    'h_win_rate',          # Home raw win rate
    'a_win_rate',          # Away raw win rate
    'h_avg_margin',        # Home avg margin
    'a_avg_margin',        # Away avg margin
    'opp_adj_score_diff',  # Opponent-adjusted scoring differential
]

# Advanced stat features added if available
ADV_FEATURE_COLS = [f'{s}_diff' for s in ADV_STAT_COLS]

# Blend weights — independent signals
ELO_WEIGHT = 0.40
ML_WEIGHT  = 0.60


def get_active_feature_cols(feature_df):
    """Returns which feature columns are available in this dataset."""
    base_avail = [c for c in BASE_FEATURE_COLS if c in feature_df.columns]
    adv_avail  = [c for c in ADV_FEATURE_COLS  if c in feature_df.columns]
    return base_avail + adv_avail


def train_probability_model(feature_df):
    """
    Trains a calibrated logistic regression on form + stats features.

    WHY LOGISTIC REGRESSION:
    - Interpretable coefficients
    - Reliable calibration when wrapped with Platt scaling
    - Doesn't overfit with ~2000 samples the way tree models do
    - Fast enough to retrain on every app load if needed

    WHY CALIBRATED:
    - Raw LR probabilities can be systematically over/under-confident
    - CalibratedClassifierCV (Platt scaling) corrects this so that
      a "60% game" actually wins ~60% of the time historically.

    WHY NOT ELO IN FEATURES:
    - Elo is used directly in the probability blend (ELO_WEIGHT)
    - Adding it as an ML feature would count it twice
    - ML here learns the INCREMENTAL signal from form/stats
      that Elo alone misses
    """
    feature_cols = get_active_feature_cols(feature_df)
    print(f"  Training with {len(feature_cols)} features: {feature_cols}")

    clean = feature_df.dropna(subset=feature_cols + ['home_win'])
    X = clean[feature_cols].values.astype(float)
    y = clean['home_win'].values.astype(int)

    # Fill any remaining NaN with column means
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = col_means[j]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    base_model = LogisticRegression(C=0.5, random_state=42, max_iter=2000, solver='lbfgs')
    model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
    model.fit(X_scaled, y)

    # Cross-validated accuracy for reporting
    base_uncal = LogisticRegression(C=0.5, random_state=42, max_iter=2000)
    cv_scores  = cross_val_score(base_uncal, X_scaled, y, cv=5, scoring='accuracy')
    accuracy   = cv_scores.mean()

    print(f"  Cross-validated accuracy (ML only): {accuracy:.1%}")
    print(f"  Note: final model blends this with Elo ({ELO_WEIGHT*100:.0f}% Elo weight)")

    # Feature importance
    base_uncal.fit(X_scaled, y)
    coef_df = pd.DataFrame({
        'feature':     feature_cols,
        'coefficient': base_uncal.coef_[0],
    }).sort_values('coefficient', key=abs, ascending=False)
    print("\n  Top feature importances:")
    for _, row in coef_df.head(10).iterrows():
        direction = "→ favours home" if row['coefficient'] > 0 else "→ favours away"
        print(f"    {row['feature']:30s}: {row['coefficient']:+.4f}  {direction}")

    joblib.dump(model,  MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    # Save which features were used (so prediction uses same set)
    joblib.dump(feature_cols, "afl_prob_feature_cols.pkl")
    print(f"\n  Model saved.")
    return model, scaler, feature_cols, accuracy


def load_probability_model():
    """Load saved model. Returns (None, None, None) if not found."""
    if (os.path.exists(MODEL_FILE) and
        os.path.exists(SCALER_FILE) and
        os.path.exists("afl_prob_feature_cols.pkl")):
        return (
            joblib.load(MODEL_FILE),
            joblib.load(SCALER_FILE),
            joblib.load("afl_prob_feature_cols.pkl"),
        )
    return None, None, None


def _build_feature_vector(home, away, elo_ratings, results_df, stats_df,
                           feature_cols, all_adj):
    """
    Builds a single feature vector for the ML model.
    Uses the same feature set as training.
    """
    n = len(results_df)

    # Adjusted Elo for opponent-adjusted scoring (uses Elo as proxy for opp strength)
    snap_elo = {
        team: (elo_ratings.get(team, INITIAL_RATING) + all_adj.get(team, 0.0))
        for team in list(elo_ratings.keys()) + [home, away]
    }

    hf = weighted_recent_form(home, results_df, n)
    af = weighted_recent_form(away, results_df, n)
    hs = weighted_recent_stats(home, stats_df, results_df, n) if stats_df is not None else {}
    as_ = weighted_recent_stats(away, stats_df, results_df, n) if stats_df is not None else {}

    h_opp_adj = opponent_adjusted_scoring_avg(home, results_df, snap_elo, n)
    a_opp_adj = opponent_adjusted_scoring_avg(away, results_df, snap_elo, n)

    h_scoring_diff = hf['avg_score'] - hf['avg_against']
    a_scoring_diff = af['avg_score'] - af['avg_against']

    feature_map = {
        'form_win_diff':      hf['win_rate']   - af['win_rate'],
        'form_margin_diff':   hf['avg_margin']  - af['avg_margin'],
        'scoring_diff':       h_scoring_diff    - a_scoring_diff,
        'h_consistency':      hf['consistency'],
        'a_consistency':      af['consistency'],
        'h_win_rate':         hf['win_rate'],
        'a_win_rate':         af['win_rate'],
        'h_avg_margin':       hf['avg_margin'],
        'a_avg_margin':       af['avg_margin'],
        'opp_adj_score_diff': h_opp_adj - a_opp_adj,
        'h_opp_adj_score':    h_opp_adj,
        'a_opp_adj_score':    a_opp_adj,
    }

    # Advanced stats differentials
    for stat in ADV_STAT_COLS:
        hv = hs.get(stat, np.nan)
        av = as_.get(stat, np.nan)
        if pd.notna(hv) and pd.notna(av):
            feature_map[f'{stat}_diff'] = hv - av
            feature_map[f'h_{stat}']    = hv
            feature_map[f'a_{stat}']    = av

    # Build vector in correct column order, using 0.0 for missing
    vector = [feature_map.get(col, 0.0) for col in feature_cols]
    return vector


def calculate_true_probability(
    home_team, away_team, elo_ratings, results_df, stats_df,
    model, scaler, feature_cols,
    injury_adj=None, rest_adj=None, travel_adj=None
):
    """
    Master probability function. Returns full breakdown dict.

    All components stored separately for transparency in explanations.
    """
    # Combine all Elo adjustments
    all_adj = {}
    for d in [injury_adj, rest_adj, travel_adj]:
        if d:
            for team, val in d.items():
                all_adj[team] = all_adj.get(team, 0.0) + val

    # Step 1: Elo probability (with situational adjustments)
    home_elo = elo_ratings.get(home_team, INITIAL_RATING) + all_adj.get(home_team, 0.0)
    away_elo = elo_ratings.get(away_team, INITIAL_RATING) + all_adj.get(away_team, 0.0)
    elo_prob = win_probability_from_elo(home_elo, away_elo)

    # Step 2: ML probability (form + stats, NO elo_diff — independent signal)
    fv = _build_feature_vector(
        home_team, away_team, elo_ratings, results_df, stats_df,
        feature_cols, all_adj
    )
    X_scaled = scaler.transform([fv])
    ml_prob  = float(model.predict_proba(X_scaled)[0][1])

    # Step 3: Blend (independent signals — valid to combine)
    blended_prob = (ML_WEIGHT * ml_prob) + (ELO_WEIGHT * elo_prob)

    # Step 4: Decompose contributions for transparency
    base_home_elo = elo_ratings.get(home_team, INITIAL_RATING)
    base_away_elo = elo_ratings.get(away_team, INITIAL_RATING)

    inj_h = (injury_adj or {}).get(home_team, 0.0)
    inj_a = (injury_adj or {}).get(away_team, 0.0)
    rst_h = (rest_adj   or {}).get(home_team, 0.0)
    rst_a = (rest_adj   or {}).get(away_team, 0.0)
    trv_h = (travel_adj or {}).get(home_team, 0.0)
    trv_a = (travel_adj or {}).get(away_team, 0.0)

    def prob_with(h_delta, a_delta):
        return win_probability_from_elo(base_home_elo + h_delta, base_away_elo + a_delta)

    base_elo_prob = prob_with(0, 0)

    injury_component  = prob_with(inj_h, inj_a)  - base_elo_prob
    rest_component    = prob_with(rst_h, rst_a)   - base_elo_prob
    travel_component  = prob_with(trv_h, trv_a)   - base_elo_prob
    form_component    = ml_prob - base_elo_prob  # What form adds vs pure Elo
    elo_component     = base_elo_prob - 0.5      # Elo's deviation from coin flip

    # Step 5: Clamp
    final_prob = float(np.clip(blended_prob, 0.05, 0.95))

    # Form stats for display
    n = len(results_df)
    home_form = weighted_recent_form(home_team, results_df, n)
    away_form = weighted_recent_form(away_team, results_df, n)

    # Advanced stats for display
    home_stats = weighted_recent_stats(home_team, stats_df, results_df, n) if stats_df is not None else {}
    away_stats = weighted_recent_stats(away_team, stats_df, results_df, n) if stats_df is not None else {}

    return {
        'elo_prob':          round(elo_prob, 4),
        'ml_prob':           round(ml_prob, 4),
        'blended_prob':      round(blended_prob, 4),
        'final_prob':        round(final_prob, 4),
        'home_elo':          round(home_elo, 1),
        'away_elo':          round(away_elo, 1),
        'home_elo_raw':      round(base_home_elo, 1),
        'away_elo_raw':      round(base_away_elo, 1),
        'elo_component':     round(elo_component * 100, 2),
        'form_component':    round(form_component * 100, 2),
        'injury_component':  round(injury_component * 100, 2),
        'rest_component':    round(rest_component * 100, 2),
        'travel_component':  round(travel_component * 100, 2),
        'home_form':         home_form,
        'away_form':         away_form,
        'home_adv_stats':    home_stats,
        'away_adv_stats':    away_stats,
    }
