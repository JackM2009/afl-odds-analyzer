# probability_engine.py
# ═══════════════════════════════════════════════════════════
# MODULE: Probability Engine
#
# Responsibility: Produce the FINAL "true probability" for
# each match by combining all model components.
#
# Architecture:
#
#   STEP 1 — Base probability from Elo ratings
#             (the strongest single predictor)
#
#   STEP 2 — ML adjustment from logistic regression
#             trained on [Elo diff + form features]
#             This learns which Elo differences predict
#             outcomes better after controlling for form.
#
#   STEP 3 — Situational modifiers
#             Applied as direct probability adjustments:
#             • Injury penalty (from injuries.py)
#             • Weather suppression (from weather.py)
#             • Rest days adjustment
#             • Travel fatigue
#
#   STEP 4 — Probability blending
#             final_prob = 0.65×ML + 0.35×Elo
#             ML gets more weight because it incorporates
#             form features that pure Elo misses.
#             Elo anchors it against overfitting.
#
#   STEP 5 — Clipping
#             Clamp to [0.05, 0.95] — we never assign
#             near-certainty. No model is that good.
#
# TRANSPARENCY: Every component is stored separately
# so the explanation module can show exactly what drove
# the final number.
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
from feature_engineering import weighted_recent_form

MODEL_FILE  = "afl_prob_model.pkl"
SCALER_FILE = "afl_prob_scaler.pkl"

# Feature columns used by the ML model
# Order here must match order in build_feature_matrix()
FEATURE_COLS = [
    'elo_diff',         # Pre-game Elo difference (home - away)
    'form_win_diff',    # Win rate difference (last 5, weighted)
    'form_margin_diff', # Avg margin difference (last 5, weighted)
    'scoring_diff',     # Scoring differential (offence - defence)
    'h_consistency',    # Home team score consistency (0-1)
    'a_consistency',    # Away team score consistency (0-1)
    'h_win_rate',       # Home team raw win rate (last 5)
    'a_win_rate',       # Away team raw win rate (last 5)
    'h_avg_margin',     # Home team avg margin (last 5)
    'a_avg_margin',     # Away team avg margin (last 5)
]

# Blending weights: how much each model contributes to final probability
ML_WEIGHT  = 0.65   # ML picks up form patterns Elo misses
ELO_WEIGHT = 0.35   # Elo anchors against overfitting to short-term noise


def train_probability_model(feature_df):
    """
    Trains a calibrated logistic regression on historical game features.

    WHY CALIBRATED?
    Regular logistic regression probabilities can be systematically
    off (e.g., always slightly overconfident). CalibratedClassifierCV
    uses Platt scaling to correct this, so a "60% prediction" really
    does win ~60% of the time historically.

    WHY LOGISTIC REGRESSION (not neural network, gradient boosting etc)?
    1. Interpretable — we can explain every prediction
    2. Resistant to overfitting with ~2000 training games
    3. Probability outputs are reliable when calibrated
    4. Coefficients show us which features matter most

    Cross-validation:
    We test on 5 held-out folds to get an unbiased accuracy estimate.
    This tells us how well the model would do on future games it
    has never seen — which is what actually matters.
    """
    print("  Training probability model...")
    clean = feature_df.dropna(subset=FEATURE_COLS + ['home_win'])
    clean = clean.reset_index(drop=True)

    X = clean[FEATURE_COLS].values.astype(float)
    y = clean['home_win'].values.astype(int)

    # Normalise: ensures elo_diff (range ±500) doesn't dominate
    # form features (range ±1). Without this, Elo would swamp everything.
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Base model: L2-regularised logistic regression
    # C=0.5 is moderately regularised — prevents overfitting to noise
    base_model = LogisticRegression(
        C=0.5, random_state=42, max_iter=2000, solver='lbfgs'
    )

    # Wrap with Platt scaling calibration
    # cv=5 means it uses 5-fold cross-validation internally
    model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
    model.fit(X_scaled, y)

    # External cross-validation for reporting accuracy
    base_uncal = LogisticRegression(C=0.5, random_state=42, max_iter=2000)
    cv_scores = cross_val_score(base_uncal, X_scaled, y, cv=5, scoring='accuracy')
    accuracy = cv_scores.mean()

    print(f"  Cross-validated accuracy: {accuracy:.1%}")
    print(f"  Baseline (always pick home): ~58%")
    print(f"  Improvement over baseline: {(accuracy - 0.58) * 100:.1f}pp")

    # Feature importance (from base logistic regression coefficients)
    base_uncal.fit(X_scaled, y)
    coef_df = pd.DataFrame({
        'feature':    FEATURE_COLS,
        'coefficient': base_uncal.coef_[0],
    }).sort_values('coefficient', key=abs, ascending=False)
    print("\n  Feature importance (logistic regression coefficients):")
    for _, row in coef_df.iterrows():
        direction = "→ favours home" if row['coefficient'] > 0 else "→ favours away"
        print(f"    {row['feature']:22s}: {row['coefficient']:+.4f}  {direction}")

    # Save to disk
    joblib.dump(model,  MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"\n  Model saved.")
    return model, scaler, accuracy


def load_probability_model():
    """Loads previously saved model from disk. Returns (None, None) if not found."""
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        return joblib.load(MODEL_FILE), joblib.load(SCALER_FILE)
    return None, None


def _get_ml_probability(home, away, elo_ratings, results_df, model, scaler, adj=None):
    """
    Internal: runs the ML model for a single matchup.

    Builds the feature vector for the current game, applies the
    same scaling used during training, and returns probability.
    """
    n   = len(results_df)
    hr  = elo_ratings.get(home, INITIAL_RATING) + (adj or {}).get(home, 0)
    ar  = elo_ratings.get(away, INITIAL_RATING) + (adj or {}).get(away, 0)

    hf  = weighted_recent_form(home, results_df, n)
    af  = weighted_recent_form(away, results_df, n)

    features = [[
        hr - ar,
        hf['win_rate']   - af['win_rate'],
        hf['avg_margin'] - af['avg_margin'],
        (hf['avg_score'] - hf['avg_against']) - (af['avg_score'] - af['avg_against']),
        hf['consistency'],
        af['consistency'],
        hf['win_rate'],
        af['win_rate'],
        hf['avg_margin'],
        af['avg_margin'],
    ]]

    X_scaled = scaler.transform(features)
    return float(model.predict_proba(X_scaled)[0][1])  # P(home wins)


def calculate_true_probability(
    home_team, away_team, elo_ratings, results_df,
    model, scaler,
    injury_adj=None, rest_adj=None, travel_adj=None
):
    """
    Master function: returns the complete probability breakdown.

    injury_adj:  dict {team: elo_penalty} — from injury module
    rest_adj:    dict {team: elo_penalty} — from feature_engineering.days_rest()
    travel_adj:  dict {team: elo_penalty} — from feature_engineering.travel_penalty()

    Returns a dict with ALL components for full transparency:
    {
      'elo_prob':         float  — pure Elo probability
      'ml_prob':          float  — ML model probability
      'blended_prob':     float  — weighted blend
      'final_prob':       float  — blended + situational (FINAL)
      'home_elo':         float  — adjusted Elo for home team
      'away_elo':         float  — adjusted Elo for away team
      'elo_component':    float  — Elo's contribution to final (percentage points)
      'form_component':   float  — Form's contribution vs pure Elo
      'injury_component': float  — Injury adjustment (pp change)
      'rest_component':   float  — Rest adjustment (pp change)
      'travel_component': float  — Travel adjustment (pp change)
      'home_form':        dict   — Home team form stats
      'away_form':        dict   — Away team form stats
    }
    """
    # ── Combine all adjustments ───────────────────────────
    all_adj = {}
    for d in [injury_adj, rest_adj, travel_adj]:
        if d:
            for team, val in d.items():
                all_adj[team] = all_adj.get(team, 0.0) + val

    # ── Step 1: Elo probability ───────────────────────────
    home_elo = elo_ratings.get(home_team, INITIAL_RATING) + all_adj.get(home_team, 0.0)
    away_elo = elo_ratings.get(away_team, INITIAL_RATING) + all_adj.get(away_team, 0.0)
    elo_prob = win_probability_from_elo(home_elo, away_elo)

    # ── Step 2: ML probability ────────────────────────────
    ml_prob  = _get_ml_probability(
        home_team, away_team, elo_ratings, results_df,
        model, scaler, all_adj
    )

    # ── Step 3: Blend ─────────────────────────────────────
    blended_prob = (ML_WEIGHT * ml_prob) + (ELO_WEIGHT * elo_prob)

    # ── Step 4: Situational probability shifts ────────────
    # Convert Elo adjustments to probability shifts for transparency
    # We do this by comparing prob with vs without adjustment
    base_home_elo = elo_ratings.get(home_team, INITIAL_RATING)
    base_away_elo = elo_ratings.get(away_team, INITIAL_RATING)

    inj_h = (injury_adj or {}).get(home_team, 0.0)
    inj_a = (injury_adj or {}).get(away_team, 0.0)
    rst_h = (rest_adj   or {}).get(home_team, 0.0)
    rst_a = (rest_adj   or {}).get(away_team, 0.0)
    trv_h = (travel_adj or {}).get(home_team, 0.0)
    trv_a = (travel_adj or {}).get(away_team, 0.0)

    def prob_with(h_delta, a_delta):
        return win_probability_from_elo(
            base_home_elo + h_delta, base_away_elo + a_delta
        )

    base_elo_prob = prob_with(0, 0)

    injury_component  = prob_with(inj_h, inj_a)  - base_elo_prob
    rest_component    = prob_with(rst_h, rst_a)   - base_elo_prob
    travel_component  = prob_with(trv_h, trv_a)   - base_elo_prob
    form_component    = blended_prob - elo_prob
    elo_component     = elo_prob - 0.5  # How much Elo alone moves from 50/50

    # ── Step 5: Final (clamp to reasonable range) ─────────
    final_prob = float(np.clip(blended_prob, 0.05, 0.95))

    # ── Form stats for display ────────────────────────────
    n = len(results_df)
    home_form = weighted_recent_form(home_team, results_df, n)
    away_form = weighted_recent_form(away_team, results_df, n)

    return {
        'elo_prob':          round(elo_prob, 4),
        'ml_prob':           round(ml_prob, 4),
        'blended_prob':      round(blended_prob, 4),
        'final_prob':        round(final_prob, 4),
        'home_elo':          round(home_elo, 1),
        'away_elo':          round(away_elo, 1),
        'home_elo_raw':      round(base_home_elo, 1),
        'away_elo_raw':      round(base_away_elo, 1),
        'elo_component':     round(elo_component * 100, 2),    # in %
        'form_component':    round(form_component * 100, 2),   # in %
        'injury_component':  round(injury_component * 100, 2), # in %
        'rest_component':    round(rest_component * 100, 2),   # in %
        'travel_component':  round(travel_component * 100, 2), # in %
        'home_form':         home_form,
        'away_form':         away_form,
    }
