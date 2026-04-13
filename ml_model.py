# ml_model.py
# Trains a logistic regression model on real AFL data.
# Logistic regression finds the best mathematical formula
# to predict P(home win) from Elo + form + scoring features.

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib, os, warnings
warnings.filterwarnings('ignore')

MODEL_FILE  = "afl_model.pkl"
SCALER_FILE = "afl_scaler.pkl"

FEATURE_COLS = [
    'elo_diff',       # Home Elo minus Away Elo
    'win_rate_diff',  # Recent win rate difference (last 5 games)
    'score_diff',     # Average score difference
    'margin_diff',    # Average margin difference
    'h_recent_wins',  # Home team's recent win rate (0–1)
    'a_recent_wins',  # Away team's recent win rate (0–1)
    'h_avg_margin',   # Home team's average recent margin
    'a_avg_margin',   # Away team's average recent margin
]


def _team_stats(results_df, team, before_idx, n=5):
    """Get a team's stats from their last N games before a given index."""
    hg = results_df[(results_df['hteam']==team) & (results_df.index < before_idx)].tail(n)
    ag = results_df[(results_df['ateam']==team) & (results_df.index < before_idx)].tail(n)

    records = []
    for _, g in hg.iterrows():
        records.append({'won': int(g['hscore']>g['ascore']),
                        'score': g['hscore'], 'against': g['ascore'],
                        'margin': g['hscore']-g['ascore']})
    for _, g in ag.iterrows():
        records.append({'won': int(g['ascore']>g['hscore']),
                        'score': g['ascore'], 'against': g['hscore'],
                        'margin': g['ascore']-g['hscore']})

    if not records:
        return 0.5, 80, 80, 0

    recent = sorted(records, key=lambda x: 0)[-n:]
    return (np.mean([r['won']    for r in recent]),
            np.mean([r['score']  for r in recent]),
            np.mean([r['against']for r in recent]),
            np.mean([r['margin'] for r in recent]))


def build_training_data(results_df, elo_history_df):
    """Build a feature table from results + Elo history."""
    print("Building training features (takes ~1 minute)...")
    rows = []
    results_df = results_df.reset_index(drop=True)

    for idx, game in results_df.iterrows():
        hw, hs, hag, hm = _team_stats(results_df, game['hteam'], idx)
        aw, as_, aag, am = _team_stats(results_df, game['ateam'], idx)

        pre_h = elo_history_df['pre_home_elo'].iloc[idx] if idx < len(elo_history_df) else 1500
        pre_a = elo_history_df['pre_away_elo'].iloc[idx] if idx < len(elo_history_df) else 1500

        rows.append({
            'elo_diff':      pre_h - pre_a,
            'win_rate_diff': hw - aw,
            'score_diff':    hs - as_,
            'margin_diff':   hm - am,
            'h_recent_wins': hw,
            'a_recent_wins': aw,
            'h_avg_margin':  hm,
            'a_avg_margin':  am,
            'home_win':      game['home_win'],
        })

    return pd.DataFrame(rows)


def train_model(training_df):
    """Train logistic regression and save to disk."""
    print("Training model...")
    clean = training_df.dropna(subset=FEATURE_COLS+['home_win']).iloc[100:]

    X = clean[FEATURE_COLS].values
    y = clean['home_win'].values

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
    model.fit(X_scaled, y)

    cv = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy').mean()
    print(f"Model accuracy: {cv:.1%} (random guessing ≈ 55% due to home advantage)")

    joblib.dump(model,  MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print("Model saved.")
    return model, scaler, cv


def load_model():
    """Load a previously saved model."""
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        return joblib.load(MODEL_FILE), joblib.load(SCALER_FILE)
    return None, None


def predict_with_ml(home_team, away_team, elo_ratings,
                    results_df, model, scaler, injury_adj=None):
    """
    Predict win probability for home team using the trained model.
    injury_adj: dict of {team: elo_penalty}
    """
    hr = elo_ratings.get(home_team, 1500)
    ar = elo_ratings.get(away_team, 1500)
    if injury_adj:
        hr += injury_adj.get(home_team, 0)
        ar += injury_adj.get(away_team, 0)

    n = len(results_df)
    hw, hs, _, hm = _team_stats(results_df, home_team, n)
    aw, as_, _, am = _team_stats(results_df, away_team, n)

    features = [[
        hr - ar,
        hw - aw,
        hs - as_,
        hm - am,
        hw, aw, hm, am,
    ]]

    prob = model.predict_proba(scaler.transform(features))[0][1]
    return round(float(prob), 4)
