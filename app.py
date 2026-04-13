# app.py — AFL Odds Analyzer
# The complete Streamlit app. Run this and everything else loads automatically.
# Streamlit reads this file top to bottom and draws each element on screen.

import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
from datetime import date, timedelta

st.set_page_config(page_title="AFL Odds Analyzer 🏉", page_icon="🏉", layout="wide")

# ── Import our modules ───────────────────────────────────
from data_fetcher import load_data, download_all_data
from elo_model    import calculate_elo_ratings, get_win_probability, expected_prob, HOME_ADVANTAGE
from ml_model     import build_training_data, train_model, load_model, predict_with_ml
from weather      import fetch_weather_for_venue, weather_elo_adjustment, VENUE_COORDS

# ────────────────────────────────────────────────────────
# INJURY DATABASE
# SuperCoach prices used to auto-calculate Elo penalties.
# Price = how much they're worth in the fantasy game.
# Higher price = more important player = bigger penalty when missing.
# ────────────────────────────────────────────────────────
KNOWN_PLAYERS = {
    "Nick Daicos":         {"price": 880_000, "position": "Midfielder"},
    "Patrick Cripps":      {"price": 860_000, "position": "Midfielder"},
    "Marcus Bontempelli":  {"price": 860_000, "position": "Midfielder"},
    "Lachie Neale":        {"price": 840_000, "position": "Midfielder"},
    "Errol Gulden":        {"price": 820_000, "position": "Midfielder"},
    "Sam Walsh":           {"price": 820_000, "position": "Midfielder"},
    "Christian Petracca":  {"price": 830_000, "position": "Midfielder"},
    "Chad Warner":         {"price": 830_000, "position": "Midfielder"},
    "Clayton Oliver":      {"price": 810_000, "position": "Midfielder"},
    "Caleb Serong":        {"price": 810_000, "position": "Midfielder"},
    "Zak Butters":         {"price": 800_000, "position": "Midfielder"},
    "Andrew Brayshaw":     {"price": 800_000, "position": "Midfielder"},
    "Connor Rozee":        {"price": 790_000, "position": "Midfielder"},
    "Isaac Heeney":        {"price": 790_000, "position": "Forward"},
    "Max Gawn":            {"price": 790_000, "position": "Ruck"},
    "Charlie Curnow":      {"price": 790_000, "position": "Key Forward"},
    "Josh Kelly":          {"price": 770_000, "position": "Midfielder"},
    "Zach Merrett":        {"price": 780_000, "position": "Midfielder"},
    "Ollie Wines":         {"price": 760_000, "position": "Midfielder"},
    "Patrick Dangerfield": {"price": 750_000, "position": "Midfielder"},
    "Adam Treloar":        {"price": 760_000, "position": "Midfielder"},
    "Touk Miller":         {"price": 790_000, "position": "Midfielder"},
    "Matt Rowell":         {"price": 770_000, "position": "Midfielder"},
    "Jordan Dawson":       {"price": 750_000, "position": "Midfielder"},
    "Rory Laird":          {"price": 760_000, "position": "Defender"},
    "Bailey Smith":        {"price": 740_000, "position": "Midfielder"},
    "Harris Andrews":      {"price": 740_000, "position": "Key Defender"},
    "Darcy Moore":         {"price": 710_000, "position": "Key Defender"},
    "James Sicily":        {"price": 720_000, "position": "Defender"},
    "Jai Newcombe":        {"price": 750_000, "position": "Midfielder"},
    "Jordan De Goey":      {"price": 720_000, "position": "Forward"},
    "Toby Greene":         {"price": 730_000, "position": "Forward"},
    "Jeremy Cameron":      {"price": 740_000, "position": "Key Forward"},
    "Joe Daniher":         {"price": 700_000, "position": "Key Forward"},
    "Harry McKay":         {"price": 720_000, "position": "Key Forward"},
    "Tom Hawkins":         {"price": 640_000, "position": "Key Forward"},
    "Charlie Dixon":       {"price": 680_000, "position": "Key Forward"},
    "Jesse Hogan":         {"price": 620_000, "position": "Key Forward"},
    "Peter Wright":        {"price": 620_000, "position": "Key Forward"},
    "Oscar Allen":         {"price": 680_000, "position": "Key Forward"},
    "Taylor Walker":       {"price": 600_000, "position": "Key Forward"},
    "Mark Blicavs":        {"price": 680_000, "position": "Utility"},
    "Nat Fyfe":            {"price": 640_000, "position": "Midfielder"},
    "Scott Pendlebury":    {"price": 650_000, "position": "Midfielder"},
    "Christian Salem":     {"price": 720_000, "position": "Defender"},
    "Tim Kelly":           {"price": 700_000, "position": "Midfielder"},
    "Rowan Marshall":      {"price": 710_000, "position": "Ruck"},
    "Luke Davies-Uniacke": {"price": 710_000, "position": "Midfielder"},
    "Jason Horne-Francis": {"price": 720_000, "position": "Midfielder"},
    "Jack Sinclair":       {"price": 740_000, "position": "Defender"},
    "Dustin Martin":       {"price": 690_000, "position": "Midfielder"},
    "Tom Lynch":           {"price": 660_000, "position": "Key Forward"},
}

POSITION_MULT = {
    "Midfielder": 1.2, "Mid": 1.2,
    "Key Forward": 1.1, "Forward": 1.0,
    "Key Defender": 1.0, "Defender": 0.95,
    "Ruck": 1.05, "Utility": 0.85, "Unknown": 1.0,
}


def player_elo_penalty(price, position="Unknown"):
    """Convert SuperCoach price + position into an Elo rating penalty."""
    pct  = (max(100_000, min(950_000, price)) - 100_000) / 850_000
    mult = POSITION_MULT.get(position, 1.0)
    return -round(min(pct * 55 * mult, 55), 1)


def lookup_player(name):
    """Find a player in the database (case-insensitive partial match)."""
    nl = name.lower()
    for k, v in KNOWN_PLAYERS.items():
        if nl in k.lower() or k.lower() in nl:
            return v["price"], v["position"]
    return None


# ─────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────
st.title("🏉 AFL Odds Analyzer")
st.markdown("*Find value bets — where bookmakers underestimate a team's real chance of winning*")

# ─────────────────────────────────────────────────────────
# SIDEBAR — DATA MANAGEMENT
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Data Controls")
    st.markdown("""
    **How this works:**
    1. Click **Load Data** the first time
    2. It downloads 10 years of AFL results
    3. Trains the ML model (takes ~2 minutes)
    4. After that, loads instantly from cache

    Click **Refresh Data** each week to get the latest results.
    """)

    load_btn    = st.button("📥 Load / Refresh Data", use_container_width=True, type="primary")
    retrain_btn = st.button("🔄 Force Retrain Model", use_container_width=True)
    st.divider()
    stake = st.number_input("💰 Stake per bet ($)", 1, 1000, 10)
    st.divider()
    st.caption("Data: Squiggle AFL API (free)\nWeather: Open-Meteo (free)\nModel: scikit-learn logistic regression")

# ─────────────────────────────────────────────────────────
# LOAD DATA & TRAIN MODEL
# ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data():
    return load_data()

@st.cache_data(show_spinner=False)
def get_elo(_df):
    return calculate_elo_ratings(_df)

@st.cache_resource(show_spinner=False)
def get_model(_df, _elo_hist):
    m, s = load_model()
    if m is None:
        tr = build_training_data(_df, _elo_hist)
        m, s, _ = train_model(tr)
    return m, s

if load_btn:
    st.cache_data.clear()
    st.cache_resource.clear()
    with st.spinner("Downloading AFL data (this takes about 30 seconds)..."):
        download_all_data()
    st.success("Data downloaded!")
    st.rerun()

if retrain_btn:
    for f in ["afl_model.pkl","afl_scaler.pkl"]:
        if os.path.exists(f): os.remove(f)
    st.cache_resource.clear()
    st.rerun()

results_df = get_data()

if results_df is None:
    st.info("👈 Click **Load / Refresh Data** in the sidebar to get started.")
    st.stop()

with st.spinner("Calculating Elo ratings..."):
    elo_ratings, elo_history = get_elo(results_df)

with st.spinner("Loading ML model (first time takes ~2 minutes)..."):
    model, scaler = get_model(results_df, elo_history)

# ─────────────────────────────────────────────────────────
# ELO LADDER (collapsible)
# ─────────────────────────────────────────────────────────
with st.expander("📊 Current Elo Power Rankings (click to expand)", expanded=False):
    elo_rows = sorted(elo_ratings.items(), key=lambda x: -x[1])
    elo_df = pd.DataFrame([
        {"#": i+1, "Team": t, "Elo": round(r),
         "vs Average": f"{r-1500:+.0f} pts",
         "Form indicator": "🔥 Strong" if r>1560 else ("✅ Good" if r>1510 else ("⚠️ Average" if r>1470 else "❌ Weak"))}
        for i,(t,r) in enumerate(elo_rows)
    ])
    st.dataframe(elo_df, hide_index=True, use_container_width=True)
    st.caption("Ratings calculated automatically from all results since 2015. Higher = stronger team right now.")

st.divider()

# ─────────────────────────────────────────────────────────
# STEP 1: MATCH ENTRY
# ─────────────────────────────────────────────────────────
st.subheader("Step 1 — Enter This Week's Matches & Bookmaker Odds")
st.markdown(
    "For each game: select the teams and venue, then enter the bookmaker's odds "
    "for the **home team** to win. Get these from Sportsbet, TAB, or Ladbrokes — "
    "look for the decimal odds (e.g. 1.85, 2.40) next to the home team's name."
)

all_teams  = sorted(elo_ratings.keys())
all_venues = sorted([v for v in VENUE_COORDS if v != "Unknown"])

DEFAULTS = [
    ("Collingwood",    "Carlton",       "MCG",           1.72),
    ("Brisbane Lions", "GWS Giants",    "Gabba",         2.10),
    ("Melbourne",      "Geelong",       "MCG",           1.90),
    ("Port Adelaide",  "Essendon",      "Adelaide Oval", 1.65),
    ("Sydney",         "Richmond",      "SCG",           1.55),
    ("Western Bulldogs","Fremantle",    "Marvel Stadium",2.00),
    ("Hawthorn",       "Gold Coast",    "MCG",           1.80),
    ("St Kilda",       "North Melbourne","Marvel Stadium",1.45),
    ("West Coast",     "Adelaide",      "Optus Stadium", 2.20),
]

matches_raw = []
for i,(home,away,venue,odds) in enumerate(DEFAULTS):
    c1,c2,c3,c4,c5 = st.columns([2,2,2,1.3,1.3])
    h = c1.selectbox("🏠 Home", all_teams,
                     index=all_teams.index(home) if home in all_teams else 0, key=f"h{i}")
    a = c2.selectbox("✈️ Away", all_teams,
                     index=all_teams.index(away) if away in all_teams else 1, key=f"a{i}")
    v = c3.selectbox("📍 Venue", all_venues,
                     index=all_venues.index(venue) if venue in all_venues else 0, key=f"v{i}")
    o = c4.number_input("Bookie odds", 1.01, 20.0, float(odds), 0.05, key=f"o{i}",
                        help="Decimal odds for HOME team. e.g. 1.85 means bet $1 to win $1.85 total.")
    gd = c5.date_input("Game date", value=date.today()+timedelta(days=3), key=f"d{i}")
    matches_raw.append({"home":h,"away":a,"venue":v,"odds":o,"date":gd})
    if i < len(DEFAULTS)-1:
        st.markdown("---")

st.divider()

# ─────────────────────────────────────────────────────────
# STEP 2: INJURIES
# ─────────────────────────────────────────────────────────
st.subheader("Step 2 — Injury & Team Selection Changes")
st.markdown(
    "If key players are **ruled out** this week, enter them here. "
    "The model will reduce that team's rating accordingly. "
    "Leave this blank if you don't know or there are no major changes."
)

with st.expander("📖 How injury penalties work", expanded=False):
    st.markdown("""
    We use **SuperCoach price** as a measure of player value.
    SuperCoach is the official AFL fantasy game — a $900k player
    is a superstar, a $150k player is a rookie.

    **Finding SuperCoach prices:** Go to **supercoach.com.au** and search the player's name.

    **Rough guide if you don't want to look it up:**
    | Player type | Approx price | Elo penalty |
    |---|---|---|
    | Fringe / bench player | $150–250k | –3 to –8 pts |
    | Regular contributor | $300–450k | –12 to –18 pts |
    | Important player | $500–650k | –20 to –28 pts |
    | Star / All-Australian | $700–800k | –30 to –38 pts |
    | Best player in comp | $850–950k | –42 to –55 pts |
    """)

injury_adj = {}  # {team: total_penalty}

n_inj = st.number_input("How many teams have injury concerns this week?", 0, 18, 0)

for i in range(int(n_inj)):
    with st.container():
        st.markdown(f"**Injury concern #{i+1}:**")
        col_t, col_rest = st.columns([2,4])
        inj_team = col_t.selectbox("Team", all_teams, key=f"it{i}")

        with col_rest:
            n_players = st.number_input(f"Players out for {inj_team}", 1, 8, 1, key=f"np{i}")
            total_pen = 0

            for j in range(int(n_players)):
                pc1,pc2,pc3 = st.columns([2,1,1])
                pname = pc1.text_input(f"Player {j+1} name", key=f"pn{i}{j}",
                                       placeholder="e.g. Nick Daicos")

                # Auto-lookup defaults
                default_price = 400
                default_pos   = "Midfielder"
                if pname:
                    found = lookup_player(pname)
                    if found:
                        default_price = found[0]//1000
                        default_pos   = found[1]
                        pc1.caption("✅ Found in database — price & position auto-filled")

                pprice = pc2.number_input(
                    "SC price ($k)", 100, 950, default_price, 10, key=f"pp{i}{j}",
                    help="SuperCoach price in thousands. e.g. 850 = $850,000"
                ) * 1000
                ppos = pc3.selectbox(
                    "Position", list(POSITION_MULT.keys()),
                    index=list(POSITION_MULT.keys()).index(default_pos)
                          if default_pos in POSITION_MULT else 0,
                    key=f"ppos{i}{j}"
                )

                if pname:
                    pen = player_elo_penalty(pprice, ppos)
                    total_pen += pen
                    pc1.caption(f"Penalty: {pen:+.0f} Elo pts")

            if total_pen:
                injury_adj[inj_team] = total_pen
                st.info(f"**{inj_team}** total injury penalty: **{total_pen:+.0f} Elo points**")
        st.markdown("---")

st.divider()

# ─────────────────────────────────────────────────────────
# STEP 3: RUN ANALYSIS
# ─────────────────────────────────────────────────────────
st.subheader("Step 3 — Run Analysis")
st.markdown(
    "Click the button below. The model will calculate the **real probability** "
    "of each team winning, convert that to **fair odds**, then compare against "
    "the bookmaker's odds to find where **you have the edge**."
)

run = st.button("🔮 Find Value Bets Now", type="primary", use_container_width=True)

if run:
    results_out = []

    # Fetch weather for each unique venue
    wx_cache = {}
    wx_status = st.status("🌤️ Fetching weather forecasts...", expanded=False)
    with wx_status:
        seen = {}
        for m in matches_raw:
            if m["venue"] not in seen:
                seen[m["venue"]] = m["date"]
        for venue, gdate in seen.items():
            st.write(f"  {venue} on {gdate}...")
            wx_cache[venue] = fetch_weather_for_venue(venue, gdate)
    wx_status.update(label="✅ Weather loaded", state="complete")

    # Run predictions
    pred_status = st.status("🤖 Running predictions...", expanded=False)
    with pred_status:
        for m in matches_raw:
            home, away, venue, bookie = m["home"], m["away"], m["venue"], m["odds"]

            # Injury-adjusted Elo ratings
            home_inj = injury_adj.get(home, 0)
            away_inj = injury_adj.get(away, 0)
            h_elo = elo_ratings.get(home, 1500) + home_inj
            a_elo = elo_ratings.get(away, 1500) + away_inj

            # Weather adjustment
            wx = wx_cache.get(venue, {})
            wx_h, wx_a = weather_elo_adjustment(wx, h_elo, a_elo)
            h_elo_final = h_elo + wx_h
            a_elo_final = a_elo + wx_a

            # Combined injury + weather adjustment dict for ML model
            combined_adj = {
                home: home_inj + wx_h,
                away: away_inj + wx_a,
            }

            # ML probability
            ml_prob = predict_with_ml(home, away, elo_ratings,
                                      results_df, model, scaler, combined_adj)

            # Pure Elo probability
            elo_prob = expected_prob(h_elo_final + HOME_ADVANTAGE, a_elo_final)

            # Blend: 70% ML + 30% Elo
            final_prob = round(0.7 * ml_prob + 0.3 * elo_prob, 4)

            # ── The key betting calculations ──────────────────
            # Fair odds = what the odds SHOULD be if bookmaker had no margin
            fair_odds = round(1 / final_prob, 2)

            # Edge = how much better our odds are vs the bookmaker
            # Positive = bookmaker is paying MORE than they should → VALUE BET
            # Negative = bookmaker is paying less → avoid
            edge = round((final_prob * bookie - 1) * 100, 2)

            # Expected Value = average profit per bet over many bets
            # Positive EV = profitable long-term
            ev = round(final_prob * bookie * stake - stake, 2)

            is_value = edge > 0

            # Confidence: how much do ML and Elo agree?
            agreement = 1 - abs(ml_prob - elo_prob)
            conf = "🟢 High" if agreement > 0.95 else ("🟡 Medium" if agreement > 0.90 else "🔴 Low")

            results_out.append({
                "home": home, "away": away, "venue": venue,
                "h_elo": round(h_elo_final), "a_elo": round(a_elo_final),
                "ml_prob": ml_prob, "elo_prob": elo_prob,
                "final_prob": final_prob, "fair_odds": fair_odds,
                "bookie_odds": bookie, "edge": edge, "ev": ev,
                "is_value": is_value, "conf": conf, "wx": wx,
                "home_inj": home_inj, "away_inj": away_inj,
            })
            st.write(f"  ✅ {home} vs {away}: {final_prob*100:.1f}% home, edge {edge:+.1f}%")

    pred_status.update(label="✅ Predictions complete", state="complete")

    st.divider()

    # ─────────────────────────────────────────────────────
    # RESULTS: SUMMARY METRICS
    # ─────────────────────────────────────────────────────
    n_val = sum(1 for r in results_out if r["is_value"])
    best  = max(results_out, key=lambda r: r["edge"])
    total_ev = sum(r["ev"] for r in results_out if r["is_value"])

    st.subheader("📊 Results Summary")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Games analysed",     len(results_out))
    m2.metric("Value bets found",   n_val,
              help="Games where bookmaker odds are higher than our fair odds")
    m3.metric("Best edge",          f"+{best['edge']:.1f}%",
              delta=f"{best['home']} vs {best['away']}")
    m4.metric(f"Total EV (${stake} each)", f"${total_ev:+.2f}",
              help="Combined expected profit if you bet $"+str(stake)+" on every value bet")

    # ─────────────────────────────────────────────────────
    # RESULTS: COLOUR-CODED TABLE
    # ─────────────────────────────────────────────────────
    st.markdown("### Full Results Table")
    st.markdown(
        "🟢 **Green rows** = value bet (bookmaker odds > our fair odds)\n\n"
        "🔴 **Red rows** = no value (bookmaker has the edge — avoid)"
    )

    tbl = []
    for r in results_out:
        away_prob = round((1 - r["final_prob"]) * 100, 1)
        tbl.append({
            "Home Team":     r["home"],
            "Away Team":     r["away"],
            "Venue":         r["venue"],
            "Home Win %":    f"{r['final_prob']*100:.1f}%",
            "Away Win %":    f"{away_prob}%",
            "Fair Odds (H)": r["fair_odds"],
            "Bookie Odds":   r["bookie_odds"],
            "Edge":          f"{r['edge']:+.1f}%",
            f"EV (${stake})":f"${r['ev']:+.2f}",
            "Weather":       r["wx"].get("condition","—"),
            "Confidence":    r["conf"],
            "Verdict":       "✅ BET" if r["is_value"] else "❌ Avoid",
        })

    tdf = pd.DataFrame(tbl)

    def row_colors(row):
        green = "background-color:#d4edda;color:#155724"
        red   = "background-color:#f8d7da;color:#721c24"
        return [green if "BET" in str(row["Verdict"]) else red] * len(row)

    st.dataframe(
        tdf.style.apply(row_colors, axis=1),
        hide_index=True, use_container_width=True
    )

    # ─────────────────────────────────────────────────────
    # RESULTS: HOW TO READ THIS
    # ─────────────────────────────────────────────────────
    with st.expander("📖 How to read this table", expanded=False):
        st.markdown(f"""
        | Column | What it means |
        |---|---|
        | **Home Win %** | Our model's probability of the home team winning |
        | **Fair Odds** | What the odds *should* be at that probability. e.g. 60% chance = 1.67 fair odds |
        | **Bookie Odds** | What the bookmaker is actually offering |
        | **Edge** | How much better the bookie odds are vs our fair odds. **Positive = value.** |
        | **EV (${stake})** | Expected profit per ${stake} bet over the long run |
        | **Verdict** | ✅ BET = value bet. ❌ Avoid = bookmaker has the edge |

        **Example of a value bet:**
        - Our model says: 60% chance home team wins → fair odds = 1.67
        - Bookmaker offers: 1.90
        - Edge = (0.60 × 1.90 − 1) × 100 = **+14%** ← bookmaker is overpaying
        - This is a **value bet** — bet here and you profit long-term

        **Example of no value:**
        - Our model says: 60% chance → fair odds = 1.67
        - Bookmaker offers: 1.45
        - Edge = (0.60 × 1.45 − 1) × 100 = **−13%** ← bookmaker is paying less than fair value
        - **Avoid** — the bookmaker has the mathematical edge here
        """)

    st.divider()

    # ─────────────────────────────────────────────────────
    # RESULTS: PER-GAME DEEP DIVES
    # ─────────────────────────────────────────────────────
    st.markdown("### 🔍 Game-by-Game Breakdown")
    st.caption("Click any game to see exactly what drove the prediction.")

    for r in results_out:
        verdict_icon = "✅ VALUE BET" if r["is_value"] else "❌ No Value"
        label = f"{verdict_icon} — {r['home']} vs {r['away']} @ {r['venue']}"

        with st.expander(label):
            c1,c2,c3 = st.columns(3)

            with c1:
                st.markdown("**📈 Probability Breakdown**")
                st.metric("ML model says",  f"{r['ml_prob']*100:.1f}%",
                          help="Logistic regression trained on 10 years of AFL data")
                st.metric("Elo model says", f"{r['elo_prob']*100:.1f}%",
                          help="Pure Elo rating system (like chess ratings)")
                st.metric("Final (blended)",f"{r['final_prob']*100:.1f}%",
                          help="70% ML + 30% Elo — more stable than either alone")
                st.metric("Model confidence", r["conf"])

            with c2:
                st.markdown("**💰 Betting Value**")
                st.metric("Fair odds",    r["fair_odds"],
                          help="1 ÷ win probability. What odds SHOULD be.")
                st.metric("Bookie offers",r["bookie_odds"])
                diff = round(r["bookie_odds"] - r["fair_odds"], 2)
                st.metric("Odds difference", f"{diff:+.2f}",
                          delta="You're getting overpaid!" if diff>0 else "Bookmaker has edge")
                st.metric("Edge",         f"{r['edge']:+.1f}%")
                st.metric(f"EV on ${stake}",f"${r['ev']:+.2f}")

            with c3:
                st.markdown("**⚙️ Adjustments Applied**")
                st.write(f"**Elo ratings (post-adjustment):**")
                st.write(f"• {r['home']}: {r['h_elo']} pts")
                st.write(f"• {r['away']}: {r['a_elo']} pts")

                if r["home_inj"] or r["away_inj"]:
                    st.markdown("**🏥 Injury adjustments:**")
                    if r["home_inj"]: st.write(f"• {r['home']}: {r['home_inj']:+.0f} pts")
                    if r["away_inj"]: st.write(f"• {r['away']}: {r['away_inj']:+.0f} pts")

                wx = r["wx"]
                st.markdown("**☁️ Weather:**")
                if wx.get("is_indoor"):
                    st.write("• Indoor venue — no adjustment")
                else:
                    st.write(f"• Condition: {wx.get('condition','—')}")
                    st.write(f"• 🌡️ {wx.get('temperature_c')}°C  🌧️ {wx.get('rain_mm')}mm  💨 {wx.get('wind_kmh')} km/h")
                    st.write(f"• {wx.get('impact','—')}")

            # Plain English verdict
            st.divider()
            if r["is_value"]:
                st.success(f"""
**✅ This is a value bet.**

Our model prices {r['home']} at fair odds of **{r['fair_odds']}**.
The bookmaker is offering **{r['bookie_odds']}** — that's {abs(round(r['bookie_odds']-r['fair_odds'],2))} more than fair value.

Edge: **{r['edge']:+.1f}%** — for every ${stake} you bet here, you expect to make **${abs(r['ev']):.2f} profit** on average over time.

This doesn't mean they'll win every time. It means the **price is right** — you're being paid more than the real risk.
                """)
            else:
                st.error(f"""
**❌ No value here.**

Our model prices {r['home']} at fair odds of **{r['fair_odds']}**.
The bookmaker is only offering **{r['bookie_odds']}** — that's LESS than fair value.

Edge: **{r['edge']:+.1f}%** — the bookmaker has the mathematical edge on this one.
Even if {r['home']} wins, you were underpaid for the risk you took. Pass on this.
                """)

st.divider()
st.caption("⚠️ For educational purposes only. Bet responsibly. Past accuracy doesn't guarantee future results.")
