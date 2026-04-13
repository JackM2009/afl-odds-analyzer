# app.py — AFL Odds Analyzer (with live bookmaker odds)
import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
from datetime import date, timedelta, datetime

st.set_page_config(page_title="AFL Odds Analyzer 🏉", page_icon="🏉", layout="wide")

from data_fetcher  import load_data, download_all_data
from elo_model     import calculate_elo_ratings, expected_prob, HOME_ADVANTAGE
from ml_model      import build_training_data, train_model, load_model, predict_with_ml
from weather       import fetch_weather_for_venue, weather_elo_adjustment, VENUE_COORDS
from odds_fetcher  import fetch_afl_odds, match_game_to_model, get_best_bookmaker_for_team

# ── Injury database (same as before) ────────────────────
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
    "Nat Fyfe":            {"price": 640_000, "position": "Midfielder"},
    "Scott Pendlebury":    {"price": 650_000, "position": "Midfielder"},
    "Rowan Marshall":      {"price": 710_000, "position": "Ruck"},
    "Dustin Martin":       {"price": 690_000, "position": "Midfielder"},
    "Tom Lynch":           {"price": 660_000, "position": "Key Forward"},
    "Luke Davies-Uniacke": {"price": 710_000, "position": "Midfielder"},
    "Jason Horne-Francis": {"price": 720_000, "position": "Midfielder"},
    "Jack Sinclair":       {"price": 740_000, "position": "Defender"},
    "Tim Kelly":           {"price": 700_000, "position": "Midfielder"},
    "Oscar Allen":         {"price": 680_000, "position": "Key Forward"},
    "Taylor Walker":       {"price": 600_000, "position": "Key Forward"},
    "Mark Blicavs":        {"price": 680_000, "position": "Utility"},
}

POSITION_MULT = {
    "Midfielder":1.2,"Mid":1.2,"Key Forward":1.1,"Forward":1.0,
    "Key Defender":1.0,"Defender":0.95,"Ruck":1.05,"Utility":0.85,"Unknown":1.0,
}

def player_elo_penalty(price, position="Unknown"):
    pct  = (max(100_000, min(950_000, price)) - 100_000) / 850_000
    return -round(min(pct * 55 * POSITION_MULT.get(position, 1.0), 55), 1)

def lookup_player(name):
    nl = name.lower()
    for k, v in KNOWN_PLAYERS.items():
        if nl in k.lower() or k.lower() in nl:
            return v["price"], v["position"]
    return None

# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.title("🏉 AFL Odds Analyzer")
st.markdown("*Compare our model's predicted odds vs real bookmaker odds — find where you have the edge*")

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    load_btn    = st.button("📥 Load/Refresh AFL Data",   use_container_width=True, type="primary")
    odds_btn    = st.button("💰 Fetch Live Bookmaker Odds", use_container_width=True)
    retrain_btn = st.button("🔄 Retrain ML Model",        use_container_width=True)
    st.divider()
    stake = st.number_input("💰 Stake per bet ($)", 1, 1000, 10)
    st.divider()
    # Show how many API requests are left
    if "api_status" in st.session_state:
        st.caption(st.session_state["api_status"])

# ─────────────────────────────────────────────────────────
# LOAD DATA
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
    st.cache_data.clear(); st.cache_resource.clear()
    with st.spinner("Downloading AFL data..."):
        download_all_data()
    st.rerun()

if retrain_btn:
    for f in ["afl_model.pkl","afl_scaler.pkl"]:
        if os.path.exists(f): os.remove(f)
    st.cache_resource.clear()
    st.rerun()

results_df = get_data()
if results_df is None:
    st.info("👈 Click **Load/Refresh AFL Data** in the sidebar to get started.")
    st.stop()

with st.spinner("Building Elo ratings..."):
    elo_ratings, elo_history = get_elo(results_df)

with st.spinner("Loading ML model..."):
    model, scaler = get_model(results_df, elo_history)

all_teams  = sorted(elo_ratings.keys())
all_venues = sorted([v for v in VENUE_COORDS if v != "Unknown"])

# ─────────────────────────────────────────────────────────
# FETCH LIVE ODDS
# ─────────────────────────────────────────────────────────

# Store fetched odds in session_state so they persist
# without re-fetching on every UI interaction
if "live_odds" not in st.session_state:
    st.session_state["live_odds"] = None

if odds_btn:
    with st.spinner("Fetching live odds from Sportsbet, TAB, Neds, Ladbrokes..."):
        games, status = fetch_afl_odds()
        st.session_state["live_odds"]  = games
        st.session_state["api_status"] = status

live_odds = st.session_state["live_odds"]

# ─────────────────────────────────────────────────────────
# ELO LADDER
# ─────────────────────────────────────────────────────────
with st.expander("📊 Current Elo Power Rankings", expanded=False):
    elo_rows = sorted(elo_ratings.items(), key=lambda x: -x[1])
    elo_df = pd.DataFrame([
        {"#":i+1,"Team":t,"Elo":round(r),"vs Avg":f"{r-1500:+.0f}",
         "Strength":"🔥 Top"if r>1560 else("✅ Good"if r>1510 else("⚠️ Mid"if r>1470 else"❌ Weak"))}
        for i,(t,r) in enumerate(elo_rows)
    ])
    st.dataframe(elo_df, hide_index=True, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────
# LIVE ODDS PANEL
# Shows real bookmaker odds pulled from The Odds API
# ─────────────────────────────────────────────────────────
st.subheader("💰 Live Bookmaker Odds")

if live_odds is None:
    st.info(
        "Click **💰 Fetch Live Bookmaker Odds** in the sidebar to pull real-time odds "
        "from Sportsbet, TAB, Neds, Ladbrokes and more.\n\n"
        "This uses 1 of your 500 free monthly API requests."
    )
elif live_odds == [] or len(live_odds) == 0:
    st.warning(
        "No AFL games are currently listed by bookmakers. "
        "This usually happens between rounds — try again closer to the next round."
    )
else:
    # Show the status message (includes requests remaining)
    if "api_status" in st.session_state:
        st.caption(st.session_state["api_status"])

    # Build a comparison table: our odds vs bookmaker odds
    st.markdown("### Our model vs the market — all games this round")
    st.markdown(
        "**How to read this:** *Our Odds* is what we think the odds should be. "
        "*Best Bookie* is the highest odds any bookmaker is offering. "
        "When *Best Bookie* > *Our Odds*, that team is **underpriced** by the market — potential value."
    )

    comparison_rows = []
    for game in live_odds:
        home = game["home_team"]
        away = game["away_team"]

        # Skip if team not in our model
        if home not in elo_ratings or away not in elo_ratings:
            continue

        # Get our model's probability
        h_elo = elo_ratings.get(home, 1500)
        a_elo = elo_ratings.get(away, 1500)
        our_home_prob = expected_prob(h_elo + HOME_ADVANTAGE, a_elo)
        our_away_prob = 1 - our_home_prob
        our_home_odds = round(1 / our_home_prob, 2)
        our_away_odds = round(1 / our_away_prob, 2)

        # Market consensus probability (already margin-removed in odds_fetcher)
        mkt_home_prob = game["market_home_prob"]
        mkt_away_prob = game["market_away_prob"]

        best_home_bm, best_home_odds = get_best_bookmaker_for_team(game, home, is_home=True)
        best_away_bm, best_away_odds = get_best_bookmaker_for_team(game, away, is_home=False)

        # Edge for home team: are we more confident than the market?
        home_edge = round((our_home_prob * best_home_odds - 1) * 100, 1)
        away_edge = round((our_away_prob * best_away_odds - 1) * 100, 1)

        # Game time
        ct = game.get("commence_time", "")
        try:
            dt  = datetime.fromisoformat(ct.replace("Z", "+00:00"))
            game_time = dt.strftime("%a %d %b, %I:%M %p")
        except:
            game_time = ct[:10] if ct else "TBC"

        comparison_rows.append({
            "Game":              f"{home} vs {away}",
            "Date":              game_time,
            "Our Odds (H)":      our_home_odds,
            "Best Bookie (H)":   best_home_odds,
            "Best at":           best_home_bm,
            "Market Prob (H)":   f"{mkt_home_prob*100:.1f}%",
            "Our Prob (H)":      f"{our_home_prob*100:.1f}%",
            "Home Edge":         f"{home_edge:+.1f}%",
            "Home Verdict":      "✅ VALUE" if home_edge > 0 else "❌ Avoid",
            "Our Odds (A)":      our_away_odds,
            "Best Bookie (A)":   best_away_odds,
            "Away Edge":         f"{away_edge:+.1f}%",
            "Away Verdict":      "✅ VALUE" if away_edge > 0 else "❌ Avoid",
            "Bookie Margin":     f"{game['bookmaker_margin']}%",
            "# Bookmakers":      game["bookmaker_count"],
            "_home_edge":        home_edge,
            "_away_edge":        away_edge,
        })

    if comparison_rows:
        cdf = pd.DataFrame(comparison_rows)
        display_cols = [
            "Game","Date","Our Odds (H)","Best Bookie (H)","Best at",
            "Our Prob (H)","Market Prob (H)","Home Edge","Home Verdict",
            "Bookie Margin","# Bookmakers"
        ]

        def color_comparison(row):
            if "VALUE" in str(row.get("Home Verdict","")):
                return ["background-color:#d4edda;color:#155724"] * len(row)
            return ["background-color:#f8d7da;color:#721c24"] * len(row)

        st.dataframe(
            cdf[display_cols].style.apply(color_comparison, axis=1),
            hide_index=True, use_container_width=True
        )

        # Per-bookmaker breakdown
        st.markdown("### 📋 Odds from every bookmaker — game by game")
        st.markdown("Compare exactly what each bookmaker is offering so you can find the best price.")

        for game in live_odds:
            home = game["home_team"]
            away = game["away_team"]
            if home not in elo_ratings or away not in elo_ratings:
                continue

            our_home_prob = expected_prob(
                elo_ratings.get(home,1500) + HOME_ADVANTAGE,
                elo_ratings.get(away,1500)
            )
            our_home_odds = round(1/our_home_prob, 2)
            our_away_odds = round(1/(1-our_home_prob), 2)

            with st.expander(f"🏉 {home} vs {away}"):
                # Header row with our model's odds
                st.markdown(
                    f"**Our model's fair odds:** "
                    f"{home} = **{our_home_odds}** | "
                    f"{away} = **{our_away_odds}**"
                )
                st.caption(
                    f"Bookmaker margin: {game['bookmaker_margin']}% — "
                    f"this is how much extra the bookmakers take on average"
                )

                # Build bookmaker table
                bm_rows = []
                for bm in game["bookmakers"]:
                    ho = bm["home_odds"]
                    ao = bm["away_odds"]
                    h_edge = round((our_home_prob * ho - 1) * 100, 1)
                    a_edge = round(((1-our_home_prob) * ao - 1) * 100, 1)
                    bm_rows.append({
                        "Bookmaker":     bm["bookmaker"],
                        f"{home} odds":  ho,
                        f"{away} odds":  ao,
                        "Home edge":     f"{h_edge:+.1f}%",
                        "Away edge":     f"{a_edge:+.1f}%",
                        "Home verdict":  "✅ BET" if h_edge > 0 else "❌",
                        "Away verdict":  "✅ BET" if a_edge > 0 else "❌",
                        "_h_edge":       h_edge,
                    })

                bm_df = pd.DataFrame(bm_rows)

                def color_bm(row):
                    if "BET" in str(row.get("Home verdict","")):
                        return ["background-color:#d4edda;color:#155724"]*len(row)
                    return ["background-color:#f8d7da;color:#721c24"]*len(row)

                show_cols = [
                    "Bookmaker",
                    f"{home} odds", "Home edge","Home verdict",
                    f"{away} odds", "Away edge","Away verdict",
                ]
                st.dataframe(
                    bm_df[show_cols].style.apply(color_bm, axis=1),
                    hide_index=True, use_container_width=True
                )

                # Best odds highlight
                bh_bm, bh_odds = get_best_bookmaker_for_team(game, home, True)
                ba_bm, ba_odds = get_best_bookmaker_for_team(game, away, False)
                st.markdown(
                    f"🏆 **Best odds:** "
                    f"{home} → **{bh_odds}** at {bh_bm} | "
                    f"{away} → **{ba_odds}** at {ba_bm}"
                )

st.divider()

# ─────────────────────────────────────────────────────────
# MANUAL MATCH ENTRY
# (Used when live odds aren't available, or to add injuries/weather)
# ─────────────────────────────────────────────────────────
st.subheader("🔧 Deep Analysis — Add Injuries & Weather")
st.markdown(
    "The live odds section above shows the raw comparison. "
    "Use this section to run the **full ML model** with injury adjustments and weather. "
    "The games auto-populate from live odds if available, or you can enter them manually."
)

# Pre-populate matches from live odds if available
if live_odds and len(live_odds) > 0:
    valid_games = [g for g in live_odds if g["home_team"] in all_teams and g["away_team"] in all_teams]
    DEFAULT_MATCHES = []
    for g in valid_games[:9]:
        # Default venue based on home team
        venue_defaults = {
            "Collingwood":"MCG","Melbourne":"MCG","Richmond":"MCG","Carlton":"Marvel Stadium",
            "St Kilda":"Marvel Stadium","Western Bulldogs":"Marvel Stadium","Essendon":"Marvel Stadium",
            "Hawthorn":"MCG","North Melbourne":"Marvel Stadium","Geelong":"GMHBA Stadium",
            "Brisbane Lions":"Gabba","Gold Coast":"People First Stadium",
            "Adelaide":"Adelaide Oval","Port Adelaide":"Adelaide Oval",
            "Fremantle":"Optus Stadium","West Coast":"Optus Stadium",
            "Sydney":"SCG","GWS Giants":"Giants Stadium",
        }
        v = venue_defaults.get(g["home_team"], "MCG")
        o = g["best_home_odds"]
        DEFAULT_MATCHES.append((g["home_team"], g["away_team"], v, o))
    if not DEFAULT_MATCHES:
        DEFAULT_MATCHES = [
            ("Collingwood","Carlton","MCG",1.72),
            ("Brisbane Lions","GWS Giants","Gabba",2.10),
            ("Melbourne","Geelong","MCG",1.90),
        ]
else:
    DEFAULT_MATCHES = [
        ("Collingwood","Carlton","MCG",1.72),
        ("Brisbane Lions","GWS Giants","Gabba",2.10),
        ("Melbourne","Geelong","MCG",1.90),
        ("Port Adelaide","Essendon","Adelaide Oval",1.65),
        ("Sydney","Richmond","SCG",1.55),
    ]

matches_raw = []
for i,(home,away,venue,odds) in enumerate(DEFAULT_MATCHES):
    c1,c2,c3,c4,c5 = st.columns([2,2,2,1.3,1.3])
    h  = c1.selectbox("🏠 Home",  all_teams, index=all_teams.index(home)  if home  in all_teams else 0, key=f"h{i}")
    a  = c2.selectbox("✈️ Away",  all_teams, index=all_teams.index(away)  if away  in all_teams else 1, key=f"a{i}")
    v  = c3.selectbox("📍 Venue", all_venues,index=all_venues.index(venue)if venue in all_venues else 0,key=f"v{i}")
    o  = c4.number_input("Bookie odds (H)", 1.01, 20.0, float(odds), 0.05, key=f"o{i}")
    gd = c5.date_input("Date", value=date.today()+timedelta(days=3), key=f"d{i}")
    matches_raw.append({"home":h,"away":a,"venue":v,"odds":o,"date":gd})
    if i < len(DEFAULT_MATCHES)-1:
        st.markdown("---")

st.divider()

# ─────────────────────────────────────────────────────────
# INJURIES
# ─────────────────────────────────────────────────────────
st.subheader("🏥 Injury & Team Selection")

with st.expander("How injury penalties work", expanded=False):
    st.markdown("""
    We convert SuperCoach price + position into an Elo penalty.
    Find SC prices at **supercoach.com.au** — search the player name, it's free.

    | Player type | Price range | Elo penalty |
    |---|---|---|
    | Fringe player | $150–250k | –3 to –8 pts |
    | Regular contributor | $300–450k | –12 to –18 pts |
    | Important player | $500–650k | –20 to –28 pts |
    | Star player | $700–800k | –30 to –38 pts |
    | Best in comp | $850–950k | –42 to –55 pts |
    """)

injury_adj = {}
n_inj = st.number_input("Teams with injury concerns this week:", 0, 18, 0)

for i in range(int(n_inj)):
    with st.container():
        st.markdown(f"**Injury #{i+1}**")
        col_t, col_rest = st.columns([2,4])
        inj_team = col_t.selectbox("Team", all_teams, key=f"it{i}")
        with col_rest:
            n_pl = st.number_input(f"Players out for {inj_team}", 1, 8, 1, key=f"np{i}")
            total_pen = 0
            for j in range(int(n_pl)):
                pc1,pc2,pc3 = st.columns([2,1,1])
                pname = pc1.text_input(f"Player {j+1}", key=f"pn{i}{j}", placeholder="e.g. Nick Daicos")
                dp, dpos = 400, "Midfielder"
                if pname:
                    found = lookup_player(pname)
                    if found:
                        dp, dpos = found[0]//1000, found[1]
                        pc1.caption("✅ Found in database")
                pprice = pc2.number_input("SC price ($k)", 100, 950, dp, 10, key=f"pp{i}{j}") * 1000
                ppos   = pc3.selectbox("Position", list(POSITION_MULT.keys()),
                                       index=list(POSITION_MULT.keys()).index(dpos)
                                             if dpos in POSITION_MULT else 0,
                                       key=f"ppos{i}{j}")
                if pname:
                    pen = player_elo_penalty(pprice, ppos)
                    total_pen += pen
                    pc1.caption(f"Penalty: {pen:+.0f} Elo pts")
            if total_pen:
                injury_adj[inj_team] = total_pen
                st.info(f"{inj_team} total penalty: **{total_pen:+.0f} Elo pts**")
        st.markdown("---")

st.divider()

# ─────────────────────────────────────────────────────────
# FULL ANALYSIS BUTTON
# ─────────────────────────────────────────────────────────
st.subheader("🔮 Full ML Analysis")
run = st.button("Run Full Analysis (ML + Weather + Injuries)", type="primary", use_container_width=True)

if run:
    results_out = []

    # Fetch weather
    wx_cache = {}
    with st.status("🌤️ Fetching weather...", expanded=False) as wx_status:
        seen = {m["venue"]: m["date"] for m in matches_raw}
        for venue, gdate in seen.items():
            wx_cache[venue] = fetch_weather_for_venue(venue, gdate)
            st.write(f"  {venue}: {wx_cache[venue].get('condition','—')}")
    wx_status.update(label="✅ Weather loaded", state="complete")

    # Run predictions
    with st.status("🤖 Running predictions...", expanded=False) as pred_status:
        for m in matches_raw:
            home, away, venue, bookie = m["home"], m["away"], m["venue"], m["odds"]
            home_inj = injury_adj.get(home, 0)
            away_inj = injury_adj.get(away, 0)
            h_elo = elo_ratings.get(home, 1500) + home_inj
            a_elo = elo_ratings.get(away, 1500) + away_inj

            wx = wx_cache.get(venue, {})
            wx_h, wx_a = weather_elo_adjustment(wx, h_elo, a_elo)
            h_elo_final = h_elo + wx_h
            a_elo_final = a_elo + wx_a

            combined_adj = {home: home_inj + wx_h, away: away_inj + wx_a}

            ml_prob  = predict_with_ml(home, away, elo_ratings, results_df, model, scaler, combined_adj)
            elo_prob = expected_prob(h_elo_final + HOME_ADVANTAGE, a_elo_final)
            final_prob = round(0.7 * ml_prob + 0.3 * elo_prob, 4)

            fair_odds = round(1 / final_prob, 2)
            edge      = round((final_prob * bookie - 1) * 100, 2)
            ev        = round(final_prob * bookie * stake - stake, 2)
            is_value  = edge > 0
            conf      = "🟢 High" if abs(ml_prob-elo_prob)<0.05 else ("🟡 Medium" if abs(ml_prob-elo_prob)<0.10 else "🔴 Low")

            # Pull live odds for this match if available
            live_match = None
            if live_odds:
                for g in live_odds:
                    if g["home_team"] == home and g["away_team"] == away:
                        live_match = g
                        break

            results_out.append({
                "home":home,"away":away,"venue":venue,
                "h_elo":round(h_elo_final),"a_elo":round(a_elo_final),
                "ml_prob":ml_prob,"elo_prob":elo_prob,"final_prob":final_prob,
                "fair_odds":fair_odds,"bookie_odds":bookie,
                "edge":edge,"ev":ev,"is_value":is_value,"conf":conf,
                "wx":wx,"home_inj":home_inj,"away_inj":away_inj,
                "live_match":live_match,
            })
            st.write(f"  ✅ {home} vs {away}: {final_prob*100:.1f}% home, edge {edge:+.1f}%")

    pred_status.update(label="✅ Done", state="complete")

    # ── Summary ──────────────────────────────────────────
    n_val    = sum(1 for r in results_out if r["is_value"])
    best     = max(results_out, key=lambda r: r["edge"])
    total_ev = sum(r["ev"] for r in results_out if r["is_value"])

    st.subheader("📊 Results")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Games analysed", len(results_out))
    m2.metric("Value bets",     n_val)
    m3.metric("Best edge",      f"+{best['edge']:.1f}%", best["home"])
    m4.metric(f"Total EV (${stake} each)", f"${total_ev:+.2f}")

    # ── Full comparison table ─────────────────────────────
    tbl = []
    for r in results_out:
        lm = r["live_match"]
        tbl.append({
            "Home":          r["home"],
            "Away":          r["away"],
            "Our Prob":      f"{r['final_prob']*100:.1f}%",
            "Our Fair Odds": r["fair_odds"],
            "You Entered":   r["bookie_odds"],
            "Best Bookie":   lm["best_home_odds"] if lm else "—",
            "Best at":       (get_best_bookmaker_for_team(lm,r["home"],True)[0] if lm else "—"),
            "Mkt Prob":      f"{lm['market_home_prob']*100:.1f}%" if lm else "—",
            "Edge":          f"{r['edge']:+.1f}%",
            f"EV(${stake})": f"${r['ev']:+.2f}",
            "Weather":       r["wx"].get("condition","—"),
            "Confidence":    r["conf"],
            "Verdict":       "✅ BET" if r["is_value"] else "❌ Avoid",
        })

    tdf = pd.DataFrame(tbl)

    def row_col(row):
        c = "background-color:#d4edda;color:#155724" if "BET" in str(row["Verdict"]) \
            else "background-color:#f8d7da;color:#721c24"
        return [c] * len(row)

    st.dataframe(tdf.style.apply(row_col, axis=1), hide_index=True, use_container_width=True)

    st.divider()

    # ── Per-game deep dives ───────────────────────────────
    st.markdown("### 🔍 Game Breakdowns")

    for r in results_out:
        icon = "✅" if r["is_value"] else "❌"
        with st.expander(f"{icon} {r['home']} vs {r['away']} @ {r['venue']}"):
            c1,c2,c3 = st.columns(3)
            lm = r["live_match"]

            with c1:
                st.markdown("**📈 Probability**")
                st.metric("ML model",    f"{r['ml_prob']*100:.1f}%")
                st.metric("Elo model",   f"{r['elo_prob']*100:.1f}%")
                st.metric("Final blend", f"{r['final_prob']*100:.1f}%")
                if lm:
                    st.metric("Market says", f"{lm['market_home_prob']*100:.1f}%",
                              delta=f"{'We agree' if abs(r['final_prob']-lm['market_home_prob'])<0.05 else 'We disagree ← edge here'}")
                st.metric("Confidence", r["conf"])

            with c2:
                st.markdown("**💰 Value Analysis**")
                st.metric("Our fair odds",  r["fair_odds"])
                st.metric("You entered",    r["bookie_odds"])
                if lm:
                    bh_bm, bh_odds = get_best_bookmaker_for_team(lm, r["home"], True)
                    st.metric("Best available", bh_odds, delta=f"at {bh_bm}")
                st.metric("Edge",           f"{r['edge']:+.1f}%",
                          delta="VALUE" if r["is_value"] else "No value")
                st.metric(f"EV on ${stake}", f"${r['ev']:+.2f}")

            with c3:
                st.markdown("**⚙️ Adjustments**")
                st.write(f"**Elo:** {r['home']} {r['h_elo']} | {r['away']} {r['a_elo']}")
                if r["home_inj"]: st.write(f"🏥 {r['home']} injury: {r['home_inj']:+.0f}")
                if r["away_inj"]: st.write(f"🏥 {r['away']} injury: {r['away_inj']:+.0f}")
                wx = r["wx"]
                if wx.get("is_indoor"):
                    st.write("🏟️ Indoor — no weather adj")
                else:
                    st.write(f"☁️ {wx.get('condition','—')}")
                    st.write(f"🌡️ {wx.get('temperature_c')}°C  🌧️ {wx.get('rain_mm')}mm  💨 {wx.get('wind_kmh')}km/h")
                if lm:
                    st.write(f"📊 Bookie margin: {lm['bookmaker_margin']}%")
                    st.write(f"📚 {lm['bookmaker_count']} bookmakers listed this game")

            st.divider()

            # Live odds per bookmaker for this game
            if lm:
                st.markdown("**All bookmaker odds for this game:**")
                bm_rows = []
                for bm in lm["bookmakers"]:
                    ho = bm["home_odds"]
                    ao = bm["away_odds"]
                    h_edge = round((r["final_prob"] * ho - 1) * 100, 1)
                    a_edge = round(((1-r["final_prob"]) * ao - 1) * 100, 1)
                    bm_rows.append({
                        "Bookmaker":        bm["bookmaker"],
                        f"{r['home']}":     ho,
                        f"Edge":            f"{h_edge:+.1f}%",
                        f"Verdict (H)":     "✅ BET" if h_edge>0 else "❌",
                        f"{r['away']}":     ao,
                        f"Away edge":       f"{a_edge:+.1f}%",
                        f"Verdict (A)":     "✅ BET" if a_edge>0 else "❌",
                    })
                bdf = pd.DataFrame(bm_rows)
                def bc(row):
                    if "BET" in str(row.get("Verdict (H)","")):
                        return ["background-color:#d4edda;color:#155724"]*len(row)
                    return ["background-color:#f8d7da;color:#721c24"]*len(row)
                st.dataframe(bdf.style.apply(bc, axis=1), hide_index=True, use_container_width=True)

            # Final plain-English verdict
            if r["is_value"]:
                st.success(
                    f"**✅ Value bet.** Our model prices {r['home']} at fair odds of **{r['fair_odds']}**. "
                    f"The best available odds are **{lm['best_home_odds'] if lm else r['bookie_odds']}**. "
                    f"Edge: **{r['edge']:+.1f}%**. "
                    f"On a ${stake} bet, expected profit: **${abs(r['ev']):.2f}** over time."
                )
            else:
                st.error(
                    f"**❌ No value.** Fair odds = **{r['fair_odds']}**, "
                    f"bookmaker offers **{r['bookie_odds']}** — less than fair. "
                    f"Bookmaker edge: **{abs(r['edge']):.1f}%**. Pass."
                )

st.divider()
st.caption("⚠️ Educational purposes only. Bet responsibly. Model accuracy doesn't guarantee future results.")
