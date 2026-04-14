# app.py — AFL Odds Analyzer
# ═══════════════════════════════════════════════════════════
# Professional-grade market-aware betting model.
# Finds where bookmakers are wrong using:
#   • Margin-adjusted Elo ratings (multi-season)
#   • Calibrated ML model (form + advanced stats)
#   • Opponent-adjusted scoring (not raw averages)
#   • Real historical odds for backtesting (CLV measurement)
#   • Live bookmaker odds (The Odds API)
#   • Weather, travel, rest, injury adjustments
# ═══════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import date, timedelta, datetime

st.set_page_config(
    page_title="AFL Odds Analyzer",
    page_icon="🏉",
    layout="wide",
    initial_sidebar_state="expanded",
)

from data_fetcher        import load_data, download_all_data
from feature_engineering import build_feature_matrix, weighted_recent_form, travel_penalty, days_rest, VENUE_STATE
from rating_model        import calculate_elo_ratings, get_elo_probability, win_probability_from_elo, INITIAL_RATING
from probability_engine  import train_probability_model, load_probability_model, calculate_true_probability
from odds_comparison     import full_odds_analysis, remove_overround
from decision_engine     import classify_bet, generate_explanation, backtest_model
from odds_fetcher        import fetch_afl_odds, get_best_bookmaker_for_team
from weather             import fetch_weather_for_venue, weather_elo_adjustment, VENUE_COORDS

# ── Injury database ──────────────────────────────────────
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
    "Midfielder": 1.2, "Mid": 1.2,
    "Key Forward": 1.1, "Forward": 1.0,
    "Key Defender": 1.0, "Defender": 0.95,
    "Ruck": 1.05, "Utility": 0.85, "Unknown": 1.0,
}


def player_elo_penalty(price, position="Unknown"):
    pct = (max(100_000, min(950_000, price)) - 100_000) / 850_000
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
st.markdown(
    "*Professional-grade market-aware betting model — "
    "find where bookmakers are wrong*"
)

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ System Controls")

    load_btn     = st.button("📥 Load / Refresh Data",        use_container_width=True, type="primary")
    odds_btn     = st.button("💰 Fetch Live Bookmaker Odds",  use_container_width=True)
    retrain_btn  = st.button("🔄 Force Retrain Model",        use_container_width=True)
    backtest_btn = st.button("📊 Run Backtest",               use_container_width=True)

    st.divider()
    stake    = st.number_input("Stake per bet ($)", 1, 1000, 50)
    bankroll = st.number_input("Total bankroll ($)", 100, 100_000, 1000,
                               help="Used for Kelly Criterion bet sizing")

    st.divider()
    st.markdown("**Bet thresholds:**")
    min_edge = st.slider("Min edge to show bets (%)", 0, 15, 2) / 100
    min_prob = st.slider("Min model probability (%)", 50, 70, 52) / 100

    st.divider()
    if "api_status" in st.session_state:
        st.caption(st.session_state["api_status"])


# ─────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data():
    return load_data()  # Returns (results_df, stats_df, hist_odds_df)


@st.cache_data(show_spinner=False)
def get_ratings(_df):
    return calculate_elo_ratings(_df)


@st.cache_data(show_spinner=False)
def get_features(_df, _elo_hist, _stats_df):
    return build_feature_matrix(_df, _elo_hist, _stats_df)


@st.cache_resource(show_spinner=False)
def get_prob_model(_feat_df):
    m, s, fc = load_probability_model()
    if m is None:
        m, s, fc, _ = train_probability_model(_feat_df)
    return m, s, fc


if load_btn:
    st.cache_data.clear()
    st.cache_resource.clear()
    with st.spinner("Downloading AFL data (results + stats + historical odds)..."):
        download_all_data()
    st.rerun()

if retrain_btn:
    for f in ["afl_prob_model.pkl", "afl_prob_scaler.pkl", "afl_prob_feature_cols.pkl"]:
        if os.path.exists(f):
            os.remove(f)
    st.cache_resource.clear()
    st.rerun()

# Load all data
data_tuple = get_data()
if data_tuple is None or data_tuple[0] is None:
    st.info("👈 Click **Load / Refresh Data** in the sidebar to get started.")
    st.stop()

results_df, stats_df, hist_odds_df = data_tuple

with st.spinner("Building Elo ratings..."):
    elo_ratings, elo_history = get_ratings(results_df)

with st.spinner("Engineering features..."):
    feature_df = get_features(results_df, elo_history, stats_df)

with st.spinner("Loading probability model..."):
    prob_model, prob_scaler, feature_cols = get_prob_model(feature_df)

all_teams  = sorted(elo_ratings.keys())
all_venues = sorted([v for v in VENUE_COORDS if v != "Unknown"])

# Model info banner
has_stats = stats_df is not None
has_hist_odds = hist_odds_df is not None and len(hist_odds_df) > 0
adv_stat_count = len([c for c in feature_cols if any(s in c for s in ['inside50s','clearances','tackles'])])

col_info1, col_info2, col_info3, col_info4 = st.columns(4)
col_info1.metric("Games in model", f"{len(results_df):,}")
col_info2.metric("Advanced stats", "✅ Active" if has_stats else "⚠️ Unavailable")
col_info3.metric("Historical odds", f"✅ {len(hist_odds_df):,} records" if has_hist_odds else "⚠️ Simulated only")
col_info4.metric("ML features", len(feature_cols))

# ─────────────────────────────────────────────────────────
# LIVE ODDS
# ─────────────────────────────────────────────────────────
if "live_odds" not in st.session_state:
    st.session_state["live_odds"] = None

if odds_btn:
    with st.spinner("Fetching live odds..."):
        games, status = fetch_afl_odds()
        st.session_state["live_odds"]  = games
        st.session_state["api_status"] = status
    st.rerun()

live_odds = st.session_state["live_odds"]


# ─────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Predictions",
    "📊 Power Rankings",
    "💰 Live Odds",
    "📈 Backtest",
    "❓ How It Works",
])


# ══════════════════════════════════════════════════════════
# TAB 1: PREDICTIONS
# ══════════════════════════════════════════════════════════
with tab1:
    st.subheader("Match Setup")

    # Auto-populate from live odds if available
    if live_odds and len(live_odds) > 0:
        valid = [g for g in live_odds if g["home_team"] in all_teams and g["away_team"] in all_teams]
        DEFAULTS = []
        venue_map = {
            "Collingwood": "MCG", "Melbourne": "MCG", "Richmond": "MCG",
            "Carlton": "Marvel Stadium", "St Kilda": "Marvel Stadium",
            "Western Bulldogs": "Marvel Stadium", "Essendon": "Marvel Stadium",
            "Hawthorn": "MCG", "North Melbourne": "Marvel Stadium",
            "Geelong": "GMHBA Stadium", "Brisbane Lions": "Gabba",
            "Gold Coast": "People First Stadium", "Adelaide": "Adelaide Oval",
            "Port Adelaide": "Adelaide Oval", "Fremantle": "Optus Stadium",
            "West Coast": "Optus Stadium", "Sydney": "SCG",
            "GWS Giants": "Giants Stadium",
        }
        for g in valid[:9]:
            v = venue_map.get(g["home_team"], "MCG")
            DEFAULTS.append((g["home_team"], g["away_team"], v, g["best_home_odds"]))
        if not DEFAULTS:
            DEFAULTS = [("Collingwood", "Carlton", "MCG", 1.72)]
    else:
        DEFAULTS = [
            ("Collingwood", "Carlton",       "MCG",            1.72),
            ("Brisbane Lions", "GWS Giants", "Gabba",          2.10),
            ("Melbourne", "Geelong",         "MCG",            1.90),
            ("Port Adelaide", "Essendon",    "Adelaide Oval",  1.65),
            ("Sydney", "Richmond",           "SCG",            1.55),
        ]

    matches_raw = []
    with st.expander("📋 Enter Matches", expanded=True):
        for i, (home, away, venue, odds) in enumerate(DEFAULTS):
            c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 1.3, 1.3])
            h  = c1.selectbox("Home",  all_teams,  index=all_teams.index(home)    if home  in all_teams  else 0, key=f"h{i}")
            a  = c2.selectbox("Away",  all_teams,  index=all_teams.index(away)    if away  in all_teams  else 1, key=f"a{i}")
            v  = c3.selectbox("Venue", all_venues, index=all_venues.index(venue)  if venue in all_venues else 0, key=f"v{i}")
            o  = c4.number_input("Bookie odds (H)", 1.01, 20.0, float(odds), 0.05, key=f"o{i}")
            gd = c5.date_input("Date", value=date.today() + timedelta(days=4), key=f"d{i}")
            matches_raw.append({"home": h, "away": a, "venue": v, "odds": o, "date": gd})
            st.markdown("---")

    # ── Injuries ─────────────────────────────────────────
    with st.expander("🏥 Injury & Team Selection", expanded=False):
        st.markdown(
            "Enter ruled-out players. If in the database, price fills automatically. "
            "Get SuperCoach prices at **supercoach.com.au**."
        )
        n_inj = st.number_input("Teams with injuries:", 0, 18, 0, key="n_inj")
        injury_adj            = {}
        injury_notes_by_team  = {}

        for i in range(int(n_inj)):
            col_t, col_r = st.columns([2, 4])
            inj_team = col_t.selectbox("Team", all_teams, key=f"it{i}")
            with col_r:
                n_pl = st.number_input("Players out:", 1, 8, 1, key=f"np{i}")
                total_pen = 0
                notes = []
                for j in range(int(n_pl)):
                    pc1, pc2, pc3 = st.columns([2, 1, 1])
                    pname = pc1.text_input(f"Player {j+1}", key=f"pn{i}{j}")
                    dp, dpos = 400, "Midfielder"
                    if pname:
                        found = lookup_player(pname)
                        if found:
                            dp, dpos = found[0] // 1000, found[1]
                            pc1.caption("✅ In database")
                    pprice = pc2.number_input("SC $k", 100, 950, dp, 10, key=f"pp{i}{j}") * 1000
                    ppos   = pc3.selectbox("Pos", list(POSITION_MULT.keys()),
                                           index=list(POSITION_MULT.keys()).index(dpos)
                                                 if dpos in POSITION_MULT else 0,
                                           key=f"pps{i}{j}")
                    if pname:
                        pen = player_elo_penalty(pprice, ppos)
                        total_pen += pen
                        notes.append(f"{pname} ({ppos}, ${pprice//1000}k): {pen:+.0f} Elo pts")
                if total_pen:
                    injury_adj[inj_team] = total_pen
                    injury_notes_by_team[inj_team] = notes
                    st.info(f"{inj_team}: **{total_pen:+.0f} Elo pts** total injury penalty")

    # ── Run ───────────────────────────────────────────────
    run = st.button("🔮 Analyse All Matches", type="primary", use_container_width=True)

    if run:
        all_results = []

        # Weather
        wx_cache = {}
        with st.status("🌤️ Fetching weather...", expanded=False) as s:
            for m in matches_raw:
                vn = m["venue"]
                if vn not in wx_cache:
                    wx_cache[vn] = fetch_weather_for_venue(vn, m["date"])
                    s.write(f"  {vn}: {wx_cache[vn].get('condition','—')}")
        s.update(label="✅ Weather loaded", state="complete")

        with st.status("🤖 Running probability engine...", expanded=False) as s:
            for m in matches_raw:
                home, away, venue = m["home"], m["away"], m["venue"]
                gdate       = m["date"]
                bookie_odds = m["odds"]

                # Rest
                r_h_days, r_h_pen = days_rest(home, results_df, gdate, len(results_df) - 1)
                r_a_days, r_a_pen = days_rest(away, results_df, gdate, len(results_df) - 1)
                rest_adj_dict = {}
                if r_h_pen: rest_adj_dict[home] = r_h_pen
                if r_a_pen: rest_adj_dict[away] = r_a_pen

                # Travel
                trv_h = travel_penalty(home, venue)
                trv_a = travel_penalty(away, venue)
                travel_adj_dict = {}
                if trv_h: travel_adj_dict[home] = trv_h
                if trv_a: travel_adj_dict[away] = trv_a

                # Weather
                wx = wx_cache.get(venue, {})
                home_elo_raw = elo_ratings.get(home, INITIAL_RATING)
                away_elo_raw = elo_ratings.get(away, INITIAL_RATING)
                wx_h_adj, wx_a_adj = weather_elo_adjustment(wx, home_elo_raw, away_elo_raw)
                wx_adj_dict = {}
                if wx_h_adj: wx_adj_dict[home] = wx_h_adj
                if wx_a_adj: wx_adj_dict[away] = wx_a_adj

                # Injuries
                inj_adj = {k: v for k, v in injury_adj.items() if k in [home, away]}

                # Probability
                prob_bd = calculate_true_probability(
                    home, away, elo_ratings, results_df, stats_df,
                    prob_model, prob_scaler, feature_cols,
                    injury_adj=inj_adj,
                    rest_adj=rest_adj_dict,
                    travel_adj=travel_adj_dict,
                )

                # Weather probability shift
                if wx_h_adj or wx_a_adj:
                    base_p = win_probability_from_elo(home_elo_raw, away_elo_raw)
                    wx_adj_p = win_probability_from_elo(
                        home_elo_raw + wx_h_adj,
                        away_elo_raw + wx_a_adj,
                    )
                    wx_shift = wx_adj_p - base_p
                    prob_bd['final_prob'] = float(np.clip(
                        prob_bd['final_prob'] + wx_shift * 0.3, 0.05, 0.95
                    ))
                    prob_bd['weather_component'] = round(wx_shift * 0.3 * 100, 2)
                else:
                    prob_bd['weather_component'] = 0.0

                final_prob = prob_bd['final_prob']

                # Bookmaker list
                bm_list = []
                proxy_away = round(1.0 / max(0.05, (1 - final_prob) * 0.95), 2)
                bm_list.append({
                    'bookmaker': 'You entered',
                    'home_odds': bookie_odds,
                    'away_odds': proxy_away,
                })

                live_match = None
                if live_odds:
                    for g in live_odds:
                        if g["home_team"] == home and g["away_team"] == away:
                            live_match = g
                            for bm in g["bookmakers"]:
                                if bm["bookmaker"] != "You entered":
                                    bm_list.append({
                                        'bookmaker': bm["bookmaker"],
                                        'home_odds': bm["home_odds"],
                                        'away_odds': bm["away_odds"],
                                    })
                            break

                odds_bd = full_odds_analysis(home, away, final_prob, bm_list, stake)

                inj_notes_combined = (
                    injury_notes_by_team.get(home, []) +
                    injury_notes_by_team.get(away, [])
                )
                explanation, classification, label, colour = generate_explanation(
                    home, away, prob_bd, odds_bd, inj_notes_combined, wx, stake
                )

                rest_notes = []
                if r_h_pen: rest_notes.append(f"{home}: {r_h_days}d rest ({r_h_pen:+.0f} Elo)")
                if r_a_pen: rest_notes.append(f"{away}: {r_a_days}d rest ({r_a_pen:+.0f} Elo)")
                travel_notes = []
                if trv_h: travel_notes.append(f"{home}: {trv_h:+.0f} Elo (travel)")
                if trv_a: travel_notes.append(f"{away}: {trv_a:+.0f} Elo (travel)")

                all_results.append({
                    "home": home, "away": away, "venue": venue,
                    "prob_bd": prob_bd, "odds_bd": odds_bd,
                    "explanation": explanation,
                    "classification": classification,
                    "label": label, "colour": colour,
                    "wx": wx, "live_match": live_match,
                    "rest_notes": rest_notes,
                    "travel_notes": travel_notes,
                })
                s.write(f"  ✅ {home} vs {away}: {final_prob*100:.1f}% home, edge {odds_bd['home_edge']*100:+.1f}%")

        s.update(label="✅ Analysis complete", state="complete")

        # ── Summary bar ───────────────────────────────────
        st.divider()

        strong_bets = [r for r in all_results if r["classification"] == "STRONG_BET"]
        bets        = [r for r in all_results if r["classification"] in ("BET","SMALL_BET")]
        no_bets     = [r for r in all_results if r["classification"] == "NO_BET"]
        all_val     = [r for r in all_results if r["classification"] != "NO_BET"]
        total_ev    = sum(
            r["odds_bd"]["home_ev"] if r["odds_bd"]["home_edge"] > r["odds_bd"]["away_edge"]
            else r["odds_bd"]["away_ev"]
            for r in all_val if r["odds_bd"]
        )

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Games",         len(all_results))
        m2.metric("🔥 Strong bets",len(strong_bets))
        m3.metric("✅ Bets",        len(bets))
        m4.metric("❌ No bet",      len(no_bets))
        m5.metric(f"Total EV (${stake})", f"${total_ev:+.2f}")

        # ── Summary table ─────────────────────────────────
        st.markdown("### Results — Quick View")

        tbl_rows = []
        for r in all_results:
            p  = r["prob_bd"]
            o  = r["odds_bd"]
            ho = o["home_edge"]
            ao = o["away_edge"]
            if ho >= ao:
                edge = ho; odds = o["best_home_odds"]; bm = o["best_home_bm"]
                ev = o["home_ev"]; mkt = o["market_home_prob"]
            else:
                edge = ao; odds = o["best_away_odds"]; bm = o["best_away_bm"]
                ev = o["away_ev"]; mkt = o["market_away_prob"]

            tbl_rows.append({
                "Match":          f"{r['home']} vs {r['away']}",
                "Our Prob (H)":   f"{p['final_prob']*100:.1f}%",
                "Mkt Prob (H)":   f"{mkt*100:.1f}%",
                "Prob diff":      f"{(p['final_prob']-o['market_home_prob'])*100:+.1f}pp",
                "Our Fair Odds":  round(1.0/p['final_prob'],2),
                "Best Odds":      odds,
                "Best at":        bm,
                "Edge":           f"{edge*100:+.2f}%",
                f"EV (${stake})": f"${ev:+.2f}",
                "Overround":      f"{o['avg_overround']:.1f}%",
                "Decision":       r["label"],
            })

        tdf = pd.DataFrame(tbl_rows)

        def row_colour(row):
            cls = all_results[list(tdf.index).index(row.name)]["classification"]
            if cls == "STRONG_BET":
                return ["background-color:#155724;color:#d4edda"] * len(row)
            elif cls == "BET":
                return ["background-color:#d4edda;color:#155724"] * len(row)
            elif cls == "SMALL_BET":
                return ["background-color:#fff3cd;color:#856404"] * len(row)
            elif cls == "FADE":
                return ["background-color:#e2e3e5;color:#383d41"] * len(row)
            else:
                return ["background-color:#f8d7da;color:#721c24"] * len(row)

        st.dataframe(
            tdf.style.apply(row_colour, axis=1),
            hide_index=True, use_container_width=True
        )

        # ── Deep dives ────────────────────────────────────
        st.divider()
        st.markdown("### 🔍 Full Game Breakdowns")

        for r in all_results:
            p = r["prob_bd"]
            o = r["odds_bd"]
            icon = {"STRONG_BET":"🔥","BET":"✅","SMALL_BET":"🟡","NO_BET":"❌","FADE":"🔄"}.get(r["classification"],"❓")

            with st.expander(f"{icon} {r['home']} vs {r['away']}  —  {r['label']}"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**📊 Probability**")
                    st.metric("Elo model",       f"{p['elo_prob']*100:.1f}%")
                    st.metric("ML model",        f"{p['ml_prob']*100:.1f}%")
                    st.metric("Final (blended)", f"{p['final_prob']*100:.1f}%")
                    st.metric("Market consensus",f"{o['market_home_prob']*100:.1f}%")
                    diff = (p['final_prob'] - o['market_home_prob']) * 100
                    st.metric("Model vs market", f"{diff:+.1f}pp",
                              delta="Bullish on home" if diff > 0 else "Bullish on away")

                with col2:
                    st.markdown("**💰 Value**")
                    st.metric("Fair odds (H)",   o['fair_home_odds'])
                    st.metric("Best odds (H)",   o['best_home_odds'], delta=f"at {o['best_home_bm']}")
                    st.metric("Home edge",       f"{o['home_edge']*100:+.2f}%")
                    st.metric("Away edge",       f"{o['away_edge']*100:+.2f}%")
                    st.metric(f"Best EV (${stake})", f"${max(o['home_ev'],o['away_ev']):+.2f}")
                    kelly_val = max(o['home_kelly'], o['away_kelly'])
                    st.metric("Kelly bet size",  f"${bankroll*kelly_val:.0f}",
                              delta=f"{kelly_val*100:.1f}% of bankroll")

                with col3:
                    st.markdown("**⚙️ Adjustments**")
                    st.write(f"**Elo:**  {r['home']} {p['home_elo']:.0f}  |  {r['away']} {p['away_elo']:.0f}")
                    st.write(f"**Breakdown (home %):**")
                    st.write(f"• Elo effect:    {p['elo_component']:+.1f}pp")
                    st.write(f"• Form/ML:       {p['form_component']:+.1f}pp")
                    if abs(p['injury_component']) > 0.1:
                        st.write(f"• Injury:        {p['injury_component']:+.1f}pp")
                    if abs(p['rest_component']) > 0.1:
                        st.write(f"• Rest/fatigue:  {p['rest_component']:+.1f}pp")
                    if abs(p['travel_component']) > 0.1:
                        st.write(f"• Travel:        {p['travel_component']:+.1f}pp")
                    wx_comp = p.get('weather_component', 0)
                    if abs(wx_comp) > 0.1:
                        st.write(f"• Weather:       {wx_comp:+.1f}pp")

                    # Advanced stats comparison
                    hs = p.get('home_adv_stats', {})
                    as_ = p.get('away_adv_stats', {})
                    if hs.get('inside50s') or hs.get('clearances'):
                        st.write("")
                        st.write("**Advanced stats (recent avg):**")
                        for stat in ['inside50s','clearances','tackles','hitouts']:
                            hv = hs.get(stat)
                            av = as_.get(stat)
                            if hv and av:
                                st.write(f"• {stat}: {r['home']} {hv:.1f} vs {r['away']} {av:.1f} ({hv-av:+.1f})")

                if r["rest_notes"] or r["travel_notes"]:
                    st.markdown("**🚌 Situational:**")
                    for n in r["rest_notes"] + r["travel_notes"]:
                        st.write(f"  • {n}")

                wx = r["wx"]
                if not wx.get("is_indoor", False):
                    st.markdown(f"**☁️ Weather at {r['venue']}:**")
                    st.write(
                        f"  {wx.get('condition','—')} | "
                        f"🌡️{wx.get('temperature_c')}°C  "
                        f"🌧️{wx.get('rain_mm')}mm  "
                        f"💨{wx.get('wind_kmh')}km/h"
                    )
                    st.write(f"  {wx.get('impact','—')}")

                if o.get("bias_flags"):
                    st.markdown("**🔍 Market Signals:**")
                    for flag in o["bias_flags"]:
                        st.warning(flag)

                if o.get("per_bookmaker"):
                    st.markdown("**📋 All bookmaker odds:**")
                    bm_rows = []
                    for bm in o["per_bookmaker"]:
                        bm_rows.append({
                            "Bookmaker":           bm["bookmaker"],
                            f"{r['home']} odds":   bm["home_odds"],
                            "H edge":              f"{bm['home_edge']*100:+.2f}%",
                            f"{r['away']} odds":   bm["away_odds"],
                            "A edge":              f"{bm['away_edge']*100:+.2f}%",
                            "Overround":           f"{bm['overround']:.1f}%",
                            "Verdict":             "✅ BET" if bm["home_edge"] > min_edge else "❌",
                        })
                    bdf = pd.DataFrame(bm_rows)
                    def bm_col(row):
                        if "✅" in str(row.get("Verdict","")):
                            return ["background-color:#d4edda;color:#155724"] * len(row)
                        return ["background-color:#f8d7da;color:#721c24"] * len(row)
                    st.dataframe(bdf.style.apply(bm_col, axis=1), hide_index=True, use_container_width=True)

                st.markdown("**📝 Full explanation:**")
                st.code(r["explanation"], language=None)

                st.divider()
                if r["classification"] in ("STRONG_BET","BET","SMALL_BET"):
                    best_team = r["home"] if o["home_edge"] >= o["away_edge"] else r["away"]
                    best_edge = max(o["home_edge"], o["away_edge"])
                    best_odds = o["best_home_odds"] if o["home_edge"] >= o["away_edge"] else o["best_away_odds"]
                    best_bm   = o["best_home_bm"]   if o["home_edge"] >= o["away_edge"] else o["best_away_bm"]
                    kelly_bet = bankroll * max(o["home_kelly"], o["away_kelly"])
                    st.success(
                        f"**{r['label']}** — Bet **{best_team}** at **{best_odds}** ({best_bm})\n\n"
                        f"Edge: **{best_edge*100:+.2f}%** | "
                        f"Kelly suggested: **${kelly_bet:.0f}** of ${bankroll:,}"
                    )
                else:
                    st.error("**No value found.** The bookmaker has the edge on this game. Pass.")


# ══════════════════════════════════════════════════════════
# TAB 2: POWER RANKINGS
# ══════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Elo Power Rankings")
    st.markdown(
        "Ratings computed by replaying every AFL game since 2015 with a "
        "**margin-of-victory adjusted Elo** system. Higher = stronger right now."
    )

    elo_rows = sorted(elo_ratings.items(), key=lambda x: -x[1])
    elo_tbl  = pd.DataFrame([
        {
            "#":            i + 1,
            "Team":         t,
            "Elo Rating":   round(r),
            "vs Average":   f"{r-1500:+.0f}",
            "Win% vs avg":  f"{win_probability_from_elo(r, 1500)*100:.1f}%",
            "Tier": (
                "🔥 Elite"    if r > 1580 else
                "✅ Strong"   if r > 1530 else
                "🟡 Average"  if r > 1480 else
                "🔴 Weak"
            ),
        }
        for i, (t, r) in enumerate(elo_rows)
    ])
    st.dataframe(elo_tbl, hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("**Model parameters:**")
    st.markdown("""
    | Parameter | Value | Reason |
    |---|---|---|
    | K-factor | 20 | Team sports standard (NBA/NFL validated) |
    | Scale | 400 | Standard logistic |
    | Home advantage | +50 pts | AFL home ground effect |
    | Season reversion | 25% | Partial regression to 1500 each off-season |
    | Margin multiplier | log(margin+1)×2.2/... | FiveThirtyEight NFL formula |
    """)


# ══════════════════════════════════════════════════════════
# TAB 3: LIVE ODDS
# ══════════════════════════════════════════════════════════
with tab3:
    st.subheader("💰 Live Bookmaker Odds vs Model")

    if live_odds is None:
        st.info("Click **💰 Fetch Live Bookmaker Odds** in the sidebar to load real-time odds.")
    elif not live_odds:
        st.warning("No games currently listed — market may be between rounds.")
    else:
        if "api_status" in st.session_state:
            st.caption(st.session_state["api_status"])

        comp_rows = []
        for game in live_odds:
            home, away = game["home_team"], game["away_team"]
            if home not in elo_ratings or away not in elo_ratings:
                continue
            h_elo = elo_ratings.get(home, INITIAL_RATING)
            a_elo = elo_ratings.get(away, INITIAL_RATING)
            our_prob    = win_probability_from_elo(h_elo, a_elo)
            our_fair_h  = round(1 / our_prob, 2)
            true_h, _, ovr = remove_overround(game["avg_home_odds"], game["avg_away_odds"])
            h_edge = (our_prob * game["best_home_odds"]) - 1
            _, best_h_odds = get_best_bookmaker_for_team(game, home, True)
            _, best_a_odds = get_best_bookmaker_for_team(game, away, False)

            comp_rows.append({
                "Game":            f"{home} vs {away}",
                "Elo Prob (H)":    f"{our_prob*100:.1f}%",
                "Market Prob (H)": f"{true_h*100:.1f}%",
                "Model diff":      f"{(our_prob-true_h)*100:+.1f}pp",
                "Fair Odds":       our_fair_h,
                "Best Bookie (H)": best_h_odds,
                "Edge (H)":        f"{h_edge*100:+.2f}%",
                "Overround":       f"{ovr:.1f}%",
                "# Bookmakers":    game["bookmaker_count"],
                "Signal":          "✅ Check" if h_edge > min_edge else "❌",
            })

        cdf = pd.DataFrame(comp_rows)
        def col_live(row):
            if "✅" in str(row["Signal"]):
                return ["background-color:#d4edda;color:#155724"] * len(row)
            return ["background-color:#f8d7da;color:#721c24"] * len(row)
        st.dataframe(cdf.style.apply(col_live, axis=1), hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 4: BACKTEST
# ══════════════════════════════════════════════════════════
with tab4:
    st.subheader("📈 Model Backtest — Historical Performance")

    if has_hist_odds:
        st.success(
            f"✅ **Real historical odds loaded** ({len(hist_odds_df):,} records from aussportsbetting.com.au). "
            f"Backtest uses actual bookmaker prices — not simulated."
        )
    else:
        st.warning(
            "⚠️ Real historical odds unavailable. Backtest uses simulated 5% overround. "
            "Click **Load / Refresh Data** to attempt download from aussportsbetting.com.au."
        )

    st.markdown("""
    **How it works:**  
    Every historical game is replayed. Where our Elo model had sufficient edge 
    (using real or simulated bookmaker odds), we place a simulated $10 bet and 
    record the result. This tells us if the model genuinely beats the market.

    **Closing Line Value (CLV):** Average edge on placed bets. Positive CLV 
    is the gold-standard signal that a model has genuine long-run edge — 
    it means you consistently got better-than-closing odds.
    """)

    if backtest_btn or st.button("Run Backtest Now"):
        with st.spinner("Backtesting all seasons... (may take 30–60s)"):
            bt = backtest_model(
                results_df, elo_history,
                hist_odds_df=hist_odds_df if has_hist_odds else None,
                min_edge=min_edge,
                min_prob=min_prob,
            )

        if bt['total_bets'] == 0:
            st.warning("No bets passed the threshold. Lower the minimum edge or probability in the sidebar.")
        else:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total bets",    bt['total_bets'])
            m2.metric("Win rate",      f"{bt['win_rate']*100:.1f}%")
            m3.metric("Total staked",  f"${bt['total_staked']:,.0f}")
            m4.metric("Total return",  f"${bt['total_return']:,.0f}")
            roi_delta = "Profitable ✅" if bt['roi'] > 0 else "Loss-making ❌"
            m5.metric("ROI",           f"{bt['roi']:+.2f}%", delta=roi_delta)

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Avg CLV (edge)",   f"{bt['avg_clv']:+.2f}%",
                         help="Positive = beating the market consistently")
            col_b.metric("Brier Score",      bt['brier_score'],
                         help="Model calibration. Lower = better. Random = 0.25")
            col_c.metric("Real odds used",   f"{bt['real_odds_pct']}%",
                         help="% of bets using real historical odds vs simulated")

            if bt.get('message'):
                st.caption(bt['message'])

            st.markdown("### By Season")
            yr_rows = []
            for yr, d in sorted(bt['by_year'].items()):
                yr_rows.append({
                    "Season": yr,
                    "Bets":   d['bets'],
                    "Wins":   d['wins'],
                    "Win%":   f"{d['wins']/max(d['bets'],1)*100:.1f}%",
                    "P&L ($)":f"${d['profit']:+.2f}",
                    "ROI":    f"{d['roi']:+.2f}%",
                })
            ydf = pd.DataFrame(yr_rows)
            def yr_col(row):
                if "+" in str(row["ROI"]):
                    return ["background-color:#d4edda;color:#155724"] * len(row)
                return ["background-color:#f8d7da;color:#721c24"] * len(row)
            st.dataframe(ydf.style.apply(yr_col, axis=1), hide_index=True, use_container_width=True)

            st.markdown("### Sample Bets (most recent 20)")
            bdf = bt['bets_df'].tail(20)[
                ['year','home','away','bet_team','model_prob','sim_odds','edge','profit','won','real_odds']
            ].copy()
            bdf.columns = ['Year','Home','Away','Bet on','Prob','Odds','Edge','P&L ($)','Won','Real odds']
            bdf['Edge']    = bdf['Edge'].apply(lambda x: f"{x*100:+.2f}%")
            bdf['Prob']    = bdf['Prob'].apply(lambda x: f"{x*100:.1f}%")
            bdf['P&L ($)'] = bdf['P&L ($)'].apply(lambda x: f"${x:+.2f}")
            bdf['Real odds']= bdf['Real odds'].apply(lambda x: "✅" if x else "~sim")
            st.dataframe(bdf, hide_index=True, use_container_width=True)
    else:
        st.info("Click **Run Backtest Now** above, or use the sidebar button.")


# ══════════════════════════════════════════════════════════
# TAB 5: HOW IT WORKS
# ══════════════════════════════════════════════════════════
with tab5:
    st.subheader("❓ How the Model Works")
    st.markdown(f"""
    ## Core philosophy

    The goal is NOT to predict winners. The goal is to find where the
    **bookmaker's odds are wrong** — where our probability estimate is
    higher than the market implies.

    ---

    ## Step 1 — Elo Rating System

    Every team starts at 1500 points. After each game:
    - Winner gains points; loser loses the same amount (zero-sum)
    - Amount depends on how surprising the result was **AND** the margin
    - Margin multiplier uses the FiveThirtyEight NFL formula: `log(margin+1) × 2.2 / (elo_diff × 0.001 + 2.2)`
    - Each off-season, ratings revert 25% toward 1500

    **Home advantage:** +50 Elo points applied to all home teams.

    ---

    ## Step 2 — ML Model (form + advanced stats)

    A calibrated logistic regression trained on **{len(feature_cols)} features** 
    including recent form (last 5 games, exponentially weighted) and advanced 
    stats where available (inside 50s, clearances, tackles).

    Key fix: **Elo diff is NOT in the ML feature set.** This prevents 
    double-counting — Elo and ML are independent signals that get blended.

    **Opponent-adjusted scoring:** Instead of raw scoring averages, we 
    normalise each team's scores by their opponents' Elo ratings, so scoring 
    110 against a top-8 side is worth more than scoring 110 against a bottom team.

    ---

    ## Step 3 — Probability Blend

    ```
    final_prob = 0.60 × ML_prob + 0.40 × Elo_prob
    ```

    Both signals are independent — this blend is mathematically valid.
    ML captures form patterns Elo misses; Elo anchors against overfitting.

    ---

    ## Step 4 — Situational Modifiers

    Applied as Elo adjustments BEFORE probability calculation:
    - **Injuries:** Player price × position multiplier → Elo penalty
    - **Short rest:** <6 days → −12 Elo, <8 days → −5 Elo
    - **Long break:** >21 days → −3 Elo (rust)
    - **Interstate travel:** −5 Elo; cross-country −10 Elo
    - **Weather:** Rain/wind reduce the stronger team's advantage

    ---

    ## Step 5 — Market Comparison

    Bookmaker overround is removed to get true market probabilities:
    ```
    true_home = (1/home_odds) / ((1/home_odds) + (1/away_odds))
    ```

    Edge is calculated the same way for live predictions AND backtesting:
    ```
    edge = (model_probability × bookmaker_odds) - 1
    ```

    ---

    ## Step 6 — Kelly Criterion

    Bet sizing: 25% fractional Kelly
    ```
    Kelly = (odds × prob − (1−prob)) / odds × 25%
    ```

    ---

    ## Data Sources

    | Data | Source | Cost |
    |---|---|---|
    | AFL results (2015–now) | Squiggle API | Free |
    | Advanced stats (inside 50s etc) | Squiggle API | Free |
    | Historical bookmaker odds | aussportsbetting.com.au | Free |
    | Live bookmaker odds | The Odds API | 500 req/month free |
    | Weather forecasts | Open-Meteo | Free |
    """)

st.divider()
st.caption("⚠️ For educational purposes only. Bet responsibly. Model accuracy doesn't guarantee future results.")
