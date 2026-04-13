# weather.py
# Fetches weather forecasts from Open-Meteo (free, no API key needed).
# Rain and wind reduce the stronger team's advantage slightly.
# Indoor venues (Marvel Stadium) are unaffected.

import requests
from datetime import datetime, date

# GPS coordinates for every current AFL venue
# Format: "Name": (latitude, longitude, is_indoor)
VENUE_COORDS = {
    "MCG":                     (-37.8200, 144.9834, False),
    "Melbourne Cricket Ground":(-37.8200, 144.9834, False),
    "Marvel Stadium":          (-37.8165, 144.9469, True),
    "Docklands":               (-37.8165, 144.9469, True),
    "GMHBA Stadium":           (-38.1574, 144.3552, False),
    "Kardinia Park":           (-38.1574, 144.3552, False),
    "Mars Stadium":            (-37.5622, 143.8503, False),
    "Gabba":                   (-27.4858, 153.0381, False),
    "The Gabba":               (-27.4858, 153.0381, False),
    "People First Stadium":    (-27.9978, 153.3713, False),
    "Adelaide Oval":           (-34.9157, 138.5961, False),
    "Optus Stadium":           (-31.9505, 115.8882, False),
    "Perth Stadium":           (-31.9505, 115.8882, False),
    "SCG":                     (-33.8914, 151.2249, False),
    "Sydney Cricket Ground":   (-33.8914, 151.2249, False),
    "Giants Stadium":          (-33.8063, 150.9965, False),
    "ENGIE Stadium":           (-33.8474, 151.0633, False),
    "Blundstone Arena":        (-42.8839, 147.3609, False),
    "York Park":               (-41.4312, 147.1380, False),
    "Manuka Oval":             (-35.3192, 149.1296, False),
    "Norwood Oval":            (-34.9157, 138.6200, False),
    "Unknown":                 (-37.8200, 144.9834, False),
}


def get_venue_coords(venue_name):
    if venue_name in VENUE_COORDS:
        return VENUE_COORDS[venue_name]
    vl = venue_name.lower()
    for k, v in VENUE_COORDS.items():
        if k.lower() in vl or vl in k.lower():
            return v
    return VENUE_COORDS["Unknown"]


def fetch_weather_for_venue(venue_name, game_date=None):
    """
    Returns weather dict with: temperature_c, rain_mm, wind_kmh,
    is_indoor, condition, impact, elo_adjustment.
    """
    lat, lon, is_indoor = get_venue_coords(venue_name)

    if is_indoor:
        return {"temperature_c":20,"rain_mm":0,"wind_kmh":0,
                "is_indoor":True,"condition":"Indoor venue — no weather effect",
                "impact":"No adjustment","elo_adjustment":0,"rain_probability_pct":0}

    if game_date is None:
        game_date = date.today()

    try:
        params = {
            "latitude": lat, "longitude": lon,
            "daily": ["precipitation_sum","windspeed_10m_max",
                      "temperature_2m_max","temperature_2m_min"],
            "timezone": "Australia/Sydney",
            "forecast_days": 7,
        }
        r = requests.get("https://api.open-meteo.com/v1/forecast",
                         params=params, timeout=10)
        r.raise_for_status()
        daily = r.json().get("daily", {})
        dates = daily.get("time", [])
        gds   = str(game_date)
        i     = dates.index(gds) if gds in dates else 0

        rain    = daily["precipitation_sum"][i] or 0
        wind    = daily["windspeed_10m_max"][i] or 0
        tmax    = daily["temperature_2m_max"][i] or 20
        tmin    = daily["temperature_2m_min"][i] or 15
        temp    = (tmax + tmin) / 2

        # Build plain-English conditions
        parts = []
        if rain > 10:   parts.append("Heavy rain")
        elif rain > 3:  parts.append("Light rain")
        if wind > 40:   parts.append("Strong wind")
        elif wind > 25: parts.append("Moderate wind")
        if temp < 10:   parts.append("Cold")
        elif temp > 30: parts.append("Hot")
        condition = " & ".join(parts) if parts else "Fine"

        # How much does this suppress the stronger team?
        suppression = 0
        if rain > 10:   suppression += 15
        elif rain > 3:  suppression += 7
        if wind > 40:   suppression += 12
        elif wind > 25: suppression += 6

        if suppression >= 20:   impact = "⛈️ Major scoring suppression"
        elif suppression >= 10: impact = "🌧️ Moderate weather impact"
        elif suppression >= 5:  impact = "🌦️ Minor weather factor"
        else:                   impact = "☀️ Minimal impact"

        return {"temperature_c": round(temp,1), "rain_mm": round(rain,1),
                "wind_kmh": round(wind,1), "is_indoor": False,
                "condition": condition, "impact": impact,
                "elo_adjustment": suppression, "rain_probability_pct": 0}

    except Exception as e:
        return {"temperature_c":18,"rain_mm":0,"wind_kmh":10,"is_indoor":False,
                "condition":"Weather unavailable","impact":"No adjustment",
                "elo_adjustment":0,"rain_probability_pct":0}


def weather_elo_adjustment(wx, home_elo, away_elo):
    """
    Returns (home_adj, away_adj) — negative values reduce that team's advantage.
    Bad weather helps the underdog by making the game more random.
    """
    s = wx.get("elo_adjustment", 0)
    if s == 0: return 0, 0
    diff = home_elo - away_elo
    if abs(diff) < 30: return 0, 0
    if diff > 0:
        adj = min(s * 0.5, diff * 0.3)
        return -adj, 0
    else:
        adj = min(s * 0.5, abs(diff) * 0.3)
        return 0, -adj
