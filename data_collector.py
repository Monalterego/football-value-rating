"""
data_collector.py — Veri Toplama
=================================
Transfermarkt + Understat + FotMob verilerini çeker, CSV'ye kaydeder.
Süper Lig dahil 6 lig destekler.

Kullanım:
    python data_collector.py

Süre: ~15-20dk
Çıktılar: data/ klasörüne CSV dosyaları
"""

import os
import re
import time
import pandas as pd
import requests
from io import StringIO

HEADERS_TM = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
HEADERS_FM = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ── Transfermarkt pozisyon listesi ───────────────────────
TM_POSITIONS = [
    "Second Striker", "Centre-Forward", "Left Winger", "Right Winger",
    "Attacking Midfield", "Central Midfield", "Defensive Midfield",
    "Right Midfield", "Left Midfield", "Right-Back", "Left-Back",
    "Centre-Back", "Sweeper", "Goalkeeper",
]

# ── Lig tanımları ────────────────────────────────────────
TM_LEAGUES = {
    "GB1": "Premier League", "ES1": "La Liga", "IT1": "Serie A",
    "L1": "Bundesliga", "FR1": "Ligue 1", "TR1": "Süper Lig",
}

UNDERSTAT_LEAGUES = {
    "EPL": "Premier League", "La_Liga": "La Liga", "Serie_A": "Serie A",
    "Bundesliga": "Bundesliga", "Ligue_1": "Ligue 1",
}

FOTMOB_LEAGUES = {
    "47": {"name": "Premier League", "seasons": {"2020": "15382", "2021": "16390", "2022": "17664", "2023": "20720", "2024": "23685"}},
    "87": {"name": "La Liga", "seasons": {"2020": "15585", "2021": "16520", "2022": "17852", "2023": "21053", "2024": "23686"}},
    "55": {"name": "Serie A", "seasons": {"2020": "15604", "2021": "16621", "2022": "17866", "2023": "20956", "2024": "23819"}},
    "54": {"name": "Bundesliga", "seasons": {"2020": "15481", "2021": "16494", "2022": "17801", "2023": "20946", "2024": "23794"}},
    "53": {"name": "Ligue 1", "seasons": {"2020": "15293", "2021": "16499", "2022": "17810", "2023": "20868", "2024": "23724"}},
    "71": {"name": "Süper Lig", "seasons": {"2020": "15568", "2021": "16616", "2022": "17938", "2023": "21459", "2024": "23864"}},
}

FOTMOB_STATS = {
    "goals": "goals", "goal_assist": "assists", "expected_goals": "xG",
    "expected_assists": "xA", "total_scoring_att": "shots_p90",
    "ontarget_scoring_att": "shots_on_target_p90",
    "won_contest": "successful_dribbles_p90",
    "big_chance_created": "big_chances_created",
    "total_att_assist": "chances_created", "total_tackle": "tackles_p90",
    "interception": "interceptions_p90",
    "effective_clearance": "clearances_p90",
    "outfielder_block": "blocks_p90",
    "poss_won_att_3rd": "poss_won_final3rd_p90",
    "accurate_pass": "accurate_passes_p90",
    "accurate_long_balls": "accurate_long_balls_p90",
    "_save_percentage": "save_pct", "_goals_prevented": "goals_prevented",
    "rating": "fotmob_rating", "mins_played": "minutes_played",
}

UNDERSTAT_SEASONS = [2020, 2021, 2022, 2023, 2024]
MIN_MINUTES = 900


def _parse_tm_name(raw):
    for pos in TM_POSITIONS:
        if str(raw).endswith(pos):
            return str(raw)[: -len(pos)].strip(), pos
    return str(raw), "Unknown"


def _parse_tm_value(val):
    if pd.isna(val):
        return None
    val = str(val).replace("€", "").strip()
    if val.endswith("m"):
        return float(val.replace("m", "").replace(",", ""))
    elif val.endswith("k"):
        return float(val.replace("k", "").replace(",", "")) / 1000
    return None


def _parse_tm_age(dob_str):
    if pd.isna(dob_str):
        return None
    m = re.search(r"\((\d+)\)", str(dob_str))
    return int(m.group(1)) if m else None


# ══════════════════════════════════════════════════════════
# 1. TRANSFERMARKT
# ══════════════════════════════════════════════════════════
def fetch_transfermarkt():
    path = os.path.join(DATA_DIR, "transfermarkt.csv")
    if os.path.exists(path):
        print(f"✅ {path} zaten mevcut — atlanıyor.")
        return pd.read_csv(path)

    print("=" * 50)
    print("1️⃣  TRANSFERMARKT")
    print("=" * 50)
    all_players = []

    for code, lname in TM_LEAGUES.items():
        print(f"\n⏳ {lname}")
        try:
            r = requests.get(
                f"https://www.transfermarkt.com/wettbewerb/startseite/wettbewerb/{code}",
                headers=HEADERS_TM, timeout=30,
            )
            links = re.findall(
                r'href="(/[^/]+/startseite/verein/(\d+)/saison_id/\d+)"', r.text
            )
            seen = set()
            teams = []
            for link, tid in links:
                if tid not in seen:
                    seen.add(tid)
                    teams.append((tid, link))
            print(f"  {len(teams)} takım")
        except Exception as e:
            print(f"  ❌ {e}")
            continue

        for i, (tid, link) in enumerate(teams):
            for attempt in range(2):
                try:
                    r2 = requests.get(
                        f"https://www.transfermarkt.com{link}",
                        headers=HEADERS_TM, timeout=30,
                    )
                    if r2.status_code != 200:
                        break
                    tables = pd.read_html(StringIO(r2.text))
                    main = None
                    for t in tables:
                        if "Market value" in t.columns and "Player" in t.columns:
                            main = t
                            break
                    if main is None:
                        break
                    clean = main[main["#"].notna()].copy()
                    for _, row in clean.iterrows():
                        name, pos = _parse_tm_name(row["Player"])
                        all_players.append({
                            "player_name": name,
                            "age": _parse_tm_age(row.get("Date of birth/Age")),
                            "market_value_m": _parse_tm_value(row.get("Market value")),
                            "position_tm": pos,
                            "tm_league": lname,
                        })
                    print(f"  [{i+1}/{len(teams)}] ✓ {len(clean)}")
                    break
                except Exception:
                    if attempt == 0:
                        time.sleep(5)
                    else:
                        print(f"  [{i+1}/{len(teams)}] ❌")
            time.sleep(3)

    df = pd.DataFrame(all_players)
    df.to_csv(path, index=False)
    print(f"\n📊 TM: {len(df)} oyuncu → {path}")
    return df


# ══════════════════════════════════════════════════════════
# 2. UNDERSTAT (Süper Lig yok)
# ══════════════════════════════════════════════════════════
def fetch_understat():
    path = os.path.join(DATA_DIR, "understat.csv")
    if os.path.exists(path):
        print(f"✅ {path} zaten mevcut — atlanıyor.")
        return pd.read_csv(path)

    print("\n" + "=" * 50)
    print("2️⃣  UNDERSTAT")
    print("=" * 50)
    rows = []

    for season in UNDERSTAT_SEASONS:
        print(f"\n📅 {season}-{season+1}")
        for code, display in UNDERSTAT_LEAGUES.items():
            try:
                r = requests.post(
                    "https://understat.com/main/getPlayersStats/",
                    data={"league": code, "season": str(season)},
                    headers={"User-Agent": "Mozilla/5.0"}, timeout=30,
                )
                data = r.json()
                players = data.get("players", []) if data.get("success") else []
            except Exception:
                continue

            count = 0
            for p in players:
                minutes = int(p.get("time", 0))
                if minutes < MIN_MINUTES:
                    continue
                n90 = minutes / 90.0
                g = int(p.get("goals", 0))
                a = int(p.get("assists", 0))
                xG = float(p.get("xG", 0))
                xA = float(p.get("xA", 0))
                sh = int(p.get("shots", 0))
                kp = int(p.get("key_passes", 0))
                npg = int(p.get("npg", 0))
                npxG = float(p.get("npxG", 0))
                xGC = float(p.get("xGChain", 0))
                xGB = float(p.get("xGBuildup", 0))
                gm = int(p.get("games", 0))
                rows.append({
                    "player_name": p.get("player_name"), "team": p.get("team_title"),
                    "position": p.get("position"), "league": display,
                    "season": season, "games": gm, "minutes": minutes,
                    "goals": g, "assists": a, "xG": round(xG, 2), "xA": round(xA, 2),
                    "npg": npg, "npxG": round(npxG, 2), "shots": sh, "key_passes": kp,
                    "xGChain": round(xGC, 2), "xGBuildup": round(xGB, 2),
                    "goals_p90": round(g / n90, 4), "assists_p90": round(a / n90, 4),
                    "xG_p90": round(xG / n90, 4), "xA_p90": round(xA / n90, 4),
                    "npg_p90": round(npg / n90, 4), "npxG_p90": round(npxG / n90, 4),
                    "shots_p90": round(sh / n90, 4),
                    "key_passes_p90": round(kp / n90, 4),
                    "xGChain_p90": round(xGC / n90, 4),
                    "xGBuildup_p90": round(xGB / n90, 4),
                    "goal_conversion": round(g / sh, 4) if sh > 0 else 0,
                    "xG_overperf": round((g - xG) / n90, 4),
                    "xA_overperf": round((a - xA) / n90, 4),
                })
                count += 1
            print(f"  {display}: ✓ {count}")

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"\n📊 Understat: {len(df)} kayıt → {path}")
    return df


# ══════════════════════════════════════════════════════════
# 3. FOTMOB (Süper Lig dahil)
# ══════════════════════════════════════════════════════════
def fetch_fotmob():
    path = os.path.join(DATA_DIR, "fotmob.csv")
    if os.path.exists(path):
        print(f"✅ {path} zaten mevcut — atlanıyor.")
        return pd.read_csv(path)

    print("\n" + "=" * 50)
    print("3️⃣  FOTMOB")
    print("=" * 50)
    all_data = []

    for lid, league_info in FOTMOB_LEAGUES.items():
        lname = league_info["name"]
        for season_year, season_id in league_info["seasons"].items():
            print(f"⏳ {lname} {season_year}/{int(season_year)+1}", end=" ")
            players = {}

            for stat_key, stat_label in FOTMOB_STATS.items():
                url = f"https://data.fotmob.com/stats/{lid}/season/{season_id}/{stat_key}.json"
                try:
                    r = requests.get(url, headers=HEADERS_FM, timeout=15)
                    if r.status_code != 200:
                        continue
                    data = r.json()
                    top_lists = data.get("TopLists", [])
                    if not top_lists:
                        continue
                    for p in top_lists[0].get("StatList", []):
                        pid = p.get("ParticiantId")
                        if not pid:
                            continue
                        if pid not in players:
                            players[pid] = {
                                "player_id": pid,
                                "player_name": p.get("ParticipantName"),
                                "team_name": p.get("TeamName", ""),
                                "league": lname,
                                "season": int(season_year),
                            }
                        players[pid][stat_label] = p.get("StatValue")
                    time.sleep(0.3)
                except Exception:
                    pass

            all_data.extend(players.values())
            print(f"✓ {len(players)}")

    df = pd.DataFrame(all_data)
    df.to_csv(path, index=False)
    print(f"\n📊 FotMob: {len(df)} kayıt → {path}")
    return df


# ══════════════════════════════════════════════════════════
# ANA
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    fetch_transfermarkt()
    fetch_understat()
    fetch_fotmob()
    print("\n" + "=" * 50)
    print("🎉 Veri toplama tamamlandı!")
    print("=" * 50)
