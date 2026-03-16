"""
model.py — ML Model Pipeline
==============================
3 veri kaynağını birleştirir, pozisyon bazlı modeller eğitir,
oyuncuları underrated/overrated olarak sınıflar.
"""

import html as html_lib
import re
import unicodedata
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# ══════════════════════════════════════════════════════════
# SABİTLER
# ══════════════════════════════════════════════════════════

CURRENT_SEASON = 2024
MIN_MINUTES = 900
MAX_AGE = 32

UEFA_COEF = {
    "Premier League": 1.00,
    "La Liga": 0.92,
    "Serie A": 0.88,
    "Bundesliga": 0.85,
    "Ligue 1": 0.75,
    "Süper Lig": 0.58,
}

COMMON_FEATURES = ["age", "minutes", "league_coef"]

POS_FEATURES = {
    "GK": COMMON_FEATURES + [
        "save_pct", "goals_prevented",
        "accurate_passes_p90", "accurate_long_balls_p90",
    ],
    "DF": COMMON_FEATURES + [
        "tackles_p90", "interceptions_p90", "clearances_p90", "blocks_p90",
        "accurate_passes_p90", "accurate_long_balls_p90",
        "xGChain_p90_adj", "xGBuildup_p90",
        "goals_p90", "assists_p90", "xA_p90",
        "tackles_p90_trend", "interceptions_p90_trend",
    ],
    "DM": COMMON_FEATURES + [
        "tackles_p90", "interceptions_p90",
        "accurate_passes_p90", "accurate_long_balls_p90",
        "poss_won_final3rd_p90",
        "xGChain_p90_adj", "xGBuildup_p90",
        "key_passes_p90", "xA_p90", "assists_p90",
        "tackles_p90_trend", "xGChain_p90_trend",
    ],
    "MF": COMMON_FEATURES + [
        "chances_created", "big_chances_created",
        "successful_dribbles_p90", "accurate_passes_p90",
        "poss_won_final3rd_p90",
        "xA_p90", "key_passes_p90",
        "xGChain_p90", "xGBuildup_p90",
        "goals_p90", "xG_p90_adj", "npxG_p90",
        "xG_p90_trend", "xA_p90_trend",
    ],
    "FW": COMMON_FEATURES + [
        "shots_on_target_p90", "successful_dribbles_p90",
        "big_chances_created", "poss_won_final3rd_p90",
        "npxG_p90", "goal_conversion",
        "xG_p90", "goals_p90_adj",
        "xA_p90", "key_passes_p90",
        "xGChain_p90_adj", "assists_p90",
        "goals_p90_trend", "npxG_p90_trend",
    ],
}

TREND_METRICS = [
    "goals_p90", "assists_p90", "xG_p90", "xA_p90", "npxG_p90",
    "key_passes_p90", "xGChain_p90", "xGBuildup_p90",
    "tackles_p90", "interceptions_p90",
]

POS_LABELS = {
    "GK": {"en": "Goalkeepers", "tr": "Kaleciler"},
    "DF": {"en": "Defenders", "tr": "Stoper & Bekler"},
    "DM": {"en": "Defensive Midfielders", "tr": "Defansif Orta Sahalar"},
    "MF": {"en": "Midfielders & Attackers", "tr": "Orta Saha & Ofansif"},
    "FW": {"en": "Forwards & Wingers", "tr": "Forvet & Kanat"},
}

POS_COLORS = {
    "GK": "#9b59b6", "DF": "#3498db", "DM": "#1abc9c",
    "MF": "#f39c12", "FW": "#e74c3c",
}

POS_ICONS = {"GK": "🧤", "DF": "🛡️", "DM": "⚙️", "MF": "🎯", "FW": "⚡"}


# ══════════════════════════════════════════════════════════
# İSİM NORMALİZASYON
# ══════════════════════════════════════════════════════════

def normalize_name(name: str) -> str:
    name = html_lib.unescape(str(name).strip()).lower()
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = re.sub(r"[-'`]", " ", name)
    name = re.sub(r"[^a-z\s]", "", name)
    return re.sub(r"\s+", " ", name).strip()


def _build_surname_index(names):
    idx = {}
    for nn in names:
        parts = nn.split()
        sur = parts[-1] if parts else nn
        idx.setdefault(sur, []).append(nn)
    return idx


def fuzzy_match(us_nn, tm_set, tm_by_surname):
    if us_nn in tm_set:
        return us_nn

    parts = us_nn.split()
    first = parts[0] if parts else ""
    sur = parts[-1] if len(parts) > 1 else parts[0] if parts else ""

    # Soyadı aynı + isim benzer
    for cand in tm_by_surname.get(sur, []):
        tm_first = cand.split()[0] if cand.split() else ""
        score = SequenceMatcher(None, us_nn, cand).ratio()
        first_sim = SequenceMatcher(None, first, tm_first).ratio()
        if score > 0.7 and first_sim >= 0.5:
            return cand

    # Contains
    for cand in tm_by_surname.get(sur, []):
        if us_nn in cand or cand in us_nn:
            if SequenceMatcher(None, us_nn, cand).ratio() > 0.55:
                return cand

    # Truncated
    if len(parts) >= 3:
        for shorter in [" ".join(parts[:2]), f"{parts[0]} {parts[-1]}"]:
            if shorter in tm_set:
                return shorter

    # Single name contains
    if len(parts) == 1 and len(us_nn) >= 5:
        for tm_nn in tm_set:
            if us_nn in tm_nn and SequenceMatcher(None, us_nn, tm_nn).ratio() > 0.5:
                return tm_nn

    return None


# ══════════════════════════════════════════════════════════
# POZİSYON GRUPLAMA
# ══════════════════════════════════════════════════════════

def classify_position(pos: str) -> str:
    pos = str(pos).strip()
    if "GK" in pos:
        return "GK"
    has_f = "F" in pos
    has_m = "M" in pos
    has_d = "D" in pos
    if has_f and has_m:
        return "FW"
    if has_f and not has_m and not has_d:
        return "FW"
    if has_f and has_d and not has_m:
        return "DF"
    if has_m and not has_f and not has_d:
        return "MF"
    if has_d and has_m and not has_f:
        return "DM"
    if has_d and not has_m and not has_f:
        return "DF"
    return "MF"


# ══════════════════════════════════════════════════════════
# ANA PİPELINE
# ══════════════════════════════════════════════════════════

def run_pipeline(tm_path, us_path, fm_path):
    """
    End-to-end: veri yükle → birleştir → model eğit → sınıfla.

    Returns dict:
        data      : final DataFrame (tüm oyuncular + rating)
        models    : {pos_group: {"r2": float, "importance": Series}}
        stats     : genel istatistikler
    """
    # ── Veri Yükle ───────────────────────────────────────
    tm = pd.read_csv(tm_path)
    us = pd.read_csv(us_path)
    fm = pd.read_csv(fm_path)

    tm["nn"] = tm["player_name"].apply(normalize_name)
    us["nn"] = us["player_name"].apply(normalize_name)
    fm["nn"] = fm["player_name"].apply(normalize_name)

    # ── Understat pozisyon ───────────────────────────────
    if "position" in us.columns:
        us["pos_group"] = us["position"].apply(classify_position)

    # ── FotMob birleştirme (sezon bazlı) ─────────────────
    fm_stats_cols = [
        "tackles_p90", "interceptions_p90", "clearances_p90", "blocks_p90",
        "accurate_passes_p90", "accurate_long_balls_p90",
        "successful_dribbles_p90", "big_chances_created", "chances_created",
        "poss_won_final3rd_p90", "shots_on_target_p90",
        "save_pct", "goals_prevented", "fotmob_rating", "minutes_played",
    ]
    fm_merge_cols = ["nn", "season"] + [c for c in fm_stats_cols if c in fm.columns]
    fm_merge = fm[fm_merge_cols].drop_duplicates(subset=["nn", "season"], keep="first")

    # US + FM
    combined = us.merge(fm_merge, on=["nn", "season"], how="left", suffixes=("", "_fm"))

    # FotMob-only oyuncular (Süper Lig gibi Understat'ta olmayan ligler)
    us_keys = set(zip(us["nn"], us["season"]))
    fm_only = fm[~fm.apply(lambda r: (r["nn"], r["season"]) in us_keys, axis=1)].copy()

    if len(fm_only) > 0:
        # FM-only oyuncuları combined formatına dönüştür
        fm_only_rows = []
        for _, row in fm_only.iterrows():
            r = {
                "player_name": row.get("player_name"),
                "team": row.get("team_name", ""),
                "league": row.get("league", ""),
                "season": row.get("season"),
                "nn": row.get("nn"),
                "minutes": row.get("minutes_played", 0),
            }
            # FM statlarını ekle
            for c in fm_stats_cols:
                if c in row.index:
                    r[c] = row[c]
            # FM'den gelen xG/xA gibi alanları US formatına kopyala
            for c in ["goals", "assists", "xG", "xA"]:
                if c in row.index:
                    r[c] = row[c]
            # Per90 hesapla
            mins = row.get("minutes_played", 0)
            if pd.notna(mins) and mins > 0:
                n90 = mins / 90.0
                for c in ["goals", "assists", "xG", "xA"]:
                    val = row.get(c)
                    if pd.notna(val):
                        r[f"{c}_p90"] = round(val / n90, 4)
            fm_only_rows.append(r)

        fm_only_df = pd.DataFrame(fm_only_rows)
        combined = pd.concat([combined, fm_only_df], ignore_index=True)

    # ── TM birleştirme (fuzzy match) ─────────────────────
    tm_dedup = tm.drop_duplicates(subset=["nn"], keep="first")
    tm_set = set(tm_dedup["nn"].unique())
    tm_by_surname = _build_surname_index(tm_set)

    tm_match_map = {}
    for nn in combined["nn"].unique():
        match = fuzzy_match(nn, tm_set, tm_by_surname)
        if match:
            tm_match_map[nn] = match

    tm_vals = tm_dedup.set_index("nn")[["age", "market_value_m"]].to_dict("index")
    combined["age"] = combined["nn"].map(
        lambda x: tm_vals.get(tm_match_map.get(x, ""), {}).get("age")
    )
    combined["market_value_m"] = combined["nn"].map(
        lambda x: tm_vals.get(tm_match_map.get(x, ""), {}).get("market_value_m")
    )

    # ── Pozisyon gruplama ────────────────────────────────
    if "pos_group" not in combined.columns:
        combined["pos_group"] = "MF"
    combined["pos_group"] = combined["pos_group"].fillna("MF")

    # FotMob-only oyuncular için pozisyon tahmini
    # (Savunma metrikleri yüksekse DF/DM, hücum yüksekse FW/MF)
    mask_no_pos = combined["pos_group"].isna() | (combined["pos_group"] == "")
    # Basit kural: tackles yüksekse defans, goals yüksekse forvet
    # Şimdilik MF varsayalım
    combined.loc[mask_no_pos, "pos_group"] = "MF"

    # ── Lig katsayısı ───────────────────────────────────
    combined["league_coef"] = combined["league"].map(UEFA_COEF).fillna(0.70)
    combined["xGChain_p90_adj"] = combined.get("xGChain_p90", 0) * combined["league_coef"]
    combined["goals_p90_adj"] = combined.get("goals_p90", 0) * combined["league_coef"]
    combined["xG_p90_adj"] = combined.get("xG_p90", 0) * combined["league_coef"]

    # ── Trend metrikleri ─────────────────────────────────
    avail_trend = [m for m in TREND_METRICS if m in combined.columns]
    if avail_trend:
        sa = combined.groupby(["nn", "season"])[avail_trend].mean().reset_index()
        trows = []
        for _, row in combined.iterrows():
            prev = sa[(sa["nn"] == row["nn"]) & (sa["season"] == row["season"] - 1)]
            t = {}
            for m in TREND_METRICS:
                col = f"{m}_trend"
                if m in combined.columns and len(prev) > 0:
                    rv = row.get(m)
                    pv = prev.iloc[0].get(m)
                    if pd.notna(rv) and pd.notna(pv):
                        t[col] = rv - pv
                    else:
                        t[col] = 0.0
                else:
                    t[col] = 0.0
                trows.append(t)
        # Only add if not already present
        trend_cols = [f"{m}_trend" for m in TREND_METRICS]
        existing = [c for c in trend_cols if c in combined.columns]
        if not existing:
            trend_df = pd.DataFrame(trows[: len(combined)])
            combined = pd.concat([combined.reset_index(drop=True), trend_df], axis=1)

    # ── Model eğitimi ───────────────────────────────────
    current = combined[combined["season"] == CURRENT_SEASON].copy()
    model_results = {}
    all_predictions = []

    for pg, feats in POS_FEATURES.items():
        train = combined[
            (combined["pos_group"] == pg)
            & combined["market_value_m"].notna()
            & (combined["market_value_m"] > 0)
            & combined["age"].notna()
        ].copy()

        pred = current[
            (current["pos_group"] == pg)
            & current["market_value_m"].notna()
            & (current["market_value_m"] > 0)
            & current["age"].notna()
            & (current["age"] <= MAX_AGE)
        ].copy()

        if len(train) < 30:
            model_results[pg] = {"r2": 0, "n_train": len(train), "importance": None}
            continue

        for f in feats:
            if f not in train.columns:
                train[f] = 0.0
            if f not in pred.columns:
                pred[f] = 0.0

        X = train[feats].fillna(0)
        y = np.log1p(train["market_value_m"])

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        model = GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.08,
            subsample=0.8, random_state=42,
        )
        cv = cross_val_score(model, X_s, y, cv=5, scoring="r2")
        model.fit(X_s, y)

        imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)

        model_results[pg] = {
            "r2": cv.mean(),
            "r2_std": cv.std(),
            "n_train": len(train),
            "n_pred": len(pred),
            "importance": imp,
        }

        if len(pred) > 0:
            pred["predicted_m"] = np.expm1(
                model.predict(scaler.transform(pred[feats].fillna(0)))
            ).round(2)
            pred["diff_m"] = (pred["predicted_m"] - pred["market_value_m"]).round(2)
            pred["diff_pct"] = (
                pred["diff_m"] / pred["market_value_m"]
            ).round(4)
            all_predictions.append(pred)

    # ── Rating ───────────────────────────────────────────
    if all_predictions:
        final = pd.concat(all_predictions, ignore_index=True)
        upper = final["diff_pct"].quantile(0.85)
        lower = final["diff_pct"].quantile(0.15)
        final["rating"] = "Fairly Rated"
        final.loc[final["diff_pct"] > upper, "rating"] = "Underrated"
        final.loc[final["diff_pct"] < lower, "rating"] = "Overrated"
    else:
        final = pd.DataFrame()
        upper, lower = 0.15, -0.10

    # ── İstatistikler ────────────────────────────────────
    stats = {
        "total_players": len(final),
        "underrated": (final["rating"] == "Underrated").sum() if len(final) > 0 else 0,
        "fairly_rated": (final["rating"] == "Fairly Rated").sum() if len(final) > 0 else 0,
        "overrated": (final["rating"] == "Overrated").sum() if len(final) > 0 else 0,
        "avg_r2": np.mean([v["r2"] for v in model_results.values() if v["r2"] > 0]),
        "upper_threshold": upper,
        "lower_threshold": lower,
        "tm_count": len(tm),
        "us_count": len(us),
        "fm_count": len(fm),
    }

    return {
        "data": final,
        "models": model_results,
        "stats": stats,
    }
