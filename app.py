"""
app.py — Player Value Rating System
=====================================
Streamlit dashboard with The Athletic Soccer Analytics theme.
Bilingual (EN/TR). Deploy on Streamlit Cloud via GitHub.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from model import (
    run_pipeline, POS_LABELS, POS_COLORS, POS_ICONS,
    normalize_name, classify_position,
)
from translations import t

# ══════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Player Value Rating System",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════
# CUSTOM CSS — The Athletic Style
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap');

/* The Athletic: editorial serif headers + clean sans body */
.main { font-family: 'Inter', sans-serif; }

.hero-title {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
    line-height: 1.1;
    margin-bottom: 2px;
}

.hero-accent {
    color: #c4122f;
}

.hero-sub {
    font-family: 'Inter', sans-serif;
    color: #8a8f98;
    font-size: 0.95rem;
    margin-bottom: 24px;
    border-bottom: 1px solid #2a2d35;
    padding-bottom: 16px;
}

.metric-row { display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap; }

.metric-box {
    background: #1e2029;
    border: 1px solid #2a2d35;
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
    flex: 1;
    min-width: 130px;
}

.metric-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 500;
}

.metric-label {
    font-family: 'Inter', sans-serif;
    color: #8a8f98;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 2px;
}

.pos-badge {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 6px;
    display: inline-block;
    margin-bottom: 8px;
}

.model-stat {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #8a8f98;
    background: #1e2029;
    border: 1px solid #2a2d35;
    border-radius: 6px;
    padding: 6px 10px;
    display: inline-block;
    margin: 2px;
}

.player-card {
    background: #1e2029;
    border: 1px solid #2a2d35;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 12px;
}

.bmc-btn {
    display: inline-block;
    background: #FFDD00;
    color: #000000 !important;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 8px 18px;
    border-radius: 6px;
    text-decoration: none;
    margin-top: 8px;
}
.bmc-btn:hover { background: #e6c700; }

.footer-text {
    text-align: center;
    color: #555;
    font-size: 0.72rem;
    margin-top: 24px;
    padding-top: 16px;
    border-top: 1px solid #2a2d35;
}

/* Hide default streamlit hamburger */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# LANGUAGE SELECTOR
# ══════════════════════════════════════════════════════════
lang = st.sidebar.radio("🌐", ["EN", "TR"], horizontal=True, label_visibility="collapsed")
lang = lang.lower()


# ══════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════
DATA_DIR = "data"


@st.cache_data(ttl=7200, show_spinner=False)
def load_data():
    tm_path = os.path.join(DATA_DIR, "transfermarkt.csv")
    us_path = os.path.join(DATA_DIR, "understat.csv")
    fm_path = os.path.join(DATA_DIR, "fotmob.csv")

    missing = [p for p in [tm_path, us_path, fm_path] if not os.path.exists(p)]
    if missing:
        return None, missing

    results = run_pipeline(tm_path, us_path, fm_path)
    return results, []


with st.spinner("Loading data & training models..." if lang == "en" else "Veri yükleniyor & model eğitiliyor..."):
    results, missing = load_data()

if results is None:
    st.error(
        f"Missing data files: {missing}\n\n"
        "Run `python data_collector.py` first to download the data."
    )
    st.stop()

df = results["data"]
models = results["models"]
stats = results["stats"]

if df.empty:
    st.error("No data available after processing.")
    st.stop()


# ══════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════
st.markdown(
    f'<p class="hero-title">⚽ {t("title", lang)}</p>',
    unsafe_allow_html=True,
)
st.markdown(f'<p class="hero-sub">{t("subtitle", lang)}</p>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"### ⚙️ {t('filter_league', lang)}")

    all_leagues = sorted(df["league"].dropna().unique())
    leagues = st.multiselect(t("filter_league", lang), all_leagues, default=all_leagues, label_visibility="collapsed")

    age_range = st.slider(t("filter_age", lang), 17, 32, (17, 32))
    min_min = st.slider(t("filter_min_minutes", lang), 900, 3500, 900, 100)

    pos_options = [pg for pg in ["FW", "MF", "DM", "DF", "GK"] if pg in df["pos_group"].unique()]
    pos_labels_display = {
        pg: f"{POS_ICONS.get(pg, '')} {POS_LABELS.get(pg, {}).get(lang, pg)}"
        for pg in pos_options
    }
    selected_pos = st.multiselect(
        t("filter_position", lang),
        pos_options,
        default=pos_options,
        format_func=lambda x: pos_labels_display.get(x, x),
    )

    rating_map = {
        "Underrated": t("rating_underrated", lang),
        "Fairly Rated": t("rating_fairly", lang),
        "Overrated": t("rating_overrated", lang),
    }
    selected_ratings = st.multiselect(
        t("filter_rating", lang),
        list(rating_map.keys()),
        default=list(rating_map.keys()),
        format_func=lambda x: rating_map[x],
    )

    st.markdown("---")
    st.markdown(f"### 📊 {t('model_performance', lang)}")
    for pg, info in models.items():
        if info.get("r2", 0) > 0:
            icon = POS_ICONS.get(pg, "")
            label = POS_LABELS.get(pg, {}).get(lang, pg)
            r2 = info["r2"]
            color = "#00c853" if r2 > 0.6 else "#ffd700" if r2 > 0.45 else "#ff5252"
            st.markdown(
                f'<span class="model-stat">{icon} {label}: '
                f'<span style="color:{color}">R²={r2:.3f}</span></span>',
                unsafe_allow_html=True,
            )

    # Buy Me a Coffee
    st.markdown("---")
    st.markdown(f"☕ {t('bmc_text', lang)}")
    # Kullanıcı kendi BMC linkini buraya koyacak
    BMC_URL = "https://buymeacoffee.com/yourname"  # ← BUNU DEĞİŞTİR
    st.markdown(
        f'<a href="{BMC_URL}" target="_blank" class="bmc-btn">☕ Buy me a coffee</a>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════
# FILTER
# ══════════════════════════════════════════════════════════
mask = (
    df["league"].isin(leagues)
    & (df["age"] >= age_range[0])
    & (df["age"] <= age_range[1])
    & (df["minutes"] >= min_min)
    & df["pos_group"].isin(selected_pos)
    & df["rating"].isin(selected_ratings)
)
filtered = df[mask].copy()


# ══════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════
n_total = len(filtered)
n_under = (filtered["rating"] == "Underrated").sum()
n_fair = (filtered["rating"] == "Fairly Rated").sum()
n_over = (filtered["rating"] == "Overrated").sum()

st.markdown(f"""
<div class="metric-row">
    <div class="metric-box"><div class="metric-num" style="color:#4fc3f7">{n_total}</div><div class="metric-label">{t("total_players", lang)}</div></div>
    <div class="metric-box"><div class="metric-num" style="color:#00c853">{n_under}</div><div class="metric-label">{t("metric_underrated", lang)}</div></div>
    <div class="metric-box"><div class="metric-num" style="color:#ffd700">{n_fair}</div><div class="metric-label">{t("metric_fairly", lang)}</div></div>
    <div class="metric-box"><div class="metric-num" style="color:#ff5252">{n_over}</div><div class="metric-label">{t("metric_overrated", lang)}</div></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# TABLE CONFIG
# ══════════════════════════════════════════════════════════
display_cols = [
    "player_name", "age", "team", "pos_group", "league",
    "minutes", "goals", "assists", "xG", "xA",
    "market_value_m", "predicted_m", "diff_m", "diff_pct", "rating",
]
display_cols = [c for c in display_cols if c in filtered.columns]

col_config = {
    "player_name": st.column_config.TextColumn(t("col_player", lang), width="medium"),
    "age": st.column_config.NumberColumn(t("col_age", lang), format="%d"),
    "team": st.column_config.TextColumn(t("col_team", lang), width="medium"),
    "pos_group": st.column_config.TextColumn(t("col_pos", lang)),
    "league": st.column_config.TextColumn(t("col_league", lang)),
    "minutes": st.column_config.NumberColumn(t("col_minutes", lang), format="%d"),
    "goals": st.column_config.NumberColumn(t("col_goals", lang), format="%d"),
    "assists": st.column_config.NumberColumn(t("col_assists", lang), format="%d"),
    "xG": st.column_config.NumberColumn("xG", format="%.1f"),
    "xA": st.column_config.NumberColumn("xA", format="%.1f"),
    "market_value_m": st.column_config.NumberColumn(t("col_value", lang), format="€%.1f"),
    "predicted_m": st.column_config.NumberColumn(t("col_predicted", lang), format="€%.1f"),
    "diff_m": st.column_config.NumberColumn(t("col_diff", lang), format="%+.1f"),
    "diff_pct": st.column_config.NumberColumn(t("col_diff_pct", lang), format="%+.0%%"),
    "rating": st.column_config.TextColumn(t("col_rating", lang)),
}

RATING_DISPLAY = {
    "Underrated": t("rating_underrated", lang),
    "Fairly Rated": t("rating_fairly", lang),
    "Overrated": t("rating_overrated", lang),
}


# ══════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════
tabs = st.tabs([
    t("tab_underrated", lang),
    t("tab_overrated", lang),
    t("tab_all", lang),
    t("tab_position", lang),
    t("tab_charts", lang),
    t("tab_search", lang),
    t("tab_about", lang),
])

# ── Tab: Underrated ──────────────────────────────────────
with tabs[0]:
    st.markdown(f"### {t('tab_underrated', lang)}")
    st.caption(t("underrated_desc", lang))
    u = filtered[filtered["rating"] == "Underrated"].sort_values("diff_pct", ascending=False)
    show = u[display_cols].copy()
    show["rating"] = show["rating"].map(RATING_DISPLAY)
    st.dataframe(show, column_config=col_config, hide_index=True, use_container_width=True)

# ── Tab: Overrated ───────────────────────────────────────
with tabs[1]:
    st.markdown(f"### {t('tab_overrated', lang)}")
    st.caption(t("overrated_desc", lang))
    o = filtered[filtered["rating"] == "Overrated"].sort_values("diff_pct")
    show = o[display_cols].copy()
    show["rating"] = show["rating"].map(RATING_DISPLAY)
    st.dataframe(show, column_config=col_config, hide_index=True, use_container_width=True)

# ── Tab: All ─────────────────────────────────────────────
with tabs[2]:
    st.markdown(f"### {t('tab_all', lang)}")
    show = filtered.sort_values("diff_pct", ascending=False)[display_cols].copy()
    show["rating"] = show["rating"].map(RATING_DISPLAY)
    st.dataframe(show, column_config=col_config, hide_index=True, use_container_width=True)

# ── Tab: By Position ─────────────────────────────────────
with tabs[3]:
    for pg in ["FW", "MF", "DM", "DF", "GK"]:
        if pg not in selected_pos:
            continue
        pos_data = filtered[filtered["pos_group"] == pg]
        if pos_data.empty:
            continue

        info = models.get(pg, {})
        color = POS_COLORS.get(pg, "#888")
        icon = POS_ICONS.get(pg, "")
        label = POS_LABELS.get(pg, {}).get(lang, pg)
        r2 = info.get("r2", 0)

        st.markdown(
            f'<div class="pos-badge" style="background:{color}22;border:1px solid {color};color:{color}">'
            f"{icon} {label} — R²: {r2:.3f}</div>",
            unsafe_allow_html=True,
        )

        col_u, col_o = st.columns(2)
        mini = ["player_name", "age", "team", "market_value_m", "predicted_m", "diff_pct"]
        mini = [c for c in mini if c in pos_data.columns]

        with col_u:
            st.markdown(f"**{t('tab_underrated', lang)}**")
            top_u = pos_data[pos_data["rating"] == "Underrated"].sort_values("diff_pct", ascending=False).head(10)
            if len(top_u) > 0:
                st.dataframe(top_u[mini], column_config=col_config, hide_index=True, use_container_width=True)
            else:
                st.caption("—")

        with col_o:
            st.markdown(f"**{t('tab_overrated', lang)}**")
            top_o = pos_data[pos_data["rating"] == "Overrated"].sort_values("diff_pct").head(10)
            if len(top_o) > 0:
                st.dataframe(top_o[mini], column_config=col_config, hide_index=True, use_container_width=True)
            else:
                st.caption("—")

        # Feature importance expander
        imp = info.get("importance")
        if imp is not None:
            with st.expander(f"{t('chart_importance', lang)} — {label}"):
                imp_df = imp.head(8).reset_index()
                imp_df.columns = ["Feature", "Importance"]
                fig = px.bar(
                        imp_df, x="Importance", y="Feature", orientation="h",
                        color="Importance",
                        color_continuous_scale="blues",
                    )
                )
                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(autorange="reversed"),
                    showlegend=False, height=250,
                    margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig, use_container_width="stretch")
        st.markdown("")

# ── Tab: Charts ──────────────────────────────────────────
with tabs[4]:
    # Scatter
    fig1 = px.scatter(
        filtered, x="market_value_m", y="predicted_m",
        color="pos_group", symbol="rating",
        hover_name="player_name",
        hover_data={"team": True, "age": True, "league": True},
        color_discrete_map=POS_COLORS,
        labels={
            "market_value_m": t("col_value", lang),
            "predicted_m": t("col_predicted", lang),
            "pos_group": t("col_pos", lang),
        },
        title=t("chart_real_vs_pred", lang),
    )
    mx = max(
        filtered["market_value_m"].max() if len(filtered) > 0 else 1,
        filtered["predicted_m"].max() if len(filtered) > 0 else 1,
    ) * 1.1
    fig1.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx,
                   line=dict(color="white", width=1, dash="dash"))
    fig1.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                       paper_bgcolor="rgba(0,0,0,0)", height=500)
    st.plotly_chart(fig1, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        league_stats = filtered.groupby(["league", "rating"]).size().reset_index(name="count")
        fig2 = px.bar(
            league_stats, x="league", y="count", color="rating",
            color_discrete_map={
                "Underrated": "#00c853", "Fairly Rated": "#ffd700", "Overrated": "#ff5252",
            },
            title=t("chart_league_dist", lang),
            labels={"league": "", "count": ""},
        )
        fig2.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        pos_stats = filtered.groupby(["pos_group", "rating"]).size().reset_index(name="count")
        fig3 = px.bar(
            pos_stats, x="pos_group", y="count", color="rating",
            color_discrete_map={
                "Underrated": "#00c853", "Fairly Rated": "#ffd700", "Overrated": "#ff5252",
            },
            title=t("chart_pos_dist", lang),
            labels={"pos_group": "", "count": ""},
        )
        fig3.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

    # Age vs diff
    fig4 = px.scatter(
        filtered, x="age", y="diff_pct",
        color="pos_group", hover_name="player_name",
        hover_data={"team": True, "market_value_m": True},
        color_discrete_map=POS_COLORS,
        labels={"age": t("col_age", lang), "diff_pct": t("col_diff_pct", lang)},
        title=t("chart_age_diff", lang),
    )
    fig4.add_hline(y=stats.get("upper_threshold", 0.15), line_dash="dash", line_color="#00c853", opacity=0.4)
    fig4.add_hline(y=stats.get("lower_threshold", -0.10), line_dash="dash", line_color="#ff5252", opacity=0.4)
    fig4.add_hline(y=0, line_dash="solid", line_color="white", opacity=0.2)
    fig4.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                       paper_bgcolor="rgba(0,0,0,0)", height=400)
    st.plotly_chart(fig4, use_container_width=True)

# ── Tab: Search ──────────────────────────────────────────
with tabs[5]:
    search = st.text_input("", placeholder=t("search_placeholder", lang))

    if search:
        sn = normalize_name(search)
        found = filtered[
            filtered["player_name"].apply(normalize_name).str.contains(sn, na=False)
            | filtered["team"].apply(lambda x: sn in normalize_name(str(x)))
        ]

        if len(found) > 0:
            for _, p in found.iterrows():
                rating_color = {"Underrated": "#00c853", "Fairly Rated": "#ffd700", "Overrated": "#ff5252"}.get(p.get("rating", ""), "#888")
                pos_color = POS_COLORS.get(p.get("pos_group", ""), "#888")
                pos_icon = POS_ICONS.get(p.get("pos_group", ""), "")
                pos_label = POS_LABELS.get(p.get("pos_group", ""), {}).get(lang, "")
                rating_label = RATING_DISPLAY.get(p.get("rating", ""), p.get("rating", ""))

                age_val = int(p["age"]) if pd.notna(p.get("age")) else "?"
                mins_val = int(p["minutes"]) if pd.notna(p.get("minutes")) else "?"
                goals_val = int(p["goals"]) if pd.notna(p.get("goals")) else "?"
                assists_val = int(p["assists"]) if pd.notna(p.get("assists")) else "?"
                xg_val = f'{p["xG"]:.1f}' if pd.notna(p.get("xG")) else "?"
                xa_val = f'{p["xA"]:.1f}' if pd.notna(p.get("xA")) else "?"
                mv = f'€{p["market_value_m"]:.1f}M' if pd.notna(p.get("market_value_m")) else "?"
                pv = f'€{p["predicted_m"]:.1f}M' if pd.notna(p.get("predicted_m")) else "?"
                diff = f'{p["diff_pct"]*100:+.1f}%' if pd.notna(p.get("diff_pct")) else "?"

                st.markdown(f"""
                <div class="player-card">
                    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap">
                        <div>
                            <span style="font-family:'Source Serif 4',serif;font-size:1.3rem;font-weight:700">{p['player_name']}</span>
                            <span style="color:{pos_color};margin-left:8px;font-size:0.85rem">{pos_icon} {pos_label}</span>
                            <br><span style="color:#8a8f98;font-size:0.85rem">{p.get('team','')} · {p.get('league','')} · {t('col_age',lang)}: {age_val} · {mins_val} {t('col_minutes',lang).lower()}</span>
                        </div>
                        <span style="color:{rating_color};font-size:1rem;font-weight:600">{rating_label}</span>
                    </div>
                    <div style="display:flex;gap:32px;margin-top:16px;flex-wrap:wrap">
                        <div><span style="color:#8a8f98;font-size:0.75rem;text-transform:uppercase">{t('col_value',lang)}</span><br><span style="font-family:'JetBrains Mono';font-size:1.3rem">{mv}</span></div>
                        <div><span style="color:#8a8f98;font-size:0.75rem;text-transform:uppercase">{t('col_predicted',lang)}</span><br><span style="font-family:'JetBrains Mono';font-size:1.3rem;color:{rating_color}">{pv}</span></div>
                        <div><span style="color:#8a8f98;font-size:0.75rem;text-transform:uppercase">{t('col_diff_pct',lang)}</span><br><span style="font-family:'JetBrains Mono';font-size:1.3rem;color:{rating_color}">{diff}</span></div>
                        <div><span style="color:#8a8f98;font-size:0.75rem;text-transform:uppercase">{t('col_goals',lang)}/{t('col_assists',lang)}</span><br><span style="font-family:'JetBrains Mono';font-size:1.3rem">{goals_val}/{assists_val}</span></div>
                        <div><span style="color:#8a8f98;font-size:0.75rem;text-transform:uppercase">xG/xA</span><br><span style="font-family:'JetBrains Mono';font-size:1.3rem">{xg_val}/{xa_val}</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(
                "Player not found. Try adjusting filters."
                if lang == "en"
                else "Oyuncu bulunamadı. Filtreleri genişletmeyi deneyin."
            )

# ── Tab: About ───────────────────────────────────────────
with tabs[6]:
    st.markdown(t("about_content", lang))

# ── Footer ───────────────────────────────────────────────
st.markdown(f'<div class="footer-text">{t("footer", lang)}</div>', unsafe_allow_html=True)
