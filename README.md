# ⚽ Player Value Rating System

**Performance-based market value analysis for football players across Europe's top leagues.**

Uses machine learning to predict what a player's market value *should be* based on on-pitch performance metrics, then compares with actual Transfermarkt valuations to identify underrated and overrated players.

## 🔗 Live Demo
[**→ Open Dashboard**](https://your-app-name.streamlit.app)

## 📊 Data Sources
| Source | What it provides | Coverage |
|--------|-----------------|----------|
| **Understat** | xG, xA, key passes, shot quality, xGChain, xGBuildup | 5 leagues, 5 seasons |
| **FotMob** | Tackles, interceptions, clearances, blocks, passes, dribbles, saves | 6 leagues (incl. Süper Lig), 5 seasons |
| **Transfermarkt** | Market values, player ages | 6 leagues |

## 🧠 Model
- **Position-based Gradient Boosting** — separate models for GK, DF, DM, MF, FW
- **5 seasons** of training data (2020-2024)
- **UEFA league coefficients** for cross-league normalization
- **Trend analysis** — tracks performance improvement/decline
- **R² ≈ 0.60** average across positions

## 🚀 Setup

### 1. Collect Data (~15 min)
```bash
pip install -r requirements.txt
python data_collector.py
```

### 2. Run Dashboard
```bash
streamlit run app.py
```

### 3. Deploy to Streamlit Cloud
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → Select `app.py` → Deploy

## 📁 Structure
```
├── app.py                 # Streamlit dashboard
├── model.py               # ML pipeline
├── translations.py        # EN/TR language support
├── data_collector.py       # Data fetching script
├── requirements.txt       # Dependencies
├── .streamlit/config.toml # Theme config
└── data/                  # CSV data files (gitignored)
```

## ☕ Support
If you find this useful: [Buy me a coffee](https://buymeacoffee.com/yourname)

## ⚠️ Disclaimer
This is an analysis tool, not investment advice. Market value is influenced by many factors beyond on-pitch performance.
