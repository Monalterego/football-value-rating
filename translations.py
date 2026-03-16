"""
translations.py — İki dil desteği (EN/TR)
"""

T = {
    "title": {
        "en": "Player Value Rating System",
        "tr": "Oyuncu Değer Analiz Sistemi",
    },
    "subtitle": {
        "en": "Position-based ML models · 5 seasons · Trend analysis · Performance vs Market Value",
        "tr": "Pozisyon bazlı ML modelleri · 5 sezon · Trend analizi · Performans vs Piyasa Değeri",
    },
    "tab_underrated": {"en": "🟢 Underrated", "tr": "🟢 Değerinin Altında"},
    "tab_overrated": {"en": "🔴 Overrated", "tr": "🔴 Değerinin Üstünde"},
    "tab_all": {"en": "📋 All Players", "tr": "📋 Tüm Oyuncular"},
    "tab_position": {"en": "📊 By Position", "tr": "📊 Pozisyon Bazlı"},
    "tab_charts": {"en": "📈 Charts", "tr": "📈 Grafikler"},
    "tab_search": {"en": "🔍 Search", "tr": "🔍 Oyuncu Ara"},
    "tab_about": {"en": "ℹ️ About", "tr": "ℹ️ Hakkında"},
    "filter_league": {"en": "League", "tr": "Lig"},
    "filter_age": {"en": "Age Range", "tr": "Yaş Aralığı"},
    "filter_min_minutes": {"en": "Min. Minutes", "tr": "Min. Dakika"},
    "filter_position": {"en": "Position", "tr": "Pozisyon"},
    "filter_rating": {"en": "Rating", "tr": "Sınıf"},
    "model_performance": {"en": "Model Performance", "tr": "Model Performansı"},
    "total_players": {"en": "Total Players", "tr": "Toplam Oyuncu"},
    "metric_underrated": {"en": "Underrated", "tr": "Değerinin Altında"},
    "metric_fairly": {"en": "Fairly Rated", "tr": "Doğru Değerlenmiş"},
    "metric_overrated": {"en": "Overrated", "tr": "Değerinin Üstünde"},
    "col_player": {"en": "Player", "tr": "Oyuncu"},
    "col_age": {"en": "Age", "tr": "Yaş"},
    "col_team": {"en": "Team", "tr": "Takım"},
    "col_pos": {"en": "Pos", "tr": "Poz"},
    "col_league": {"en": "League", "tr": "Lig"},
    "col_minutes": {"en": "Min", "tr": "Dk"},
    "col_goals": {"en": "Goals", "tr": "Gol"},
    "col_assists": {"en": "Assists", "tr": "Asist"},
    "col_value": {"en": "Value (M€)", "tr": "Değer (M€)"},
    "col_predicted": {"en": "Predicted (M€)", "tr": "Tahmin (M€)"},
    "col_diff": {"en": "Diff (M€)", "tr": "Fark (M€)"},
    "col_diff_pct": {"en": "Diff %", "tr": "Fark %"},
    "col_rating": {"en": "Rating", "tr": "Sınıf"},
    "search_placeholder": {"en": "Search player or team...", "tr": "Oyuncu veya takım ara..."},
    "underrated_desc": {
        "en": "Players whose on-pitch performance suggests a higher value than their current market price",
        "tr": "Saha içi performansı, mevcut piyasa değerinin üzerinde olan oyuncular",
    },
    "overrated_desc": {
        "en": "Players whose market value exceeds what their performance metrics suggest",
        "tr": "Piyasa değeri, performans metriklerinin öngördüğünün üzerinde olan oyuncular",
    },
    "chart_real_vs_pred": {
        "en": "Actual vs Predicted Market Value",
        "tr": "Gerçek vs Tahmin Edilen Piyasa Değeri",
    },
    "chart_league_dist": {
        "en": "Rating Distribution by League",
        "tr": "Lig Bazında Rating Dağılımı",
    },
    "chart_pos_dist": {
        "en": "Rating Distribution by Position",
        "tr": "Pozisyon Bazında Rating Dağılımı",
    },
    "chart_age_diff": {
        "en": "Age vs Value Difference",
        "tr": "Yaş vs Değer Farkı",
    },
    "chart_importance": {
        "en": "Feature Importance",
        "tr": "Feature Önem Sırası",
    },
    "about_title": {
        "en": "About This Project",
        "tr": "Bu Proje Hakkında",
    },
    "about_content": {
        "en": """
### What does this tool do?

This tool analyzes **on-pitch performance** of football players across Europe's top leagues and compares it to their **market value** on Transfermarkt. Using machine learning, it predicts what a player's market value *should be* based purely on their performance metrics — then flags players who are priced significantly above or below that prediction.

### How does it work?

**1. Data Collection**
We collect data from three independent sources:
- **Understat** — Advanced performance metrics: xG (expected goals), xA (expected assists), key passes, shot quality, and contribution to attacking chains. 5 seasons of data.
- **FotMob** — Defensive and technical metrics: tackles, interceptions, clearances, blocks, passing accuracy, dribbles. Covers all leagues including Süper Lig.
- **Transfermarkt** — Market values and player ages.

**2. Position-Based Models**
A goalkeeper and a striker do completely different jobs, so we evaluate them differently. We train separate machine learning models for each position group:
- **Goalkeepers**: Save %, goals prevented, passing quality
- **Defenders**: Tackles, interceptions, clearances, blocks, build-up play
- **Defensive Midfielders**: Ball recovery, passing under pressure, positional play
- **Midfielders**: Chance creation, key passes, goals, assists, dribbling
- **Forwards**: Goals, shot efficiency, xG, assists, dribbling

**3. League Difficulty Adjustment**
10 goals in the Premier League is harder than 10 goals in Ligue 1. We apply UEFA league coefficients to normalize performance across leagues.

**4. Trend Analysis**
A player improving season-over-season deserves a higher valuation. We track performance trends across 5 seasons.

**5. Rating Classification**
The model predicts what a player's market value *should be*. If the prediction is significantly higher than reality → **Underrated**. If lower → **Overrated**. The thresholds are set at the 15th and 85th percentiles of the difference distribution.

### What this is NOT
This is not investment advice. Market value is influenced by many factors our model cannot see: contract length, passport/nationality, injury history, club prestige, marketability. Our model captures ~60% of value variation through on-pitch performance alone. The remaining ~40% is the "non-performance premium" — and that's exactly what makes the underrated/overrated distinction interesting.

### Data Sources
- [Understat](https://understat.com) — xG, xA, shot data
- [FotMob](https://fotmob.com) — Comprehensive match statistics
- [Transfermarkt](https://transfermarkt.com) — Market valuations

### Model
Gradient Boosting Regressor with 5-fold cross-validation, trained on 5 seasons of data across 6 leagues.
""",
        "tr": """
### Bu araç ne yapıyor?

Bu araç, Avrupa'nın büyük liglerindeki futbolcuların **saha içi performansını** analiz edip Transfermarkt'taki **piyasa değerleriyle** karşılaştırıyor. Makine öğrenmesi kullanarak, bir oyuncunun piyasa değerinin salt performansına göre *ne olması gerektiğini* tahmin ediyor — ve bu tahminden önemli ölçüde sapan oyuncuları işaretliyor.

### Nasıl çalışıyor?

**1. Veri Toplama**
Üç bağımsız kaynaktan veri topluyoruz:
- **Understat** — İleri düzey performans metrikleri: xG (beklenen gol), xA (beklenen asist), kilit paslar, şut kalitesi ve atak zincirlerine katkı. 5 sezonluk veri.
- **FotMob** — Savunma ve teknik metrikler: top kapma, pas arası, uzaklaştırma, blok, pas isabeti, dripling. Süper Lig dahil tüm ligleri kapsıyor.
- **Transfermarkt** — Piyasa değerleri ve oyuncu yaşları.

**2. Pozisyon Bazlı Modeller**
Bir kaleci ile bir forvet tamamen farklı işler yapıyor, bu yüzden onları farklı değerlendiriyoruz. Her pozisyon grubu için ayrı makine öğrenmesi modeli eğitiyoruz:
- **Kaleciler**: Kurtarış yüzdesi, engellenen goller, pas kalitesi
- **Defans**: Top kapma, pas arası, uzaklaştırma, blok, oyun kurma
- **Defansif Orta Saha**: Top kazanma, baskı altında pas, pozisyon oyunu
- **Orta Saha**: Şans yaratma, kilit paslar, gol, asist, dripling
- **Forvet**: Gol, şut verimliliği, xG, asist, dripling

**3. Lig Zorluk Katsayısı**
Premier League'de 10 gol atmak Ligue 1'de 10 gol atmaktan daha zordur. Ligler arası performansı normalleştirmek için UEFA lig katsayılarını uyguluyoruz.

**4. Trend Analizi**
Sezondan sezona gelişen bir oyuncu daha yüksek değerlemeyi hak eder. 5 sezon boyunca performans trendlerini takip ediyoruz.

**5. Sınıflandırma**
Model, bir oyuncunun piyasa değerinin *ne olması gerektiğini* tahmin eder. Tahmin gerçekten önemli ölçüde yüksekse → **Değerinin Altında (Underrated)**. Düşükse → **Değerinin Üstünde (Overrated)**. Eşikler fark dağılımının %15 ve %85 persentillerine göre belirleniyor.

### Bu araç ne DEĞİLDİR
Bu bir yatırım tavsiyesi değildir. Piyasa değerini modelimizin göremediği birçok faktör etkiler: sözleşme süresi, pasaport/uyruk, sakatlık geçmişi, kulüp prestiji, pazarlanabilirlik. Modelimiz saha içi performans ile değer varyasyonunun ~%60'ını yakalar. Kalan ~%40 "performans dışı prim"dir — ve tam da bu, değerinin altında/üstünde ayrımını ilginç kılan şeydir.

### Veri Kaynakları
- [Understat](https://understat.com) — xG, xA, şut verisi
- [FotMob](https://fotmob.com) — Kapsamlı maç istatistikleri
- [Transfermarkt](https://transfermarkt.com) — Piyasa değerlemeleri

### Model
5 katlı çapraz doğrulama ile Gradient Boosting Regressor, 6 lig boyunca 5 sezonluk veri üzerinde eğitildi.
""",
    },
    "footer": {
        "en": "Data: Understat · FotMob · Transfermarkt | Model: Position-based Gradient Boosting · 5 seasons | ⚠️ This is an analysis tool, not investment advice.",
        "tr": "Veri: Understat · FotMob · Transfermarkt | Model: Pozisyon bazlı Gradient Boosting · 5 sezon | ⚠️ Bu bir analiz aracıdır, yatırım tavsiyesi değildir.",
    },
    "rating_underrated": {"en": "🟢 Underrated", "tr": "🟢 Değerinin Altında"},
    "rating_fairly": {"en": "🟡 Fairly Rated", "tr": "🟡 Doğru Değerlenmiş"},
    "rating_overrated": {"en": "🔴 Overrated", "tr": "🔴 Değerinin Üstünde"},
    "bmc_text": {
        "en": "If you find this useful, consider buying me a coffee!",
        "tr": "Faydalı bulduysan bir kahve ısmarlayabilirsin!",
    },
}


def t(key, lang="en"):
    """Get translated string."""
    entry = T.get(key, {})
    return entry.get(lang, entry.get("en", key))
