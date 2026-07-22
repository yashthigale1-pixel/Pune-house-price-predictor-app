"""
app.py
Pune House Price Predictor — polished Streamlit UI with custom styling,
interactive charts, and a locality map.

Run with: streamlit run app.py

NOTE: Trained on a synthetic dataset calibrated to real 2026 Pune
price-per-sqft benchmarks — NOT real transaction data. See
DATA_SOURCING.md for methodology.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from xgboost import XGBRegressor
import plotly.express as px
import plotly.graph_objects as go

# Resolve paths relative to this script's location, not the process's
# working directory. Streamlit Cloud (and some other hosts) don't
# always launch with cwd set to the repo root, which breaks bare
# relative paths like "models/preprocessor.pkl" even though they work
# fine when running `streamlit run app.py` locally from the project
# folder.
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# =====================================================================
# PAGE CONFIG + CUSTOM CSS
# =====================================================================
st.set_page_config(
    page_title="Pune House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Poppins', sans-serif; }

.stApp {
    background: linear-gradient(180deg, #0f0c29 0%, #1a1650 100%);
}

/* Hero header */
.hero {
    background: linear-gradient(120deg, #6a3093 0%, #a044ff 45%, #ff9966 100%);
    border-radius: 20px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 40px rgba(160, 68, 255, 0.35);
}
.hero h1 {
    color: white;
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero p {
    color: rgba(255,255,255,0.92);
    font-size: 1.05rem;
    margin-top: 0.4rem;
    margin-bottom: 0;
}

/* Glass cards */
.glass-card {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 16px;
    padding: 1.5rem 1.7rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1rem;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    border-radius: 18px;
    padding: 1.8rem 2rem;
    text-align: center;
    box-shadow: 0 8px 30px rgba(56, 239, 125, 0.3);
    margin: 1rem 0;
}
.result-card .price {
    font-family: 'Poppins', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: #062e29;
    margin: 0;
}
.result-card .subtext {
    color: #063d33;
    font-size: 0.95rem;
    font-weight: 500;
    margin-top: 0.3rem;
}

/* Warning banner */
.warn-banner {
    background: rgba(255, 193, 7, 0.15);
    border: 1px solid rgba(255, 193, 7, 0.4);
    border-radius: 12px;
    padding: 0.9rem 1.3rem;
    color: #ffe9a8;
    font-size: 0.9rem;
    margin-bottom: 1.2rem;
}

/* Metric-style badges */
.badge-row { display: flex; gap: 0.8rem; flex-wrap: wrap; margin-top: 0.8rem; }
.badge {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    flex: 1;
    min-width: 140px;
    text-align: center;
}
.badge .label { color: rgba(255,255,255,0.6); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; }
.badge .value { color: white; font-size: 1.25rem; font-weight: 700; font-family: 'Poppins', sans-serif; }

/* Section title */
.section-title {
    color: white;
    font-weight: 700;
    font-size: 1.3rem;
    margin: 1.5rem 0 0.8rem 0;
}

/* Text elements generally white on dark bg */
p, label, .stMarkdown, .stCaption { color: rgba(255,255,255,0.9) !important; }
.stSelectbox label, .stNumberInput label, .stSlider label { color: rgba(255,255,255,0.85) !important; font-weight: 500; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.5rem 1.2rem;
    color: white;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(120deg, #a044ff, #ff9966);
}

/* Button */
.stButton>button {
    background: linear-gradient(120deg, #a044ff 0%, #ff9966 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 1.5rem;
    font-weight: 700;
    font-size: 1rem;
    width: 100%;
    box-shadow: 0 4px 20px rgba(160, 68, 255, 0.4);
    transition: transform 0.1s ease;
}
.stButton>button:hover { transform: translateY(-2px); }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# DATA / MODEL LOADING
# =====================================================================
LOCALITY_INFO = {
    # name: (zone, lat, lon)  -- coordinates approximate, for map display only
    "Baner":            ("Premium West", 18.5601, 73.7900),
    "Aundh":            ("Premium West", 18.5590, 73.8070),
    "Kalyani Nagar":    ("Premium West", 18.5480, 73.9020),
    "Koregaon Park":    ("Premium West", 18.5362, 73.8938),
    "Boat Club Road":   ("Premium West", 18.5390, 73.8850),
    "Pashan":           ("Premium West", 18.5320, 73.7890),
    "Kharadi":          ("IT Corridor", 18.5522, 73.9436),
    "Viman Nagar":      ("IT Corridor", 18.5679, 73.9143),
    "Kothrud":          ("IT Corridor", 18.5333, 73.8514),
    "Balewadi":         ("IT Corridor", 18.5740, 73.7770),
    "Hinjewadi":        ("Mid-tier", 18.5946, 73.7185),
    "Wakad":            ("Mid-tier", 18.5975, 73.7645),
    "Bavdhan":          ("Mid-tier", 18.5089, 73.7815),
    "Ravet":            ("Mid-tier", 18.6480, 73.7550),
    "Magarpatta":       ("Mid-tier", 18.5158, 73.9280),
    "Camp":             ("Central", 18.5122, 73.8792),
    "Narayan Peth":     ("Central", 18.5150, 73.8520),
    "Somwar Peth":      ("Central", 18.5175, 73.8620),
    "Swargate":         ("Central", 18.5000, 73.8570),
    "Shivaji Nagar":    ("Central", 18.5304, 73.8474),
    "Wagholi":          ("Outskirts", 18.5679, 73.9791),
    "Moshi":            ("Outskirts", 18.6807, 73.8940),
    "Undri":            ("Outskirts", 18.4550, 73.9350),
    "Narhe":            ("Outskirts", 18.4550, 73.8080),
    "Kondhwa":          ("Outskirts", 18.4650, 73.8900),
    "Hadapsar":         ("Outskirts", 18.5089, 73.9260),
    "Chinchwad":        ("Outskirts", 18.6298, 73.7997),
    "Kiwale":           ("Outskirts", 18.6520, 73.7280),
    "NIBM Road":        ("Outskirts", 18.4690, 73.9020),
    "Sinhgad Road":     ("Outskirts", 18.4650, 73.8250),
}
ZONE_COLORS = {
    "Premium West": "#a044ff",
    "IT Corridor": "#3fbcf4",
    "Mid-tier": "#38ef7d",
    "Central": "#ffcc33",
    "Outskirts": "#ff6f61",
}

@st.cache_resource
def load_model():
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    xgb_model = XGBRegressor()
    xgb_model.load_model(str(MODELS_DIR / "xgb_model.json"))
    with open(MODELS_DIR / "final_metrics.json") as f:
        metrics = json.load(f)
    return preprocessor, xgb_model, metrics

@st.cache_data
def load_data():
    return pd.read_csv(DATA_DIR / "pune_housing.csv")

preprocessor, xgb_model, metrics = load_model()
df = load_data()

# =====================================================================
# HERO HEADER
# =====================================================================
st.markdown("""
<div class="hero">
    <h1>🏠 Pune House Price Predictor</h1>
    <p>Instant price estimates across 30 Pune localities, powered by a tuned XGBoost model</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="warn-banner">
    ⚠️ <strong>Synthetic data notice:</strong> this model is trained on a synthetic dataset
    calibrated to real 2026 Pune per-sq-ft benchmarks (NoBroker, 99acres), not actual transaction
    records. Great for exploring how location, size, and amenities drive price — not for real
    valuation. Full methodology in <code>DATA_SOURCING.md</code>.
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔮  Predict Price", "📊  Market Insights", "ℹ️  About This Model"])

# =====================================================================
# TAB 1 — PREDICT
# =====================================================================
with tab1:
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### 📍 Location")
        locality = st.selectbox("Locality", sorted(LOCALITY_INFO.keys()))
        zone = LOCALITY_INFO[locality][0]
        st.caption(f"Zone: **{zone}**")

        c1, c2 = st.columns(2)
        with c1:
            bhk = st.selectbox("BHK", [1, 2, 3, 4, 5], index=1)
            bathrooms = st.number_input("Bathrooms", min_value=1, max_value=6, value=2)
            balconies = st.number_input("Balconies", min_value=0, max_value=3, value=1)
        with c2:
            total_sqft = st.number_input("Built-up area (sq ft)", min_value=200, value=1050, step=50)
            property_type = st.selectbox("Property type", ["Apartment", "Villa", "Builder Floor"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### 🏗️ Building details")
        c3, c4 = st.columns(2)
        with c3:
            age_years = st.slider("Property age (years)", 0, 30, 5)
            floor_no = st.number_input("Floor number", min_value=0, value=3)
        with c4:
            total_floors = st.number_input("Total floors", min_value=1, value=12)
            amenities_score = st.slider("Amenities score (0-5)", 0, 5, 3)
        c5, c6 = st.columns(2)
        with c5:
            parking = st.selectbox("Parking included?", ["Yes", "No"]) == "Yes"
        with c6:
            distance_to_it_park_km = st.number_input("Distance to nearest IT park (km)", min_value=0.0, value=6.0)
        st.markdown('</div>', unsafe_allow_html=True)

        predict_clicked = st.button("✨ Predict Price", type="primary")

    with right:
        if predict_clicked:
            input_df = pd.DataFrame([{
                "locality": locality, "zone": zone, "bhk": bhk, "total_sqft": total_sqft,
                "bathrooms": bathrooms, "balconies": balconies, "age_years": age_years,
                "floor_no": floor_no, "total_floors": total_floors,
                "floor_ratio": floor_no / max(total_floors, 1),
                "sqft_per_bhk": total_sqft / bhk, "property_type": property_type,
                "amenities_score": amenities_score, "parking": int(parking),
                "distance_to_it_park_km": distance_to_it_park_km,
            }])
            transformed = preprocessor.transform(input_df)
            prediction = xgb_model.predict(transformed)[0]
            rate_per_sqft = prediction * 100000 / total_sqft
            margin = metrics["test_rmse_lakhs"]

            zone_avg = df[df["zone"] == zone]["price_lakhs"].mean()
            city_avg = df["price_lakhs"].mean()
            delta_vs_zone = (prediction - zone_avg) / zone_avg * 100

            st.markdown(f"""
            <div class="result-card">
                <div class="price">₹{prediction:.1f} Lakhs</div>
                <div class="subtext">≈ ₹{rate_per_sqft:,.0f} / sq ft &nbsp;|&nbsp; estimate range: ₹{max(prediction-margin,0):.1f} – ₹{prediction+margin:.1f} lakhs</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="badge-row">
                <div class="badge"><div class="label">vs {zone} avg</div><div class="value">{'+' if delta_vs_zone>=0 else ''}{delta_vs_zone:.0f}%</div></div>
                <div class="badge"><div class="label">vs Pune avg</div><div class="value">₹{city_avg:.0f}L avg</div></div>
                <div class="badge"><div class="label">Model R²</div><div class="value">{metrics['test_r2']:.2f}</div></div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge showing where this price sits within the zone's range
            zone_prices = df[df["zone"] == zone]["price_lakhs"]
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                number={"prefix": "₹", "suffix": "L", "font": {"color": "white", "size": 28}},
                gauge={
                    "axis": {"range": [zone_prices.min(), zone_prices.max()], "tickcolor": "white"},
                    "bar": {"color": ZONE_COLORS.get(zone, "#a044ff")},
                    "bgcolor": "rgba(255,255,255,0.05)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [zone_prices.min(), zone_prices.quantile(0.33)], "color": "rgba(255,255,255,0.08)"},
                        {"range": [zone_prices.quantile(0.33), zone_prices.quantile(0.66)], "color": "rgba(255,255,255,0.15)"},
                        {"range": [zone_prices.quantile(0.66), zone_prices.max()], "color": "rgba(255,255,255,0.22)"},
                    ],
                },
                title={"text": f"Where this sits within {zone} prices", "font": {"color": "white", "size": 14}},
            ))
            fig.update_layout(height=260, margin=dict(t=50, b=10, l=20, r=20),
                               paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center; padding: 3rem 1rem;">
                <div style="font-size: 3rem;">🔮</div>
                <p style="margin-top: 0.5rem;">Fill in the property details and hit <strong>Predict Price</strong> to see the estimate here.</p>
            </div>
            """, unsafe_allow_html=True)

# =====================================================================
# TAB 2 — MARKET INSIGHTS
# =====================================================================
with tab2:
    st.markdown('<div class="section-title">Median price by locality</div>', unsafe_allow_html=True)
    locality_medians = df.groupby(["locality", "zone"])["price_lakhs"].median().reset_index().sort_values("price_lakhs", ascending=True)
    fig_bar = px.bar(
        locality_medians, x="price_lakhs", y="locality", color="zone",
        color_discrete_map=ZONE_COLORS, orientation="h",
        labels={"price_lakhs": "Median Price (Lakhs)", "locality": "", "zone": "Zone"},
        height=700,
    )
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"}, legend={"orientation": "h", "y": -0.05},
        xaxis={"gridcolor": "rgba(255,255,255,0.1)"}, yaxis={"gridcolor": "rgba(255,255,255,0.1)"},
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">Price distribution by zone</div>', unsafe_allow_html=True)
        fig_box = px.box(
            df, x="zone", y="price_lakhs", color="zone", color_discrete_map=ZONE_COLORS,
            labels={"price_lakhs": "Price (Lakhs)", "zone": ""},
            category_orders={"zone": ["Premium West", "IT Corridor", "Mid-tier", "Central", "Outskirts"]},
        )
        fig_box.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"}, showlegend=False,
            xaxis={"gridcolor": "rgba(255,255,255,0.1)"}, yaxis={"gridcolor": "rgba(255,255,255,0.1)"},
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Price vs built-up area</div>', unsafe_allow_html=True)
        sample = df.sample(min(1500, len(df)), random_state=42)
        fig_scatter = px.scatter(
            sample, x="total_sqft", y="price_lakhs", color="bhk",
            labels={"total_sqft": "Total Sqft", "price_lakhs": "Price (Lakhs)", "bhk": "BHK"},
            color_continuous_scale="Viridis", opacity=0.6,
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "white"},
            xaxis={"gridcolor": "rgba(255,255,255,0.1)"}, yaxis={"gridcolor": "rgba(255,255,255,0.1)"},
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown('<div class="section-title">Locality map (bubble size = median price)</div>', unsafe_allow_html=True)
    map_df = locality_medians.copy()
    map_df["lat"] = map_df["locality"].map(lambda x: LOCALITY_INFO[x][1])
    map_df["lon"] = map_df["locality"].map(lambda x: LOCALITY_INFO[x][2])
    fig_map = px.scatter_map(
        map_df, lat="lat", lon="lon", size="price_lakhs", color="zone",
        color_discrete_map=ZONE_COLORS, hover_name="locality",
        hover_data={"price_lakhs": ":.0f", "lat": False, "lon": False},
        zoom=10, height=500, size_max=35,
    )
    fig_map.update_layout(
        map_style="carto-darkmatter", paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"}, margin=dict(t=10, b=10, l=10, r=10),
        legend={"orientation": "h", "y": -0.05},
    )
    st.plotly_chart(fig_map, use_container_width=True)
    st.caption("Locality coordinates are approximate, for visualization only.")

# =====================================================================
# TAB 3 — ABOUT
# =====================================================================
with tab3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f"""
##### Model performance
- **Algorithm**: tuned {metrics['best_model']}
- **Test R²**: {metrics['test_r2']:.3f} — explains ~{metrics['test_r2']*100:.0f}% of price variance
- **Test RMSE**: ₹{metrics['test_rmse_lakhs']:.1f} lakhs
- **Test MAE**: ₹{metrics['test_mae_lakhs']:.1f} lakhs

##### How the training data was built
There's no reliable, freely downloadable transaction-level dataset for
Pune real estate — several "Pune house price" files circulating online
are actually the Bangalore dataset with locality names swapped. Instead,
this dataset is synthetic, generated with a documented pricing model
where each locality's base ₹/sqft rate is drawn from **real, sourced
2026 market-rate benchmarks** (NoBroker, 99acres, Pune real-estate
reports), combined with realistic adjustments for age, floor, amenities,
property type, BHK, and distance to IT parks.

See `DATA_SOURCING.md` in the project folder for full sources and
methodology.

##### What this is useful for
Practicing an end-to-end ML regression pipeline with realistic,
India-specific feature relationships.

##### What this is NOT useful for
Actual property valuation or investment decisions. Treat the
*relationships* the model learned (location > size > property type >
age) as the useful takeaway — not the exact rupee figures.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
