import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════
st.set_page_config(
    page_title="Kisan Price Oracle",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════
# CUSTOM CSS — earthy harvest palette
# ═══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Inter:wght@400;500;600&display=swap');

/* ── Root palette ── */
:root {
  --soil:     #2C1A0E;
  --harvest:  #D47C0F;
  --wheat:    #F2C46D;
  --cream:    #FBF5E6;
  --leaf:     #3A7D44;
  --sage:     #6B9E6F;
  --mist:     #E8F4EA;
  --charcoal: #1E2428;
  --slate:    #4A5568;
}

/* ── Global ── */
html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  background-color: var(--cream);
  color: var(--charcoal);
}
[data-testid="stAppViewContainer"] {
  background: var(--cream);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(170deg, var(--soil) 0%, #3D2512 100%);
  border-right: 3px solid var(--harvest);
}
[data-testid="stSidebar"] * {
  color: var(--wheat) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label {
  color: var(--wheat) !important;
  font-weight: 500;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
  color: #C9A96E !important;
}

/* ── Sidebar selectbox/radio elements ── */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: rgba(255,255,255,0.1) !important;
  border-color: var(--harvest) !important;
  color: white !important;
}

/* ── Hero header ── */
.hero-banner {
  background: linear-gradient(135deg, var(--soil) 0%, #5C3010 50%, var(--harvest) 100%);
  border-radius: 20px;
  padding: 40px 48px;
  margin-bottom: 32px;
  position: relative;
  overflow: hidden;
}
.hero-banner::before {
  content: "🌾";
  position: absolute;
  right: 40px;
  top: 50%;
  transform: translateY(-50%);
  font-size: 120px;
  opacity: 0.15;
}
.hero-title {
  font-family: 'Playfair Display', serif;
  font-size: 2.6rem;
  font-weight: 900;
  color: var(--wheat);
  margin: 0 0 8px 0;
  line-height: 1.15;
}
.hero-sub {
  font-size: 1.05rem;
  color: #C9A96E;
  margin: 0;
  font-weight: 400;
}
.hero-badge {
  display: inline-block;
  background: rgba(212,124,15,0.3);
  border: 1px solid var(--harvest);
  border-radius: 20px;
  padding: 4px 14px;
  font-size: 0.78rem;
  color: var(--wheat);
  margin-bottom: 16px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

/* ── Stat cards ── */
.stat-row { display: flex; gap: 16px; margin-bottom: 28px; }
.stat-card {
  flex: 1;
  background: white;
  border-radius: 14px;
  padding: 20px 24px;
  border-left: 4px solid var(--harvest);
  box-shadow: 0 2px 12px rgba(44,26,14,0.08);
}
.stat-label {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--slate);
  margin-bottom: 4px;
  font-weight: 600;
}
.stat-value {
  font-family: 'Playfair Display', serif;
  font-size: 1.8rem;
  font-weight: 700;
  color: var(--soil);
}
.stat-card.green { border-left-color: var(--leaf); }
.stat-card.green .stat-value { color: var(--leaf); }

/* ── Prediction result ── */
.predict-result {
  background: linear-gradient(135deg, var(--leaf) 0%, #2A6132 100%);
  border-radius: 18px;
  padding: 36px 40px;
  text-align: center;
  margin: 20px 0;
  box-shadow: 0 8px 32px rgba(58,125,68,0.3);
}
.predict-label {
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: #A8D5B5;
  margin-bottom: 8px;
  font-weight: 600;
}
.predict-price {
  font-family: 'Playfair Display', serif;
  font-size: 3.5rem;
  font-weight: 900;
  color: white;
  line-height: 1;
}
.predict-unit {
  font-size: 1rem;
  color: #A8D5B5;
  margin-top: 6px;
}

/* ── Input card ── */
.input-card {
  background: white;
  border-radius: 16px;
  padding: 28px 32px;
  box-shadow: 0 2px 16px rgba(44,26,14,0.07);
  margin-bottom: 20px;
  border-top: 3px solid var(--harvest);
}
.section-eyebrow {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: var(--harvest);
  font-weight: 700;
  margin-bottom: 4px;
}
.section-title {
  font-family: 'Playfair Display', serif;
  font-size: 1.4rem;
  font-weight: 700;
  color: var(--soil);
  margin-bottom: 20px;
}

/* ── Model badge ── */
.model-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--mist);
  border: 1px solid var(--sage);
  border-radius: 8px;
  padding: 6px 12px;
  font-size: 0.82rem;
  color: var(--leaf);
  font-weight: 600;
  margin: 4px 2px;
  cursor: pointer;
}
.model-badge.active {
  background: var(--leaf);
  color: white;
  border-color: var(--leaf);
}

/* ── Tab styling ── */
[data-baseweb="tab-list"] {
  background: transparent;
  border-bottom: 2px solid #E2D5C3;
  gap: 4px;
}
[data-baseweb="tab"] {
  font-weight: 600;
  font-size: 0.9rem;
  color: var(--slate) !important;
  border-radius: 8px 8px 0 0;
  padding: 10px 20px;
}
[aria-selected="true"][data-baseweb="tab"] {
  color: var(--harvest) !important;
  border-bottom: 3px solid var(--harvest);
}

/* ── Chart containers ── */
.chart-wrap {
  background: white;
  border-radius: 14px;
  padding: 20px;
  box-shadow: 0 2px 12px rgba(44,26,14,0.06);
  margin-bottom: 16px;
}

/* ── Streamlit overrides ── */
.stButton > button {
  background: linear-gradient(135deg, var(--harvest), #B8680A) !important;
  color: white !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 14px 28px !important;
  font-size: 1rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.02em !important;
  box-shadow: 0 4px 16px rgba(212,124,15,0.35) !important;
  transition: all 0.2s ease !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 24px rgba(212,124,15,0.5) !important;
}

/* metrics */
[data-testid="metric-container"] {
  background: white;
  border-radius: 12px;
  padding: 12px 16px;
  border: 1px solid #E8DDD0;
  box-shadow: 0 1px 6px rgba(44,26,14,0.05);
}
[data-testid="stMetricValue"] { color: var(--soil) !important; font-weight: 700; }
[data-testid="stMetricLabel"] { color: var(--slate) !important; font-size: 0.8rem; }

/* divider */
hr { border-color: #E2D5C3; }

/* selectbox */
[data-baseweb="select"] > div {
  border-radius: 10px !important;
  border-color: #D4C4AE !important;
}
[data-baseweb="select"] > div:focus-within {
  border-color: var(--harvest) !important;
  box-shadow: 0 0 0 2px rgba(212,124,15,0.2) !important;
}

/* number input */
[data-testid="stNumberInput"] input {
  border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# DATA & MODEL (cached)
# ═══════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("agriculture.csv")
    return df.dropna().drop_duplicates()

@st.cache_resource(show_spinner="Training models — this only happens once…")
def train_all(n_rows):
    df = load_data()
    X = df.drop("Modal Price", axis=1)
    y = df["Modal Price"]
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", MinMaxScaler(), num_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    specs = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost":           XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6,
                                          random_state=42, verbosity=0),
    }
    trained, metrics, preds = {}, {}, {}
    for name, reg in specs.items():
        pipe = Pipeline([("pre", pre), ("model", reg)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        trained[name] = pipe
        preds[name]   = pred
        metrics[name] = {"MAE": mean_absolute_error(y_test, pred),
                         "R2":  r2_score(y_test, pred)}
    return trained, metrics, preds, np.array(y_test), X_test

df = load_data()
with st.spinner("Loading models…"):
    models, metrics, test_preds, y_test_arr, X_test = train_all(len(df))


# ═══════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:24px 0 8px'>
      <div style='font-size:48px'>🌾</div>
      <div style='font-family:Playfair Display,serif;font-size:1.3rem;font-weight:900;
                  color:#F2C46D;line-height:1.2'>Kisan Price<br>Oracle</div>
      <div style='font-size:0.72rem;color:#C9A96E;letter-spacing:.1em;
                  text-transform:uppercase;margin-top:6px'>India Agri Markets</div>
    </div>
    <hr style='border-color:rgba(212,124,15,.3);margin:16px 0'>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:.75rem;letter-spacing:.1em;text-transform:uppercase;"
                "color:#C9A96E;font-weight:600;margin-bottom:8px'>Choose Model</div>",
                unsafe_allow_html=True)

    model_choice = st.radio(
        "",
        list(models.keys()),
        index=1,
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:rgba(212,124,15,.3);margin:20px 0'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.75rem;letter-spacing:.1em;text-transform:uppercase;"
                "color:#C9A96E;font-weight:600;margin-bottom:12px'>Model Accuracy</div>",
                unsafe_allow_html=True)

    icons = {"Linear Regression": "📐", "Random Forest": "🌳", "XGBoost": "⚡"}
    for name, m in metrics.items():
        is_best = name == "Random Forest"
        st.markdown(f"""
        <div style='background:rgba(255,255,255,{0.12 if is_best else 0.06});
                    border:1px solid rgba(212,124,15,{0.6 if is_best else 0.2});
                    border-radius:10px;padding:10px 14px;margin-bottom:8px'>
          <div style='font-size:.82rem;font-weight:700;color:#F2C46D'>
            {icons[name]} {name} {'✅' if is_best else ''}
          </div>
          <div style='font-size:.75rem;color:#C9A96E;margin-top:4px'>
            MAE ₹{m['MAE']:,.0f} &nbsp;·&nbsp; R² {m['R2']:.4f}
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(212,124,15,.3);margin:20px 0'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:.72rem;color:#8a7060;line-height:1.6'>
      📦 {len(df):,} market records<br>
      🗺️ {df['State'].nunique()} states across India<br>
      🥕 {df['Commodity'].nunique()} commodities<br>
      🏪 {df['Market'].nunique()} mandis
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
  <div class="hero-badge">AI-Powered · India Agriculture · 23K+ Records</div>
  <div class="hero-title">Kisan Price Oracle</div>
  <p class="hero-sub">Predict commodity modal prices across Indian mandis —<br>
  powered by Linear Regression, Random Forest &amp; XGBoost.</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔮  Predict Price", "📊  Market Insights", "📈  Model Performance"])


# ════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════
with tab1:
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:10px;margin-bottom:24px'>
      <div style='font-family:Playfair Display,serif;font-size:1.6rem;
                  font-weight:700;color:#2C1A0E'>Predict Modal Price</div>
      <div style='background:#E8F4EA;border:1px solid #6B9E6F;border-radius:8px;
                  padding:3px 12px;font-size:.8rem;color:#3A7D44;font-weight:600'>
        using {model_choice}
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_loc, col_com = st.columns(2, gap="large")

    with col_loc:
        st.markdown('<div class="section-eyebrow">Location</div>'
                    '<div class="section-title">Where is the market?</div>', unsafe_allow_html=True)
        state = st.selectbox("State", sorted(df["State"].unique()), key="state")
        districts = sorted(df[df["State"] == state]["District"].unique())
        district  = st.selectbox("District", districts, key="dist")
        markets   = sorted(df[(df["State"] == state) & (df["District"] == district)]["Market"].unique())
        market    = st.selectbox("Market / Mandi", markets, key="market")
        arrival   = st.selectbox("Arrival Date", sorted(df["Arrival_Date"].unique(), reverse=True), key="date")

    with col_com:
        st.markdown('<div class="section-eyebrow">Commodity</div>'
                    '<div class="section-title">What are you selling?</div>', unsafe_allow_html=True)
        commodity = st.selectbox("Commodity", sorted(df["Commodity"].unique()), key="comm")
        varieties = sorted(df[df["Commodity"] == commodity]["Variety"].unique())
        variety   = st.selectbox("Variety", varieties, key="var")
        grade     = st.selectbox("Grade", sorted(df["Grade"].unique()), key="grade")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        pc1, pc2 = st.columns(2)
        with pc1:
            min_price = st.number_input("Min Price (₹)", min_value=0, value=2000, step=100)
        with pc2:
            max_price = st.number_input("Max Price (₹)", min_value=0, value=5000, step=100)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    predict_btn = st.button("🔮  Predict Modal Price", use_container_width=True)

    if predict_btn:
        if min_price > max_price:
            st.error("⚠️ Min Price cannot exceed Max Price.")
        else:
            inp = pd.DataFrame([{
                "State": state, "District": district, "Market": market,
                "Commodity": commodity, "Variety": variety, "Grade": grade,
                "Arrival_Date": arrival,
                "Min Price": float(min_price), "Max Price": float(max_price),
            }])
            pred = models[model_choice].predict(inp)[0]
            hist = df[df["Commodity"] == commodity]["Modal Price"]

            # Big result card
            st.markdown(f"""
            <div class="predict-result">
              <div class="predict-label">Predicted Modal Price · {model_choice}</div>
              <div class="predict-price">₹ {pred:,.0f}</div>
              <div class="predict-unit">per quintal (100 kg)</div>
            </div>
            """, unsafe_allow_html=True)

            # Context metrics
            m1, m2, m3, m4 = st.columns(4)
            delta_pct = ((pred - hist.mean()) / hist.mean() * 100)
            m1.metric("Your Prediction",  f"₹{pred:,.0f}")
            m2.metric("Market Average",   f"₹{hist.mean():,.0f}",
                      f"{delta_pct:+.1f}% vs avg")
            m3.metric("Historical Low",   f"₹{hist.min():,.0f}")
            m4.metric("Historical High",  f"₹{hist.max():,.0f}")

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            # Gauge + price band chart
            gc, bc = st.columns([1, 1], gap="large")
            with gc:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=pred,
                    delta={"reference": hist.mean(), "valueformat": ",.0f",
                           "prefix": "₹", "relative": False},
                    number={"prefix": "₹", "valueformat": ",.0f",
                            "font": {"size": 32, "color": "#2C1A0E"}},
                    title={"text": f"{commodity} · Price Gauge", "font": {"size": 13, "color": "#4A5568"}},
                    gauge={
                        "axis": {"range": [hist.min(), hist.max()],
                                 "tickprefix": "₹", "tickformat": ",.0f",
                                 "tickfont": {"size": 9}},
                        "bar":  {"color": "#3A7D44", "thickness": 0.28},
                        "bgcolor": "#F8F5F0",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [hist.min(),          hist.quantile(0.33)], "color": "#FDECC8"},
                            {"range": [hist.quantile(0.33), hist.quantile(0.66)], "color": "#F2C46D"},
                            {"range": [hist.quantile(0.66), hist.max()],          "color": "#D4884F"},
                        ],
                        "threshold": {"line": {"color": "#2C1A0E", "width": 3},
                                      "thickness": 0.8, "value": hist.mean()},
                    }
                ))
                fig_g.update_layout(height=270, margin=dict(t=50, b=10, l=20, r=20),
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_g, use_container_width=True)

            with bc:
                # Violin / box of this commodity
                fig_v = go.Figure()
                fig_v.add_trace(go.Violin(
                    y=hist, name=commodity,
                    box_visible=True, meanline_visible=True,
                    fillcolor="#6B9E6F", opacity=0.6,
                    line_color="#3A7D44"
                ))
                fig_v.add_hline(y=pred, line_dash="dash", line_color="#D47C0F", line_width=2,
                                annotation_text=f"  Your prediction ₹{pred:,.0f}",
                                annotation_font_color="#D4880F")
                fig_v.update_layout(
                    title=dict(text=f"{commodity} · Price Distribution", font=dict(size=13, color="#4A5568")),
                    yaxis_title="Modal Price (₹)", showlegend=False,
                    height=270, margin=dict(t=50, b=10, l=10, r=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(gridcolor="#EEE6D8")
                )
                st.plotly_chart(fig_v, use_container_width=True)

            # Similar listings
            similar = df[
                (df["Commodity"] == commodity) & (df["State"] == state)
            ][["Market", "Variety", "Grade", "Min Price", "Max Price", "Modal Price"]].head(6)
            if not similar.empty:
                st.markdown("<div style='font-size:.85rem;font-weight:600;"
                            "color:#4A5568;margin:12px 0 8px'>Similar recent listings in this state</div>",
                            unsafe_allow_html=True)
                st.dataframe(similar.style.format({
                    "Min Price": "₹{:,.0f}", "Max Price": "₹{:,.0f}", "Modal Price": "₹{:,.0f}"
                }), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════
# TAB 2 — INSIGHTS
# ════════════════════════════════════════════════
with tab2:
    st.markdown("<div style='font-family:Playfair Display,serif;font-size:1.6rem;"
                "font-weight:700;color:#2C1A0E;margin-bottom:24px'>Market Insights</div>",
                unsafe_allow_html=True)

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Records",    f"{len(df):,}")
    k2.metric("States",           df["State"].nunique())
    k3.metric("Commodities",      df["Commodity"].nunique())
    k4.metric("Avg Modal Price",  f"₹{df['Modal Price'].mean():,.0f}")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Row 1: Top commodities + state map
    r1a, r1b = st.columns([1.2, 1], gap="large")
    with r1a:
        top15 = (df.groupby("Commodity")["Modal Price"]
                   .mean().sort_values(ascending=False).head(15).reset_index())
        fig1 = px.bar(top15, x="Modal Price", y="Commodity", orientation="h",
                      color="Modal Price", color_continuous_scale=["#F2C46D", "#D4880F", "#8B3A0F"],
                      labels={"Modal Price": "Avg Price (₹)", "Commodity": ""},
                      title="Top 15 Commodities · Average Modal Price")
        fig1.update_layout(yaxis=dict(autorange="reversed"), showlegend=False,
                           height=420, margin=dict(t=50,b=10,l=10,r=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           title_font=dict(size=13, color="#4A5568"),
                           xaxis=dict(gridcolor="#EEE6D8"),
                           coloraxis_showscale=False)
        st.plotly_chart(fig1, use_container_width=True)

    with r1b:
        state_avg = (df.groupby("State")["Modal Price"]
                       .mean().sort_values(ascending=False).reset_index())
        fig2 = px.bar(state_avg, x="Modal Price", y="State", orientation="h",
                      color="Modal Price", color_continuous_scale=["#C8E6C9", "#3A7D44", "#1B4D22"],
                      labels={"Modal Price": "Avg Price (₹)", "State": ""},
                      title="Average Modal Price · by State")
        fig2.update_layout(yaxis=dict(autorange="reversed"), showlegend=False,
                           height=420, margin=dict(t=50,b=10,l=10,r=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           title_font=dict(size=13, color="#4A5568"),
                           xaxis=dict(gridcolor="#EEE6D8"),
                           coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2: distribution + grade box
    r2a, r2b = st.columns(2, gap="large")
    with r2a:
        fig3 = px.histogram(df, x="Modal Price", nbins=70,
                            color_discrete_sequence=["#D4880F"],
                            title="Modal Price Distribution",
                            labels={"Modal Price": "Modal Price (₹)", "count": "Frequency"})
        fig3.update_layout(height=320, margin=dict(t=50,b=10,l=10,r=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           title_font=dict(size=13, color="#4A5568"),
                           xaxis=dict(gridcolor="#EEE6D8"), yaxis=dict(gridcolor="#EEE6D8"))
        st.plotly_chart(fig3, use_container_width=True)

    with r2b:
        grade_palette = {"FAQ": "#3A7D44", "Large": "#D4880F", "Medium": "#F2C46D", "Small": "#8B3A0F"}
        fig4 = px.box(df, x="Grade", y="Modal Price",
                      color="Grade", color_discrete_map=grade_palette,
                      title="Price Range by Grade",
                      labels={"Modal Price": "Modal Price (₹)"})
        fig4.update_layout(height=320, showlegend=False, margin=dict(t=50,b=10,l=10,r=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           title_font=dict(size=13, color="#4A5568"),
                           yaxis=dict(range=[0, df["Modal Price"].quantile(0.97)],
                                      gridcolor="#EEE6D8"))
        st.plotly_chart(fig4, use_container_width=True)

    # Row 3: treemap + scatter
    r3a, r3b = st.columns([1.2, 1], gap="large")
    with r3a:
        top_comm = df["Commodity"].value_counts().head(20).reset_index()
        top_comm.columns = ["Commodity", "Count"]
        fig5 = px.treemap(top_comm, path=["Commodity"], values="Count",
                          color="Count", color_continuous_scale=["#FDECC8", "#D4880F"],
                          title="20 Most Traded Commodities (by listing count)")
        fig5.update_layout(height=340, margin=dict(t=50,b=0,l=0,r=0),
                           paper_bgcolor="rgba(0,0,0,0)",
                           title_font=dict(size=13, color="#4A5568"),
                           coloraxis_showscale=False)
        st.plotly_chart(fig5, use_container_width=True)

    with r3b:
        samp = df.sample(min(800, len(df)), random_state=7)
        fig6 = px.scatter(samp, x="Min Price", y="Max Price",
                          color="Grade", size_max=6,
                          color_discrete_map=grade_palette, opacity=0.55,
                          title="Min vs Max Price by Grade",
                          labels={"Min Price": "Min Price (₹)", "Max Price": "Max Price (₹)"})
        fig6.update_layout(height=340, margin=dict(t=50,b=10,l=10,r=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           title_font=dict(size=13, color="#4A5568"),
                           xaxis=dict(gridcolor="#EEE6D8"), yaxis=dict(gridcolor="#EEE6D8"))
        st.plotly_chart(fig6, use_container_width=True)


# ════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════
with tab3:
    st.markdown("<div style='font-family:Playfair Display,serif;font-size:1.6rem;"
                "font-weight:700;color:#2C1A0E;margin-bottom:24px'>Model Performance</div>",
                unsafe_allow_html=True)

    # Comparison table
    comp_rows = []
    for name, m in metrics.items():
        comp_rows.append({
            "Model": name,
            "MAE (₹)": f"₹{m['MAE']:,.2f}",
            "R² Score": f"{m['R2']:.4f}",
            "Accuracy": f"{m['R2']*100:.2f}%",
            "🏆": "Best" if name == "Random Forest" else ""
        })
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # MAE + R2 comparison bars
    p1, p2 = st.columns(2, gap="large")
    model_colors = {"Linear Regression": "#1565C0", "Random Forest": "#3A7D44", "XGBoost": "#E65100"}

    with p1:
        fig_mae = go.Figure()
        for name, m in metrics.items():
            fig_mae.add_trace(go.Bar(
                name=name, x=[name], y=[m["MAE"]],
                marker_color=model_colors[name],
                text=[f"₹{m['MAE']:,.0f}"], textposition="outside",
                textfont=dict(size=12, color=model_colors[name])
            ))
        fig_mae.update_layout(
            title=dict(text="MAE — lower is better", font=dict(size=13, color="#4A5568")),
            showlegend=False, height=300,
            margin=dict(t=50,b=10,l=10,r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="#EEE6D8", title="₹ Error"),
            xaxis=dict(tickfont=dict(size=11))
        )
        st.plotly_chart(fig_mae, use_container_width=True)

    with p2:
        fig_r2 = go.Figure()
        for name, m in metrics.items():
            fig_r2.add_trace(go.Bar(
                name=name, x=[name], y=[m["R2"]],
                marker_color=model_colors[name],
                text=[f"{m['R2']:.4f}"], textposition="outside",
                textfont=dict(size=12, color=model_colors[name])
            ))
        fig_r2.update_layout(
            title=dict(text="R² Score — higher is better", font=dict(size=13, color="#4A5568")),
            showlegend=False, height=300,
            margin=dict(t=50,b=10,l=10,r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="#EEE6D8", range=[0.97, 1.0]),
            xaxis=dict(tickfont=dict(size=11))
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    # Per-model deep dive
    st.markdown("<hr>", unsafe_allow_html=True)
    selected_eval = st.selectbox("Deep-dive into model", list(models.keys()))
    pred_ev = test_preds[selected_eval]
    col_ev = model_colors[selected_eval]

    e1, e2 = st.columns(2, gap="large")
    with e1:
        samp_idx = np.random.default_rng(42).choice(len(y_test_arr), size=min(600, len(y_test_arr)), replace=False)
        fig_sc = px.scatter(
            x=y_test_arr[samp_idx], y=pred_ev[samp_idx],
            labels={"x": "Actual Price (₹)", "y": "Predicted Price (₹)"},
            title=f"Actual vs Predicted · {selected_eval}",
            opacity=0.45, color_discrete_sequence=[col_ev]
        )
        mn = min(y_test_arr[samp_idx].min(), pred_ev[samp_idx].min())
        mx = max(y_test_arr[samp_idx].max(), pred_ev[samp_idx].max())
        fig_sc.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                         line=dict(color="red", dash="dot", width=2))
        fig_sc.update_layout(height=340, margin=dict(t=50,b=10,l=10,r=10),
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             title_font=dict(size=13, color="#4A5568"),
                             xaxis=dict(gridcolor="#EEE6D8"), yaxis=dict(gridcolor="#EEE6D8"))
        st.plotly_chart(fig_sc, use_container_width=True)

    with e2:
        residuals = y_test_arr - pred_ev
        fig_res = px.histogram(
            x=residuals, nbins=60,
            labels={"x": "Residual (Actual − Predicted) ₹", "count": "Frequency"},
            title=f"Residual Distribution · {selected_eval}",
            color_discrete_sequence=[col_ev]
        )
        fig_res.add_vline(x=0, line_dash="dot", line_color="red", line_width=2)
        fig_res.update_layout(height=340, margin=dict(t=50,b=10,l=10,r=10),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              title_font=dict(size=13, color="#4A5568"),
                              xaxis=dict(gridcolor="#EEE6D8"), yaxis=dict(gridcolor="#EEE6D8"))
        st.plotly_chart(fig_res, use_container_width=True)


# ═══════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════
st.markdown("""
<div style='text-align:center;padding:32px 0 16px;color:#A09080;font-size:.8rem;
            border-top:1px solid #E2D5C3;margin-top:40px'>
  🌾 <strong>Kisan Price Oracle</strong> &nbsp;·&nbsp;
  Built with Streamlit &amp; scikit-learn &nbsp;·&nbsp;
  Linear Regression · Random Forest · XGBoost
</div>
""", unsafe_allow_html=True)
