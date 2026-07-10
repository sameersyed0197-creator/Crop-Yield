import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_DIR, RESULTS_DIR, PLOT_DIR, SEQUENCE_LENGTH, NUM_WEATHER_FEATURES

st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾", layout="wide")

# ── Styling ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f5f7f0; }
    h1, h2, h3 { color: #2d5a27; }
    .stMetric label { color: #5a7a52; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_fusion_model():
    import keras
    path = os.path.join(MODEL_DIR, "fusion_model.keras")
    if os.path.exists(path):
        return keras.models.load_model(path)
    return None


@st.cache_data
def load_results():
    path = os.path.join(RESULTS_DIR, "all_model_results.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None


@st.cache_data
def load_processed_data():
    from config import PROCESSED_DIR
    try:
        yields = np.load(os.path.join(PROCESSED_DIR, "yields.npy"))
        weather = np.load(os.path.join(PROCESSED_DIR, "weather.npy"))
        ndvi = np.load(os.path.join(PROCESSED_DIR, "ndvi.npy"))
        crop_labels = np.load(os.path.join(PROCESSED_DIR, "crop_labels.npy"))
        return yields, weather, ndvi, crop_labels
    except FileNotFoundError:
        return None, None, None, None


# ── Sidebar navigation ─────────────────────────────────────────────────────
page = st.sidebar.selectbox(
    "📋 Navigation",
    ["🏠 Home", "📊 EDA", "🔮 Prediction", "📈 Model Comparison", "ℹ️ About"]
)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🌾 AI-Based Crop Yield Prediction System")
    st.markdown("""
    **Architecture:** CNN (spatial) + LSTM (temporal) → Fusion → Yield Regression  
    **Goal:** Predict crop yield (tons/hectare) from satellite imagery + weather data  
    **Crops:** Wheat · Rice · Maize  
    """)

    yields, weather, ndvi, crop_labels = load_processed_data()
    if yields is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Fields", f"{len(yields):,}")
        col2.metric("Avg Yield (t/ha)", f"{yields.mean():.2f}")
        col3.metric("Max Yield (t/ha)", f"{yields.max():.2f}")
        col4.metric("Min Yield (t/ha)", f"{yields.min():.2f}")

        crop_names = {0: "Wheat", 1: "Rice", 2: "Maize"}
        df_crop = pd.DataFrame({
            "Crop": [crop_names[c] for c in crop_labels],
            "Yield (t/ha)": yields
        })
        fig = px.box(df_crop, x="Crop", y="Yield (t/ha)", color="Crop",
                     color_discrete_sequence=["#4a7c59", "#8fbc8f", "#c8a96e"],
                     title="Yield Distribution by Crop Type")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Run `python src/data_preprocessing.py` first to generate data.")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    yields, weather, ndvi, crop_labels = load_processed_data()

    if yields is None:
        st.warning("⚠️ Run data preprocessing first.")
    else:
        tab1, tab2, tab3 = st.tabs(["NDVI Seasonal Trend", "Weather Correlations", "Yield Distribution"])

        with tab1:
            ndvi_spatial = ndvi.mean(axis=(2, 3))  # (N, T)
            mean_curve = ndvi_spatial.mean(axis=0)
            std_curve  = ndvi_spatial.std(axis=0)
            weeks = list(range(1, SEQUENCE_LENGTH + 1))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=weeks, y=mean_curve, name="Mean NDVI",
                                     line=dict(color="#2d8a4e", width=2)))
            fig.add_trace(go.Scatter(
                x=weeks + weeks[::-1],
                y=list(mean_curve + std_curve) + list((mean_curve - std_curve)[::-1]),
                fill="toself", fillcolor="rgba(45,138,78,0.2)",
                line=dict(color="rgba(255,255,255,0)"), name="±1 std"
            ))
            fig.update_layout(title="Seasonal NDVI Trend", xaxis_title="Week", yaxis_title="NDVI")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            wx_mean = weather.mean(axis=1)  # (N, F)
            wx_df = pd.DataFrame(wx_mean, columns=["Rainfall", "Temp Max", "Temp Min",
                                                    "Humidity", "Solar Rad", "Soil Moisture"])
            wx_df["Yield"] = yields
            corr = wx_df.corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn",
                            title="Weather–Yield Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            crop_names = {0: "Wheat", 1: "Rice", 2: "Maize"}
            df_v = pd.DataFrame({"Crop": [crop_names[c] for c in crop_labels], "Yield": yields})
            fig = px.violin(df_v, x="Crop", y="Yield", color="Crop", box=True,
                            color_discrete_sequence=["#4a7c59", "#8fbc8f", "#c8a96e"],
                            title="Yield Distribution per Crop Type")
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediction":
    st.title("🔮 Real-Time Yield Prediction")
    model = load_fusion_model()

    if model is None:
        st.warning("⚠️ Train the fusion model first (`python src/train.py`).")
    else:
        st.markdown("Adjust the sliders to simulate field conditions:")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🌦 Weather Parameters")
            rainfall    = st.slider("Total Rainfall (mm)", 100, 800, 400)
            temp_max    = st.slider("Avg Max Temperature (°C)", 15, 40, 28)
            temp_min    = st.slider("Avg Min Temperature (°C)", 5, 30, 18)
            humidity    = st.slider("Avg Humidity (%)", 20, 100, 65)
            solar_rad   = st.slider("Avg Solar Radiation (MJ/m²)", 5, 30, 18)
            soil_moist  = st.slider("Avg Soil Moisture", 0.1, 0.6, 0.35)

        with col2:
            st.subheader("🛰 Vegetation Parameters")
            peak_ndvi   = st.slider("Peak NDVI", 0.2, 0.9, 0.65)
            mean_ndvi   = st.slider("Mean NDVI", 0.1, 0.8, 0.45)
            crop_type   = st.selectbox("Crop Type", ["Wheat", "Rice", "Maize"])

        if st.button("🌾 Predict Yield", type="primary"):
            import joblib
            # Build synthetic inputs matching model input shapes
            wx_vals = np.array([rainfall / SEQUENCE_LENGTH, temp_max, temp_min,
                                humidity, solar_rad, soil_moist], dtype=np.float32)
            wx_input = np.tile(wx_vals, (1, SEQUENCE_LENGTH, 1))  # (1, T, F)

            # Normalize weather
            wx_sc_path = os.path.join(MODEL_DIR, "weather_scaler.pkl")
            if os.path.exists(wx_sc_path):
                wx_sc = joblib.load(wx_sc_path)
                wx_flat = wx_input.reshape(-1, NUM_WEATHER_FEATURES)
                wx_input = wx_sc.transform(wx_flat).reshape(1, SEQUENCE_LENGTH, NUM_WEATHER_FEATURES).astype(np.float32)

            # Build synthetic image input
            from config import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS
            nir_val = peak_ndvi * 0.8
            red_val = nir_val * (1 - peak_ndvi) / (1 + peak_ndvi + 1e-8)
            img_input = np.zeros((1, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS + 2), dtype=np.float32)
            img_input[..., 1] = nir_val
            img_input[..., 0] = red_val
            img_input[..., -2] = peak_ndvi  # NDVI channel
            img_input[..., -1] = mean_ndvi  # EVI channel

            pred = float(model.predict([img_input, wx_input], verbose=0).flatten()[0])
            pred = max(0.5, pred)
            std_est = pred * 0.08  # ~8% confidence interval

            st.success(f"### 🌾 Predicted Yield: **{pred:.2f} ± {std_est:.2f} t/ha**")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred,
                title={"text": "Predicted Yield (t/ha)"},
                gauge={"axis": {"range": [0, 10]},
                       "bar": {"color": "#2d8a4e"},
                       "steps": [
                           {"range": [0, 3], "color": "#ffcccc"},
                           {"range": [3, 6], "color": "#ffffcc"},
                           {"range": [6, 10], "color": "#ccffcc"}
                       ]}
            ))
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Comparison":
    st.title("📈 Model Performance Comparison")
    df = load_results()

    if df is None:
        st.warning("⚠️ Run `python src/evaluate.py` first.")
    else:
        st.dataframe(df.style.highlight_min(subset=["RMSE", "MAE", "MAPE"], color="#c8f7c5")
                              .highlight_max(subset=["R2"], color="#c8f7c5"), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(df.reset_index(), x="Model", y="RMSE", color="Model",
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         title="RMSE by Model (lower is better)")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(df.reset_index(), x="Model", y="R2", color="Model",
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         title="R² Score by Model (higher is better)")
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ══════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    ## AI-Based Crop Yield Prediction using CNN-LSTM Fusion

    **Architecture:**
    - **CNN Branch:** TimeDistributed CNN encoder extracts spatial features from satellite image patches per timestep → LSTM aggregates temporal spatial features
    - **LSTM Branch:** Stacked LSTM encodes weather time-series
    - **Fusion:** Concatenation → Dense regression head → Yield (t/ha)

    **Tech Stack:**
    - `TensorFlow 2.13` · `scikit-learn` · `NumPy` · `Pandas`
    - `Streamlit` · `Plotly` · `Matplotlib` · `Seaborn`

    **Dataset:** Synthetic Sentinel-2 style multispectral imagery + weather time-series  
    **Crops:** Wheat (base 3.5 t/ha) · Rice (4.2 t/ha) · Maize (5.1 t/ha)  
    **Sequence Length:** 16 weeks · **Image Size:** 32×32 px · **Bands:** 5 + NDVI + EVI

    **Models Compared:**
    1. Standalone CNN (spatial only)
    2. Standalone LSTM (weather only)
    3. CNN-LSTM Fusion (proposed)
    4. Random Forest (baseline)
    5. SVR (baseline)
    """)

    arch_path = os.path.join(PLOT_DIR, "fusion_architecture.png")
    if os.path.exists(arch_path):
        st.image(arch_path, caption="Fusion Model Architecture", use_column_width=True)
