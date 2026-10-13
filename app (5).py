import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="AgroPulse AI Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.5rem; color: #1e5631; font-weight: bold; margin-bottom: 5px; }
    .sub-title { font-size: 1.1rem; color: #4c9a2a; margin-bottom: 25px; }
    .metric-card { background-color: #f1f8e9; padding: 15px; border-radius: 8px; border-left: 5px solid #4c9a2a; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Core Aims Data
# ----------------------------------------------------
aims = [
    "🚀 To improve yields: Scale crop production per hectare using data-driven fertilizer and soil intelligence.",
    "♻️ To reduce waste: Cut resource over-allocation (water, chemicals, labor) via precision zoning.",
    "⚙️ To optimize operations in that process: Streamline farm logistics and real-time field monitoring from orbit.",
    "🌤️ To predict the weather to protect the crops: Forecast micro-climate variations and storms for preemptive defenses.",
    "🛡️ To detect disease and take appropriate measures: Track plant biological evolution automatically to flag infections early."
]

# ----------------------------------------------------
# Sidebar Controls & Inputs
# ----------------------------------------------------
st.sidebar.image("https://images.unsplash.com/photo-1500937386664-56d1dfef3854?auto=format&fit=crop&q=80&w=200", caption="AgroPulse AI Orbit Tracking")
st.sidebar.header("🌍 Location & Settings")

country_list = ["Nigeria", "Ghana", "Kenya", "South Africa", "United States", "India", "Brazil", "France", "Australia", "Canada"]
selected_country = st.sidebar.selectbox("Select Target Country / Region", country_list)

st.sidebar.markdown("---")
st.sidebar.header("⏱️ WarriorHacks 2.0 Route Config")
route_time_input = st.sidebar.slider("Historical Data Route/Time Level", min_value=1.0, max_value=12.0, value=6.5, step=0.5)

# ----------------------------------------------------
# Main Dashboard Layout
# ----------------------------------------------------
st.markdown("<div class='main-title'>🌱 AgroPulse AI Platform</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Satellite Data, Weather Models, and Predictive Analytics for Global Food Security</div>", unsafe_allow_html=True)

# Core Aims Layout Component
with st.expander("📌 Core Platform Aims & Directives", expanded=True):
    for aim in aims:
        st.write(aim)

# ----------------------------------------------------
# Simulation Data Engine
# ----------------------------------------------------
@st.cache_data
def generate_base_data():
    np.random.seed(42)
    n_samples = 2000
    data = {
        'rainfall_mm': np.random.normal(1200, 300, n_samples).clip(300, 2500),
        'ndvi_satellite_index': np.random.uniform(0.2, 0.9, n_samples),
        'soil_nitrogen_ppm': np.random.normal(45, 15, n_samples).clip(10, 90),
        'temperature_avg_c': np.random.normal(26, 4, n_samples).clip(15, 40),
        'fertilizer_kg_ha': np.random.normal(120, 40, n_samples).clip(20, 300),
        'route_time_index': np.random.uniform(1, 12, n_samples)
    }
    df = pd.DataFrame(data)
    df['yield_t_ha'] = (
        0.002 * df['rainfall_mm'] +
        3.5 * df['ndvi_satellite_index'] +
        0.04 * df['soil_nitrogen_ppm'] -
        0.05 * np.abs(df['temperature_avg_c'] - 25) +
        0.008 * df['fertilizer_kg_ha'] +
        np.random.normal(0, 0.3, n_samples)
    ).clip(0.5, 12.0)
    return df

df = generate_base_data()

# Live Model Training Loop
X = df[['rainfall_mm', 'ndvi_satellite_index', 'soil_nitrogen_ppm', 'temperature_avg_c', 'fertilizer_kg_ha', 'route_time_index']]
y = df['yield_t_ha']
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X, y)

# ----------------------------------------------------
# Real-Time Interactive Parcel Briefing
# ----------------------------------------------------
st.markdown("### 🛰️ Live Orbit Parcel Simulation & Suitability Briefing")
col1, col2, col3 = st.columns(3)

with col1:
    sim_ndvi = st.slider("Live Satellite NDVI Index (Crop Biomass)", 0.0, 1.0, 0.65, 0.05)
    sim_rain = st.slider("Microclimate Rainfall Model (mm)", 300, 2500, 1400, 50)
with col2:
    sim_nitro = st.slider("Soil Nitrogen Concentration (ppm)", 10, 90, 50, 5)
    sim_temp = st.slider("Predictive Weather Forecast Temp (°C)", 15, 40, 28, 1)
with col3:
    sim_fert = st.slider("Optimized Fertilizer Allocation (kg/ha)", 20, 300, 150, 10)
    
    # Run dynamic inference
    input_features = pd.DataFrame([{
        'rainfall_mm': sim_rain,
        'ndvi_satellite_index': sim_ndvi,
        'soil_nitrogen_ppm': sim_nitro,
        'temperature_avg_c': sim_temp,
        'fertilizer_kg_ha': sim_fert,
        'route_time_index': route_time_input
    }])
    
    predicted_yield = model.predict(input_features)[0]
    
    # Simple rule-based early disease discovery simulation
    expected_ndvi_for_rain = 0.3 + (sim_rain / 5000)
    is_anomaly = sim_ndvi < (expected_ndvi_for_rain - 0.15)
    
    st.markdown("#### **Prediction Insights**")
    st.metric(label=f"Predicted Crop Yield ({selected_country})", value=f"{predicted_yield:.2f} Tons/Ha")

# ----------------------------------------------------
# Notification Panel / Early Disease Discovery
# ----------------------------------------------------
if is_anomaly:
    st.error("🚨 **Early Disease Discovery Alert:** Crop biological development is significantly below historical baseline parameters for this local rainfall pattern. Immediate scouting recommended to secure food production!")
else:
    st.success("✅ **Parcel Status Stable:** Crop biological growth tracks perfectly against healthy model metrics.")

st.markdown("---")

# ----------------------------------------------------
# Analytics & Model Performance Section
# ----------------------------------------------------
st.markdown("### 📊 Platform Core Metrics & Evaluation Engine")
m_col1, m_col2, m_col3 = st.columns(3)

# Predictions vs actual set for performance metrics calculation
y_pred_all = model.predict(X)
mae = mean_absolute_error(y, y_pred_all)
rmse = np.sqrt(mean_squared_error(y, y_pred_all))
r2 = r2_score(y, y_pred_all)

with m_col1:
    st.metric("Mean Absolute Error (MAE)", f"{mae:.4f} t/ha")
with m_col2:
    st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f} t/ha")
with m_col3:
    st.metric("R² Score Accuracy", f"{r2:.4f}")

# Chart Feature
fig, ax = plt.subplots(figsize=(6, 3))
ax.scatter(y, y_pred_all, alpha=0.3, color='#2e7d32', s=5)
ax.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='#d32f2f', lw=1.5)
ax.set_xlabel('Actual Historical Yield (Tons/Ha)', fontsize=8)
ax.set_ylabel('Predicted Platform Yield (Tons/Ha)', fontsize=8)
ax.set_title('AgroPulse AI Validation Space', fontsize=10)
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# ----------------------------------------------------
# Footer / Contact Info
# ----------------------------------------------------
st.markdown("---")
st.markdown("""
🌐 **AgroPulse AI Hub** | **Region:** West Africa  
📞 **Phone:** +2348063353863 | 📧 **Email:** godstimeenang4@gmail.com  
_Supporting food security for a growing global population._
""")
