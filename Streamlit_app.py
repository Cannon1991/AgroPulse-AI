import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================================================================
# 1. GLOBAL PAGE CONFIGURATION & THEME STYLING
# ==============================================================================
st.set_page_config(
    page_title="AgroPulse AI | Precision Digital Agriculture MVP",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium interface CSS injections for UI/UX rendering
st.markdown("""
    <style>
    .main { background-color: #FAFAFA; }
    h1, h2, h3, h4 { color: #1B4332 !important; font-family: 'Inter', sans-serif; font-weight: 700; }
    .stButton>button { 
        background-color: #1B4332; 
        color: white; 
        border-radius: 8px; 
        font-weight: bold; 
        width: 100%;
        padding: 10px;
    }
    .stButton>button:hover { background-color: #52B788; color: white; }
    .metric-card {
        background-color: #ffffff;
        padding: 22px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        border-left: 6px solid #1B4332;
        margin-bottom: 20px;
    }
    .cta-container {
        background-color: #E8F5E9;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #C8E6C9;
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. MACHINE LEARNING ENGINE & PIPELINE INITIALIZATION
# ==============================================================================
@st.cache_resource
def train_production_model():
    """Simulates multi-spectral historical parcel telemetry data and trains the engine."""
    np.random.seed(42)
    samples = 1200
    
    # Core Hackathon & Agro Components Matrix
    route_ids = np.random.randint(101, 199, size=samples)
    route_times = np.random.uniform(0.5, 24.0, size=samples)
    ndvi_hist = np.random.uniform(0.15, 0.90, size=samples)
    rain_hist = np.random.uniform(200, 1800, size=samples)
    fert_hist = np.random.uniform(50, 300, size=samples)
    disease_hist = np.random.uniform(0.0, 1.0, size=samples)
    
    # Continuous output generation for Digital Agricultural Level Index
    y_level = (ndvi_hist * 5.2) + (rain_hist / 400) + (fert_hist / 70) - (disease_hist * 3.2)
    y_level = np.clip(y_level + np.random.normal(0, 0.15, samples), 0.5, 12.0)
    
    df = pd.DataFrame({
        'Route_ID': route_ids, 'Route_Time_Hrs': route_times, 'Satellite_NDVI': ndvi_hist,
        'Rainfall_mm': rain_hist, 'Fertilizer_kg_ha': fert_hist, 'Disease_Anomaly_Score': disease_hist,
        'Digital_Agri_Yield_Level': y_level
    })
    
    X = df.drop(columns=['Digital_Agri_Yield_Level'])
    y = df['Digital_Agri_Yield_Level']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return model, X.columns, y_test, y_pred

model, feature_cols, y_test, y_pred = train_production_model()

# ==============================================================================
# 3. INTERACTIVE SIMULATION API UTILITY MODULES
# ==============================================================================
def live_openweather_fetch(lat, lon, api_key=None):
    """Fetches diagnostic data from OpenWeather or returns a fallback array."""
    if not api_key:
        return {"temp": 28.5, "humidity": 72.0, "rain_estimate": 1150.0, "mode": "Fallback Baseline"}
    try:
        url = f"https://openweathermap.org{lat}&lon={lon}&appid={api_key}&units=metric"
        res = requests.get(url, timeout=5).json()
        main = res['list'][0]['main']
        return {"temp": main['temp'], "humidity": main['humidity'], "rain_estimate": 1200.0, "mode": "Live OpenWeather API"}
    except:
        return {"temp": 27.0, "humidity": 70.0, "rain_estimate": 1050.0, "mode": "Fallback Baseline Connection Failure"}

# ==============================================================================
# 4. APP NAVIGATION & CONTROL SIDEBAR
# ==============================================================================
st.sidebar.markdown("<h2 style='text-align: center; color: #1B4332;'>🌾 AgroPulse AI</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
workspace = st.sidebar.radio("Navigate Application Dashboard", [
    "🏠 Landing Page & Strategy", 
    "🔮 Village from Orbit (Live Demo)", 
    "📈 WarriorHacks 2.0 Evaluation Metrics"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📞 Contact & Support Matrix")
st.sidebar.info("""
**Regional Branch Hub:**  
West Africa  

**Phone Number:**  
`+2348063353863`  

**Email Address:**  
`godstimeenang4@gmail.com`
""")

# ==============================================================================
# WORKSPACE 1: HIGH-CONVERTING LANDING PAGE & STRATEGY
# ==============================================================================
if workspace == "🏠 Landing Page & Strategy":
    st.title("Village from Orbit: Global Food Security, Powered by AI 🌍")
    st.markdown("##### *Optimizing agricultural production and reducing food insecurity in Nigeria and the world at large.*")
    
    st.markdown("""
    With the global human population rapidly increasing to **eight billion people** and projected to climb even higher, 
    ensuring sustainable, high-volume food production is one of humanity's greatest challenges. 
    **AgroPulse AI** leverages satellite data, automated weather models, and predictive ML analytics to map unrecognized 
    crop parcels, intercept diseases early, and maximize yield efficiency.
    """)
    
    st.markdown("<div class='cta-container'><h3>🚀 Ready to check your field coordinates?</h3><p>Open the sidebar and jump into the <b>Village from Orbit Live Demo</b> to generate instant suitability briefs.</p></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("📋 Dashboard Strategic Aims & Objectives")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h4>🌾 1. To Improve Yields</h4>
            <p>Deploy specialized fertilizer formulations and predictive insights to dramatically step up output per hectare, often doubling historical baselines.</p>
        </div>
        <div class='metric-card'>
            <h4>♻️ 2. To Reduce Waste</h4>
            <p>Target custom micro-dosing strategies to minimize input application expenses and prevent toxic environmental runtime runoff.</p>
        </div>
        <div class='metric-card'>
            <h4>⚙️ 3. To Optimize Operations</h4>
            <p>Streamline systemic end-to-end harvesting and distribution paths based on remote canopy health mapping.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h4>🌤️ 4. To Predict the Weather to Protect Crops</h4>
            <p>Integrate automated live meteorological feeds to anticipate weather fluctuations and apply timely protective countermeasures.</p>
        </div>
        <div class='metric-card'>
            <h4>🦠 5. To Detect Disease & Secure Production</h4>
            <p>Track plant development against healthy structural growth models. Automated anomaly detection isolates diseases early, preserving local harvests.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🖼️ Interactive UI Visual Mockup Reference Index")
    st.markdown("""
    * 📱 **Digital Agriculture Mobile App Viewport** — *Visualizing field operations team executing real-time high-resolution crop scanning on smartphones.*
    * 🛰️ **Satellite Automated Parcel Boundary Mapping** — *Showing algorithmic detection drawing definitive borders across unrecognized smallholder fields.*
    * 🌤️ **Meteorological Hazard & Suitability Models** — *Layering heat maps and moisture vectors over regional agricultural coordinate blocks.*
    * 📈 **Predictive Production Yield Metrics Canvas** — *Visualizing steep line charts demonstrating doubled output metrics achieved through optimized precision protocols.*
    * 🌍 **Global Farmer Outcome Improvement Layout** — *Illustrating data streaming across emerging markets to secure high-efficiency agronomic resilience.*
    """)

# ==============================================================================
# WORKSPACE 2: VILLAGE FROM ORBIT (LIVE GENERATIVE DEMO)
# ==============================================================================
elif workspace == "🔮 Village from Orbit (Live Demo)":
    st.title("🔮 Village from Orbit: Satellite Simulation Console")
    st.markdown("Our AI simulates a satellite parcel map and generates a comprehensive crop-suitability briefing in seconds.")
    
    # Input Optional API Verification String
    api_key_input = st.text_input("OpenWeather API Key (Optional)", type="password", help="Leave blank to run on preset fallback data matrices.")
    
    with st.form("simulation_form"):
        st.markdown("### 🛠️ Configurable Target Route, Time & Field Parameters")
        c1, c2, c3 = st.columns(3)
        with c1:
            country_name = st.text_input("Target Country Context", "Nigeria")
            route_id_input = st.number_input("WarriorHacks Route ID Input", min_value=100, max_value=200, value=145)
            route_time_input = st.slider("Route Logistics Time Input (Hours)", 0.5, 24.0, 4.5)
        with c2:
            lat = st.number_input("Target Field Latitude", value=7.4912, format="%.4f")
