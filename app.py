# app.py  — polished Streamlit UI for ETA prediction

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------- page + styles ----------
st.set_page_config(page_title="Food Delivery ETA", page_icon="⏱️", layout="wide")

# light CSS polish
st.markdown("""
<style>
/* center content and add breathing room */
.main > div { padding-top: 1rem; }
.card {background: #111827; border: 1px solid #1f2937; padding: 1rem 1.25rem; border-radius: 14px;}
.result {background: #064e3b; border: 1px solid #065f46; padding: 1rem 1.25rem; border-radius: 12px; color: #e7ffe7;}
.subtle {color:#9CA3AF;}
.small {font-size:0.9rem;}
hr {border: none; height: 1px; background: #1f2937; margin: .8rem 0 1.2rem;}
</style>
""", unsafe_allow_html=True)

# ---------- model loading ----------
@st.cache_resource(show_spinner=False)
def load_model(p: Path):
    return joblib.load(p)

MODEL_PATH = Path("models/eta_model_rf.pkl")
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load model from `{MODEL_PATH}`.\n\n{e}")
    st.stop()

st.title("⏱️ Food Delivery ETA Predictor")
st.markdown("<span class='subtle small'>Baseline ML model (Random Forest) with simple feature engineering.</span>", unsafe_allow_html=True)
st.markdown("<hr/>", unsafe_allow_html=True)

# ---------- sidebar: presets & info ----------
with st.sidebar:
    st.header("⚙️ Presets")
    if st.button("Short trip • Low traffic"):
        st.session_state.update({
            "Distance_km": 3.0, "Preparation_Time_min": 12, "Courier_Experience_yrs": 4.0,
            "Weather": "Clear", "Traffic_Level": "Low", "Time_of_Day": "Afternoon", "Vehicle_Type": "Bike"
        })
    if st.button("Rush hour • Long trip"):
        st.session_state.update({
            "Distance_km": 12.0, "Preparation_Time_min": 22, "Courier_Experience_yrs": 1.0,
            "Weather": "Rainy", "Traffic_Level": "High", "Time_of_Day": "Evening", "Vehicle_Type": "Scooter"
        })

    st.markdown("---")
    st.caption("**Tip:** Distance & Prep Time impact ETA most in our baseline. Traffic and Time of Day matter less but still help.")

# ---------- inputs ----------
col1, col2 = st.columns(2)

with col1:
    # remove empty extra container before Distance
    distance = st.number_input("Distance (km)", min_value=0.1, max_value=50.0, value=5.0, step=0.1, key="Distance_km")
    prep = st.number_input("Preparation Time (min)", min_value=0, max_value=120, value=20, step=1, key="Preparation_Time_min")
    exp = st.number_input("Courier Experience (yrs)", min_value=0.0, max_value=20.0, value=0.0, step=0.5, key="Courier_Experience_yrs")

with col2:
    weather = st.selectbox("Weather", ["Clear","Rainy","Snowy","Foggy","Windy","Unknown"], key="Weather")
    traffic = st.selectbox("Traffic Level", ["Low","Medium","High","Unknown"], key="Traffic_Level")
    tod = st.selectbox("Time of Day", ["Morning","Afternoon","Evening","Night","Unknown"], key="Time_of_Day")
    vehicle = st.selectbox("Vehicle Type", ["Bike","Scooter","Car"], key="Vehicle_Type")


# ---------- feature engineering (must mirror training) ----------
is_peak = 1 if tod in ["Morning", "Evening"] else 0
prep_eff = prep / (exp + 1)

row = pd.DataFrame([{
    "Distance_km": float(distance),
    "Weather": weather,
    "Traffic_Level": traffic,
    "Time_of_Day": tod,
    "Vehicle_Type": vehicle,
    "Preparation_Time_min": float(prep),
    "Courier_Experience_yrs": float(exp),
    "Is_Peak": is_peak,
    "Prep_Efficiency": prep_eff
}])

# ---------- predict ----------
left, mid, right = st.columns([1,1,2])
with left:
    do_pred = st.button("🔮 Predict ETA", use_container_width=True)

if do_pred:
    eta = float(model.predict(row)[0])

    # Optional: show a simple confidence band using your validation MAE (~6.7)
    # Adjust this if you measured a different MAE.
    APPROX_MAE = 6.7
    lo = max(0, eta - APPROX_MAE)
    hi = eta + APPROX_MAE

    st.markdown("<div class='result'>", unsafe_allow_html=True)
    st.subheader(f"Estimated delivery time: **{eta:.1f} minutes**")
    st.write(f"Expected range (±MAE): **{lo:.1f} – {hi:.1f} minutes**")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("See the features we sent to the model"):
        st.write(row)

st.markdown("<hr/>", unsafe_allow_html=True)
st.caption("Built with Streamlit • Model: RandomForestRegressor • Preprocessing: OneHotEncoder + numeric passthrough")
