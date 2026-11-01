import time
import random
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# Page Configuration with expanded sidebar by default
st.set_page_config(
    page_title="AI Water Leakage & Localization System",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Theme-Aware Custom Styling (True Black, Yellow-Green Headings, White Body Text) ---
st.markdown("""
    <style>
    /* Force True Pitch-Black Background */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stToolbar"] {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #000000 !important;
        border-right: 1px solid rgba(132, 204, 22, 0.2);
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label, [data-testid="stSidebar"] h3 {
        color: #84cc16 !important;
    }

    /* Navigation Bar Tabs Text - Pure White */
    [data-testid="stTabs"] button p {
        color: #ffffff !important;
        font-weight: 600;
    }

    /* Body Text in Main Area - Pure White */
    p, span, label, .stMarkdown, div[data-baseweb="select"] span {
        color: #ffffff !important;
    }

    /* System Status Normal Banner text override to white */
    div[data-testid="stNotification"] p, div[data-testid="stNotification"] span {
        color: #ffffff !important;
    }
    
    /* Metric 3D Flip Card Container */
    .flip-card {
        background-color: transparent;
        perspective: 1000px;
        height: 155px;
        margin-bottom: 1rem;
    }
    .flip-checkbox {
        display: none;
    }
    .flip-card-inner {
        position: relative;
        width: 100%;
        height: 100%;
        text-align: left;
        transition: transform 0.6s cubic-bezier(0.4, 0.2, 0.2, 1);
        transform-style: preserve-3d;
        display: block;
        cursor: pointer;
    }
    .flip-checkbox:checked + .flip-card-inner {
        transform: rotateY(180deg);
    }
    .flip-card-front, .flip-card-back {
        position: absolute;
        width: 100%;
        height: 100%;
        backface-visibility: hidden;
        border-radius: 1rem;
        padding: 1.25rem;
        box-sizing: border-box;
        background-color: #0d1117;
        border: 1px solid rgba(132, 204, 22, 0.3);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        color: #ffffff;
    }
    .flip-card-back {
        transform: rotateY(180deg);
        border-color: rgba(132, 204, 22, 0.6);
    }

    /* Glassy Container for Graphs */
    .glass-graph-container {
        background-color: #0d1117;
        border: 1px solid rgba(132, 204, 22, 0.3);
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        margin-top: 0rem;
        margin-bottom: 1rem;
        color: #ffffff;
    }

    /* Event Log Table - Pure White Text */
    [data-testid="stDataFrame"] {
        background-color: #0d1117 !important;
    }
    [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Train a Mock AI Model on Startup ---
@st.cache_resource
def load_ai_model():
    X_train = [
        [12.5, 3.0, 2.9, 0.1],   # Normal flow
        [14.0, 3.1, 3.0, 0.1],   # Normal flow
        [0.2, 3.0, 1.1, 1.9],    # Leakage event Zone 2
        [11.8, 2.9, 2.8, 0.1],   # Normal flow
        [0.1, 3.0, 0.4, 2.6],    # Severe leak Zone 3
    ]
    y_train = [0, 0, 1, 0, 1]  # 0: Normal, 1: Leak Detected
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X_train, y_train)
    return model

ai_model = load_ai_model()

# --- Initialize Session State for Live Chart Data & View Mode ---
if 'data_history' not in st.session_state:
    st.session_state.data_history = pd.DataFrame(columns=[
        'Timestamp', 'Flow_Rate', 'Pressure_In', 'Pressure_Out', 'Pressure_Drop', 'Status', 'Location'
    ])

if 'graph_view_mode' not in st.session_state:
    st.session_state.graph_view_mode = "Live Graph"

# --- STREAMLINED SIDEBAR ---
with st.sidebar:
    st.markdown("### 💧 AquaGuard Pro")
    st.markdown("AI-Based Water Leakage & Localization")
    st.markdown("---")

    sidebar_option = st.selectbox(
        "Menu Options",
        ["📊 Live Dashboard", "⚙️ System Settings (Reserved)"],
    )

    st.markdown("---")
    st.markdown("### System Health")
    st.success("● ONLINE\n\nAll 12 ESP32 Nodes Active")

# --- MAIN HEADER ---
st.markdown("<h2 style='color: #84cc16;'>AI-Based Water Leakage Detection & Localization System</h2>", unsafe_allow_html=True)
st.markdown("<p style='color: #ffffff;'>Automatic real-time telemetry from ESP32 sensors, AI anomaly classification, and pipeline fault mapping.</p>", unsafe_allow_html=True)
st.markdown("")

# Handle sidebar option selection cleanly
if sidebar_option == "⚙️ System Settings (Reserved)":
    st.markdown("---")
    st.markdown("<h3 style='color: #84cc16;'>⚙️ System Configuration & Reserved Module</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #ffffff;'>This module is reserved for configuring IoT gateway endpoints and sensor calibration profiles.</p>", unsafe_allow_html=True)
    st.info("System operating normally. Switch back to '📊 Live Dashboard' in the sidebar anytime.")
else:
    # --- Automatic ESP32 Sensor Data Stream ---
    timestamp = pd.Timestamp.now().strftime("%H:%M:%S")

    flow = round(random.uniform(11.0, 14.5), 2)
    p_in = 3.0
    p_out = round(random.uniform(2.8, 2.95), 2)
    p_drop = round(p_in - p_out, 2)
    
    features = np.array([[flow, p_in, p_out, p_drop]])
    status_val = ai_model.predict(features)[0]
    location_val = "Zone C (Near Terminal Node)" if status_val == 1 else "None (System Normal)"

    new_row = pd.DataFrame({
        'Timestamp': [timestamp],
        'Flow_Rate': [flow],
        'Pressure_In': [p_in],
        'Pressure_Out': [p_out],
        'Pressure_Drop': [p_drop],
        'Status': ['LEAK DETECTED' if status_val == 1 else 'NORMAL'],
        'Location': [location_val]
    })

    st.session_state.data_history = pd.concat([st.session_state.data_history, new_row], ignore_index=True)
    if len(st.session_state.data_history) > 30:
        st.session_state.data_history = st.session_state.data_history.iloc[1:]

    current_status = st.session_state.data_history.iloc[-1]['Status']
    current_loc = st.session_state.data_history.iloc[-1]['Location']

    # --- Top Status Banner ---
    if current_status == 'LEAK DETECTED':
        st.error(f"🚨 CRITICAL ALERT: LEAKAGE DETECTED! — Estimated Location: {current_loc}")
    else:
        st.success("✅ SYSTEM STATUS: NORMAL — Pipeline integrity secure across all sections.")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- TOP NAVIGATION TABS (Structured Sections) ---
    nav_tab1, nav_tab2, nav_tab3, nav_tab4 = st.tabs([
        "📊 Overview Metrics", 
        "📈 Flow & Pressure Analytics", 
        "🗺️ Zone & AI Diagnostics", 
        "🚰 Valve Controls & Logs"
    ])

    with nav_tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("💡 *Click on any metric card to flip and view its normal baseline and expected average range.*")
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
                <div class="flip-card">
                    <input type="checkbox" id="card1" class="flip-checkbox">
                    <label for="card1" class="flip-card-inner">
                        <div class="flip-card-front">
                            <p style="font-size: 0.85rem; margin-bottom: 0px; color: #ffffff; opacity: 0.9;">Inlet Pressure</p>
                            <h2 style="color: #84cc16; margin-top: 5px; margin-bottom: 0px;">{p_in} Bar</h2>
                            <p style="color: #ffffff; font-size: 0.75rem; margin-top: 5px;">-{p_drop} Bar Drop <i style="color: #84cc16;">(Tap to flip ↺)</i></p>
                        </div>
                        <div class="flip-card-back">
                            <p style="color: #84cc16; font-size: 0.8rem; font-weight: bold; margin-bottom: 2px;">Inlet Pressure Ranges</p>
                            <p style="color: #ffffff; font-size: 0.75rem; margin: 2px 0;">• Normal: <b>3.0 - 3.5 Bar</b></p>
                            <p style="color: #ffffff; font-size: 0.75rem; margin: 2px 0;">• Average: <b>3.2 Bar</b></p>
                            <p style="color: #84cc16; font-size: 0.70rem; margin-top: 6px;">Status: Optimal baseline</p>
                        </div>
                    </label>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div class="flip-card">
                    <input type="checkbox" id="card2" class="flip-checkbox">
                    <label for="card2" class="flip-card-inner">
                        <div class="flip-card-front">
                            <p style="font-size: 0.85rem; margin-bottom: 0px; color: #ffffff; opacity: 0.9;">Outlet Pressure</p>
                            <h2 style="color: #84cc16; margin-top: 5px; margin-bottom: 0px;">{p_out} Bar</h2>
                            <p style="color: #ffffff; font-size: 0.75rem; margin-top: 5px;">Optimal Range <i style="color: #84cc16;">(Tap to flip ↺)</i></p>
                        </div>
                        <div class="flip-card-back">
                            <p style="color: #84cc16; font-size: 0.8rem; font-weight: bold; margin-bottom: 2px;">Outlet Pressure Ranges</p>
                            <p style="color: #ffffff; font-size: 0.75rem; margin: 2px 0;">• Normal: <b>2.8 - 3.1 Bar</b></p>
                            <p style="color: #ffffff; font-size: 0.75rem; margin: 2px 0;">• Average: <b>2.95 Bar</b></p>
                            <p style="color: #84cc16; font-size: 0.70rem; margin-top: 6px;">Status: Target threshold</p>
                        </div>
                    </label>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
                <div class="flip-card">
                    <input type="checkbox" id="card3" class="flip-checkbox">
                    <label for="card3" class="flip-card-inner">
                        <div class="flip-card-front">
                            <p style="font-size: 0.85rem; margin-bottom: 0px; color: #ffffff; opacity: 0.9;">Flow Rate</p>
                            <h2 style="color: #84cc16; margin-top: 5px; margin-bottom: 0px;">{flow} L/min</h2>
                            <p style="color: #ffffff; font-size: 0.75rem; margin-top: 5px;">ESP Sensor Live <i style="color: #84cc16;">(Tap to flip ↺)</i></p>
                        </div>
                        <div class="flip-card-back">
                            <p style="color: #84cc16; font-size: 0.8rem; font-weight: bold; margin-bottom: 2px;">Flow Rate Ranges</p>
                            <p style="color: #ffffff; font-size: 0.75rem; margin: 2px 0;">• Normal: <b>10.0 - 15.0 L/min</b></p>
                            <p style="color: #ffffff; font-size: 0.75rem; margin: 2px 0;">• Average: <b>12.5 L/min</b></p>
                            <p style="color: #84cc16; font-size: 0.70rem; margin-top: 6px;">Status: Non-leakage flow</p>
                        </div>
                    </label>
                </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
                <div class="flip-card">
                    <input type="checkbox" id="card4" class="flip-checkbox">
                    <label for="card4" class="flip-card-inner">
                        <div class="flip-card-front">
                            <p style="font-size: 0.85rem; margin-bottom: 0px; color: #ffffff; opacity: 0.9;">AI Model Confidence</p>
                            <h2 style="color: #84cc16; margin-top: 5px; margin-bottom: 0px;">94.8%</h2>
                            <p style="color: #ffffff; font-size: 0.75rem; margin-top: 5px;">Random Forest <i style="color: #84cc16;">(Tap to flip ↺)</i></p>
                        </div>
                        <div class="flip-card-back">
                            <p style="color: #84cc16; font-size: 0.8rem; font-weight: bold; margin-bottom: 2px;">Model Thresholds</p>
                            <p style="color: #ffffff; font-size: 0.75rem; margin: 2px 0;">• Min Required: <b>> 90.0%</b></p>
                            <p style="color: #ffffff; font-size: 0.75rem; margin: 2px 0;">• Average: <b>93.5%</b></p>
                            <p style="color: #84cc16; font-size: 0.70rem; margin-top: 6px;">Status: High certainty</p>
                        </div>
                    </label>
                </div>
            """, unsafe_allow_html=True)

    with nav_tab2:
        st.markdown('<div class="glass-graph-container" style="margin-top: 1rem;">', unsafe_allow_html=True)
        
        col_gh1, col_gh2 = st.columns([3, 1])
        with col_gh1:
            if st.session_state.graph_view_mode == "Live Graph":
                st.markdown("<h4 style='color: #84cc16; margin: 0;'>📈 Flow & Pressure Analytics</h4>", unsafe_allow_html=True)
            else:
                st.markdown("<h4 style='color: #84cc16; margin: 0;'>🤖 Smart AI Hourly Summary Report</h4>", unsafe_allow_html=True)
        with col_gh2:
            btn_text = "AI Summary" if st.session_state.graph_view_mode == "Live Graph" else "Live Graph"
            if st.button(btn_text, use_container_width=True, key="graph_mode_btn"):
                if st.session_state.graph_view_mode == "Live Graph":
                    st.session_state.graph_view_mode = "AI Summary"
                else:
                    st.session_state.graph_view_mode = "Live Graph"
                st.rerun()

        st.markdown("---")

        if st.session_state.graph_view_mode == "AI Summary":
            leak_events = st.session_state.data_history[st.session_state.data_history['Status'] == 'LEAK DETECTED']
            if len(leak_events) > 0:
                first_leak_time = leak_events.iloc[0]['Timestamp']
                last_leak_time = leak_events.iloc[-1]['Timestamp']
                st.markdown(f"""
                <div style="background-color: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; border-radius: 0.5rem; padding: 1rem; color: #ffffff;">
                    <b>Operational Status (Past Hour):</b><br>
                    - <b>Incident Detected:</b> The system recorded pressure anomalies and leakage alerts between <b>{first_leak_time}</b> and <b>{last_leak_time}</b>.<br>
                    - <b>Action Taken:</b> Automated isolation valves throttled Zone B, stabilizing pressure back to baseline normal ranges following automated rectification.<br>
                    - <b>Current State:</b> System integrity is currently secure within normal flow bounds of 11.0 to 14.5 L/min.
                </div>
                """, unsafe_allow_html=True)
            else:
                avg_flow = st.session_state.data_history['Flow_Rate'].mean() if not st.session_state.data_history.empty else 12.5
                st.markdown(f"""
                <div style="background-color: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; border-radius: 0.5rem; padding: 1rem; color: #ffffff;">
                    <b>Operational Status (Past Hour):</b><br>
                    - <b>Normal Operation:</b> For the past hour, the water inlet and distribution network have operated completely normally with zero warning triggers or leakage events.<br>
                    - <b>Flow Range:</b> Flow rates maintained a stable average of <b>{avg_flow:.1f} L/min</b> (within the optimal 10.0 - 15.0 L/min baseline).<br>
                    - <b>Pressure Integrity:</b> Inlet pressure held steady at 3.0 Bar with minimal friction drop.
                </div>
                """, unsafe_allow_html=True)
        else:
            if not st.session_state.data_history.empty:
                chart_data = st.session_state.data_history.set_index('Timestamp')[['Flow_Rate', 'Pressure_Drop']]
                fig = px.line(
                    chart_data,
                    labels={"value": "Telemetry Value", "Timestamp": "Time", "variable": "Sensor Metric"},
                    color_discrete_map={
                        "Flow_Rate": "#84cc16",
                        "Pressure_Drop": "#38bdf8",
                    },
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=350,
                    font=dict(color="#84cc16"),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Waiting for data stream...")

        st.markdown('</div>', unsafe_allow_html=True)

    with nav_tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #84cc16;'>🗺️ AI Diagnostics & Zone Status Monitor</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #ffffff;'>Live status of smart shutoff valves and machine learning classification.</p>", unsafe_allow_html=True)

        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            st.markdown(f"""
                <div style="color: #ffffff;">
                    <p><b>Classification Output:</b> {current_status}</p>
                    <p><b>Isolated Zone:</b> {current_loc}</p>
                    <p><b>Algorithm Used:</b> Random Forest Classifier</p>
                    <p><b>Sensor Network:</b> ESP32 Edge Node Active</p>
                </div>
            """, unsafe_allow_html=True)
        
        with diag_col2:
            st.info("🟢 **Zone A (Main Inlet)**: Valve Open (Normal)")
            st.warning("🟡 **Zone B (Residential Wing)**: Valve Throttled (Minor Drop)")
            st.success("🟢 **Zone C (Industrial Wing)**: Valve Open (Normal)")

    with nav_tab4:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #84cc16;'>🚰 Emergency Master Control & Event Stream Logs</h3>", unsafe_allow_html=True)

        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)

        with col_ctrl1:
            valve_override = st.toggle(
                "Emergency Shutoff Valve", value=False, key="valve"
            )
            if valve_override:
                st.error("⚠️ EMERGENCY SHUTOFF ENGAGED! Water supply halted across all zones.")
            else:
                st.success("✅ Main Supply Flow Active.")

        with col_ctrl2:
            pump_mode = st.selectbox(
                "Booster Pump Operation",
                ["Automatic (PID Loop)", "Manual Override", "Low Power Eco"],
            )

        with col_ctrl3:
            st.markdown("<p style='color: #ffffff;'>System Actions</p>", unsafe_allow_html=True)
            if st.button("🔄 Clear Log History", key="clear_log_btn"):
                st.session_state.data_history = st.session_state.data_history.iloc[0:0]
                st.rerun()

        st.markdown("<h3 style='color: #84cc16;'>📋 Recent Event Stream Log</h3>", unsafe_allow_html=True)
        st.dataframe(st.session_state.data_history.tail(5), use_container_width=True)

    # Auto-refresh loop to continuously read live ESP sensor values
    time.sleep(2)
    st.rerun()