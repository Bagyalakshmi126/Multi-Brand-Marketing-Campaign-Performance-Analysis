import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Marketing Campaign AI Decision Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished layout and high-contrast KPI cards
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .stMetric {
        background-color: #ffffff;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    .metric-card-positive {
        border-left: 6px solid #10b981 !important;
    }
    .metric-card-negative {
        border-left: 6px solid #ef4444 !important;
    }
    .metric-card-info {
        border-left: 6px solid #3b82f6 !important;
    }
    .status-badge-profit {
        background-color: #d1fae5;
        color: #065f46;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .status-badge-loss {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ARTIFACT LOADING WITH FALLBACK ENGINE
# ==========================================
@st.cache_resource
def load_artifacts():
    """Attempts to load pickle models; returns None if files aren't found."""
    try:
        if (os.path.exists('revenue_regressor.pkl') and 
            os.path.exists('profit_classifier.pkl') and 
            os.path.exists('regression_scaler.pkl') and 
            os.path.exists('classification_scaler.pkl') and 
            os.path.exists('feature_columns.pkl')):

            with open('revenue_regressor.pkl', 'rb') as f:
                reg_model = pickle.load(f)
            with open('profit_classifier.pkl', 'rb') as f:
                cls_model = pickle.load(f)
            with open('regression_scaler.pkl', 'rb') as f:
                reg_scaler = pickle.load(f)
            with open('classification_scaler.pkl', 'rb') as f:
                cls_scaler = pickle.load(f)
            with open('feature_columns.pkl', 'rb') as f:
                features = pickle.load(f)

            return reg_model, cls_model, reg_scaler, cls_scaler, features, True
        else:
            return None, None, None, None, None, False
    except Exception as e:
        st.warning(f"Note: Error loading .pkl models ({e}). Operating in demonstration Mode.")
        return None, None, None, None, None, False

reg_model, cls_model, reg_scaler, cls_scaler, feature_meta, models_loaded = load_artifacts()

# ==========================================
# 3. SIDEBAR - CAMPAIGN INPUTS
# ==========================================
st.sidebar.image("https://img.icons8.com/color/96/000000/bullish.png", width=70)
st.sidebar.title("Campaign Setup")
st.sidebar.write("Configure your marketing parameters to generate ML insights.")

with st.sidebar.expander("📍 Channel & Metadata", expanded=True):
    channel = st.selectbox("Marketing Channel", ['Paid Search', 'Social Media', 'Email Marketing', 'Influencer', 'Display Ads'])
    target_audience = st.selectbox("Target Audience", ['Young Adults (18-24)', 'Working Professionals (25-40)', 'Parents & Families', 'B2B Buyers'])
    language = st.selectbox("Language / Region", ['English', 'Spanish', 'Hindi', 'French', 'German'])

with st.sidebar.expander("💰 Financials & Budget", expanded=True):
    spend = st.number_input("Total Spend ($)", min_value=100.0, value=2500.0, step=100.0)
    target_roi = st.slider("Target ROI Multiplier", min_value=0.5, max_value=10.0, value=2.5, step=0.1)

with st.sidebar.expander("📊 Engagement & Funnel Metrics", expanded=True):
    impressions = st.number_input("Impressions", min_value=1000, value=65000, step=1000)
    clicks = st.number_input("Clicks", min_value=10, value=3200, step=50)
    leads = st.number_input("Leads Generated", min_value=0, value=180, step=10)
    conversions = st.number_input("Conversions / Sales", min_value=0, value=85, step=5)
    engagement_score = st.slider("Engagement Rating (0-10)", min_value=0.0, max_value=10.0, value=6.8, step=0.1)

# ==========================================
# 4. PREPROCESSING & FEATURE ENGINEERING
# ==========================================
# Calculate Derived Features
ctr = (clicks / impressions) * 100 if impressions > 0 else 0.0
cpc = spend / clicks if clicks > 0 else 0.0
cpa = spend / conversions if conversions > 0 else 0.0
conversion_rate = (conversions / clicks) * 100 if clicks > 0 else 0.0

# Encode Categorical Variables
channel_map = {'Paid Search': 0, 'Social Media': 1, 'Email Marketing': 2, 'Influencer': 3, 'Display Ads': 4}
audience_map = {'Young Adults (18-24)': 0, 'Working Professionals (25-40)': 1, 'Parents & Families': 2, 'B2B Buyers': 3}
lang_map = {'English': 0, 'Spanish': 1, 'Hindi': 2, 'French': 3, 'German': 4}

raw_input_dict = {
    'Channel': channel_map[channel],
    'Target_Audience': audience_map[target_audience],
    'Language': lang_map[language],
    'Spend': spend,
    'Impressions': impressions,
    'Clicks': clicks,
    'Leads': leads,
    'Conversions': conversions,
    'Engagement_Score': engagement_score,
    'Target_ROI': target_roi,
    'CTR': ctr,
    'CPC': cpc,
    'CPA': cpa,
    'Conversion_Rate': conversion_rate
}

input_df = pd.DataFrame([raw_input_dict])

# ==========================================
# 5. MODEL PREDICTION INFERENCE
# ==========================================
if models_loaded:
    try:
        # Reorder columns to match expected pickle schema
        reg_cols = feature_meta['reg_features']
        cls_cols = feature_meta['cls_features']

        # Fill missing features if any
        reg_inputs = input_df[[c for c in reg_cols if c in input_df.columns]].copy()
        for c in reg_cols:
            if c not in reg_inputs.columns: reg_inputs[c] = 0
        reg_inputs = reg_inputs[reg_cols]

        cls_inputs = input_df[[c for c in cls_cols if c in input_df.columns]].copy()
        for c in cls_cols:
            if c not in cls_inputs.columns: cls_inputs[c] = 0
        cls_inputs = cls_inputs[cls_cols]

        # Scale and Predict
        reg_scaled = reg_scaler.transform(reg_inputs)
        pred_revenue = float(reg_model.predict(reg_scaled)[0])

        cls_scaled = cls_scaler.transform(cls_inputs)
        pred_profit_class = int(cls_model.predict(cls_scaled)[0])
        pred_profit_prob = float(cls_model.predict_proba(cls_scaled)[0][1])

    except Exception as e:
        st.error(f"Inference error with pickle model: {e}. Defaulting to heuristic computation.")
        models_loaded = False

if not models_loaded:
    # Heuristic surrogate calculation for standalone testing
    est_aov = 85.0  # Estimated Average Order Value
    pred_revenue = float((conversions * est_aov) + (clicks * engagement_score * 0.25))
    net_diff = pred_revenue - spend
    pred_profit_class = 1 if net_diff > 0 else 0
    # Calculate synthetic probability curve
    margin_ratio = (pred_revenue - spend) / max(spend, 1)
    pred_profit_prob = min(max(1 / (1 + np.exp(-2 * margin_ratio)), 0.05), 0.98)

# Calculated financial outcomes
pred_net_profit = pred_revenue - spend
pred_roi = ((pred_revenue - spend) / spend) * 100 if spend > 0 else 0.0
profit_margin = (pred_net_profit / pred_revenue) * 100 if pred_revenue > 0 else 0.0

# ==========================================
# 6. MAIN DISPLAY & DASHBOARD
# ==========================================
st.title("🎯 Marketing Campaign Decision Engine")
st.markdown("Automated Machine Learning forecast for revenue, ROI, and profitability risk assessment.")

if not models_loaded:
    st.info("💡 **Demonstration Mode Active:** Custom `.pkl` model files were not detected in the working directory. Using heuristic scoring engine. Save your trained `.pkl` models to enable trained predictions.")
else:
    st.success("✅ **Pickle Machine Learning Models Loaded Successfully!**")

st.divider()

# --- KPI METRIC ROW ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        label="Predicted Gross Revenue",
        value=f"${pred_revenue:,.2f}",
        delta=f"{((pred_revenue - spend)/spend)*100:+.1f}% vs Spend"
    )

with kpi2:
    st.metric(
        label="Predicted Net Profit / Loss",
        value=f"${pred_net_profit:,.2f}",
        delta=f"${pred_net_profit:,.2f}",
        delta_color="normal" if pred_net_profit >= 0 else "inverse"
    )

with kpi3:
    st.metric(
        label="Forecasted ROI",
        value=f"{pred_roi:+.1f}%",
        delta=f"{pred_roi - (target_roi*100):+.1f}% vs Target"
    )

with kpi4:
    st.write("**Profitability Status**")
    if pred_profit_class == 1:
        st.markdown('<div class="status-badge-profit">🟢 PROFITABLE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge-loss">🔴 HIGH RISK LOSS</div>', unsafe_allow_html=True)
    st.caption(f"Profit Confidence: **{pred_profit_prob:.1%}**")

st.markdown("---")

# --- DASHBOARD VISUALIZATIONS ---
tab1, tab2, tab3 = st.tabs(["📊 Financial Breakdown", "🔄 Conversion Funnel & Unit Economics", "🔮 What-If Simulator"])

with tab1:
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Spend vs. Forecasted Revenue")

        # Waterfall / Bar chart for financial flow
        fig_bar = go.Figure(go.Bar(
            x=['Ad Spend', 'Predicted Revenue', 'Net Profit/Loss'],
            y=[spend, pred_revenue, pred_net_profit],
            marker_color=['#64748b', '#3b82f6', '#10b981' if pred_net_profit >= 0 else '#ef4444'],
            text=[f"${spend:,.2f}", f"${pred_revenue:,.2f}", f"${pred_net_profit:,.2f}"],
            textposition='auto',
        ))
        fig_bar.update_layout(
            yaxis_title="USD ($)",
            height=380,
            template="plotly_white",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.subheader("Profitability Gauge")

        # Speedometer/Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_profit_prob * 100,
            number={'suffix': "%"},
            title={'text': "Probability of Profit"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#10b981" if pred_profit_prob >= 0.5 else "#ef4444"},
                'steps': [
                    {'range': [0, 40], 'color': "#fee2e2"},
                    {'range': [40, 70], 'color': "#fef3c7"},
                    {'range': [70, 100], 'color': "#d1fae5"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

with tab2:
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Campaign Conversion Funnel")
        funnel_data = dict(
            number=[impressions, clicks, leads, conversions],
            stage=["Impressions", "Clicks", "Leads", "Conversions"]
        )
        fig_funnel = px.funnel(funnel_data, x='number', y='stage', color_discrete_sequence=['#3b82f6'])
        fig_funnel.update_layout(height=380, template="plotly_white")
        st.plotly_chart(fig_funnel, use_container_width=True)

    with c2:
        st.subheader("Engineered Unit Metrics")

        unit_df = pd.DataFrame({
            'Metric Name': ['Click-Through Rate (CTR)', 'Cost Per Click (CPC)', 'Cost Per Acquisition (CPA)', 'Conversion Rate'],
            'Value': [f"{ctr:.2f}%", f"${cpc:.2f}", f"${cpa:.2f}", f"{conversion_rate:.2f}%"],
            'Benchmark Target': ['> 2.50%', '< $1.20', '< $35.00', '> 2.00%']
        })

        st.dataframe(unit_df, hide_index=True, use_container_width=True)

        st.markdown("#### Input Parameter Summary")
        st.json({
            'Channel': channel,
            'Target Audience': target_audience,
            'Engagement Rating': engagement_score,
            'Acquisition Spend': f"${spend:,.2f}"
        })

with tab3:
    st.subheader("Interactive Budget & Conversion Multiplier")
    st.write("Simulate how altering ad budget or boosting conversion rate impacts revenue.")

    sim_col1, sim_col2 = st.columns(2)

    with sim_col1:
        budget_scaler = st.slider("Scale Budget Multiplier", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
    with sim_col2:
        cr_boost = st.slider("Boost Conversion Rate (+%)", min_value=0.0, max_value=5.0, value=0.0, step=0.2)

    # Simulated outputs
    sim_spend = spend * budget_scaler
    sim_conversions = conversions * budget_scaler * (1 + (cr_boost / 100))
    sim_revenue = pred_revenue * budget_scaler * (1 + (cr_boost / 100))
    sim_profit = sim_revenue - sim_spend

    st.markdown("---")
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Simulated Spend", f"${sim_spend:,.2f}", f"{(budget_scaler-1)*100:+.0f}%")
    sc2.metric("Simulated Revenue", f"${sim_revenue:,.2f}", f"${sim_revenue - pred_revenue:+,.2f}")
    sc3.metric("Simulated Net Profit", f"${sim_profit:,.2f}", f"${sim_profit - pred_net_profit:+,.2f}",
               delta_color="normal" if sim_profit >= 0 else "inverse")