import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. PREMIUM MONOBANK UI THEME ---
st.set_page_config(page_title="FinRisk | AI Dashboard", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    /* Global UI Tweaks */
    .stApp {
        background-color: #1a1c22;
        color: #ffffff;
        font-family: 'Poppins', sans-serif;
    }

    /* Left Navigation / Sidebar */
    [data-testid="stSidebar"] {
        background-color: #13151a !important;
        border-right: 1px solid #2d2f36;
    }
    
    /* Typography Styling */
    h1, h2, h3, p, span { font-family: 'Poppins', sans-serif !important; }
    h1 { color: #ffffff !important; font-weight: 600; letter-spacing: -0.5px; }
    h2 { color: #a0a4a8 !important; font-weight: 400; font-size: 1.1rem; }

    /* Neon Metrics Styling */
    [data-testid="stMetricValue"] {
        color: #26e6a5 !important; /* Neon Green */
        font-weight: 600;
        font-size: 2.2rem !important;
    }
    
    /* Buttons - Monobank Purple Accent */
    div.stButton > button:first-child {
        background-color: #5d5fef;
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.6rem 2.5rem;
        font-weight: 600;
        transition: 0.3s ease-in-out;
    }
    div.stButton > button:hover {
        background-color: #4a4cd9;
        transform: scale(1.02);
    }
    
    /* Modern Tabs Styling */
    .stTabs [data-baseweb="tab"] { color: #787988; }
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        border-bottom-color: #5d5fef !important;
    }

    /* Custom Risk Cards */
    .risk-card {
        background-color: #13151a;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #2d2f36;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CORE LOGIC & DATA LOADING ---
@st.cache_resource
def load_model():
    return joblib.load('churn_model.pkl')

try:
    model = load_model()
except:
    st.error("Error: 'churn_model.pkl' not found. Please place it in the project root.")

features = ['avg_spend', 'total_spend', 'trans_count', 'city_pop', 'fraud_history']

# --- 3. SIDEBAR (MATCHING YOUR IMAGE) ---
with st.sidebar:
    st.markdown('<div style="text-align: center;"><img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" width="80" style="border-radius:50%; border: 2px solid #5d5fef; padding: 5px;"></div>', unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: white !important;'>Jhon Newman</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #787988 !important; font-size: 0.8rem; margin-top:-15px;'>#0004242</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.write("📂 **Dashboard**")
    st.write("👤 **Contacts**")
    st.write("📊 **Statistic**")
    st.write("📑 **Documents**")
    
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    if st.button("LOG OUT", use_container_width=True):
        st.cache_resource.clear()

# --- 4. MAIN INTERFACE ---
st.title("Dashboard")
st.markdown("## Overview of all churn risk metrics")

tab1, tab2, tab3 = st.tabs(["👤 Individual Analysis", "📑 Batch Processor", "📈 Statistic Overview"])

# --- TAB 1: INDIVIDUAL ANALYSIS ---
with tab1:
    col_input, col_metric = st.columns([1, 2])
    
    with col_input:
        st.markdown("### Profile Input")
        s_avg = st.number_input("Annual Spend (Salary)", value=75000)
        s_bal = st.number_input("Current Balance", value=20000)
        s_ten = st.slider("Tenure (Years)", 0, 10, 2)
        s_pop = st.number_input("City Population", value=500000)
        s_frd = st.selectbox("Fraud History", [0, 1])

    with col_metric:
        input_data = pd.DataFrame([[s_avg, s_bal, s_ten, s_pop, s_frd]], columns=features)
        prob = model.predict_proba(input_data)[0][1]
        
        st.markdown('<p style="color:#a0a4a8;">Predictive Churn Risk</p>', unsafe_allow_html=True)
        st.metric("", f"{prob:.2%}", delta="-431" if prob < 0.15 else "+12.1%")
        
        if prob > 0.15:
            st.markdown('<div class="risk-card" style="border: 1px solid #ff5c5c;">🚨 <span style="color:#ff5c5c;">CRITICAL:</span> High risk of exit detected based on behavior patterns.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="risk-card" style="border: 1px solid #26e6a5;">✅ <span style="color:#26e6a5;">STABLE:</span> Customer shows strong retention indicators.</div>', unsafe_allow_html=True)

# --- TAB 2: UNIVERSAL BATCH ANALYSIS (CLEAN & MERGED) ---
with tab2:
    st.header("Bulk Risk Processing")
    st.markdown("Upload your Kaggle CSV or Excel file to pinpoint high-risk individuals.")

    # 1. CSS FIX: Ensure labels are visible on dark background
    st.markdown("""
        <style>
        label, .stMarkdown p, [data-testid="stWidgetLabel"] p { color: #ffffff !important; }
        .stDataFrame { background-color: #13151a; }
        </style>
    """, unsafe_allow_html=True)

    # 2. File Uploader
    uploaded_file = st.file_uploader("Upload Customer File", type=["csv", "xlsx"])

    # 3. SINGLE LOGIC BLOCK (No Overwriting)
    if uploaded_file:
        try:
            # Load Data
            if uploaded_file.name.endswith('.csv'):
                batch_df = pd.read_csv(uploaded_file)
            else:
                batch_df = pd.read_excel(uploaded_file, engine='openpyxl')

            # --- PRE-PROCESSING ---
            mapping = {
                'EstimatedSalary': 'avg_spend', 'Balance': 'total_spend',
                'Tenure': 'trans_count', 'Geography': 'city_pop', 'HasCrCard': 'fraud_history'
            }
            
            if 'Geography' in batch_df.columns:
                st.info("💡 Kaggle format detected! Auto-mapping columns...")
                batch_df = batch_df.rename(columns=mapping)

            # Map Countries to Numbers (Option 2)
            country_to_pop = {'France': 670000, 'Germany': 830000, 'Spain': 470000}
            if batch_df['city_pop'].dtype == 'O':
                batch_df['city_pop'] = batch_df['city_pop'].map(country_to_pop).fillna(500000)

            # Force Numeric
            for col in features:
                batch_df[col] = pd.to_numeric(batch_df[col], errors='coerce').fillna(0)

            # --- PREDICTION ---
            st.success("✅ Data validated! Running model...")
            preds = model.predict_proba(batch_df[features])[:, 1]
            batch_df['Risk_Score'] = preds
            batch_df['Status'] = ["🚨 Risk" if p > 0.15 else "✅ Loyal" for p in preds]

            # --- DISPLAY RESULTS ---
            m1, m2 = st.columns(2)
            m1.metric("Total Customers", len(batch_df))
            m2.metric("At-Risk Pins", (batch_df['Risk_Score'] > 0.15).sum())

            st.subheader("🚩 High-Risk Customer Pins")
            high_risk_list = batch_df[batch_df['Risk_Score'] > 0.15].sort_values(by='Risk_Score', ascending=False)
            
            if not high_risk_list.empty:
                st.dataframe(high_risk_list.style.background_gradient(cmap='Reds', subset=['Risk_Score']), width='stretch')
            else:
                st.success("No high-risk individuals found in this batch.")

            # --- VISUALIZATION ---
            fig, ax = plt.subplots(figsize=(10, 3), facecolor='#1a1c22')
            ax.set_facecolor('#1a1c22')
            sns.histplot(batch_df['Risk_Score'], bins=20, kde=True, ax=ax, color='#5d5fef')
            plt.axvline(0.15, color='#ff5c5c', linestyle='--')
            ax.tick_params(colors='white')
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
# --- TAB 3: STATISTIC OVERVIEW ---
with tab3:
    st.subheader("Statistical Performance")
    col_chart, col_history = st.columns([2, 1])
    
    with col_chart:
        # Dark Mode Graph
        fig, ax = plt.subplots(figsize=(10, 4), facecolor='#1a1c22')
        ax.set_facecolor('#1a1c22')
        
        # Simulated curve like the image
        data_curve = np.random.normal(0.1, 0.05, 500)
        sns.kdeplot(data_curve, color='#5d5fef', fill=True, ax=ax, alpha=0.3)
        
        ax.tick_params(colors='#a0a4a8', labelsize=8)
        ax.spines['bottom'].set_color('#2d2f36')
        ax.spines['left'].set_color('#2d2f36')
        ax.set_title("Probability Distribution (Prev Month)", color='#ffffff', size=10)
        st.pyplot(fig)

    with col_history:
        st.markdown("### 🏦 Card History")
        st.markdown("""
        <div style="background-color:#13151a; padding:15px; border-radius:12px; border:1px solid #2d2f36;">
            <p style="margin:5px 0;"><span style="color:#26e6a5;">●</span> Monobank Analysis <span style="float:right; font-weight:bold; color:#26e6a5;">+$1,233</span></p>
            <p style="margin:5px 0;"><span style="color:#ff5c5c;">●</span> Kaggle Batch <span style="float:right; font-weight:bold; color:#ff5c5c;">-$32</span></p>
            <p style="margin:5px 0;"><span style="color:#26e6a5;">●</span> Regional Audit <span style="float:right; font-weight:bold; color:#26e6a5;">+$100</span></p>
        </div>
        """, unsafe_allow_html=True)
        st.button("See full history", use_container_width=True)