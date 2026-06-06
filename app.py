import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ================================
# Page Configuration
# ================================
st.set_page_config(
    page_title="Healthcare Fraud Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# Custom CSS
# ================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: #f8faff;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e3a5f 100%);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 32px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    width: 100% !important;
    box-shadow: 0 4px 15px rgba(37,99,235,0.4) !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(37,99,235,0.5) !important;
}
</style>
""", unsafe_allow_html=True)

# ================================
# Load Data
# ================================
@st.cache_data
def load_data():
    return pd.read_csv('cleaned_data.csv')

@st.cache_resource
def load_model():
    with open('fraud_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_scaler():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return scaler

claims = load_data()
model = load_model()
scaler = load_scaler()

# ================================
# Sidebar
# ================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0 10px 0;'>
        <div style='font-size:40px;'>🏥</div>
        <div style='font-size:20px; font-weight:800; color:white;'>FraudShield AI</div>
        <div style='font-size:12px; color:rgba(255,255,255,0.6); margin-top:4px;'>Healthcare Fraud Detection</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

    page = st.selectbox("Navigate To", [
        "🏠 Overview",
        "📊 Data Analysis",
        "🤖 Model Performance",
        "🔍 Predict Fraud"
    ])

    st.markdown("<hr style='border-color:rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

    total = len(claims)
    fraud = int(claims['PotentialFraud'].sum())
    legit = total - fraud
    fraud_pct = round(claims['PotentialFraud'].mean() * 100, 2)

    st.markdown(f"""
    <div style='padding:0 10px;'>
        <div style='background:rgba(255,255,255,0.08); border-radius:12px; padding:14px; margin-bottom:10px;'>
            <div style='font-size:11px; color:rgba(255,255,255,0.5); text-transform:uppercase; letter-spacing:1px;'>Total Claims</div>
            <div style='font-size:22px; font-weight:800; color:white;'>{total:,}</div>
        </div>
        <div style='background:rgba(239,68,68,0.15); border-radius:12px; padding:14px; margin-bottom:10px; border:1px solid rgba(239,68,68,0.3);'>
            <div style='font-size:11px; color:rgba(255,255,255,0.5); text-transform:uppercase; letter-spacing:1px;'>Fraud Cases</div>
            <div style='font-size:22px; font-weight:800; color:#f87171;'>{fraud:,}</div>
        </div>
        <div style='background:rgba(34,197,94,0.15); border-radius:12px; padding:14px; margin-bottom:10px; border:1px solid rgba(34,197,94,0.3);'>
            <div style='font-size:11px; color:rgba(255,255,255,0.5); text-transform:uppercase; letter-spacing:1px;'>Legitimate</div>
            <div style='font-size:22px; font-weight:800; color:#4ade80;'>{legit:,}</div>
        </div>
        <div style='background:rgba(251,191,36,0.15); border-radius:12px; padding:14px; border:1px solid rgba(251,191,36,0.3);'>
            <div style='font-size:11px; color:rgba(255,255,255,0.5); text-transform:uppercase; letter-spacing:1px;'>Fraud Rate</div>
            <div style='font-size:22px; font-weight:800; color:#fbbf24;'>{fraud_pct}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ================================
# Helper Functions
# ================================
def metric_card(label, value, color="#2563eb"):
    return f"""
    <div style='background:white; border-radius:16px; padding:24px;
                box-shadow:0 2px 15px rgba(0,0,0,0.06);
                border-top:4px solid {color};'>
        <div style='font-size:12px; font-weight:600; color:#6b7280;
                    text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;'>{label}</div>
        <div style='font-size:28px; font-weight:800; color:{color};'>{value}</div>
    </div>
    """

def section_card_start(title):
    return f"""
    <div style='background:white; border-radius:16px; padding:28px;
                box-shadow:0 2px 15px rgba(0,0,0,0.06); margin-bottom:24px;'>
        <div style='font-size:17px; font-weight:700; color:#111827;
                    margin-bottom:20px; padding-bottom:12px;
                    border-bottom:2px solid #f3f4f6;'>{title}</div>
    """

def header_banner(title, subtitle, color1="#1e3a5f", color2="#2563eb"):
    return f"""
    <div style='background:linear-gradient(135deg, {color1} 0%, {color2} 100%);
                padding:32px 40px; border-radius:20px; margin-bottom:28px;
                box-shadow:0 10px 40px rgba(37,99,235,0.25);'>
        <h1 style='color:white; font-size:28px; font-weight:800; margin:0;'>{title}</h1>
        <p style='color:rgba(255,255,255,0.75); font-size:15px; margin:8px 0 0 0;'>{subtitle}</p>
    </div>
    """

# ================================
# PAGE 1 - Overview
# ================================
if page == "🏠 Overview":

    st.markdown(header_banner(
        "🏥 Healthcare Insurance Fraud Detection",
        "End-to-end Machine Learning project detecting fraudulent insurance claims automatically"
    ), unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card("📋 Total Claims", f"{len(claims):,}", "#2563eb"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("🚨 Fraud Cases", f"{int(claims['PotentialFraud'].sum()):,}", "#dc2626"), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card("✅ Legitimate", f"{len(claims) - int(claims['PotentialFraud'].sum()):,}", "#059669"), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_card("⚠️ Fraud Rate", f"{round(claims['PotentialFraud'].mean()*100,2)}%", "#d97706"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown(section_card_start("📌 Problem Statement"), unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#eff6ff; border-left:4px solid #2563eb;
                    border-radius:8px; padding:18px; margin-bottom:16px;'>
            <p style='color:#1e40af; font-size:14px; line-height:1.8; margin:0;'>
            Healthcare insurance fraud costs billions of dollars annually, placing a massive
            burden on patients, insurers, and the healthcare system. Fraudulent claims continue
            to rise, making it increasingly difficult for insurance companies to identify
            them manually. This project uses Machine Learning to automatically detect
            fraudulent claims and provide actionable business insights.
            </p>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(section_card_start("🎯 Project Pipeline"), unsafe_allow_html=True)
        steps = [
            ("1", "Data Collection and EDA", "Analyzed 558K+ healthcare claims", "#2563eb"),
            ("2", "Data Cleaning and Engineering", "Created 50+ meaningful features", "#7c3aed"),
            ("3", "Model Building", "XGBoost with SMOTE balancing", "#059669"),
            ("4", "Model Evaluation", "90.28% accuracy, AUC 0.97", "#dc2626"),
            ("5", "Dashboard Deployment", "Interactive Streamlit dashboard", "#d97706"),
        ]
        for num, title, desc, color in steps:
            st.markdown(f"""
            <div style='display:flex; align-items:center; gap:14px;
                        padding:10px 0; border-bottom:1px solid #f9fafb;'>
                <div style='width:32px; height:32px; border-radius:50%;
                            background:{color}; color:white; font-weight:800;
                            font-size:14px; display:flex; align-items:center;
                            justify-content:center; flex-shrink:0;'>{num}</div>
                <div>
                    <div style='font-weight:700; color:#111827; font-size:14px;'>{title}</div>
                    <div style='color:#6b7280; font-size:12px;'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(section_card_start("📊 Fraud Distribution"), unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        counts = claims['PotentialFraud'].value_counts()
        bars = ax.bar(['Legitimate', 'Fraud'], counts.values,
                      color=['#059669', '#dc2626'], width=0.5,
                      edgecolor='white', linewidth=2)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
                    f'{val:,}', ha='center', fontweight='bold', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(section_card_start("🥧 Fraud Percentage"), unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('white')
        counts = claims['PotentialFraud'].value_counts()
        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=['Legitimate', 'Fraud'],
            colors=['#059669', '#dc2626'],
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 3}
        )
        for text in autotexts:
            text.set_fontweight('bold')
            text.set_fontsize(13)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(section_card_start("🗂️ Dataset Preview"), unsafe_allow_html=True)
    st.dataframe(claims.head(10), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ================================
# PAGE 2 - Data Analysis
# ================================
elif page == "📊 Data Analysis":

    st.markdown(header_banner(
        "📊 Exploratory Data Analysis",
        "Deep dive into healthcare claims data to uncover fraud patterns and insights"
    ), unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(metric_card("📋 Total Rows", f"{claims.shape[0]:,}", "#2563eb"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("📊 Features", f"{claims.shape[1]}", "#7c3aed"), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card("✅ Missing Values", "0", "#059669"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(section_card_start("💰 Claim Amount Distribution"), unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#fafbff')
        ax.hist(claims['InscClaimAmtReimbursed'], bins=60,
                color='#2563eb', alpha=0.85, edgecolor='white')
        ax.set_xlabel('Claim Amount ($)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Distribution of Claim Amounts', fontsize=13, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(section_card_start("🔍 Fraud vs Legitimate Claim Amounts"), unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#fafbff')
        fraud_amounts = claims[claims['PotentialFraud']==1]['InscClaimAmtReimbursed']
        legit_amounts = claims[claims['PotentialFraud']==0]['InscClaimAmtReimbursed']
        ax.hist(legit_amounts, bins=60, alpha=0.7, color='#059669', label='Legitimate', edgecolor='white')
        ax.hist(fraud_amounts, bins=60, alpha=0.7, color='#dc2626', label='Fraud', edgecolor='white')
        ax.set_xlabel('Claim Amount ($)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Claim Amount by Fraud Status', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if 'Age' in claims.columns:
            st.markdown(section_card_start("👤 Patient Age Distribution"), unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('#fafbff')
            ax.hist(claims['Age'].dropna(), bins=30,
                    color='#7c3aed', alpha=0.85, edgecolor='white')
            ax.set_xlabel('Age', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title('Patient Age Distribution', fontsize=13, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if 'IsDeceased' in claims.columns:
            st.markdown(section_card_start("💀 Deceased vs Living Patients"), unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('#fafbff')
            deceased_counts = claims['IsDeceased'].value_counts()
            ax.bar(['Living', 'Deceased'], deceased_counts.values,
                   color=['#059669', '#374151'], width=0.5, edgecolor='white')
            ax.set_title('Living vs Deceased Patients', fontsize=13, fontweight='bold')
            ax.set_ylabel('Count', fontsize=11)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for i, v in enumerate(deceased_counts.values):
                ax.text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(section_card_start("🔥 Feature Correlation Heatmap"), unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('white')
    numeric_cols = claims.select_dtypes(include=[np.number]).columns[:15]
    sns.heatmap(claims[numeric_cols].corr(),
                annot=False, cmap='RdYlBu_r', ax=ax,
                linewidths=0.5, square=False,
                vmin=-1, vmax=1)
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown("</div>", unsafe_allow_html=True)

# ================================
# PAGE 3 - Model Performance
# ================================
elif page == "🤖 Model Performance":

    st.markdown(header_banner(
        "🤖 Model Performance",
        "XGBoost model evaluation — accuracy, confusion matrix, ROC curve and feature importance",
        "#1a1a2e", "#16213e"
    ), unsafe_allow_html=True)

    # Use saved model directly without SMOTE
    X = claims.drop('PotentialFraud', axis=1).fillna(0)
    y = claims['PotentialFraud']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    X_test_sc = scaler.transform(X_test)
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    acc = round(accuracy_score(y_test, y_pred) * 100, 2)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = round(auc(fpr, tpr), 2)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card("🎯 Accuracy", f"{acc}%", "#2563eb"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("📈 AUC Score", f"{roc_auc}", "#059669"), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card("🌲 Algorithm", "XGBoost", "#7c3aed"), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_card("🌳 Trees", "200", "#d97706"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(section_card_start("📊 Confusion Matrix"), unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('white')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Legitimate', 'Fraud'],
                    yticklabels=['Legitimate', 'Fraud'],
                    ax=ax, linewidths=2, linecolor='white',
                    annot_kws={'size': 18, 'weight': 'bold'})
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(section_card_start("📈 ROC Curve"), unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#fafbff')
        ax.plot(fpr, tpr, color='#2563eb', linewidth=3,
                label=f'XGBoost (AUC = {roc_auc})')
        ax.fill_between(fpr, tpr, alpha=0.1, color='#2563eb')
        ax.plot([0,1], [0,1], color='#9ca3af', linestyle='--',
                linewidth=2, label='Random Guess')
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.legend(fontsize=11, loc='lower right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(section_card_start("⭐ Top 15 Important Features"), unsafe_allow_html=True)
    feature_importance = pd.DataFrame({
        'Feature': claims.drop('PotentialFraud', axis=1).columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafbff')
    colors_list = ['#dbeafe'] * 5 + ['#93c5fd'] * 5 + ['#2563eb'] * 5
    bars = ax.barh(feature_importance['Feature'],
                   feature_importance['Importance'],
                   color=colors_list, edgecolor='white', linewidth=1)
    ax.set_title('Top 15 Features for Fraud Detection',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, feature_importance['Importance']):
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9,
                fontweight='bold', color='#374151')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(section_card_start("🏆 Model Comparison"), unsafe_allow_html=True)
    comparison_data = {
        'Model': ['Random Forest', 'XGBoost (Best)'],
        'Accuracy': ['83.73%', '90.28%'],
        'AUC Score': ['0.94', '0.97'],
        'Status': ['✅ Good', '🏆 Best Model']
    }
    df_comp = pd.DataFrame(comparison_data)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ================================
# PAGE 4 - Predict Fraud
# ================================
elif page == "🔍 Predict Fraud":

    st.markdown(header_banner(
        "🔍 Real-Time Fraud Prediction",
        "Enter healthcare claim details to instantly detect whether it is fraudulent or legitimate",
        "#1a1a2e", "#dc2626"
    ), unsafe_allow_html=True)

    feature_cols = claims.drop('PotentialFraud', axis=1).columns.tolist()

    st.markdown(section_card_start("📋 Patient and Claim Information"), unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#eff6ff; border-left:4px solid #2563eb;
                border-radius:8px; padding:14px 18px; margin-bottom:20px;'>
        <p style='color:#1e40af; font-size:13px; margin:0;'>
        💡 Enter the claim details below. Default values are pre-filled with dataset averages.
        Modify values as needed and click the Predict button.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Patient Details**")
        age = st.number_input("Patient Age", min_value=0, max_value=120, value=70, step=1)
        is_deceased = st.selectbox("Is Patient Deceased?",
                                    options=[0, 1],
                                    format_func=lambda x: "❌ No" if x == 0 else "✅ Yes")
        ip_annual_reimb = st.number_input("IP Annual Reimbursement ($)",
                                           min_value=0, value=15000, step=500)

    with col2:
        st.markdown("**💰 Claim Details**")
        claim_amount = st.number_input("Claim Amount ($)",
                                        min_value=0, value=8000, step=100)
        claim_duration = st.number_input("Claim Duration (days)",
                                          min_value=0, value=14, step=1)
        deductible = st.number_input("Deductible Amount ($)",
                                      min_value=0, value=1068, step=100)

    with col3:
        st.markdown("**🏥 Hospital Details**")
        hospital_stay = st.number_input("Hospital Stay (days)",
                                         min_value=0, value=7, step=1)
        op_annual_reimb = st.number_input("OP Annual Reimbursement ($)",
                                           min_value=0, value=4000, step=500)
        num_physicians = st.number_input("Number of Physicians",
                                          min_value=0, value=3, step=1)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("🔍 PREDICT FRAUD NOW", use_container_width=True)

    if predict_btn:
        base_input = claims.drop('PotentialFraud', axis=1).mean().to_dict()

        user_inputs = {
            'Age': age,
            'IsDeceased': is_deceased,
            'InscClaimAmtReimbursed': claim_amount,
            'ClaimDuration': claim_duration,
            'DeductibleAmtPaid': deductible,
            'HospitalStayDays': hospital_stay,
            'IPAnnualReimbursementAmt': ip_annual_reimb,
            'OPAnnualReimbursementAmt': op_annual_reimb,
        }

        for key, val in user_inputs.items():
            if key in base_input:
                base_input[key] = val

        input_array = np.array([[base_input[col] for col in feature_cols]])
        input_array = np.nan_to_num(input_array, nan=0.0)
        input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        fraud_pct = round(probability * 100, 2)
        legit_pct = round((1 - probability) * 100, 2)

        st.markdown("<br>", unsafe_allow_html=True)

        if prediction == 1:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg, #fef2f2, #fee2e2);
                        border:2px solid #dc2626; border-radius:16px;
                        padding:28px; text-align:center;
                        box-shadow:0 8px 30px rgba(220,38,38,0.2);'>
                <div style='font-size:48px; margin-bottom:8px;'>🚨</div>
                <div style='font-size:26px; font-weight:800; color:#dc2626; margin-bottom:8px;'>
                    FRAUD DETECTED!
                </div>
                <div style='font-size:16px; color:#b91c1c; font-weight:600; margin-bottom:4px;'>
                    Fraud Probability: {fraud_pct}%
                </div>
                <div style='font-size:13px; color:#9b1c1c; margin-top:10px;'>
                    This claim has been flagged as potentially fraudulent. Immediate review required.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg, #f0fdf4, #dcfce7);
                        border:2px solid #059669; border-radius:16px;
                        padding:28px; text-align:center;
                        box-shadow:0 8px 30px rgba(5,150,105,0.2);'>
                <div style='font-size:48px; margin-bottom:8px;'>✅</div>
                <div style='font-size:26px; font-weight:800; color:#059669; margin-bottom:8px;'>
                    CLAIM IS LEGITIMATE
                </div>
                <div style='font-size:16px; color:#047857; font-weight:600; margin-bottom:4px;'>
                    Fraud Probability: {fraud_pct}%
                </div>
                <div style='font-size:13px; color:#065f46; margin-top:10px;'>
                    No fraudulent activity detected. This claim appears to be legitimate.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(section_card_start("📊 Prediction Probability"), unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(7, 3))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            categories = ['Legitimate', 'Fraud']
            values = [legit_pct, fraud_pct]
            colors_bar = ['#059669', '#dc2626']
            bars = ax.bar(categories, values, color=colors_bar,
                          width=0.4, edgecolor='white', linewidth=2)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.5,
                        f'{val}%', ha='center',
                        fontweight='bold', fontsize=14)
            ax.set_ylim(0, 115)
            ax.set_ylabel('Probability (%)', fontsize=11)
            ax.set_title('Prediction Probability', fontsize=13, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown(section_card_start("📋 Claim Summary"), unsafe_allow_html=True)
            summary = {
                'Field': [
                    'Patient Age', 'Is Deceased', 'Claim Amount',
                    'Claim Duration', 'Deductible', 'Hospital Stay',
                    'Physicians', 'IP Annual Reimbursement',
                    'OP Annual Reimbursement', 'Prediction', 'Fraud Probability'
                ],
                'Value': [
                    f'{age} years',
                    'Yes' if is_deceased == 1 else 'No',
                    f'${claim_amount:,}',
                    f'{claim_duration} days',
                    f'${deductible:,}',
                    f'{hospital_stay} days',
                    str(num_physicians),
                    f'${ip_annual_reimb:,}',
                    f'${op_annual_reimb:,}',
                    '🚨 FRAUD' if prediction == 1 else '✅ LEGITIMATE',
                    f'{fraud_pct}%'
                ]
            }
            df_summary = pd.DataFrame(summary)
            st.dataframe(df_summary, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)