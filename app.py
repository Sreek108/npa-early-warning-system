"""
NPA Early Warning System (EWS) - Professional Dashboard
========================================================
Enterprise-grade interface for predicting and managing potential NPAs.

Features:
- Modern, professional UI with gradient themes
- Real-time portfolio risk analysis
- Interactive visualizations
- Detailed account-level explanations
- Actionable insights and recommendations

Usage:
    streamlit run ews_dashboard_pro.py

Version: 2.0 Professional
Author: AI/ML Analytics Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#import joblib  # Not needed for cloud deployment
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="NPA Early Warning System | EWS Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# PROFESSIONAL CSS STYLING
# =============================================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    .main {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Header */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0c4a6e 100%);
        padding: 2rem 3rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(15, 23, 42, 0.15);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="40" fill="none" stroke="rgba(255,255,255,0.03)" stroke-width="8"/></svg>') repeat;
        background-size: 60px 60px;
    }
    
    .header-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: white;
        margin: 0;
        letter-spacing: -0.5px;
        position: relative;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.7);
        margin-top: 0.5rem;
        font-weight: 400;
        position: relative;
    }
    
    .header-badge {
        display: inline-block;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 30px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 1rem;
        position: relative;
    }
    
    /* Metric Cards */
    .metric-container {
        display: flex;
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
        flex: 1;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.1);
    }
    
    .metric-card.critical {
        border-left: 4px solid #ef4444;
        background: linear-gradient(135deg, #fef2f2 0%, white 100%);
    }
    
    .metric-card.warning {
        border-left: 4px solid #f97316;
        background: linear-gradient(135deg, #fff7ed 0%, white 100%);
    }
    
    .metric-card.success {
        border-left: 4px solid #10b981;
        background: linear-gradient(135deg, #ecfdf5 0%, white 100%);
    }
    
    .metric-card.info {
        border-left: 4px solid #3b82f6;
        background: linear-gradient(135deg, #eff6ff 0%, white 100%);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-change {
        font-size: 0.85rem;
        margin-top: 0.75rem;
        padding: 0.3rem 0.6rem;
        border-radius: 6px;
        display: inline-block;
    }
    
    .metric-change.positive {
        background: #dcfce7;
        color: #166534;
    }
    
    .metric-change.negative {
        background: #fee2e2;
        color: #991b1b;
    }
    
    /* Risk Badges */
    .risk-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.5rem 1rem;
        border-radius: 30px;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .risk-critical {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(249, 115, 22, 0.4);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #eab308 0%, #ca8a04 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(234, 179, 8, 0.4);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(34, 197, 94, 0.4);
    }
    
    .risk-minimal {
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(6, 182, 212, 0.4);
    }
    
    /* Cards */
    .info-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    
    .card-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f1f5f9;
    }
    
    .card-icon {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
    }
    
    .card-icon.blue { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); }
    .card-icon.green { background: linear-gradient(135deg, #10b981 0%, #059669 100%); }
    .card-icon.orange { background: linear-gradient(135deg, #f97316 0%, #ea580c 100%); }
    .card-icon.red { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); }
    .card-icon.purple { background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
    }
    
    /* Account Cards */
    .account-card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .account-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15);
    }
    
    .account-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .account-id {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f172a;
    }
    
    .account-details {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
    }
    
    .account-detail-item {
        text-align: center;
        padding: 0.75rem;
        background: #f8fafc;
        border-radius: 8px;
    }
    
    .detail-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f172a;
    }
    
    .detail-label {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.25rem;
    }
    
    /* Explanation Box */
    .explanation-box {
        background: linear-gradient(135deg, #fefce8 0%, #fef9c3 100%);
        border-left: 4px solid #eab308;
        padding: 1rem 1.25rem;
        border-radius: 0 12px 12px 0;
        margin: 0.75rem 0;
    }
    
    .explanation-title {
        font-weight: 700;
        color: #854d0e;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .risk-factor {
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(0,0,0,0.05);
    }
    
    .risk-factor:last-child {
        border-bottom: none;
    }
    
    .factor-icon {
        font-size: 1rem;
        margin-top: 0.1rem;
    }
    
    .factor-text {
        font-size: 0.9rem;
        color: #1f2937;
        line-height: 1.5;
    }
    
    /* Positive Factor */
    .positive-box {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border-left: 4px solid #10b981;
        padding: 1rem 1.25rem;
        border-radius: 0 12px 12px 0;
        margin: 0.75rem 0;
    }
    
    .positive-title {
        font-weight: 700;
        color: #065f46;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Action Box */
    .action-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.25rem;
        border-radius: 0 12px 12px 0;
        margin: 0.75rem 0;
    }
    
    .action-title {
        font-weight: 700;
        color: #1e40af;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Upload Area */
    .upload-area {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 2px dashed #cbd5e1;
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #3b82f6;
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    }
    
    .upload-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    .upload-text {
        font-size: 1.25rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 0.5rem;
    }
    
    .upload-subtext {
        font-size: 0.9rem;
        color: #64748b;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: white;
        border-radius: 12px;
        padding: 0.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        color: #64748b;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    }
    
    /* Data Table */
    .dataframe {
        border-radius: 12px !important;
        overflow: hidden;
    }
    
    /* Plotly charts background */
    .js-plotly-plot .plotly .bg {
        fill: transparent !important;
    }
    
    /* Stats Row */
    .stats-row {
        display: flex;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .stat-item {
        flex: 1;
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .stat-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0f172a;
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 3rem;
        border-top: 1px solid #e2e8f0;
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-fade-in {
        animation: fadeInUp 0.5s ease-out;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONFIGURATION
# =============================================================================

RISK_THRESHOLDS = {
    'Critical': 70,
    'High': 50,
    'Medium': 30,
    'Low': 15
}

RISK_COLORS = {
    'Critical': '#ef4444',
    'High': '#f97316',
    'Medium': '#eab308',
    'Low': '#22c55e',
    'Very Low': '#06b6d4'
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_risk_category(score):
    """Categorize risk score into risk level."""
    if score >= RISK_THRESHOLDS['Critical']:
        return 'Critical'
    elif score >= RISK_THRESHOLDS['High']:
        return 'High'
    elif score >= RISK_THRESHOLDS['Medium']:
        return 'Medium'
    elif score >= RISK_THRESHOLDS['Low']:
        return 'Low'
    else:
        return 'Very Low'

def get_risk_badge_html(category):
    """Generate HTML for risk badge."""
    icons = {
        'Critical': '🔴',
        'High': '🟠',
        'Medium': '🟡',
        'Low': '🟢',
        'Very Low': '🔵'
    }
    css_class = f"risk-{category.lower().replace(' ', '-')}"
    return f'<span class="risk-badge {css_class}">{icons.get(category, "")} {category}</span>'

def prepare_features(df):
    """Prepare features from uploaded data for prediction."""
    df_processed = df.copy()
    
    # Column mapping (handle various naming conventions)
    column_mapping = {
        'Account_ID': 'account_id',
        'AccountID': 'account_id',
        'account_id': 'account_id',
        'Bureau_Score': 'credit_score',
        'Credit_Score': 'credit_score',
        'credit_score': 'credit_score',
        'Monthly_Income': 'income',
        'Income': 'income',
        'income': 'income',
        'Loan_Amount': 'loan_amount',
        'LoanAmount': 'loan_amount',
        'loan_amount': 'loan_amount',
        'EMI_Amount': 'emi',
        'EMI': 'emi',
        'emi': 'emi',
        'Collection_Calls': 'total_contacts',
        'Total_Contacts': 'total_contacts',
        'total_contacts': 'total_contacts',
        'Bounce_Count': 'bounce_count',
        'Bounces': 'bounce_count',
        'bounce_count': 'bounce_count',
        'Current_DPD': 'current_dpd',
        'DPD': 'current_dpd',
        'current_dpd': 'current_dpd',
        'EMIs_Paid': 'success_count',
        'EMIs_Due': 'payment_count',
        'PTP_Count': 'total_promises',
        'PTP_Kept': 'promises_kept',
        'Credit_Utilization_Pct': 'credit_utilization',
        'Customer_Age': 'age',
        'Age': 'age',
        'Employment_Years': 'employment_years',
        'Product_Type': 'product',
        'Written_Off_Accounts': 'written_off_accounts',
        'Tenure_Months': 'tenure_months'
    }
    
    # Rename columns
    for old_name, new_name in column_mapping.items():
        if old_name in df_processed.columns and old_name != new_name:
            df_processed[new_name] = df_processed[old_name]
    
    # Calculate derived features
    if 'success_count' in df_processed.columns and 'payment_count' in df_processed.columns:
        df_processed['success_rate'] = df_processed['success_count'] / df_processed['payment_count'].replace(0, 1)
        df_processed['miss_rate'] = 1 - df_processed['success_rate']
    
    if 'current_dpd' in df_processed.columns:
        df_processed['avg_days_late'] = df_processed['current_dpd'] / 3
        df_processed['max_days_late'] = df_processed['current_dpd']
    
    if 'credit_score' in df_processed.columns:
        df_processed['is_low_credit'] = (df_processed['credit_score'] < 600).astype(int)
    
    if 'emi' in df_processed.columns and 'income' in df_processed.columns:
        df_processed['debt_to_income'] = df_processed['emi'] / df_processed['income'].replace(0, 1)
    
    if 'loan_amount' in df_processed.columns and 'income' in df_processed.columns:
        df_processed['loan_to_income'] = df_processed['loan_amount'] / df_processed['income'].replace(0, 1)
    
    if 'written_off_accounts' in df_processed.columns:
        df_processed['has_writeoff'] = (df_processed['written_off_accounts'] > 0).astype(int)
    
    if 'total_promises' in df_processed.columns and 'promises_kept' in df_processed.columns:
        df_processed['broken_promises'] = df_processed['total_promises'] - df_processed['promises_kept']
    
    # Employment type encoding
    if 'Employment_Type' in df.columns:
        df_processed['is_salaried'] = (df['Employment_Type'].isin(['Salaried', 'Government'])).astype(int)
    
    return df_processed

def calculate_risk_score(row):
    """Calculate risk score based on available features using rule-based approach."""
    score = 0
    factors = []
    positives = []
    
    # Payment behavior (40% weight)
    if 'success_rate' in row and pd.notna(row.get('success_rate')):
        success_rate = row['success_rate']
        if success_rate < 0.5:
            score += 35
            factors.append(f"Very low payment success rate ({success_rate:.0%})")
        elif success_rate < 0.7:
            score += 25
            factors.append(f"Low payment success rate ({success_rate:.0%})")
        elif success_rate < 0.85:
            score += 15
            factors.append(f"Below average payment success rate ({success_rate:.0%})")
        elif success_rate >= 0.95:
            positives.append(f"Excellent payment record ({success_rate:.0%})")
    
    # DPD (25% weight)
    dpd = row.get('current_dpd', row.get('Current_DPD', 0))
    if pd.notna(dpd):
        if dpd > 90:
            score += 25
            factors.append(f"Severely delinquent ({int(dpd)} days past due)")
        elif dpd > 60:
            score += 20
            factors.append(f"Significantly delinquent ({int(dpd)} days past due)")
        elif dpd > 30:
            score += 15
            factors.append(f"Delinquent ({int(dpd)} days past due)")
        elif dpd > 0:
            score += 8
            factors.append(f"Minor delinquency ({int(dpd)} days past due)")
        else:
            positives.append("Currently up-to-date on payments")
    
    # Collection activity (15% weight)
    contacts = row.get('total_contacts', row.get('Collection_Calls', 0))
    if pd.notna(contacts) and contacts > 0:
        if contacts > 10:
            score += 15
            factors.append(f"Extensive collection activity ({int(contacts)} contacts)")
        elif contacts > 5:
            score += 10
            factors.append(f"Significant collection activity ({int(contacts)} contacts)")
        elif contacts > 2:
            score += 5
            factors.append(f"Some collection activity ({int(contacts)} contacts)")
    
    # Credit score (10% weight)
    credit_score = row.get('credit_score', row.get('Bureau_Score', 700))
    if pd.notna(credit_score):
        if credit_score < 500:
            score += 12
            factors.append(f"Very poor credit score ({int(credit_score)})")
        elif credit_score < 550:
            score += 10
            factors.append(f"Poor credit score ({int(credit_score)})")
        elif credit_score < 600:
            score += 7
            factors.append(f"Below average credit score ({int(credit_score)})")
        elif credit_score >= 720:
            positives.append(f"Strong credit score ({int(credit_score)})")
    
    # Bounces (5% weight)
    bounces = row.get('bounce_count', row.get('Bounce_Count', 0))
    if pd.notna(bounces) and bounces > 0:
        if bounces > 3:
            score += 8
            factors.append(f"Multiple bounced payments ({int(bounces)} bounces)")
        elif bounces > 1:
            score += 5
            factors.append(f"Payment bounces recorded ({int(bounces)} bounces)")
    
    # Broken promises (5% weight)
    broken = row.get('broken_promises', 0)
    if pd.notna(broken) and broken > 0:
        if broken > 2:
            score += 6
            factors.append(f"Multiple broken payment promises ({int(broken)})")
        else:
            score += 3
            factors.append(f"Broken payment promise recorded")
    
    # Credit utilization
    utilization = row.get('credit_utilization', row.get('Credit_Utilization_Pct', 0))
    if pd.notna(utilization):
        if utilization > 90:
            score += 5
            factors.append(f"Very high credit utilization ({utilization:.0f}%)")
        elif utilization > 75:
            score += 3
            factors.append(f"High credit utilization ({utilization:.0f}%)")
        elif utilization < 30:
            positives.append(f"Healthy credit utilization ({utilization:.0f}%)")
    
    # Employment (positive factor)
    is_salaried = row.get('is_salaried', 0)
    emp_type = row.get('Employment_Type', '')
    if is_salaried or emp_type in ['Salaried', 'Government']:
        positives.append("Stable employment (salaried/government)")
    
    # Cap score at 100
    score = min(100, max(0, score))
    
    return score, factors, positives

def get_action_recommendation(risk_category, score):
    """Get recommended action based on risk category."""
    actions = {
        'Critical': {
            'urgency': '🚨 IMMEDIATE',
            'timeline': 'Within 24 hours',
            'action': 'Escalate to senior collection team. Consider restructuring options. Initiate legal review if applicable.',
            'color': '#ef4444'
        },
        'High': {
            'urgency': '⚠️ URGENT',
            'timeline': 'Within 48 hours',
            'action': 'Priority outbound call. Discuss payment plan options. Schedule follow-up within 1 week.',
            'color': '#f97316'
        },
        'Medium': {
            'urgency': '📋 MONITOR',
            'timeline': 'Within 1 week',
            'action': 'Add to watchlist. Send payment reminder. Review account in next collection cycle.',
            'color': '#eab308'
        },
        'Low': {
            'urgency': '✅ ROUTINE',
            'timeline': 'Standard cycle',
            'action': 'Continue normal monitoring. No immediate action required.',
            'color': '#22c55e'
        },
        'Very Low': {
            'urgency': '💚 HEALTHY',
            'timeline': 'No action needed',
            'action': 'Account in good standing. Consider for pre-approved offers or loyalty programs.',
            'color': '#06b6d4'
        }
    }
    return actions.get(risk_category, actions['Low'])

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 class="header-title">🛡️ NPA Early Warning System</h1>
        <p class="header-subtitle">AI-Powered Portfolio Risk Intelligence Platform</p>
        <span class="header-badge">✨ Version 2.0 Professional</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'predictions_made' not in st.session_state:
        st.session_state.predictions_made = False
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Upload & Predict", 
        "📊 Portfolio Analysis", 
        "🎯 Action Center",
        "📈 Risk Trends"
    ])
    
    # =========================================================================
    # TAB 1: UPLOAD & PREDICT
    # =========================================================================
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="info-card">
                <div class="card-header">
                    <div class="card-icon blue">📁</div>
                    <h3 class="card-title">Upload Portfolio Data</h3>
                </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Drag and drop or click to upload",
                type=['csv', 'xlsx', 'xls'],
                help="Upload your portfolio data in CSV or Excel format"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            if uploaded_file:
                try:
                    # Load data
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.success(f"✅ Successfully loaded **{len(df):,}** accounts")
                    
                    # Preview
                    with st.expander("📋 Preview Data", expanded=False):
                        st.dataframe(df.head(10), use_container_width=True)
                    
                    # Predict button
                    if st.button("🚀 Run Risk Analysis", use_container_width=True, type="primary"):
                        with st.spinner("🔄 Analyzing portfolio risk..."):
                            # Prepare features
                            df_processed = prepare_features(df)
                            
                            # Calculate risk scores
                            results = []
                            progress_bar = st.progress(0)
                            
                            for idx, row in df_processed.iterrows():
                                score, factors, positives = calculate_risk_score(row)
                                category = get_risk_category(score)
                                action = get_action_recommendation(category, score)
                                
                                results.append({
                                    'account_id': row.get('account_id', row.get('Account_ID', f'ACC{idx}')),
                                    'risk_score': score,
                                    'risk_category': category,
                                    'risk_factors': ' | '.join(factors) if factors else 'No significant risk factors',
                                    'positive_factors': ' | '.join(positives) if positives else 'None identified',
                                    'recommended_action': action['action'],
                                    'urgency': action['urgency'],
                                    'timeline': action['timeline']
                                })
                                
                                if idx % 100 == 0:
                                    progress_bar.progress(min(idx / len(df_processed), 1.0))
                            
                            progress_bar.progress(1.0)
                            
                            # Create results dataframe
                            results_df = pd.DataFrame(results)
                            
                            # Merge with original data
                            if 'Account_ID' in df.columns:
                                results_df = results_df.merge(
                                    df, 
                                    left_on='account_id', 
                                    right_on='Account_ID', 
                                    how='left'
                                )
                            
                            st.session_state.results_df = results_df
                            st.session_state.predictions_made = True
                            
                            st.success("✅ Risk analysis complete!")
                            st.balloons()
                            
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
        
        with col2:
            st.markdown("""
            <div class="info-card">
                <div class="card-header">
                    <div class="card-icon green">📊</div>
                    <h3 class="card-title">Quick Guide</h3>
                </div>
                <div style="color: #475569; line-height: 1.8;">
                    <p><strong>Required columns:</strong></p>
                    <ul style="margin-left: 1rem;">
                        <li>Account_ID</li>
                        <li>EMIs_Due, EMIs_Paid</li>
                        <li>Current_DPD</li>
                        <li>Bureau_Score</li>
                    </ul>
                    <p style="margin-top: 1rem;"><strong>Optional columns:</strong></p>
                    <ul style="margin-left: 1rem;">
                        <li>Collection_Calls</li>
                        <li>Bounce_Count</li>
                        <li>PTP_Count, PTP_Kept</li>
                        <li>Loan_Amount, EMI_Amount</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Download sample file
            st.markdown("""
            <div class="info-card" style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);">
                <div class="card-header">
                    <div class="card-icon purple">📥</div>
                    <h3 class="card-title">Sample Template</h3>
                </div>
            """, unsafe_allow_html=True)
            
            sample_df = pd.DataFrame({
                'Account_ID': ['LN2024000001', 'LN2024000002'],
                'Product_Type': ['Personal Loan', 'Auto Loan'],
                'Loan_Amount': [25000, 50000],
                'EMI_Amount': [1200, 1800],
                'EMIs_Due': [12, 8],
                'EMIs_Paid': [10, 8],
                'Current_DPD': [45, 0],
                'Bureau_Score': [580, 720],
                'Collection_Calls': [5, 0],
                'Bounce_Count': [2, 0]
            })
            
            st.download_button(
                label="⬇️ Download Sample CSV",
                data=sample_df.to_csv(index=False),
                file_name="ews_sample_template.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # =========================================================================
    # TAB 2: PORTFOLIO ANALYSIS
    # =========================================================================
    with tab2:
        if st.session_state.predictions_made and st.session_state.results_df is not None:
            results_df = st.session_state.results_df
            
            # Key Metrics Row
            st.markdown("""
            <div class="info-card">
                <div class="card-header">
                    <div class="card-icon blue">📈</div>
                    <h3 class="card-title">Portfolio Risk Summary</h3>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            total = len(results_df)
            critical = (results_df['risk_category'] == 'Critical').sum()
            high = (results_df['risk_category'] == 'High').sum()
            medium = (results_df['risk_category'] == 'Medium').sum()
            low_risk = (results_df['risk_category'].isin(['Low', 'Very Low'])).sum()
            
            with col1:
                st.metric("📊 Total Accounts", f"{total:,}")
            with col2:
                st.metric("🔴 Critical Risk", f"{critical:,}", delta=f"{critical/total*100:.1f}%", delta_color="inverse")
            with col3:
                st.metric("🟠 High Risk", f"{high:,}", delta=f"{high/total*100:.1f}%", delta_color="inverse")
            with col4:
                st.metric("🟡 Medium Risk", f"{medium:,}", delta=f"{medium/total*100:.1f}%", delta_color="off")
            with col5:
                st.metric("🟢 Low/Healthy", f"{low_risk:,}", delta=f"{low_risk/total*100:.1f}%")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Charts Row
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="info-card">
                    <div class="card-header">
                        <div class="card-icon orange">🎯</div>
                        <h3 class="card-title">Risk Distribution</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                risk_counts = results_df['risk_category'].value_counts()
                
                # Ensure order
                order = ['Critical', 'High', 'Medium', 'Low', 'Very Low']
                risk_counts = risk_counts.reindex([x for x in order if x in risk_counts.index])
                
                fig = go.Figure(data=[go.Pie(
                    labels=risk_counts.index,
                    values=risk_counts.values,
                    hole=0.6,
                    marker_colors=[RISK_COLORS.get(cat, '#gray') for cat in risk_counts.index],
                    textinfo='label+percent',
                    textfont_size=12,
                    hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>'
                )])
                
                fig.update_layout(
                    showlegend=False,
                    margin=dict(t=20, b=20, l=20, r=20),
                    height=350,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    annotations=[dict(
                        text=f'<b>{total:,}</b><br>Total',
                        x=0.5, y=0.5,
                        font_size=18,
                        showarrow=False
                    )]
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="info-card">
                    <div class="card-header">
                        <div class="card-icon red">📊</div>
                        <h3 class="card-title">Risk Score Distribution</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                fig = go.Figure()
                
                fig.add_trace(go.Histogram(
                    x=results_df['risk_score'],
                    nbinsx=20,
                    marker_color='#3b82f6',
                    marker_line_color='white',
                    marker_line_width=1,
                    opacity=0.8,
                    hovertemplate='Risk Score: %{x}<br>Count: %{y}<extra></extra>'
                ))
                
                # Add threshold lines
                fig.add_vline(x=70, line_dash="dash", line_color="#ef4444", 
                             annotation_text="Critical", annotation_position="top")
                fig.add_vline(x=50, line_dash="dash", line_color="#f97316",
                             annotation_text="High", annotation_position="top")
                fig.add_vline(x=30, line_dash="dash", line_color="#eab308",
                             annotation_text="Medium", annotation_position="top")
                
                fig.update_layout(
                    xaxis_title="Risk Score",
                    yaxis_title="Number of Accounts",
                    margin=dict(t=40, b=40, l=40, r=40),
                    height=350,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(gridcolor='#e2e8f0'),
                    yaxis=dict(gridcolor='#e2e8f0')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Product Analysis
            if 'Product_Type' in results_df.columns:
                st.markdown("""
                <div class="info-card">
                    <div class="card-header">
                        <div class="card-icon purple">🏦</div>
                        <h3 class="card-title">Risk by Product</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                product_risk = results_df.groupby('Product_Type').agg({
                    'risk_score': 'mean',
                    'account_id': 'count'
                }).round(1).reset_index()
                product_risk.columns = ['Product', 'Avg Risk Score', 'Count']
                product_risk = product_risk.sort_values('Avg Risk Score', ascending=True)
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    y=product_risk['Product'],
                    x=product_risk['Avg Risk Score'],
                    orientation='h',
                    marker_color=['#ef4444' if x > 50 else '#f97316' if x > 30 else '#22c55e' 
                                  for x in product_risk['Avg Risk Score']],
                    text=[f"{x:.1f}" for x in product_risk['Avg Risk Score']],
                    textposition='outside',
                    hovertemplate='<b>%{y}</b><br>Avg Risk: %{x:.1f}<extra></extra>'
                ))
                
                fig.update_layout(
                    xaxis_title="Average Risk Score",
                    margin=dict(t=20, b=40, l=120, r=40),
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(gridcolor='#e2e8f0', range=[0, 100]),
                    yaxis=dict(gridcolor='#e2e8f0')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Download Section
            st.markdown("""
            <div class="info-card">
                <div class="card-header">
                    <div class="card-icon green">📥</div>
                    <h3 class="card-title">Download Results</h3>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.download_button(
                    "📊 All Results",
                    results_df.to_csv(index=False),
                    "ews_all_results.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                critical_high = results_df[results_df['risk_category'].isin(['Critical', 'High'])]
                st.download_button(
                    f"🔴 Critical & High ({len(critical_high):,})",
                    critical_high.to_csv(index=False),
                    "ews_critical_high_risk.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col3:
                critical_only = results_df[results_df['risk_category'] == 'Critical']
                st.download_button(
                    f"🚨 Critical Only ({len(critical_only):,})",
                    critical_only.to_csv(index=False),
                    "ews_critical_risk.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col4:
                action_list = results_df[results_df['risk_category'].isin(['Critical', 'High', 'Medium'])][
                    ['account_id', 'risk_score', 'risk_category', 'urgency', 'timeline', 'recommended_action']
                ].sort_values('risk_score', ascending=False)
                st.download_button(
                    f"📋 Action List ({len(action_list):,})",
                    action_list.to_csv(index=False),
                    "ews_action_list.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("📤 Please upload portfolio data and run risk analysis first.")
    
    # =========================================================================
    # TAB 3: ACTION CENTER
    # =========================================================================
    with tab3:
        if st.session_state.predictions_made and st.session_state.results_df is not None:
            results_df = st.session_state.results_df
            
            st.markdown("""
            <div class="info-card">
                <div class="card-header">
                    <div class="card-icon red">🎯</div>
                    <h3 class="card-title">High Priority Accounts</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Filter controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                risk_filter = st.multiselect(
                    "Risk Category",
                    ['Critical', 'High', 'Medium', 'Low', 'Very Low'],
                    default=['Critical', 'High']
                )
            
            with col2:
                if 'Product_Type' in results_df.columns:
                    products = ['All'] + list(results_df['Product_Type'].unique())
                    product_filter = st.selectbox("Product", products)
                else:
                    product_filter = 'All'
            
            with col3:
                sort_by = st.selectbox("Sort By", ['Risk Score (High to Low)', 'Risk Score (Low to High)'])
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Filter data
            filtered_df = results_df[results_df['risk_category'].isin(risk_filter)]
            
            if product_filter != 'All' and 'Product_Type' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Product_Type'] == product_filter]
            
            # Sort
            ascending = 'Low to High' in sort_by
            filtered_df = filtered_df.sort_values('risk_score', ascending=ascending)
            
            st.markdown(f"**Showing {len(filtered_df):,} accounts**")
            
            # Display accounts
            for idx, row in filtered_df.head(20).iterrows():
                category = row['risk_category']
                score = row['risk_score']
                action = get_action_recommendation(category, score)
                
                with st.expander(
                    f"{row['account_id']} | {get_risk_badge_html(category)} | Score: {score:.0f}",
                    expanded=False
                ):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Account details
                        st.markdown("**📋 Account Details**")
                        
                        detail_cols = st.columns(4)
                        
                        if 'Loan_Amount' in row:
                            detail_cols[0].metric("Loan Amount", f"{row['Loan_Amount']:,.0f} SAR")
                        if 'Current_DPD' in row:
                            detail_cols[1].metric("Current DPD", f"{int(row['Current_DPD'])} days")
                        if 'Bureau_Score' in row:
                            detail_cols[2].metric("Credit Score", f"{int(row['Bureau_Score'])}")
                        if 'Collection_Calls' in row:
                            detail_cols[3].metric("Collection Calls", f"{int(row['Collection_Calls'])}")
                        
                        # Risk factors
                        if row['risk_factors'] and row['risk_factors'] != 'No significant risk factors':
                            st.markdown(f"""
                            <div class="explanation-box">
                                <div class="explanation-title">⚠️ Risk Factors</div>
                            """, unsafe_allow_html=True)
                            
                            for factor in row['risk_factors'].split(' | '):
                                st.markdown(f"• {factor}")
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Positive factors
                        if row['positive_factors'] and row['positive_factors'] != 'None identified':
                            st.markdown(f"""
                            <div class="positive-box">
                                <div class="positive-title">✅ Positive Factors</div>
                            """, unsafe_allow_html=True)
                            
                            for factor in row['positive_factors'].split(' | '):
                                st.markdown(f"• {factor}")
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col2:
                        # Action recommendation
                        st.markdown(f"""
                        <div class="action-box">
                            <div class="action-title">📌 Recommended Action</div>
                            <p><strong>Urgency:</strong> {action['urgency']}</p>
                            <p><strong>Timeline:</strong> {action['timeline']}</p>
                            <p><strong>Action:</strong> {action['action']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            if len(filtered_df) > 20:
                st.info(f"Showing top 20 of {len(filtered_df):,} accounts. Download the action list for complete data.")
        else:
            st.info("📤 Please upload portfolio data and run risk analysis first.")
    
    # =========================================================================
    # TAB 4: RISK TRENDS
    # =========================================================================
    with tab4:
        if st.session_state.predictions_made and st.session_state.results_df is not None:
            results_df = st.session_state.results_df
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="info-card">
                    <div class="card-header">
                        <div class="card-icon blue">💰</div>
                        <h3 class="card-title">Exposure at Risk</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                if 'Loan_Amount' in results_df.columns:
                    exposure = results_df.groupby('risk_category')['Loan_Amount'].sum().reset_index()
                    exposure.columns = ['Risk Category', 'Exposure']
                    
                    # Order categories
                    order = ['Critical', 'High', 'Medium', 'Low', 'Very Low']
                    exposure['order'] = exposure['Risk Category'].apply(lambda x: order.index(x) if x in order else 99)
                    exposure = exposure.sort_values('order')
                    
                    fig = go.Figure(data=[go.Bar(
                        x=exposure['Risk Category'],
                        y=exposure['Exposure'] / 1e6,
                        marker_color=[RISK_COLORS.get(cat, '#gray') for cat in exposure['Risk Category']],
                        text=[f"{x/1e6:.1f}M" for x in exposure['Exposure']],
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>Exposure: %{y:.1f}M SAR<extra></extra>'
                    )])
                    
                    fig.update_layout(
                        yaxis_title="Exposure (Million SAR)",
                        margin=dict(t=20, b=40, l=40, r=40),
                        height=350,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(gridcolor='#e2e8f0'),
                        yaxis=dict(gridcolor='#e2e8f0')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary metrics
                    total_exposure = results_df['Loan_Amount'].sum()
                    at_risk = results_df[results_df['risk_category'].isin(['Critical', 'High'])]['Loan_Amount'].sum()
                    
                    st.markdown(f"""
                    <div style="display: flex; gap: 1rem; margin-top: 1rem;">
                        <div style="flex:1; background: #f8fafc; padding: 1rem; border-radius: 10px; text-align: center;">
                            <div style="font-size: 1.5rem; font-weight: 700; color: #0f172a;">{total_exposure/1e6:.1f}M SAR</div>
                            <div style="font-size: 0.8rem; color: #64748b;">Total Portfolio</div>
                        </div>
                        <div style="flex:1; background: #fef2f2; padding: 1rem; border-radius: 10px; text-align: center;">
                            <div style="font-size: 1.5rem; font-weight: 700; color: #ef4444;">{at_risk/1e6:.1f}M SAR</div>
                            <div style="font-size: 0.8rem; color: #64748b;">At High Risk</div>
                        </div>
                        <div style="flex:1; background: #fef2f2; padding: 1rem; border-radius: 10px; text-align: center;">
                            <div style="font-size: 1.5rem; font-weight: 700; color: #ef4444;">{at_risk/total_exposure*100:.1f}%</div>
                            <div style="font-size: 0.8rem; color: #64748b;">% At Risk</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Loan amount data not available")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="info-card">
                    <div class="card-header">
                        <div class="card-icon orange">📍</div>
                        <h3 class="card-title">Risk by Geography</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                if 'City' in results_df.columns:
                    city_risk = results_df.groupby('City').agg({
                        'risk_score': 'mean',
                        'account_id': 'count'
                    }).round(1).reset_index()
                    city_risk.columns = ['City', 'Avg Risk', 'Count']
                    city_risk = city_risk.sort_values('Avg Risk', ascending=False).head(8)
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=city_risk['City'],
                        y=city_risk['Avg Risk'],
                        marker_color=['#ef4444' if x > 50 else '#f97316' if x > 30 else '#22c55e' 
                                      for x in city_risk['Avg Risk']],
                        text=[f"{x:.0f}" for x in city_risk['Avg Risk']],
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>Avg Risk: %{y:.1f}<br>Accounts: ' + 
                                      city_risk['Count'].astype(str) + '<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        yaxis_title="Average Risk Score",
                        margin=dict(t=20, b=80, l=40, r=40),
                        height=350,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(gridcolor='#e2e8f0', tickangle=45),
                        yaxis=dict(gridcolor='#e2e8f0', range=[0, 100])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("City data not available")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Risk Heatmap by Product and DPD
            if 'Product_Type' in results_df.columns and 'DPD_Bucket' in results_df.columns:
                st.markdown("""
                <div class="info-card">
                    <div class="card-header">
                        <div class="card-icon purple">🗺️</div>
                        <h3 class="card-title">Risk Heatmap: Product × DPD Bucket</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                heatmap_data = results_df.pivot_table(
                    values='risk_score',
                    index='Product_Type',
                    columns='DPD_Bucket',
                    aggfunc='mean'
                ).round(0)
                
                # Reorder columns
                dpd_order = ['Current', '1-30 DPD', '31-60 DPD', '61-90 DPD', '90+ DPD']
                heatmap_data = heatmap_data[[col for col in dpd_order if col in heatmap_data.columns]]
                
                fig = go.Figure(data=go.Heatmap(
                    z=heatmap_data.values,
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    colorscale=[[0, '#22c55e'], [0.3, '#eab308'], [0.6, '#f97316'], [1, '#ef4444']],
                    text=heatmap_data.values.astype(int),
                    texttemplate='%{text}',
                    textfont={"size": 14, "color": "white"},
                    hovertemplate='<b>%{y}</b><br>%{x}<br>Avg Risk: %{z:.0f}<extra></extra>'
                ))
                
                fig.update_layout(
                    margin=dict(t=20, b=40, l=120, r=40),
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis_title="DPD Bucket",
                    yaxis_title=""
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("📤 Please upload portfolio data and run risk analysis first.")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🛡️ <strong>NPA Early Warning System</strong> | Version 2.0 Professional</p>
        <p>Powered by AI/ML Analytics | © 2025 All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
