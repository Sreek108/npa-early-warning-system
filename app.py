"""
NPA Early Warning System - Business Intelligence Edition
=========================================================
TRUE ML Model + Comprehensive Business Insights

Model: Logistic Regression (ROC-AUC: 0.889)
Features: 70 engineered | Training: 20,000 accounts

Business Features:
- Executive Dashboard with KPIs
- Financial Exposure Analysis
- Product & Geographic Segmentation
- Collection Efficiency Metrics
- Loss Prevention Projections
- Actionable Recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="NPA Early Warning System", page_icon="🛡️", layout="wide", initial_sidebar_state="collapsed")

# EMBEDDED ML MODEL PARAMETERS
FEATURE_NAMES = ['loan_amount', 'tenure_months', 'apr', 'emi', 'age', 'income', 'credit_score', 'employment_years', 'dependents', 'debt_to_income', 'loan_to_income', 'loan_burden', 'is_high_risk_age', 'is_salaried', 'is_long_tenure_loan', 'is_low_credit', 'total_paid', 'avg_payment', 'std_payment', 'payment_count', 'avg_days_late', 'max_days_late', 'min_days_late', 'bounce_count', 'success_count', 'partial_count', 'missed_count', 'bounced_count', 'success_rate', 'miss_rate', 'bounce_rate', 'credit_score_bureau', 'total_accounts', 'active_accounts', 'overdue_accounts', 'written_off_accounts', 'total_outstanding', 'credit_utilization_pct', 'enquiries_last_30_days', 'enquiries_last_6_months', 'enquiries_last_12_months', 'worst_dpd_last_6_months', 'worst_dpd_last_12_months', 'worst_dpd_last_24_months', 'average_account_age_months', 'overdue_ratio', 'utilization_risk', 'enquiry_velocity', 'has_writeoff', 'dpd_trend', 'total_contacts', 'total_promises', 'promises_kept', 'promise_rate', 'promise_kept_rate', 'broken_promises', 'product_Jarir', 'product_Other Partners', 'gender_Male', 'employment_type_Salaried', 'employment_type_Self-employed', 'city_Jeddah', 'city_Madinah', 'city_Makkah', 'city_Other', 'city_Riyadh', 'residence_type_Owned', 'residence_type_Rented', 'marital_status_Married', 'marital_status_Single']
MODEL_COEFFICIENTS = [0.0012139695568097394, 0.1606967474364933, 0.24039768791868996, 0.06862933962204006, -0.12533954783858348, 0.008363098100807174, -0.009778870364575535, -0.13776551832552414, 0.0336634113863695, -0.016254324897132452, 0.00419992586500719, -0.05631765898794028, 0.16385446309914142, 0.008972414102203967, 0.030300825045448634, 0.08275730487337445, -0.21566108496735137, -0.03403771673756947, -0.18470538514831042, -0.12206845153826892, 0.27934341280860336, -0.2820864378446384, -0.12522424259618253, 0.3756345883061301, -0.4442758760393322, 0.24044206414265828, 0.5991307228376644, 0.3756345883061301, -0.24919162079604165, 0.2793098731946986, -0.004154783492937004, -0.009778870364575535, 1.1592657993176392, 0.9827531021817281, 1.4759705067358466, 0.41279402600181736, 0.06303312646676104, 2.4392409629357727, 0.07562388128955616, 1.876654047040677, 0.7695981581571449, 0.52849423736062, 0.6548341780050408, 0.3408926437824525, 0.02436330575984251, 0.7344427690693441, 1.1875712295531096, 1.8766540470406763, 0.412794026001832, 0.0171800468432064, 0.8434003835997356, 0.7023847527964997, 0.3611799722622628, 0.004098827425142036, -0.04628966432737171, 0.7330283594767575, 0.0572442445255684, 0.07707350058168264, 0.17768826096141765, 0.008972414102203967, 0.1739088264747969, 0.03837550292850642, -0.03567492919691276, -0.07293205727780791, 0.16538955399635816, 0.0065573492458582605, -0.007834145323690927, 0.12149518319764079, 0.061390973164710524, 0.08203019866144654]
MODEL_INTERCEPT = -2.132908689570047
SCALER_MEAN = [10354.6276875, 25.29925, 24.07276, 572.483875, 37.707625, 7320.589375, 639.59425, 4.594125, 2.0450625, 0.08332764155038215, 1.5020932413688082, 0.06642261795992963, 0.1258125, 0.8485, 0.3993125, 0.305375, 4587.981128124999, 498.5316983938694, 98.04166163133672, 9.3975, 104.05508437611203, 359.8924375, 2.273625, 0.6535625, 7.615, 0.6566875, 0.47225, 0.6535625, 0.8294709595500497, 0.10189317142722473, 0.06009897933459514, 639.59425, 5.8450625, 5.4035, 0.642625, 0.03975, 21768.1551875, 44.387175000000006, 0.4083125, 2.341875, 4.342, 12.97125, 16.08375, 25.141875, 41.659875, 0.09943824751637251, 0.1868125, 0.3903125, 0.03975, -12.170625, 4.2144375, 1.253875, 0.3269375, 0.097907948577528, 0.10321781460870935, 0.9269375, 0.150125, 0.485125, 0.6305, 0.8485, 0.1024375, 0.25125, 0.0805625, 0.1006875, 0.072625, 0.344375, 0.2975625, 0.5518125, 0.70125, 0.2473125]
SCALER_SCALE = [6106.298635329007, 7.945184040505293, 5.010198400003737, 384.3385408594673, 9.437790623836438, 4874.094702071566, 79.52284808114898, 5.035525343434883, 1.3473703540948754, 0.04014183048934514, 0.5560106859956845, 0.035839663041248214, 0.33163792733001757, 0.3585355630896327, 0.489757110559663, 0.4605660749284516, 4542.1440353114585, 365.4625119788922, 141.83069016179402, 4.927701670150092, 193.0352898426804, 474.2364663254067, 44.63847420509997, 1.405762625265642, 4.645847608348771, 0.9522730844373111, 1.177541904774518, 1.405762625265642, 0.24424504821896717, 0.1931941083550791, 0.12316614339530088, 79.52284808114898, 3.043424365922989, 2.683153508467229, 0.9639284773130214, 0.19537128115462618, 14273.627606095071, 22.287536438991523, 0.6703308156005883, 2.5527184498833786, 2.908463683802842, 21.923655567388845, 30.06098777381575, 33.089724560418674, 17.353891926146567, 0.1516078538343699, 0.38976093935096934, 0.4254530749805631, 0.19537128115462618, 17.839412465363733, 9.003656427174114, 3.176424481138344, 0.8138177751153817, 0.197573995784264, 0.2632517801917319, 2.642650065955338, 0.3571939030484703, 0.49977868539484555, 0.4826694003145424, 0.3585355630896327, 0.3032227870621699, 0.43373198809864133, 0.27216205391962706, 0.30091448510124935, 0.2595199594925215, 0.4751640341766199, 0.45718602187047447, 0.4973082191596575, 0.4577099927901946, 0.4314499129027029]
RISK_COLORS = {'Critical': '#ef4444', 'High': '#f97316', 'Medium': '#eab308', 'Low': '#22c55e', 'Very Low': '#06b6d4'}

# CSS
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.main{background:linear-gradient(180deg,#f8fafc,#f1f5f9)}*{font-family:'Inter',sans-serif}#MainMenu,footer,header{visibility:hidden}
.header{background:linear-gradient(135deg,#0f172a,#1e3a5f,#0c4a6e);padding:1.5rem 2rem;border-radius:16px;margin-bottom:1.5rem;color:white}
.header h1{font-size:2rem;font-weight:800;margin:0}.header p{opacity:0.8;margin-top:0.3rem}
.badge{display:inline-block;padding:0.3rem 0.8rem;border-radius:20px;font-size:0.75rem;font-weight:600;margin-right:0.5rem;margin-top:0.75rem}
.badge-green{background:linear-gradient(135deg,#10b981,#059669);color:white}
.badge-purple{background:linear-gradient(135deg,#8b5cf6,#7c3aed);color:white}
.kpi-card{background:white;border-radius:12px;padding:1.25rem;box-shadow:0 2px 8px rgba(0,0,0,0.04);border:1px solid #e2e8f0;text-align:center}
.kpi-value{font-size:1.75rem;font-weight:800;color:#0f172a}.kpi-label{font-size:0.75rem;color:#64748b;text-transform:uppercase;margin-top:0.25rem}
.card{background:white;border-radius:14px;padding:1.25rem;box-shadow:0 2px 10px rgba(0,0,0,0.04);border:1px solid #e2e8f0;margin-bottom:1rem}
.card-title{font-size:1rem;font-weight:700;color:#0f172a;margin-bottom:1rem;display:flex;align-items:center;gap:0.5rem}
.alert{padding:1rem;border-radius:10px;margin-bottom:0.75rem;border-left:4px solid}
.alert-critical{background:#fef2f2;border-color:#ef4444}.alert-warning{background:#fffbeb;border-color:#f59e0b}
.alert-success{background:#f0fdf4;border-color:#22c55e}.alert-info{background:#eff6ff;border-color:#3b82f6}
.metric-row{display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid #f1f5f9}
.metric-row:last-child{border:none}.stTabs [data-baseweb="tab-list"]{background:white;border-radius:10px;padding:0.3rem}
.stTabs [data-baseweb="tab"]{font-weight:600}.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#3b82f6,#2563eb);color:white!important;border-radius:8px}
</style>""", unsafe_allow_html=True)

# ML FUNCTIONS
def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def predict_ml(X):
    X = np.array(X, dtype=np.float64)
    X_scaled = (X - np.array(SCALER_MEAN)) / np.array(SCALER_SCALE)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return sigmoid(np.dot(X_scaled, np.array(MODEL_COEFFICIENTS)) + MODEL_INTERCEPT)

def prepare_features(df):
    data = df.copy()
    col_map = {'Account_ID':'account_id','Bureau_Score':'credit_score','Credit_Score':'credit_score','Monthly_Income':'income','Income':'income','Loan_Amount':'loan_amount','EMI_Amount':'emi','EMI':'emi','Collection_Calls':'total_contacts','Bounce_Count':'bounce_count','Current_DPD':'current_dpd','EMIs_Paid':'emis_paid','EMIs_Due':'emis_due','PTP_Count':'total_promises','PTP_Kept':'promises_kept','Credit_Utilization_Pct':'credit_utilization_pct','Customer_Age':'age','Age':'age','Employment_Years':'employment_years','Product_Type':'product','Written_Off_Accounts':'written_off_accounts','Tenure_Months':'tenure_months','Interest_Rate':'apr','Dependents':'dependents','Gender':'gender','City':'city','Employment_Type':'employment_type','Residence_Type':'residence_type','Marital_Status':'marital_status','Total_Trade_Lines':'total_accounts','Active_Accounts':'active_accounts','Delinquent_Accounts':'overdue_accounts','Enquiries_Last_3M':'enquiries_last_30_days','Enquiries_Last_6M':'enquiries_last_6_months','Enquiries_Last_12M':'enquiries_last_12_months','Worst_DPD_Last_12M':'worst_dpd_last_12_months','Outstanding_Balance':'total_outstanding'}
    for old,new in col_map.items():
        if old in data.columns: data[new] = data[old]
    
    defaults = {'loan_amount':10000,'tenure_months':24,'apr':24,'emi':500,'age':35,'income':7000,'credit_score':640,'employment_years':5,'dependents':2,'emis_due':10,'emis_paid':8,'current_dpd':0,'total_contacts':0,'bounce_count':0,'total_promises':0,'promises_kept':0,'credit_utilization_pct':45,'total_accounts':6,'active_accounts':5,'overdue_accounts':0,'written_off_accounts':0,'total_outstanding':20000,'enquiries_last_30_days':0,'enquiries_last_6_months':2,'enquiries_last_12_months':4,'worst_dpd_last_6_months':0,'worst_dpd_last_12_months':0,'worst_dpd_last_24_months':0,'average_account_age_months':40}
    for col,val in defaults.items():
        if col not in data.columns: data[col] = val
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(val)
    
    data['debt_to_income'] = data['emi'] / data['income'].replace(0,1)
    data['loan_to_income'] = data['loan_amount'] / data['income'].replace(0,1)
    data['loan_burden'] = data['loan_amount'] / (data['income'] * data['tenure_months']).replace(0,1)
    data['is_high_risk_age'] = ((data['age']<25)|(data['age']>55)).astype(int)
    data['is_long_tenure_loan'] = (data['tenure_months']>24).astype(int)
    data['is_low_credit'] = (data['credit_score']<600).astype(int)
    data['payment_count'] = data['emis_due']
    data['success_count'] = data['emis_paid']
    data['missed_count'] = (data['emis_due']-data['emis_paid']).clip(lower=0)
    data['success_rate'] = data['success_count'] / data['payment_count'].replace(0,1)
    data['miss_rate'] = 1 - data['success_rate']
    data['bounced_count'] = data['bounce_count']
    data['bounce_rate'] = data['bounce_count'] / data['payment_count'].replace(0,1)
    data['total_paid'] = data['success_count'] * data['emi']
    data['avg_payment'] = data['emi']
    data['std_payment'] = data['emi'] * 0.2
    data['avg_days_late'] = data['current_dpd'] / 2
    data['max_days_late'] = data['current_dpd']
    data['min_days_late'] = 0
    data['partial_count'] = 0
    data['credit_score_bureau'] = data['credit_score']
    data['overdue_ratio'] = data['overdue_accounts'] / data['total_accounts'].replace(0,1)
    data['utilization_risk'] = data['credit_utilization_pct'] / 100
    data['enquiry_velocity'] = data['enquiries_last_6_months'] / 6
    data['has_writeoff'] = (data['written_off_accounts']>0).astype(int)
    data['dpd_trend'] = data['current_dpd'] - data['worst_dpd_last_12_months']
    data['broken_promises'] = (data['total_promises']-data['promises_kept']).clip(lower=0)
    data['promise_rate'] = np.where(data['total_contacts']>0, data['total_promises']/data['total_contacts'], 0)
    data['promise_kept_rate'] = np.where(data['total_promises']>0, data['promises_kept']/data['total_promises'], 0)
    data['is_salaried'] = df['employment_type'].isin(['Salaried','Government']).astype(int) if 'employment_type' in df.columns else 1
    
    for feat in ['product_Jarir','product_Other Partners','gender_Male','employment_type_Salaried','employment_type_Self-employed','city_Jeddah','city_Madinah','city_Makkah','city_Other','city_Riyadh','residence_type_Owned','residence_type_Rented','marital_status_Married','marital_status_Single']:
        data[feat] = 0
    if 'product' in data.columns:
        data['product_Jarir'] = (data['product']=='Jarir').astype(int)
        data['product_Other Partners'] = data['product'].isin(['Other Partners','Consumer Durable','Gold Loan','Auto Loan']).astype(int)
    if 'gender' in data.columns: data['gender_Male'] = (data['gender']=='Male').astype(int)
    if 'city' in data.columns:
        for c in ['Jeddah','Madinah','Makkah','Other','Riyadh']: data[f'city_{c}'] = (data['city']==c).astype(int)
    data['employment_type_Salaried'] = data.get('is_salaried',1)
    data['marital_status_Married'] = 1
    for f in FEATURE_NAMES:
        if f not in data.columns: data[f] = 0
    return data

def get_risk_category(prob):
    s = prob * 100
    if s >= 70: return 'Critical'
    elif s >= 50: return 'High'
    elif s >= 30: return 'Medium'
    elif s >= 15: return 'Low'
    return 'Very Low'

def get_risk_factors(row):
    factors, positives = [], []
    util = row.get('credit_utilization_pct',45)
    if util > 80: factors.append(f"🔴 High credit utilization ({util:.0f}%)")
    elif util < 30: positives.append(f"✅ Healthy utilization ({util:.0f}%)")
    enq = row.get('enquiries_last_6_months',0)
    if enq > 5: factors.append(f"🔴 Many enquiries ({int(enq)})")
    contacts = row.get('total_contacts',0)
    if contacts > 10: factors.append(f"📞 Heavy collection ({int(contacts)} calls)")
    elif contacts == 0: positives.append("✅ No collection needed")
    sr = row.get('success_rate',1)
    if sr < 0.6: factors.append(f"⚠️ Low payment success ({sr:.0%})")
    elif sr >= 0.95: positives.append(f"✅ Excellent payments ({sr:.0%})")
    dpd = row.get('current_dpd',0)
    if dpd > 90: factors.append(f"🔴 Severely delinquent ({int(dpd)}d)")
    elif dpd > 30: factors.append(f"🟡 Past due ({int(dpd)}d)")
    elif dpd == 0: positives.append("✅ Current")
    cs = row.get('credit_score',650)
    if cs < 550: factors.append(f"💳 Poor credit ({int(cs)})")
    elif cs >= 720: positives.append(f"✅ Strong credit ({int(cs)})")
    return factors, positives

def get_action(cat):
    return {'Critical':{'u':'🚨 IMMEDIATE','t':'24h','a':'Escalate to senior team'},'High':{'u':'⚠️ URGENT','t':'48h','a':'Priority call'},'Medium':{'u':'📋 MONITOR','t':'1 week','a':'Add to watchlist'},'Low':{'u':'✅ ROUTINE','t':'Standard','a':'Normal monitoring'},'Very Low':{'u':'💚 HEALTHY','t':'None','a':'Consider loyalty offers'}}.get(cat,{'u':'✅','t':'-','a':'-'})

def calc_metrics(results_df, df):
    m = {'total': len(results_df)}
    m['critical'] = (results_df['risk_category']=='Critical').sum()
    m['high'] = (results_df['risk_category']=='High').sum()
    m['medium'] = (results_df['risk_category']=='Medium').sum()
    m['low'] = (results_df['risk_category'].isin(['Low','Very Low'])).sum()
    
    if 'Loan_Amount' in df.columns:
        results_df['loan_amt'] = df['Loan_Amount'].values
        m['exposure'] = results_df['loan_amt'].sum()
        m['critical_exp'] = results_df[results_df['risk_category']=='Critical']['loan_amt'].sum()
        m['high_exp'] = results_df[results_df['risk_category']=='High']['loan_amt'].sum()
        m['at_risk'] = m['critical_exp'] + m['high_exp']
    else:
        m['exposure'] = m['at_risk'] = m['critical_exp'] = m['high_exp'] = 0
    
    m['potential_loss'] = m['critical_exp']*0.6 + m['high_exp']*0.4
    m['preventable'] = m['potential_loss'] * 0.5
    m['avg_score'] = results_df['risk_score'].mean()
    
    if 'Current_DPD' in df.columns:
        results_df['dpd'] = df['Current_DPD'].values
        m['dpd_90plus'] = (results_df['dpd']>90).sum()
    else: m['dpd_90plus'] = 0
    
    if 'Collection_Calls' in df.columns:
        results_df['calls'] = df['Collection_Calls'].values
        m['contact_rate'] = (results_df['calls']>0).sum() / m['total'] * 100
    else: m['contact_rate'] = 0
    
    return m, results_df

# MAIN APP
def main():
    st.markdown("""<div class="header"><h1>🛡️ NPA Early Warning System</h1><p>AI-Powered Portfolio Risk Intelligence</p><span class="badge badge-green">✨ Production</span><span class="badge badge-purple">🤖 88.9% Accuracy</span></div>""", unsafe_allow_html=True)
    
    if 'results' not in st.session_state: st.session_state.results = None
    if 'metrics' not in st.session_state: st.session_state.metrics = None
    if 'df_orig' not in st.session_state: st.session_state.df_orig = None
    
    tabs = st.tabs(["📤 Upload", "📊 Executive Dashboard", "💰 Financial Analysis", "🎯 Segmentation", "📋 Actions"])
    
    # TAB 1: UPLOAD
    with tabs[0]:
        c1, c2 = st.columns([2,1])
        with c1:
            st.markdown('<div class="card"><div class="card-title">📁 Upload Portfolio</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader("CSV or Excel", type=['csv','xlsx'], label_visibility="collapsed")
            if uploaded:
                df = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
                st.success(f"✅ Loaded **{len(df):,}** accounts")
                if st.button("🚀 Run ML Analysis", type="primary", use_container_width=True):
                    with st.spinner("Running..."):
                        df_feat = prepare_features(df)
                        probs = predict_ml(df_feat[FEATURE_NAMES].values)
                        results = []
                        for i in range(len(df)):
                            cat = get_risk_category(probs[i])
                            factors, positives = get_risk_factors(df_feat.iloc[i])
                            act = get_action(cat)
                            results.append({'account_id': df.iloc[i].get('Account_ID',f'ACC{i+1}'), 'risk_score': round(probs[i]*100,2), 'risk_category': cat, 'risk_factors': ' | '.join(factors) if factors else 'None', 'positive_factors': ' | '.join(positives) if positives else 'None', 'urgency': act['u'], 'timeline': act['t'], 'action': act['a']})
                        results_df = pd.DataFrame(results)
                        for col in df.columns:
                            if col not in results_df.columns: results_df[col] = df[col].values
                        metrics, results_df = calc_metrics(results_df, df)
                        st.session_state.results = results_df
                        st.session_state.metrics = metrics
                        st.session_state.df_orig = df
                        st.success("✅ Complete!")
                        st.balloons()
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown("""<div class="card"><div class="card-title">🤖 Model Info</div><div class="metric-row"><span>Algorithm</span><span><b>Logistic Regression</b></span></div><div class="metric-row"><span>Accuracy</span><span><b>88.9% ROC-AUC</b></span></div><div class="metric-row"><span>Features</span><span><b>70 engineered</b></span></div><div class="metric-row"><span>Training</span><span><b>20,000 accounts</b></span></div></div>""", unsafe_allow_html=True)
    
    # TAB 2: EXECUTIVE DASHBOARD
    with tabs[1]:
        if st.session_state.results is not None:
            m = st.session_state.metrics
            results_df = st.session_state.results
            
            # Executive Summary
            st.markdown(f"""<div style="background:linear-gradient(135deg,#1e3a5f,#0c4a6e);color:white;padding:1.5rem;border-radius:14px;margin-bottom:1.5rem"><h3 style="margin:0 0 1rem 0">📈 Executive Summary</h3><div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;text-align:center"><div><div style="font-size:1.75rem;font-weight:800">{m['total']:,}</div><div style="opacity:0.8;font-size:0.8rem">Total Accounts</div></div><div><div style="font-size:1.75rem;font-weight:800">{m['exposure']/1e6:.1f}M</div><div style="opacity:0.8;font-size:0.8rem">Portfolio (SAR)</div></div><div><div style="font-size:1.75rem;font-weight:800;color:#fca5a5">{m['at_risk']/1e6:.1f}M</div><div style="opacity:0.8;font-size:0.8rem">At Risk</div></div><div><div style="font-size:1.75rem;font-weight:800;color:#86efac">{m['preventable']/1e6:.1f}M</div><div style="opacity:0.8;font-size:0.8rem">Preventable Loss</div></div></div></div>""", unsafe_allow_html=True)
            
            # KPIs
            cols = st.columns(6)
            kpis = [('🔴 Critical', m['critical'], '#ef4444'), ('🟠 High', m['high'], '#f97316'), ('🟡 Medium', m['medium'], '#eab308'), ('🟢 Healthy', m['low'], '#22c55e'), ('📊 Avg Score', f"{m['avg_score']:.1f}%", '#3b82f6'), ('⚠️ 90+ DPD', m['dpd_90plus'], '#dc2626')]
            for i, (label, val, color) in enumerate(kpis):
                cols[i].markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:{color}">{val}</div><div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)
            
            # Alerts
            st.subheader("🚨 Key Alerts")
            if m['critical'] > 0:
                st.markdown(f'<div class="alert alert-critical"><b>🚨 {m["critical"]} Critical Accounts</b> require immediate action. Exposure: {m["critical_exp"]/1e6:.2f}M SAR</div>', unsafe_allow_html=True)
            if m['at_risk']/m['exposure']*100 > 10 if m['exposure'] > 0 else False:
                st.markdown(f'<div class="alert alert-warning"><b>⚠️ {m["at_risk"]/m["exposure"]*100:.1f}% Portfolio at Risk</b> - Consider increased monitoring</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="alert alert-success"><b>💰 Early Intervention Opportunity: {m["preventable"]/1e6:.2f}M SAR</b> - Proactive outreach can prevent losses</div>', unsafe_allow_html=True)
            
            # Charts
            c1, c2 = st.columns(2)
            with c1:
                counts = results_df['risk_category'].value_counts().reindex(['Critical','High','Medium','Low','Very Low']).dropna()
                fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.6, marker_colors=[RISK_COLORS.get(c,'#gray') for c in counts.index]))
                fig.update_layout(title="Risk Distribution", showlegend=True, height=350, margin=dict(t=50,b=20,l=20,r=20))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = go.Figure(go.Histogram(x=results_df['risk_score'], nbinsx=25, marker_color='#3b82f6'))
                for t,c,l in [(70,'#ef4444','Critical'),(50,'#f97316','High'),(30,'#eab308','Medium')]:
                    fig.add_vline(x=t, line_dash="dash", line_color=c, annotation_text=l)
                fig.update_layout(title="Score Distribution", xaxis_title="Risk Score", yaxis_title="Count", height=350, margin=dict(t=50,b=40,l=40,r=20))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📤 Upload data first")
    
    # TAB 3: FINANCIAL ANALYSIS
    with tabs[2]:
        if st.session_state.results is not None:
            m = st.session_state.metrics
            results_df = st.session_state.results
            
            cols = st.columns(4)
            cols[0].metric("💰 Total Portfolio", f"{m['exposure']/1e6:.1f}M SAR")
            cols[1].metric("🔴 At Risk", f"{m['at_risk']/1e6:.1f}M SAR", f"{m['at_risk']/m['exposure']*100:.1f}%" if m['exposure']>0 else "0%")
            cols[2].metric("📉 Potential Loss", f"{m['potential_loss']/1e6:.1f}M SAR")
            cols[3].metric("💚 Preventable", f"{m['preventable']/1e6:.1f}M SAR")
            
            c1, c2 = st.columns(2)
            with c1:
                if 'loan_amt' in results_df.columns:
                    exp_data = results_df.groupby('risk_category')['loan_amt'].sum().reindex(['Critical','High','Medium','Low','Very Low']).fillna(0)
                    fig = go.Figure(go.Bar(x=exp_data.index, y=exp_data.values/1e6, marker_color=[RISK_COLORS.get(c,'#gray') for c in exp_data.index], text=[f'{v/1e6:.1f}M' for v in exp_data.values], textposition='outside'))
                    fig.update_layout(title="Exposure by Risk Category", yaxis_title="Exposure (M SAR)", height=350)
                    st.plotly_chart(fig, use_container_width=True)
            with c2:
                scenarios = ['No Action', 'Standard', 'EWS Proactive', 'Intensive']
                rates = [40, 55, 70, 85]
                recovered = [m['at_risk']*r/100/1e6 for r in rates]
                fig = go.Figure(go.Bar(x=scenarios, y=recovered, marker_color=['#ef4444','#f97316','#22c55e','#10b981'], text=[f'{r}%' for r in rates], textposition='outside'))
                fig.update_layout(title="Recovery Scenarios", yaxis_title="Recovered (M SAR)", height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            st.metric("📞 Contact Rate", f"{m['contact_rate']:.1f}%", help="% of accounts with collection activity")
        else:
            st.info("📤 Upload data first")
    
    # TAB 4: SEGMENTATION
    with tabs[3]:
        if st.session_state.results is not None:
            results_df = st.session_state.results
            df = st.session_state.df_orig
            
            c1, c2 = st.columns(2)
            with c1:
                if 'Product_Type' in df.columns:
                    results_df['product'] = df['Product_Type'].values
                    prod = results_df.groupby('product')['risk_score'].agg(['mean','count']).round(1).reset_index().sort_values('mean', ascending=False)
                    fig = go.Figure(go.Bar(y=prod['product'], x=prod['mean'], orientation='h', marker_color=['#ef4444' if r>50 else '#f97316' if r>30 else '#22c55e' for r in prod['mean']], text=[f'{r:.0f}%' for r in prod['mean']], textposition='outside'))
                    fig.update_layout(title="Risk by Product", xaxis_title="Avg Risk Score", height=300, xaxis=dict(range=[0,100]))
                    st.plotly_chart(fig, use_container_width=True)
            with c2:
                if 'City' in df.columns:
                    results_df['city'] = df['City'].values
                    city = results_df.groupby('city')['risk_score'].agg(['mean','count']).round(1).reset_index().sort_values('mean', ascending=False).head(8)
                    fig = go.Figure(go.Bar(x=city['city'], y=city['mean'], marker_color=['#ef4444' if r>50 else '#f97316' if r>30 else '#22c55e' for r in city['mean']], text=[f'{r:.0f}%' for r in city['mean']], textposition='outside'))
                    fig.update_layout(title="Risk by City", yaxis_title="Avg Risk Score", height=300, yaxis=dict(range=[0,100]))
                    st.plotly_chart(fig, use_container_width=True)
            
            if 'Product_Type' in df.columns and 'DPD_Bucket' in df.columns:
                results_df['dpd_bucket'] = df['DPD_Bucket'].values
                hm = results_df.pivot_table(values='risk_score', index='product', columns='dpd_bucket', aggfunc='mean').round(0)
                order = ['Current','1-30 DPD','31-60 DPD','61-90 DPD','90+ DPD']
                hm = hm[[c for c in order if c in hm.columns]]
                fig = go.Figure(go.Heatmap(z=hm.values, x=hm.columns, y=hm.index, colorscale=[[0,'#22c55e'],[0.5,'#eab308'],[1,'#ef4444']], text=hm.values.astype(int), texttemplate='%{text}%'))
                fig.update_layout(title="Risk Heatmap: Product × DPD", height=350)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📤 Upload data first")
    
    # TAB 5: ACTIONS
    with tabs[4]:
        if st.session_state.results is not None:
            results_df = st.session_state.results
            
            c1,c2,c3 = st.columns(3)
            risk_filter = c1.multiselect("Risk", ['Critical','High','Medium','Low','Very Low'], default=['Critical','High'])
            sort_order = c2.selectbox("Sort", ['High→Low','Low→High'])
            show_n = c3.slider("Show", 5, 30, 15)
            
            filtered = results_df[results_df['risk_category'].isin(risk_filter)].sort_values('risk_score', ascending='Low' in sort_order)
            st.write(f"**{len(filtered):,} accounts**")
            
            dc1,dc2,dc3,dc4 = st.columns(4)
            dc1.download_button("📊 All", results_df.to_csv(index=False), "all.csv", use_container_width=True)
            crit_high = results_df[results_df['risk_category'].isin(['Critical','High'])]
            dc2.download_button(f"🔴 Critical+High ({len(crit_high)})", crit_high.to_csv(index=False), "critical_high.csv", use_container_width=True)
            crit = results_df[results_df['risk_category']=='Critical']
            dc3.download_button(f"🚨 Critical ({len(crit)})", crit.to_csv(index=False), "critical.csv", use_container_width=True)
            action_df = results_df[results_df['risk_category'].isin(['Critical','High','Medium'])][['account_id','risk_score','risk_category','urgency','action']].sort_values('risk_score',ascending=False)
            dc4.download_button(f"📋 Action List ({len(action_df)})", action_df.to_csv(index=False), "actions.csv", use_container_width=True)
            
            for _,row in filtered.head(show_n).iterrows():
                with st.expander(f"**{row['account_id']}** | {row['risk_category']} | {row['risk_score']:.1f}%"):
                    c1,c2 = st.columns([2,1])
                    with c1:
                        mc = st.columns(4)
                        if 'Loan_Amount' in row and pd.notna(row.get('Loan_Amount')): mc[0].metric("💰 Loan", f"{row['Loan_Amount']:,.0f}")
                        if 'Current_DPD' in row and pd.notna(row.get('Current_DPD')): mc[1].metric("📅 DPD", f"{int(row['Current_DPD'])}d")
                        if 'Bureau_Score' in row and pd.notna(row.get('Bureau_Score')): mc[2].metric("💳 Credit", f"{int(row['Bureau_Score'])}")
                        if 'Collection_Calls' in row and pd.notna(row.get('Collection_Calls')): mc[3].metric("📞 Calls", f"{int(row['Collection_Calls'])}")
                        if row['risk_factors'] != 'None':
                            st.markdown("**⚠️ Risk Factors:**")
                            for f in row['risk_factors'].split(' | '): st.write(f"- {f}")
                        if row['positive_factors'] != 'None':
                            st.markdown("**✅ Positive:**")
                            for p in row['positive_factors'].split(' | '): st.write(f"- {p}")
                    with c2:
                        st.markdown(f"**📌 Action**\n\n**Urgency:** {row['urgency']}\n\n**Timeline:** {row['timeline']}\n\n**Action:** {row['action']}")
        else:
            st.info("📤 Upload data first")
    
    st.markdown('<div style="text-align:center;padding:2rem;color:#64748b;border-top:1px solid #e2e8f0;margin-top:2rem"><b>🛡️ NPA Early Warning System</b> | ML Model: Logistic Regression | ROC-AUC: 0.889</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
