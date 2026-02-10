# 🛡️ NPA Early Warning System

AI-powered dashboard for predicting Non-Performing Assets (NPAs) in loan portfolios.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

## 🎯 Features

- **📤 Upload Portfolio Data** - CSV or Excel format
- **🔮 AI Risk Prediction** - Instant analysis of all accounts
- **📊 Interactive Dashboards** - Beautiful visualizations
- **💡 Risk Explanations** - Understand WHY accounts are flagged
- **📥 Export Results** - Download action lists for your team

## 🚀 Quick Start

### Online Demo
Visit: [https://npa-ews.streamlit.app](https://npa-ews.streamlit.app)

### Run Locally
```bash
pip install -r requirements.txt
streamlit run ews_dashboard_pro.py
```

## 📊 How It Works

1. **Upload** your portfolio data (loan accounts with payment history)
2. **Analyze** - System predicts NPA risk for each account
3. **Review** - See risk distribution and explanations
4. **Act** - Download prioritized action lists

## 📁 Required Data Format

Your CSV/Excel should include:

| Column | Description | Required |
|--------|-------------|----------|
| Account_ID | Unique identifier | ✅ |
| EMIs_Due | Number of EMIs due | ✅ |
| EMIs_Paid | Number of EMIs paid | ✅ |
| Current_DPD | Days past due | ✅ |
| Bureau_Score | Credit score | ✅ |
| Loan_Amount | Loan amount | Optional |
| Collection_Calls | Number of calls | Optional |
| Bounce_Count | Bounced payments | Optional |

## 📈 Risk Categories

| Category | Score | Action |
|----------|-------|--------|
| 🔴 Critical | ≥70% | Immediate escalation |
| 🟠 High | ≥50% | Urgent follow-up |
| 🟡 Medium | ≥30% | Add to watchlist |
| 🟢 Low | ≥15% | Regular monitoring |
| 🔵 Very Low | <15% | Healthy account |

## 🛠️ Technology

- **Frontend**: Streamlit
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy
- **ML Model**: Trained on 70+ features

## 📝 License

© 2025 AI/ML Analytics Team. All rights reserved.

---

**Built with ❤️ for smarter lending**
