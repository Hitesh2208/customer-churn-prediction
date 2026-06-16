# Customer Churn Prediction & Retention Dashboard

An end-to-end business analytics project: from raw telco data in a SQL database, to an
explainable churn-prediction model, to a **deployed interactive app** that scores
customers, explains *why* they're at risk (SHAP), and recommends retention actions.

**Live demo:** _[paste your Streamlit Cloud URL here after deploying]_

---

## Business problem
A telecom company loses ~26.5% of its customers. Retaining a customer is far cheaper
than acquiring one. This project predicts *which* customers are likely to churn and
*why*, so the retention team can target spend at the highest-risk, highest-value
customers.

## What it does
- **SQL layer** — customer data loaded into SQLite; queries quantify churn by
  contract, tenure, service type, and payment method, plus monthly revenue at risk.
- **ML model** — XGBoost classifier (class-imbalance-weighted) predicting churn
  probability. **ROC-AUC ≈ 0.84**, churn recall ≈ 0.78 (catches ~4 of 5 churners).
- **Explainability (SHAP)** — every prediction comes with the top factors driving
  that individual customer's risk, plus a global view of what drives churn overall.
- **Decision layer** — customers bucketed into Low / Medium / High risk tiers, each
  mapped to a recommended retention action.
- **Deployed app (Streamlit)** — three tabs:
  1. **Predict a Customer** — probability, risk tier, action, and a SHAP "why".
  2. **Portfolio Dashboard** — KPIs and churn-driver charts.
  3. **Model Insights** — global SHAP feature importance.

## Key insights
- Month-to-month contracts churn at **42.7%** vs **2.8%** for two-year contracts.
- New customers (0–12 months tenure) churn far more than tenured ones.
- Contract type, tenure, and monthly charges are the strongest churn drivers (SHAP).

## Tech stack
Python · pandas · scikit-learn · XGBoost · SHAP · SQLite/SQL · Streamlit

## Project structure
```
churn_project/
├── data/telco_churn.csv     # dataset (IBM Telco, 7,043 customers)
├── data/telco.db            # pre-built SQLite database
├── build_db.py              # CSV -> SQLite (already run for you)
├── sql_analysis.sql         # BI queries
├── train_pipeline.py        # full ML + SHAP pipeline (already run for you)
├── app.py                   # Streamlit app (3 tabs)
├── model.pkl                # trained model (included)
├── shap_explainer.pkl       # SHAP explainer (included)
├── model_meta.json          # feature/encoder metadata (included)
├── requirements.txt
└── runtime.txt              # Python version pin for Streamlit Cloud
```
The model, SHAP explainer, and database are **already included** — you can run the
app immediately. If for any reason the saved model can't load (e.g. a library version
mismatch), the app **auto-retrains from the CSV on first launch**, so it always works.

---

## How to run locally
```bash
# 1. (recommended) create a clean environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. launch the app
streamlit run app.py
```
The app opens at http://localhost:8501


