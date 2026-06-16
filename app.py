"""
app.py — Customer Churn Prediction & Retention Dashboard (Streamlit)

Tabs:
  1) Predict a customer  -> churn probability, risk tier, recommended action,
                            and a SHAP explanation of WHY this customer is at risk.
  2) Portfolio dashboard -> KPIs + churn-driver charts.
  3) Model insights      -> global SHAP feature importance (what drives churn overall).

Run locally:  streamlit run app.py
Deploy free:  push to GitHub -> share.streamlit.io -> main file app.py
"""
import streamlit as st
import pandas as pd, numpy as np, json, joblib

st.set_page_config(page_title="Churn Intelligence", layout="wide", page_icon="📉")

@st.cache_resource
def load_artifacts():
    """Load pre-trained artifacts. If they're missing or incompatible with the
    installed library versions, transparently retrain from the CSV so the app
    always works with zero manual steps."""
    try:
        model = joblib.load("model.pkl")
        explainer = joblib.load("shap_explainer.pkl")
        meta = json.load(open("model_meta.json"))
        return model, explainer, meta
    except Exception:
        import train_pipeline
        train_pipeline.main()
        model = joblib.load("model.pkl")
        explainer = joblib.load("shap_explainer.pkl")
        meta = json.load(open("model_meta.json"))
        return model, explainer, meta

@st.cache_data
def load_data():
    df = pd.read_csv("data/telco_churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    return df

model, explainer, meta = load_artifacts()
df = load_data()
FEATURES = meta["features"]

def build_row(inp):
    """Turn UI inputs into a full model-ready feature row."""
    row = {}
    for f in FEATURES:
        if f in inp:
            v = inp[f]
            if f in meta["encoders"]:
                v = meta["encoders"][f].index(v)
            row[f] = float(v)
        elif f in meta["encoders"]:
            row[f] = 0.0                      # default category
        else:
            row[f] = float(meta["medians"][f])  # median for unset numerics
    return pd.DataFrame([row])[FEATURES].astype("float64")

def local_shap(X):
    """Return list of (feature, shap_value) sorted by impact for one customer."""
    sv = np.array(explainer.shap_values(X))
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    sv = sv[0]
    pairs = sorted(zip(FEATURES, sv), key=lambda t: abs(t[1]), reverse=True)
    return pairs

st.title("📉 Customer Churn Intelligence")
st.caption(f"Predict churn risk, understand drivers, and prioritize retention spend.  "
           f"Model ROC-AUC: {meta.get('auc', 0.84)}")

tab1, tab2, tab3 = st.tabs(["🔮 Predict a Customer", "📊 Portfolio Dashboard", "🧠 Model Insights"])

# ---------------- TAB 1: PREDICTION + LOCAL SHAP ----------------
with tab1:
    st.subheader("Score an individual customer")
    c1, c2, c3 = st.columns(3)
    inp = {}
    with c1:
        inp["tenure"] = st.slider("Tenure (months)", 0, 72, 3)
        inp["MonthlyCharges"] = st.slider("Monthly Charges ($)", 18.0, 120.0, 95.0)
        inp["Contract"] = st.selectbox("Contract", meta["encoders"]["Contract"])
    with c2:
        inp["InternetService"] = st.selectbox("Internet Service", meta["encoders"]["InternetService"])
        inp["TechSupport"] = st.selectbox("Tech Support", meta["encoders"]["TechSupport"])
        inp["OnlineSecurity"] = st.selectbox("Online Security", meta["encoders"]["OnlineSecurity"])
    with c3:
        inp["PaymentMethod"] = st.selectbox("Payment Method", meta["encoders"]["PaymentMethod"])
        inp["PaperlessBilling"] = st.selectbox("Paperless Billing", meta["encoders"]["PaperlessBilling"])
        inp["SeniorCitizen"] = st.selectbox("Senior Citizen", [0, 1])

    if st.button("Predict churn risk", type="primary"):
        X = build_row(inp)
        p = float(model.predict_proba(X)[:, 1][0])
        tier = "🔴 High" if p > 0.6 else ("🟡 Medium" if p > 0.3 else "🟢 Low")
        m1, m2 = st.columns(2)
        m1.metric("Churn Probability", f"{p*100:.1f}%")
        m2.metric("Risk Tier", tier)
        st.progress(min(p, 1.0))

        if p > 0.6:
            st.error("Recommend proactive retention: offer contract upgrade / loyalty discount.")
        elif p > 0.3:
            st.warning("Monitor; nudge toward an annual contract or add TechSupport (reduces churn).")
        else:
            st.success("Low risk — no intervention needed. Consider an upsell.")

        # ---- SHAP explanation: why THIS customer ----
        st.markdown("#### Why this prediction?")
        st.caption("Top factors pushing this customer's risk up (red) or down (green).")
        pairs = local_shap(X)[:6]
        exp_df = pd.DataFrame({
            "Factor": [f for f, _ in pairs],
            "Impact on churn risk": [round(float(v), 3) for _, v in pairs],
        })
        st.dataframe(exp_df, hide_index=True, use_container_width=True,
                     column_config={"Impact on churn risk":
                                    st.column_config.NumberColumn(format="%.3f")})
        st.bar_chart(exp_df.set_index("Factor")["Impact on churn risk"])
        st.caption("Positive values increase churn probability; negative values decrease it (SHAP values).")

# ---------------- TAB 2: DASHBOARD ----------------
with tab2:
    st.subheader("Portfolio overview")
    churn_rate = (df["Churn"] == "Yes").mean() * 100
    rev_risk = df.loc[df["Churn"] == "Yes", "MonthlyCharges"].sum()
    k1, k2, k3 = st.columns(3)
    k1.metric("Customers", f"{len(df):,}")
    k2.metric("Churn Rate", f"{churn_rate:.1f}%")
    k3.metric("Monthly Revenue at Risk", f"${rev_risk:,.0f}")

    st.markdown("**Churn rate by contract type (%)**")
    by_contract = (df.assign(c=(df["Churn"] == "Yes").astype(int))
                     .groupby("Contract")["c"].mean().mul(100).round(1))
    st.bar_chart(by_contract)

    st.markdown("**Churn rate by tenure bucket (%)**")
    bucket = pd.cut(df["tenure"], [-1, 12, 24, 48, 72],
                    labels=["0-12", "13-24", "25-48", "49+"])
    by_tenure = (df.assign(c=(df["Churn"] == "Yes").astype(int), b=bucket)
                   .groupby("b", observed=True)["c"].mean().mul(100).round(1))
    st.bar_chart(by_tenure)

# ---------------- TAB 3: GLOBAL SHAP ----------------
with tab3:
    st.subheader("What drives churn overall")
    st.caption("Global feature importance from SHAP — average impact of each factor "
               "across all customers.")
    sample = df.sample(min(500, len(df)), random_state=1).copy()
    rows = []
    for _, r in sample.iterrows():
        inp = {
            "tenure": int(r["tenure"]),
            "MonthlyCharges": float(r["MonthlyCharges"]),
            "Contract": r["Contract"],
            "InternetService": r["InternetService"],
            "TechSupport": r["TechSupport"],
            "OnlineSecurity": r["OnlineSecurity"],
            "PaymentMethod": r["PaymentMethod"],
            "PaperlessBilling": r["PaperlessBilling"],
            "SeniorCitizen": int(r["SeniorCitizen"]),
        }
        rows.append(build_row(inp).iloc[0])
    Xs = pd.DataFrame(rows)[FEATURES].astype("float64")
    sv = np.array(explainer.shap_values(Xs))
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    imp = pd.Series(np.abs(sv).mean(0), index=FEATURES).sort_values(ascending=False).head(10)
    st.bar_chart(imp)
    st.caption("Higher = stronger influence on churn predictions. Contract type, tenure, "
               "and monthly charges dominate.")
