"""
train_pipeline.py — End-to-end churn modeling pipeline with SHAP explainability.
Steps: load -> clean -> encode -> train XGBoost -> evaluate -> SHAP -> save artifacts.

You normally do NOT need to run this — model.pkl, shap_explainer.pkl and
model_meta.json are already included in the repo. Run it only if you want to
retrain or modify the model.

Run: python train_pipeline.py
"""
import pandas as pd, numpy as np, json, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import shap

DATA = "data/telco_churn.csv"

def load_and_clean():
    df = pd.read_csv(DATA)
    # 11 blank TotalCharges (brand-new customers, tenure=0) -> coerce to 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df = df.drop("customerID", axis=1)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df

def encode(df):
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols and c != "Churn"]
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        encoders[c] = list(le.classes_)
    return df, encoders

def main():
    df = load_and_clean()
    df, encoders = encode(df)
    X = df.drop("Churn", axis=1).astype("float64")
    y = df["Churn"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    spw = (y == 0).sum() / (y == 1).sum()  # handle class imbalance
    model = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                              scale_pos_weight=spw, eval_metric="logloss")
    model.fit(Xtr, ytr)

    proba = model.predict_proba(Xte)[:, 1]
    auc = round(roc_auc_score(yte, proba), 3)
    print("ROC-AUC:", auc)
    print(classification_report(yte, (proba > 0.5).astype(int), target_names=["Stay", "Churn"]))
    print("Confusion matrix:\n", confusion_matrix(yte, (proba > 0.5).astype(int)))

    # SHAP — exact, fast TreeExplainer for the "why this customer" narrative
    explainer = shap.TreeExplainer(model)
    base_value = float(np.array(explainer.expected_value).ravel()[0])

    sv = np.array(explainer.shap_values(Xte))
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    imp = pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)
    print("\nTop global churn drivers (mean |SHAP|):\n", imp.head(8).round(3).to_string())

    joblib.dump(model, "model.pkl")
    joblib.dump(explainer, "shap_explainer.pkl")
    json.dump({"features": list(X.columns), "encoders": encoders,
               "base_value": base_value,
               "medians": {c: float(X[c].median()) for c in X.columns},
               "auc": auc}, open("model_meta.json", "w"))
    print("\nSaved model.pkl, shap_explainer.pkl, model_meta.json")

if __name__ == "__main__":
    main()
