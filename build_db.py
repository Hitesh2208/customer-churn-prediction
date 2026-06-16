"""
build_db.py — Load the cleaned CSV into SQLite for the SQL layer.
Run: python build_db.py   then open data/telco.db and run sql_analysis.sql
"""
import pandas as pd, sqlite3

df = pd.read_csv("data/telco_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
conn = sqlite3.connect("data/telco.db")
df.to_sql("customers", conn, if_exists="replace", index=False)
print("Loaded", len(df), "rows into data/telco.db (table: customers)")
conn.close()
