
-- 1. Overall churn rate & revenue at risk
SELECT
    COUNT(*)                                              AS total_customers,
    SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END)          AS churned,
    ROUND(AVG(CASE WHEN Churn='Yes' THEN 1.0 ELSE 0 END)*100, 1) AS churn_pct,
    ROUND(SUM(CASE WHEN Churn='Yes' THEN MonthlyCharges ELSE 0 END), 0) AS monthly_revenue_lost
FROM customers;

-- 2. Churn by contract type (key driver)
SELECT Contract,
       COUNT(*) AS n,
       ROUND(AVG(CASE WHEN Churn='Yes' THEN 1.0 ELSE 0 END)*100, 1) AS churn_pct
FROM customers
GROUP BY Contract
ORDER BY churn_pct DESC;

-- 3. Churn by tenure bucket (loyalty/cohort analysis)
SELECT
    CASE
        WHEN tenure <= 12 THEN '0-12 mo'
        WHEN tenure <= 24 THEN '13-24 mo'
        WHEN tenure <= 48 THEN '25-48 mo'
        ELSE '49+ mo'
    END AS tenure_bucket,
    COUNT(*) AS n,
    ROUND(AVG(CASE WHEN Churn='Yes' THEN 1.0 ELSE 0 END)*100, 1) AS churn_pct
FROM customers
GROUP BY tenure_bucket
ORDER BY churn_pct DESC;

-- 4. Churn by internet service & payment method
SELECT InternetService, PaymentMethod,
       COUNT(*) AS n,
       ROUND(AVG(CASE WHEN Churn='Yes' THEN 1.0 ELSE 0 END)*100, 1) AS churn_pct
FROM customers
GROUP BY InternetService, PaymentMethod
ORDER BY churn_pct DESC;

-- 5. High-value at-risk segment (CLTV proxy: high charges + month-to-month)
SELECT customerID, tenure, MonthlyCharges, Contract
FROM customers
WHERE Churn='Yes'
  AND MonthlyCharges > (SELECT AVG(MonthlyCharges) FROM customers)
  AND Contract='Month-to-month'
ORDER BY MonthlyCharges DESC
LIMIT 20;
