-- name: churn_rate_overall
SELECT
  AVG(churned) AS churn_rate
FROM subscriptions;

-- name: churn_by_plan
SELECT
  plan,
  COUNT(*) AS customers,
  AVG(churned) AS churn_rate,
  AVG(monthly_fee) AS avg_fee
FROM subscriptions
GROUP BY plan
ORDER BY churn_rate DESC;

-- name: churn_by_billing_cycle
SELECT
  billing_cycle,
  COUNT(*) AS customers,
  AVG(churned) AS churn_rate
FROM subscriptions
GROUP BY billing_cycle
ORDER BY churn_rate DESC;

-- name: churn_by_channel
SELECT
  acquisition_channel,
  COUNT(*) AS customers,
  AVG(churned) AS churn_rate
FROM subscriptions
GROUP BY acquisition_channel
ORDER BY churn_rate DESC;

-- name: churn_by_tenure_bucket
SELECT
  CASE
    WHEN tenure_months < 3 THEN '<3'
    WHEN tenure_months < 6 THEN '3-5'
    WHEN tenure_months < 12 THEN '6-11'
    WHEN tenure_months < 24 THEN '12-23'
    ELSE '24+'
  END AS tenure_bucket,
  COUNT(*) AS customers,
  AVG(churned) AS churn_rate
FROM subscriptions
GROUP BY tenure_bucket
ORDER BY customers DESC;
