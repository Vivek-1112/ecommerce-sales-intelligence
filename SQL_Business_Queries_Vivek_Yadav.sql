-- ============================================================
-- E-Commerce Sales Intelligence — SQL Business Queries
-- Author: Vivek Yadav | github.com/Vivek-1112
-- Dataset: Olist E-Commerce
-- ============================================================

-- ─────────────────────────────────────────
-- TABLE SETUP (Run these first)
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS orders (
    order_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(20),
    order_status VARCHAR(20),
    order_purchase_timestamp DATETIME,
    order_delivered_customer_date DATETIME
);

CREATE TABLE IF NOT EXISTS order_items (
    order_id VARCHAR(20),
    item_id INT,
    product_category VARCHAR(50),
    price DECIMAL(10,2),
    freight_value DECIMAL(10,2)
);

CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    customer_state VARCHAR(5),
    customer_city VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS reviews (
    order_id VARCHAR(20),
    review_score INT
);

CREATE TABLE IF NOT EXISTS payments (
    order_id VARCHAR(20),
    payment_type VARCHAR(20),
    payment_installments INT,
    payment_value DECIMAL(10,2)
);

-- ─────────────────────────────────────────
-- QUERY 1: Total Revenue, Orders & AOV
-- ─────────────────────────────────────────
SELECT
    COUNT(DISTINCT o.order_id)              AS total_orders,
    ROUND(SUM(i.price + i.freight_value), 2) AS total_revenue,
    ROUND(AVG(i.price + i.freight_value), 2) AS avg_order_value
FROM orders o
JOIN order_items i ON o.order_id = i.order_id
WHERE o.order_status = 'delivered';

-- ─────────────────────────────────────────
-- QUERY 2: Top 10 Revenue-Generating Categories
-- ─────────────────────────────────────────
SELECT
    product_category,
    COUNT(DISTINCT order_id)                    AS total_orders,
    ROUND(SUM(price + freight_value), 2)        AS total_revenue,
    ROUND(AVG(price), 2)                        AS avg_price
FROM order_items
GROUP BY product_category
ORDER BY total_revenue DESC
LIMIT 10;

-- ─────────────────────────────────────────
-- QUERY 3: Month-over-Month Revenue Growth
-- ─────────────────────────────────────────
WITH monthly_revenue AS (
    SELECT
        DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m') AS month,
        ROUND(SUM(i.price + i.freight_value), 2)         AS revenue
    FROM orders o
    JOIN order_items i ON o.order_id = i.order_id
    GROUP BY month
),
growth AS (
    SELECT
        month,
        revenue,
        LAG(revenue) OVER (ORDER BY month)  AS prev_revenue
    FROM monthly_revenue
)
SELECT
    month,
    revenue,
    prev_revenue,
    ROUND(((revenue - prev_revenue) / prev_revenue) * 100, 2) AS growth_pct
FROM growth
WHERE prev_revenue IS NOT NULL
ORDER BY month;

-- ─────────────────────────────────────────
-- QUERY 4: Revenue by Customer State (Top 10)
-- ─────────────────────────────────────────
SELECT
    c.customer_state,
    COUNT(DISTINCT o.order_id)               AS total_orders,
    ROUND(SUM(i.price + i.freight_value), 2) AS total_revenue,
    ROUND(AVG(i.price), 2)                   AS avg_price
FROM orders o
JOIN order_items i  ON o.order_id = i.order_id
JOIN customers c    ON o.customer_id = c.customer_id
GROUP BY c.customer_state
ORDER BY total_revenue DESC
LIMIT 10;

-- ─────────────────────────────────────────
-- QUERY 5: Average Delivery Days vs Review Score
-- ─────────────────────────────────────────
SELECT
    r.review_score,
    COUNT(o.order_id)                                                        AS order_count,
    ROUND(AVG(DATEDIFF(o.order_delivered_customer_date,
                       o.order_purchase_timestamp)), 1)                      AS avg_delivery_days
FROM orders o
JOIN reviews r ON o.order_id = r.order_id
WHERE o.order_delivered_customer_date IS NOT NULL
GROUP BY r.review_score
ORDER BY r.review_score;

-- ─────────────────────────────────────────
-- QUERY 6: Payment Method Breakdown
-- ─────────────────────────────────────────
SELECT
    payment_type,
    COUNT(order_id)                  AS total_transactions,
    ROUND(AVG(payment_value), 2)     AS avg_payment,
    ROUND(SUM(payment_value), 2)     AS total_payment_value,
    ROUND(AVG(payment_installments), 1) AS avg_installments
FROM payments
GROUP BY payment_type
ORDER BY total_transactions DESC;

-- ─────────────────────────────────────────
-- QUERY 7: Order Status Distribution
-- ─────────────────────────────────────────
SELECT
    order_status,
    COUNT(order_id)                                                          AS order_count,
    ROUND(COUNT(order_id) * 100.0 / (SELECT COUNT(*) FROM orders), 2)       AS percentage
FROM orders
GROUP BY order_status
ORDER BY order_count DESC;

-- ─────────────────────────────────────────
-- QUERY 8: RFM Analysis (Customer Segmentation)
-- ─────────────────────────────────────────
WITH rfm_base AS (
    SELECT
        o.customer_id,
        DATEDIFF(CURRENT_DATE, MAX(o.order_purchase_timestamp)) AS recency,
        COUNT(DISTINCT o.order_id)                              AS frequency,
        ROUND(SUM(i.price + i.freight_value), 2)               AS monetary
    FROM orders o
    JOIN order_items i ON o.order_id = i.order_id
    GROUP BY o.customer_id
)
SELECT
    customer_id,
    recency,
    frequency,
    monetary,
    CASE
        WHEN recency <= 90  AND frequency >= 3 AND monetary >= 500  THEN 'Champion'
        WHEN recency <= 180 AND frequency >= 2 AND monetary >= 200  THEN 'Loyal Customer'
        WHEN recency <= 270 AND monetary >= 100                     THEN 'Potential Loyalist'
        ELSE 'At-Risk'
    END AS customer_segment
FROM rfm_base
ORDER BY monetary DESC;

-- ─────────────────────────────────────────
-- QUERY 9: Top 5 High-Value Customers
-- ─────────────────────────────────────────
SELECT
    o.customer_id,
    c.customer_state,
    COUNT(DISTINCT o.order_id)               AS total_orders,
    ROUND(SUM(i.price + i.freight_value), 2) AS lifetime_value,
    ROUND(AVG(r.review_score), 2)            AS avg_review
FROM orders o
JOIN order_items i  ON o.order_id = i.order_id
JOIN customers c    ON o.customer_id = c.customer_id
LEFT JOIN reviews r ON o.order_id = r.order_id
GROUP BY o.customer_id, c.customer_state
ORDER BY lifetime_value DESC
LIMIT 5;

-- ─────────────────────────────────────────
-- QUERY 10: Day-of-Week Order Patterns
-- ─────────────────────────────────────────
SELECT
    DAYNAME(order_purchase_timestamp) AS day_of_week,
    COUNT(order_id)                   AS total_orders,
    ROUND(COUNT(order_id) * 100.0 / (SELECT COUNT(*) FROM orders), 2) AS pct
FROM orders
GROUP BY DAYNAME(order_purchase_timestamp), DAYOFWEEK(order_purchase_timestamp)
ORDER BY DAYOFWEEK(order_purchase_timestamp);

-- ─────────────────────────────────────────
-- QUERY 11: Category Performance Scorecard
-- ─────────────────────────────────────────
SELECT
    i.product_category,
    COUNT(DISTINCT o.order_id)                           AS total_orders,
    ROUND(SUM(i.price + i.freight_value), 2)             AS total_revenue,
    ROUND(AVG(i.price), 2)                               AS avg_price,
    ROUND(AVG(i.freight_value), 2)                       AS avg_freight,
    ROUND(AVG(r.review_score), 2)                        AS avg_review_score
FROM orders o
JOIN order_items i  ON o.order_id = i.order_id
LEFT JOIN reviews r ON o.order_id = r.order_id
GROUP BY i.product_category
ORDER BY total_revenue DESC;

-- ─────────────────────────────────────────
-- QUERY 12: Repeat vs One-Time Customers
-- ─────────────────────────────────────────
WITH order_counts AS (
    SELECT customer_id, COUNT(DISTINCT order_id) AS num_orders
    FROM orders GROUP BY customer_id
)
SELECT
    CASE WHEN num_orders = 1 THEN 'One-Time Buyer'
         WHEN num_orders BETWEEN 2 AND 3 THEN 'Repeat Buyer'
         ELSE 'Loyal Buyer' END AS customer_type,
    COUNT(customer_id)          AS customer_count,
    ROUND(COUNT(customer_id) * 100.0 / (SELECT COUNT(*) FROM order_counts), 2) AS pct
FROM order_counts
GROUP BY customer_type
ORDER BY customer_count DESC;
