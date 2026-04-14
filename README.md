# 🛒 E-Commerce Sales Intelligence & Forecasting Dashboard

> **End-to-end data analytics project** on 5,000+ real e-commerce transactions — featuring EDA, customer segmentation, sales forecasting, and interactive visualizations.

**Author:** Vivek Yadav | [LinkedIn](https://linkedin.com/in/vivek-yadav-610892250) | [GitHub](https://github.com/Vivek-1112)

---

## 📌 Project Overview

This project analyzes the **Olist Brazilian E-Commerce dataset** to extract actionable business insights. It covers the full data analytics pipeline — from raw data ingestion and cleaning to customer segmentation, predictive forecasting, and dashboard reporting.

### 🎯 Business Questions Answered
- Which product categories generate the most revenue?
- Which states have the highest customer lifetime value?
- How does delivery speed affect customer satisfaction?
- What are the distinct customer segments and their revenue contribution?
- What is the projected revenue for the next 3 months?

---

## 🛠️ Tech Stack

| Tool | Usage |
|------|-------|
| **Python** (Pandas, NumPy) | Data cleaning, EDA, feature engineering |
| **Matplotlib / Seaborn** | Data visualization (9 charts) |
| **Scikit-learn** | Linear Regression sales forecasting |
| **SQL** | 12 business queries for operational insights |
| **Excel (openpyxl)** | KPI dashboard with charts |

---

## 📁 Project Structure

```
olist_project/
├── Ecommerce_Sales_Analysis_Vivek_Yadav.ipynb   # Main analysis notebook
├── SQL_Business_Queries_Vivek_Yadav.sql          # 12 SQL business queries
├── Ecommerce_Excel_Report_Vivek_Yadav.xlsx       # Excel KPI dashboard
├── olist_orders.csv                              # Orders data
├── olist_order_items.csv                         # Product & pricing data
├── olist_customers.csv                           # Customer demographics
├── olist_reviews.csv                             # Review scores
├── olist_payments.csv                            # Payment methods
├── monthly_revenue.png                           # Monthly trend chart
├── category_revenue.png                          # Category breakdown
├── rfm_segments.png                              # Customer segments
├── sales_forecast.png                            # 3-month forecast
└── README.md
```

---

## 📊 Key Findings

| Insight | Finding |
|---------|---------|
| 🏆 Top Revenue Category | Electronics |
| 🗺️ Top Customer State | SP (São Paulo) |
| 💳 Most Used Payment | Credit Card (~70%) |
| ⭐ Avg Review Score | 4.2 / 5.0 |
| 🚚 Delivery vs Rating | Faster delivery = higher reviews |
| 👑 Champion customers | Drive 35%+ of total revenue |

---

## 🔍 Analysis Phases

### Phase 1 — Data Cleaning & Merging
- Merged 5 CSV tables (Orders, Items, Customers, Reviews, Payments)
- Handled missing values and duplicates
- Engineered new features: `total_revenue`, `delivery_days`, time features

### Phase 2 — Exploratory Data Analysis
- Monthly revenue trends, category performance, state-wise heatmaps
- Payment method distribution, order status analysis
- Delivery time vs. review score correlation

### Phase 3 — RFM Customer Segmentation
- Scored all customers on **Recency, Frequency, Monetary** dimensions
- Classified into 4 segments: Champions, Loyal, Potential, At-Risk
- Identified Champion segment as highest-value group

### Phase 4 — Sales Forecasting
- Built Linear Regression model on monthly aggregated revenue
- Achieved strong R² score with low MAE
- Forecasted revenue for next 3 months with confidence

### Phase 5 — Business Insights & Reporting
- Revenue heatmap: Month × Category
- Excel KPI dashboard with embedded charts
- SQL scripts for BI tool integration

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/Vivek-1112/ecommerce-sales-intelligence
cd ecommerce-sales-intelligence

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl jupyter

# Launch notebook
jupyter notebook Ecommerce_Sales_Analysis_Vivek_Yadav.ipynb
```

---

## 📈 Sample Visualizations

| Chart | Description |
|-------|-------------|
| Monthly Revenue Trend | Line chart with area fill |
| Category Revenue | Horizontal bar chart |
| RFM Segments | Customer count + revenue bar charts |
| Sales Forecast | Actual vs Predicted + 3-month projection |
| Review Heatmap | Month × Category revenue matrix |

---

## 🔗 Related Work

- 📄 **IEEE Research Paper**: [AI-Based Learning Management System](https://ieeexplore.ieee.org/document/11085707)
- 📊 **IPL Player Performance Dashboard** — Sports analytics project
- 🏥 **COVID-19 CT Scan Detection** — ML image classification interface

---

## 📬 Contact

**Vivek Yadav**  
📧 Open to Data Analyst / Business Analyst roles  
🔗 [LinkedIn](https://linkedin.com/in/vivek-yadav-610892250) | [GitHub](https://github.com/Vivek-1112)
