# Retail Intelligence Duo – Customer Clustering + Sales Forecasting

This repository contains **two related data science projects** themed around a retail business:

1. **Minor Project – Retail Store Customer Clustering**
2. **Major Project – Sales Forecasting for a Retail Chain**

Both projects are designed to be:
- Easy to run
- Easy to understand
- Suitable for academic submission (with clear structure & comments)

---

## Project 1 – Retail Store Customer Clustering (Minor)

**Goal:**  
Group customers into meaningful segments based on their shopping behaviour.

**Tech stack:**
- Python
- pandas, numpy
- scikit-learn (KMeans, StandardScaler, PCA)
- matplotlib

**Files:**
- `minor_customer_clustering/retail_customer_clustering.ipynb` – main notebook
- `minor_customer_clustering/retail_customer_clustering.py` – same logic as a script
- `minor_customer_clustering/retail_customers.csv` – synthetic customer dataset

**Steps (notebook/script):**
1. Load the customer data
2. Clean and scale numeric features
3. Apply **K-Means clustering**
4. Use **PCA** to visualise clusters in 2D
5. Interpret each cluster (e.g., high-value, low-frequency, etc.)

---

## Project 2 – Sales Forecasting for a Retail Chain (Major)

**Goal:**  
Predict future sales based on historical daily sales data.

**Tech stack:**
- Python
- pandas, numpy
- scikit-learn (RandomForestRegressor)
- matplotlib

**Files:**
- `major_sales_forecasting/sales_forecasting_model.ipynb` – main notebook (EDA + model)
- `major_sales_forecasting/sales_forecasting_model.py` – script version of core model
- `major_sales_forecasting/retail_sales.csv` – synthetic time‑series sales data

**Main steps:**
1. Load and explore the time-series sales data
2. Engineer date-based features (day of week, month, etc.)
3. Create simple **lag features** (yesterday’s sales, last week’s sales)
4. Build a **baseline model** (naive forecast)
5. Train a **RandomForestRegressor** and compare performance
6. Plot **actual vs predicted** sales for a test period

---

## How to Run

### Option 1 – Using Jupyter Notebook
1. Install requirements (for example):
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```
2. Open Jupyter:
   ```bash
   jupyter notebook
   ```
3. Navigate to:
   - `minor_customer_clustering/retail_customer_clustering.ipynb`
   - `major_sales_forecasting/sales_forecasting_model.ipynb`
4. Run the cells in order.

### Option 2 – Using Python Scripts
From inside the respective folders:

```bash
cd minor_customer_clustering
python retail_customer_clustering.py
```

```bash
cd major_sales_forecasting
python sales_forecasting_model.py
```

Outputs will be shown as printed summaries and matplotlib plots.

---
