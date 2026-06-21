# Smart Demand Forecasting & Inventory Optimizer (Retail & E-commerce)

An AI-powered decision support system designed to help retail and supply chain managers forecast future product demand, model seasonal impact, and optimize warehouse stock levels. This project combines machine learning regressions with operations research models to minimize stockouts and reduce inventory carrying costs.

---

## 🚀 Key Features

*   **📊 Executive EDA Dashboard:** Visualizes overall store performance, tracking total revenue, transactional volume, top products, and geographical sales.
*   **🔮 Machine Learning Demand Forecasting:** Trains a Random Forest Regressor on historical sales, capturing demand inertia (lags) and seasonal patterns to project demand for the next 30 days.
*   **📅 Holiday & Seasonality Modeling:** Incorporates calendar features (day of week, month, weekends) and specific high-retail events (Black Friday, Christmas Shopping Rush) to predict spikes accurately.
*   **🧠 Explainable AI (XAI):** Visualizes feature importance scores to show users what variables (e.g. rolling averages, specific lags, or holiday flags) drive the model's predictions.
*   **📦 Operations Inventory Optimizer:** Automatically computes core logistics metrics:
    *   **Safety Stock (Buffer):** Buffer to prevent stockouts based on demand volatility and supplier lead times.
    *   **Reorder Point (ROP):** The threshold at which a new purchase order should be placed.
    *   **Economic Order Quantity (EOQ):** The cost-minimizing batch order size balancing holding costs and ordering costs.
*   **💡 What-If Scenario Simulator:** Allows business owners to simulate promotional marketing traffic boosts (0% to +100%) or pricing adjustments (price elasticity modeling) to see how future inventory levels should adjust.

---

## 🛠️ Technology Stack & Math

### 1. Demand Forecasting
We use a **Random Forest Regressor** to predict daily quantities demanded. Standard time series models (like ARIMA/Prophet) can be hard to scale across thousands of SKUs or fail during complex installation. Random Forest with engineered lag features is highly robust, trains in seconds, and handles nonlinear holiday spikes easily.

**Features Engineered:**
*   **Lags:** Quantities sold 7, 14, and 30 days ago (demand inertia).
*   **Rolling Windows:** 7-day and 30-day moving sales averages.
*   **Calendar & Seasonality:** Day of week, day of month, month, weekend flags.
*   **Promotional Events:** Specific flags for Black Friday and Christmas season rushes.

### 2. Supply Chain Optimization
Once the AI forecasts future demand, we feed the predictions into operations formulas:
*   **Safety Stock ($SS$):**
    $$SS = Z \times \sigma_D \times \sqrt{L}$$
    *Where $Z$ is the service level factor (e.g., $1.65$ for $95\%$ reliability), $\sigma_D$ is the standard deviation of daily demand, and $L$ is the supplier lead time.*
*   **Reorder Point ($ROP$):**
    $$ROP = (d \times L) + SS$$
    *Where $d$ is the average daily forecasted demand.*
*   **Economic Order Quantity ($EOQ$):**
    $$EOQ = \sqrt{\frac{2DS}{H}}$$
    *Where $D$ is the annual demand, $S$ is the administrative ordering cost per order, and $H$ is the annual holding cost per unit (calculated as a % of product price).*

---

## 📁 Repository Structure

```
├── data/
│   └── cleaned_retail_data.csv   # Aggregated daily sales per product (generated)
├── .gitignore                    # Git exclusions
├── app.py                        # Streamlit web application
├── process_data.py               # Raw data downloader, cleaner, & synthetic generator
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation
```

---

## 💻 Local Installation Guide

### Prerequisites
*   Python 3.8 or higher installed on your system.

### Step 1: Clone the Repository
```bash
git clone <your-github-repo-url>
cd Smart-Demand-Forecasting
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download & Process the Dataset
Run the data processor script to fetch the raw transaction dataset from the UCI Machine Learning Repository, clean it, and prepare the daily aggregated datasets:
```bash
python process_data.py
```
*Note: The raw file is ~45MB and will take a minute to download. If you are offline, the script will automatically fallback to generating a realistic synthetic retail dataset so that the app runs instantly.*

### Step 4: Run the Streamlit App
Launch the web interface locally:
```bash
streamlit run app.py
```
The app will open automatically in your web browser at `http://localhost:8501`.

---

## 🌐 Deploying to Streamlit Cloud

Streamlit Cloud allows you to host this interactive application for free:

1.  **Commit and Push** your code to a public GitHub repository. (We recommend excluding raw `.xlsx` files using the included `.gitignore`; only commit your scripts and code).
2.  Go to [Streamlit Share](https://share.streamlit.io/) and log in using your GitHub account.
3.  Click **New App**.
4.  Select your repository, branch (`main`), and set the main file path to `app.py`.
5.  Click **Deploy!**

*The application has an built-in fail-safe:* If the processed CSV is not found on the server, `app.py` will automatically invoke the data generator to spin up a high-quality data stream. This ensures your app compiles and deploys on Streamlit Sharing out of the box without requiring manual file uploads.

---

## 💼 Real-World Enterprise Deployment

In a production enterprise environment, this application functions as the presentation layer of a larger data infrastructure:
1.  **Extraction:** Raw transactions are fetched from CRM/ERPs (like Shopify, Salesforce, NetSuite, or SAP) and loaded into warehouse databases (BigQuery, Snowflake, or Redshift).
2.  **Orchestration (ELT):** A pipeline tool like Airflow or dbt cleans the transactional records and aggregates sales on a daily basis.
3.  **Modeling Pipelines:** Machine learning models are retrained weekly using automated pipelines (MLflow, SageMaker, or Vertex AI) and store forecasts in cache databases (Redis) or prediction tables.
4.  **Actionable Outputs:** Reorder alerts and stock warnings are pushed to procurement teams via this dashboard, Slack webhooks, or automated e-mail triggers.
