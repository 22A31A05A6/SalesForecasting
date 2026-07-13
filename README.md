# Intelligent Sales Forecasting & Demand Analytics System

An end-to-end Machine Learning project that predicts future retail sales, detects unusual sales patterns, segments products based on demand behavior, and provides interactive business insights through a Streamlit dashboard.

## Project Overview

This project analyzes historical retail sales data to help businesses make better inventory and demand planning decisions. It combines time series forecasting, anomaly detection, clustering, and interactive visualization into a single application.

## Features

- Sales data exploration and visualization
- Monthly sales forecasting using XGBoost
- Forecast by product category and region
- Weekly anomaly detection using:
  - Isolation Forest
  - Z-Score Method
- Product demand segmentation using K-Means Clustering
- Interactive Streamlit dashboard
- Business-friendly visualizations using Plotly

## Technologies Used

- Python
- Pandas
- NumPy
- XGBoost
- Scikit-learn
- Plotly
- Streamlit
- Statsmodels

## Machine Learning Techniques

- Time Series Forecasting
- Feature Engineering
- Lag Features
- Rolling Mean Features
- Isolation Forest
- K-Means Clustering
- Principal Component Analysis (PCA)

## Project Structure

```
SalesForecasting/
│
├── app.py
├── analysis.ipynb
├── train.csv
├── requirements.txt
├── summary.pdf
├── charts/
└── README.md
```

## Dashboard Pages

### 1. Sales Overview
- Total sales by year
- Monthly sales trend
- Sales by region and category

### 2. Forecast Explorer
- Forecast by Category or Region
- 1–3 month sales prediction
- Model performance metrics (MAE & RMSE)

### 3. Anomaly Report
- Weekly anomaly detection
- Isolation Forest results
- Z-Score anomaly comparison

### 4. Product Demand Segments
- Demand clustering
- PCA visualization
- Recommended inventory strategy

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/your-repository-name.git
```

Move into the project folder:

```bash
cd your-repository-name
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Dataset

- Superstore Sales Dataset (Kaggle)

The dataset contains four years of retail sales data, including product categories, regions, order dates, shipping dates, and sales information.

## Results

- Forecasted future monthly sales using XGBoost.
- Detected unusual sales spikes and drops using anomaly detection techniques.
- Segmented products into demand groups for inventory planning.
- Developed an interactive dashboard for business decision-making.

## Future Improvements

- Add SARIMA and Prophet forecasting directly into the dashboard.
- Support custom dataset uploads.
- Deploy using Docker.
- Add real-time forecasting through APIs.

## Author

**Samuel Guttula**

Bachelor of Technology (Computer Science and Engineering)

GitHub: https://github.com/22A31A05A6

LinkedIn: https://www.linkedin.com/in/samuel-guttula/
