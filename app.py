"""
Sales Forecasting & Demand Intelligence Dashboard
Internship Project - Superstore Sales Dataset (Task 7)

"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


# DATA LOADING 

@st.cache_data
def load_data():
    """"Load dataset and create date features."""
    data = pd.read_csv("train.csv")
    data["Order Date"] = pd.to_datetime(data["Order Date"], dayfirst=True)
    data["Ship Date"] = pd.to_datetime(data["Ship Date"], dayfirst=True)
    data["Year"] = data["Order Date"].dt.year
    data["Month"] = data["Order Date"].dt.month
    data["Quarter"] = data["Order Date"].dt.quarter
    return data


@st.cache_data
def get_monthly_sales(_data, column=None, value=None):
    """Return monthly sales, with optional category or region filtering."""
    subset = _data if column is None else _data[_data[column] == value]
    daily = subset.groupby("Order Date")["Sales"].sum().asfreq("D", fill_value=0)
    return daily.resample("MS").sum()


@st.cache_data
def get_weekly_sales(_data):
    daily = _data.groupby("Order Date")["Sales"].sum().asfreq("D", fill_value=0)
    return daily.resample("W").sum().reset_index()


def build_lag_features(monthly_series):
    """Create lag features for forecasting."""
    table = monthly_series.reset_index()
    table.columns = ["Date", "Sales"]
    table["Month"] = table["Date"].dt.month
    table["Quarter"] = table["Date"].dt.quarter
    table["Season"] = table["Month"].apply(lambda m: (m % 12) // 3)
    table["lag1"] = table["Sales"].shift(1)
    table["lag2"] = table["Sales"].shift(2)
    table["lag3"] = table["Sales"].shift(3)
    table["roll3"] = table["Sales"].shift(1).rolling(3).mean()
    return table.dropna().reset_index(drop=True)


@st.cache_data
def forecast_sales(monthly_series, steps=3):
    """Train the XGBoost model and forecast future sales."""
    feature_cols = ["lag1", "lag2", "lag3", "roll3", "Month", "Quarter", "Season"]
    table = build_lag_features(monthly_series)

    # Evaluate model
    train_part, test_part = table.iloc[:-steps], table.iloc[-steps:]
    eval_model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    eval_model.fit(train_part[feature_cols], train_part["Sales"])
    holdout_pred = eval_model.predict(test_part[feature_cols])
    mae = mean_absolute_error(test_part["Sales"], holdout_pred)
    rmse = mean_squared_error(test_part["Sales"], holdout_pred) ** 0.5

    # Train final model
    final_model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    final_model.fit(table[feature_cols], table["Sales"])

    history = list(monthly_series.values)
    last_date = monthly_series.index[-1]
    future_dates, future_preds = [], []

    for step in range(steps):
        next_date = last_date + pd.DateOffset(months=step + 1)
        next_row = pd.DataFrame([{
            "lag1": history[-1], "lag2": history[-2], "lag3": history[-3],
            "roll3": np.mean(history[-3:]),
            "Month": next_date.month, "Quarter": next_date.quarter,
            "Season": (next_date.month % 12) // 3,
        }])
        next_pred = final_model.predict(next_row[feature_cols])[0]
        future_dates.append(next_date)
        future_preds.append(next_pred)
        history.append(next_pred)

    return future_dates, future_preds, mae, rmse

# PAGE FUNCTIONS 

def show_sales_overview(data):
    st.header("Sales Overview")

    col1, col2 = st.columns(2)
    with col1:
        yearly = data.groupby("Year")["Sales"].sum().reset_index()
        fig = px.bar(yearly, x="Year", y="Sales", title="Total Sales by Year", text_auto=".2s")
        st.plotly_chart(fig, width="stretch")

    with col2:
        monthly = get_monthly_sales(data).reset_index()
        monthly.columns = ["Date", "Sales"]
        fig = px.line(monthly, x="Date", y="Sales", title="Monthly Sales Trend", markers=True)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Sales by Region and Category")
    filt1, filt2 = st.columns(2)
    with filt1:
        region_choice = st.multiselect("Region", data["Region"].unique(), default=list(data["Region"].unique()))
    with filt2:
        category_choice = st.multiselect("Category", data["Category"].unique(), default=list(data["Category"].unique()))

    filtered = data[data["Region"].isin(region_choice) & data["Category"].isin(category_choice)]
    grouped = filtered.groupby(["Region", "Category"])["Sales"].sum().reset_index()
    fig = px.bar(grouped, x="Region", y="Sales", color="Category", barmode="group",
                 title="Sales by Region & Category (filtered)")
    st.plotly_chart(fig, width="stretch")


def show_forecast_explorer(data):
    st.header("Forecast Explorer")
    st.caption("Forecast using the XGBoost model.")

    pick1, pick2 = st.columns(2)
    with pick1:
        dimension = st.selectbox("Forecast by", ["Category", "Region"])
    with pick2:
        value = st.selectbox("Pick one", sorted(data[dimension].unique()))

    horizon = st.slider("How many months ahead?", min_value=1, max_value=3, value=3)

    monthly_series = get_monthly_sales(data, dimension, value)

    if len(monthly_series) < 12:
        st.warning("Not enough data to generate a reliable forecast.")
        return

    future_dates, future_preds, mae, rmse = forecast_sales(monthly_series, steps=3)
    future_dates, future_preds = future_dates[:horizon], future_preds[:horizon]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_series.index, y=monthly_series.values,
                              mode="lines+markers", name="Historical Sales"))
    fig.add_trace(go.Scatter(x=future_dates, y=future_preds, mode="lines+markers",
                              name="Forecast", line=dict(dash="dash", color="orange")))
    fig.update_layout(title=f"{value} - {horizon} Month Forecast", xaxis_title="Month", yaxis_title="Sales ($)")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Forecast values")
    forecast_table = pd.DataFrame({
        "Month": [d.strftime("%b %Y") for d in future_dates],
        "Forecasted Sales": [f"${p:,.0f}" for p in future_preds],
    })
    st.table(forecast_table)

    metric1, metric2 = st.columns(2)
    metric1.metric("Model MAE (3-month holdout)", f"${mae:,.0f}")
    metric2.metric("Model RMSE (3-month holdout)", f"${rmse:,.0f}")


def show_anomaly_report(data):
    st.header("Anomaly Report")
    st.caption("Weekly sales anomalies detected using two methods.")

    weekly = get_weekly_sales(data)

    # method 1 - isolation forest
    iso_model = IsolationForest(contamination=0.08, random_state=42)
    weekly["iso_anomaly"] = iso_model.fit_predict(weekly[["Sales"]])

    # method 2 - # Z-Score method
    weekly["rolling_mean"] = weekly["Sales"].shift(1).rolling(8, min_periods=4).mean()
    weekly["rolling_std"] = weekly["Sales"].shift(1).rolling(8, min_periods=4).std()
    weekly["z_score"] = (weekly["Sales"] - weekly["rolling_mean"]) / weekly["rolling_std"]
    weekly["z_anomaly"] = weekly["z_score"].abs() > 2

    iso_anom = weekly[weekly["iso_anomaly"] == -1]
    z_anom = weekly[weekly["z_anomaly"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly["Order Date"], y=weekly["Sales"], mode="lines", name="Weekly Sales"))
    fig.add_trace(go.Scatter(x=iso_anom["Order Date"], y=iso_anom["Sales"], mode="markers",
                              marker=dict(color="red", size=10, symbol="x"), name="Isolation Forest anomaly"))
    fig.add_trace(go.Scatter(x=z_anom["Order Date"], y=z_anom["Sales"], mode="markers",
                              marker=dict(color="orange", size=12, symbol="circle-open", line=dict(width=2)),
                              name="Z-score anomaly"))
    fig.update_layout(title="Weekly Sales with Anomalies", xaxis_title="Week", yaxis_title="Sales ($)")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Detected anomaly weeks")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Isolation Forest**")
        st.dataframe(iso_anom[["Order Date", "Sales"]])
    with col2:
        st.markdown("**Z-score (more than 2 std from trailing average)**")
        st.dataframe(z_anom[["Order Date", "Sales", "z_score"]])


def show_demand_segments(data):
    st.header("Product Demand Segments")

    subcat_total = data.groupby("Sub-Category")["Sales"].sum()
    yearly_subcat = data.groupby(["Sub-Category", "Year"])["Sales"].sum().unstack()
    subcat_growth = (yearly_subcat[2018] - yearly_subcat[2015]) / yearly_subcat[2015].replace(0, np.nan)
    monthly_subcat = data.groupby(["Sub-Category", data["Order Date"].dt.to_period("M")])["Sales"].sum().reset_index()
    subcat_volatility = monthly_subcat.groupby("Sub-Category")["Sales"].std()
    subcat_orders = data.groupby("Sub-Category").size()
    subcat_avg_order_value = subcat_total / subcat_orders

    features = pd.DataFrame({
        "TotalSales": subcat_total,
        "GrowthRate": subcat_growth,
        "Volatility": subcat_volatility,
        "AvgOrderValue": subcat_avg_order_value,
    }).fillna(0)

    scaled = StandardScaler().fit_transform(features)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    features["Cluster"] = kmeans.fit_predict(scaled)

    cluster_summary = features.groupby("Cluster")[["TotalSales", "GrowthRate", "Volatility", "AvgOrderValue"]].mean()
    cluster_labels = {
        cluster_summary["TotalSales"].idxmax(): "High Volume, Stable Demand",
        cluster_summary["GrowthRate"].idxmax(): "Growing Demand",
        cluster_summary["GrowthRate"].idxmin(): "Declining Demand",
    }
    for cluster_num in cluster_summary.index:
        if cluster_num not in cluster_labels:
            cluster_labels[cluster_num] = "Low Volume, High Volatility"
    features["Label"] = features["Cluster"].map(cluster_labels)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled)
    features["PC1"], features["PC2"] = coords[:, 0], coords[:, 1]

    fig = px.scatter(features.reset_index(), x="PC1", y="PC2", color="Label",
                      text="Sub-Category", title="Product Demand Clusters (PCA view)")
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Sub-Category to Demand Cluster")
    display_table = features.reset_index()[
        ["Sub-Category", "TotalSales", "GrowthRate", "Volatility", "AvgOrderValue", "Label"]
    ].sort_values("TotalSales", ascending=False)
    st.dataframe(display_table)

    st.subheader("Recommended stocking strategy")
    st.markdown("""
    - **High Volume, Stable Demand** - steady safety stock, fixed reorder schedule
    - **Growing Demand** - increase safety stock ahead of trend, check reorder points monthly
    - **Declining Demand** - reduce future purchase orders, consider clearance of what's on hand
    - **Low Volume, High Volatility** - just-in-time / made-to-order, avoid holding heavy stock
    """)


# MAIN APP

def main():
    st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

    data = load_data()

    st.title("Sales Forecasting & Demand Intelligence Dashboard")
    st.caption("Superstore Sales Dataset (2015–2018)")

    page_choice = st.sidebar.radio(
        "Navigate",
        ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"],
    )

    if page_choice == "Sales Overview":
        show_sales_overview(data)
    elif page_choice == "Forecast Explorer":
        show_forecast_explorer(data)
    elif page_choice == "Anomaly Report":
        show_anomaly_report(data)
    elif page_choice == "Product Demand Segments":
        show_demand_segments(data)


if __name__ == "__main__":
    main()
