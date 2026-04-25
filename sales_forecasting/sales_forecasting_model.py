"""Sales Forecasting for a Retail Chain
Major Project – builds a simple time-series forecasting model using lag features
and a RandomForestRegressor, and compares it to a naive baseline.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def main():
    # 1. Load data
    df = pd.read_csv("retail_sales.csv", parse_dates=["date"])
    df = df.sort_values("date")
    print("First few rows of the dataset:")
    print(df.head())

    # 2. Create lag features on daily aggregated data
    daily = df.groupby("date", as_index=False)["sales"].sum().sort_values("date")
    daily["sales_lag1"] = daily["sales"].shift(1)
    daily["sales_lag7"] = daily["sales"].shift(7)

    daily = daily.dropna().reset_index(drop=True)

    X = daily[["sales_lag1", "sales_lag7"]]
    y = daily["sales"]

    # Time-based split: use first 80% for training, last 20% for testing
    split_idx = int(len(daily) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = daily["date"].iloc[split_idx:]

    # 3. Baseline model (naive: prediction = yesterday's sales)
    y_pred_naive = X_test["sales_lag1"].values
    mae_naive = mean_absolute_error(y_test, y_pred_naive)
    print(f"Naive baseline MAE: {mae_naive:.2f}")

    # 4. RandomForestRegressor model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae_model = mean_absolute_error(y_test, y_pred)
    print(f"Random Forest MAE: {mae_model:.2f}")
    print(f"Improvement over naive: {mae_naive - mae_model:.2f} (MAE)")

    # 5. Plot actual vs predicted for last 60 days
    N = min(60, len(y_test))
    plt.figure(figsize=(10, 4))
    plt.plot(dates_test.iloc[-N:], y_test.iloc[-N:], label="Actual")
    plt.plot(dates_test.iloc[-N:], y_pred[-N:], label="Predicted")
    plt.title("Actual vs Predicted Sales (Last 60 Days)")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
