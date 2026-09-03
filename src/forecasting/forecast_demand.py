"""
AudienceIQ — Phase 8: Demand Forecasting Engine
============================================
Builds temporal time series demand models across categories and key products:
  1. Historical timeline construction from customer interval progressions.
  2. Baseline 1: 7-Day Simple Moving Average (SMA).
  3. Baseline 2: Holt-Winters Exponential Smoothing (Trend + 7-Day Seasonality).
  4. Challenger: Seasonal ARIMA (SARIMA).
  5. Backtest evaluation (MAE, RMSE, MAPE) on holdout periods.
  6. 30-day forward operational demand forecasting with confidence intervals.
  7. Materialization of `demand_forecasts.parquet` and `forecasting_evaluation_metrics.json`.
"""

import os
import sys
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


def build_daily_time_series():
    """Construct chronological daily demand aggregates from customer order intervals."""
    print("[1/5] Constructing chronological daily transaction timelines...")
    start = time.time()
    
    df_orders = pd.read_csv(DATA_DIR / "orders.csv")
    df_orders = df_orders.sort_values(by=["user_id", "order_number"])
    
    # Cumulative days since first order per user
    df_orders["cumulative_days"] = df_orders.groupby("user_id")["days_since_prior_order"].cumsum().astype(int)
    
    # Map to calendar date timeline (anchor date: 2025-01-01)
    anchor_date = pd.Timestamp("2025-01-01")
    df_orders["order_date"] = anchor_date + pd.to_timedelta(df_orders["cumulative_days"], unit="D")
    
    # Daily overall platform demand
    daily_platform = df_orders.groupby("order_date").size().rename("order_volume").reset_index()
    daily_platform = daily_platform[(daily_platform["order_date"] >= "2025-01-01") & (daily_platform["order_date"] <= "2025-10-31")]
    daily_platform = daily_platform.set_index("order_date").asfreq("D").ffill()
    
    print(f"    [+] Generated {len(daily_platform)} days of continuous daily demand ({daily_platform.index.min().date()} to {daily_platform.index.max().date()}) in {time.time()-start:.1f}s")
    return daily_platform


def evaluate_forecast(y_true, y_pred):
    """Compute MAE, RMSE, and MAPE metrics."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    return round(float(mae), 2), round(float(rmse), 2), round(float(mape), 2)


def benchmark_forecasting_models(ts_data, test_days=30):
    """Benchmark SMA, Holt-Winters Exponential Smoothing, and SARIMA on holdout test set."""
    print(f"[2/5] Benchmarking forecasting models on {test_days}-day holdout test window...")
    
    series = ts_data["order_volume"]
    train_series = series.iloc[:-test_days]
    test_series = series.iloc[-test_days:]
    
    models_eval = []
    
    # 1. Baseline 1: 7-Day Simple Moving Average
    print("    -> [1/3] Evaluating 7-Day Moving Average Baseline...")
    sma_preds = [train_series.iloc[-7:].mean()]
    history = list(train_series.iloc[-7:])
    for val in test_series.iloc[:-1]:
        history.append(val)
        sma_preds.append(np.mean(history[-7:]))
    sma_preds = pd.Series(sma_preds, index=test_series.index)
    
    mae, rmse, mape = evaluate_forecast(test_series.values, sma_preds.values)
    models_eval.append({
        "model": "7-Day Moving Average (Baseline)",
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    })
    
    # 2. Baseline 2: Holt-Winters Exponential Smoothing (Additive Trend + Seasonality s=7)
    print("    -> [2/3] Evaluating Holt-Winters Exponential Smoothing...")
    hw_model = ExponentialSmoothing(
        train_series,
        trend="add",
        seasonal="add",
        seasonal_periods=7,
        initialization_method="estimated"
    ).fit()
    hw_preds = hw_model.forecast(test_days)
    
    mae, rmse, mape = evaluate_forecast(test_series.values, hw_preds.values)
    models_eval.append({
        "model": "Holt-Winters Exp Smoothing",
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    })
    
    # 3. Challenger: SARIMA (1, 1, 1) x (1, 1, 1, 7)
    print("    -> [3/3] Evaluating Seasonal ARIMA (SARIMA)...")
    sarima_model = SARIMAX(
        train_series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
    sarima_preds = sarima_model.forecast(test_days)
    
    mae, rmse, mape = evaluate_forecast(test_series.values, sarima_preds.values)
    models_eval.append({
        "model": "SARIMA (Promoted Challenger)",
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    })
    
    return models_eval, test_series, hw_preds, sarima_preds


def generate_future_forecast(ts_data, forecast_horizon=30):
    """Fit promoted model on complete dataset and generate 30-day forward operational demand forecast."""
    print(f"[3/5] Generating {forecast_horizon}-day forward operational forecast with confidence intervals...")
    series = ts_data["order_volume"]
    
    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
    
    forecast_res = model.get_forecast(steps=forecast_horizon)
    pred_mean = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int(alpha=0.05)  # 95% CI
    
    future_dates = pred_mean.index
    df_forecast = pd.DataFrame({
        "forecast_date": future_dates.strftime("%Y-%m-%d"),
        "predicted_demand": pred_mean.round(0).astype(int).values,
        "lower_ci_95": conf_int.iloc[:, 0].clip(lower=0).round(0).astype(int).values,
        "upper_ci_95": conf_int.iloc[:, 1].round(0).astype(int).values
    })
    
    return df_forecast


def main():
    start_total = time.time()
    print("=" * 70)
    print(" AudienceIQ — Phase 8: Demand Forecasting Engine")
    print("=" * 70)
    
    ts_data = build_daily_time_series()
    models_eval, test_actuals, hw_preds, sarima_preds = benchmark_forecasting_models(ts_data, test_days=30)
    df_future = generate_future_forecast(ts_data, forecast_horizon=30)
    
    # Save outputs
    print("[4/5] Materializing forecast tables and evaluation reports...")
    df_future.to_parquet(OUTPUT_DIR / "demand_forecasts.parquet", index=False)
    df_future.to_csv(OUTPUT_DIR / "demand_forecasts.csv", index=False)
    
    metrics_payload = {
        "forecasting_benchmark": models_eval,
        "test_window_days": 30,
        "forward_horizon_days": 30
    }
    with open(OUTPUT_DIR / "forecasting_evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
        
    print("\n" + "=" * 70)
    print(" AudienceIQ — Time Series Demand Forecasting Benchmark (30-Day Holdout)")
    print("=" * 70)
    print(f"{'Forecasting Model':<32} | {'MAE':<10} | {'RMSE':<10} | {'MAPE (%)':<10}")
    print("-" * 68)
    for m in models_eval:
        print(f"{m['model']:<32} | {m['mae']:<10.2f} | {m['rmse']:<10.2f} | {m['mape']:<10.2f}%")
        
    print("\n--- 30-Day Forward Demand Forecast Sample ---")
    print(df_future.head(7).to_string(index=False))
    print("=" * 70)
    print(f"[+] Phase 8 Demand Forecasting completed successfully in {time.time()-start_total:.1f}s.")
    print("=" * 70)


if __name__ == "__main__":
    main()
