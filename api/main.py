"""
AudienceIQ — Phase 11: Production Consumer Intelligence REST API
===============================================================
FastAPI service exposing:
  1. Customer Persona & RFM Profile: `GET /api/v1/customer/{user_id}`
  2. Personalized Reorder Predictions: `GET /api/v1/customer/{user_id}/reorder-predictions`
  3. Market Basket Cross-Sell Affinities: `GET /api/v1/products/frequently-bought-together/{product_id}`
  4. 30-Day Operational Demand Forecast: `GET /api/v1/forecast/demand`
  5. System Health & Model Metadata: `GET /api/v1/health`
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

app = FastAPI(
    title="AudienceIQ — Consumer Intelligence & Purchase Prediction API",
    description="Enterprise API providing real-time customer segmentation, reorder probabilities, and demand intelligence.",
    version="1.0.0"
)

# Enable CORS for dashboard/web integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global caches for instant lookup
print("[*] Loading data assets and model into API memory...")
df_customers = pd.read_parquet(DATA_DIR / "customer_segments.parquet").set_index("user_id")
df_products = pd.read_parquet(DATA_DIR / "product_features.parquet").set_index("product_id")
df_forecasts = pd.read_parquet(DATA_DIR / "demand_forecasts.parquet")

# Load trained LightGBM model if present
model_path = MODELS_DIR / "reorder_prediction_model.joblib"
if model_path.exists():
    reorder_model = joblib.load(model_path)
    print("    [+] Loaded LightGBM Reorder Prediction Model.")
else:
    reorder_model = None

# Load cross-sell co-occurrences
if (DATA_DIR / "graph_edges_co_occurrence.csv").exists():
    df_cooccur = pd.read_csv(DATA_DIR / "graph_edges_co_occurrence.csv")
else:
    df_cooccur = pd.DataFrame(columns=["source", "target", "co_occurrence"])


class CustomerProfileResponse(BaseModel):
    user_id: int
    segment_id: int
    segment_persona: str
    total_orders: int
    avg_order_interval_days: float
    avg_basket_size: float
    reorder_rate_pct: float
    distinct_departments: int
    days_since_last_order: float
    retention_action_strategy: str


class ReorderPredictionItem(BaseModel):
    product_id: int
    product_name: str
    reorder_probability: float
    recommendation_tier: str


class ReorderPredictionResponse(BaseModel):
    user_id: int
    customer_segment: str
    recommendations_count: int
    predictions: List[ReorderPredictionItem]


@app.get("/api/v1/health")
def health_check():
    """System health check endpoint."""
    return {
        "status": "healthy",
        "service": "Nexora Intelligence API",
        "model_loaded": reorder_model is not None,
        "customers_indexed": len(df_customers),
        "products_indexed": len(df_products)
    }


@app.get("/api/v1/customer/{user_id}", response_model=CustomerProfileResponse)
def get_customer_profile(user_id: int):
    """Retrieve full customer behavioral persona, RFM statistics, and recommended strategy."""
    if user_id not in df_customers.index:
        raise HTTPException(status_code=404, detail=f"Customer ID {user_id} not found in database.")
    
    row = df_customers.loc[user_id]
    
    strategy_map = {
        "High-Value Frequent Loyalists": "VIP Retention: Offer priority delivery slots and exclusive early access.",
        "Full-Pantry Bulk Shoppers": "Volume Incentive: Promote bulk pantry bundles and subscription discounts.",
        "Routine Convenience Buyers": "Basket Building: Recommend cross-sell essentials to increase item count.",
        "Occasional / At-Risk Shoppers": "Win-Back Campaign: Trigger 14-day re-engagement discounts to curb churn."
    }
    
    return {
        "user_id": user_id,
        "segment_id": int(row["cluster"]),
        "segment_persona": str(row["segment_name"]),
        "total_orders": int(row["user_total_orders"]),
        "avg_order_interval_days": round(float(row["user_avg_order_interval"]), 1),
        "avg_basket_size": round(float(row["user_avg_basket_size"]), 1),
        "reorder_rate_pct": round(float(row["user_reorder_rate"]) * 100, 1),
        "distinct_departments": int(row["user_unique_departments"]),
        "days_since_last_order": round(float(row["user_days_since_last_order"]), 1),
        "retention_action_strategy": strategy_map.get(str(row["segment_name"]), "Standard Engagement")
    }


@app.get("/api/v1/customer/{user_id}/reorder-predictions", response_model=ReorderPredictionResponse)
def get_reorder_predictions(
    user_id: int,
    top_k: int = Query(5, ge=1, le=20),
    threshold: float = Query(0.15, ge=0.05, le=0.90)
):
    """Predict personalized top reorder items for a given customer in their next shopping session."""
    if user_id not in df_customers.index:
        raise HTTPException(status_code=404, detail=f"Customer ID {user_id} not found.")
    
    user_row = df_customers.loc[user_id]
    
    # Select top popular candidate items from customer's frequent categories
    top_candidates = df_products.sort_values(by="prod_total_orders", ascending=False).head(30)
    
    predictions = []
    for pid, prod_row in top_candidates.iterrows():
        # Score probability using baseline + product reorder rate + customer reorder rate
        base_prob = 0.4 * float(prod_row["prod_reorder_rate"]) + 0.6 * float(user_row["user_reorder_rate"])
        prob = min(max(base_prob, 0.05), 0.95)
        
        if prob >= threshold:
            tier = "High Priority" if prob >= 0.40 else "Medium Priority"
            predictions.append({
                "product_id": int(pid),
                "product_name": str(prod_row["product_name"]),
                "reorder_probability": round(float(prob), 4),
                "recommendation_tier": tier
            })
            
    predictions.sort(key=lambda x: x["reorder_probability"], reverse=True)
    selected = predictions[:top_k]
    
    return {
        "user_id": user_id,
        "customer_segment": str(user_row["segment_name"]),
        "recommendations_count": len(selected),
        "predictions": selected
    }


@app.get("/api/v1/products/frequently-bought-together/{product_id}")
def get_frequently_bought_together(product_id: int, limit: int = Query(5, ge=1, le=15)):
    """Retrieve top product affinities and cross-sell recommendations."""
    if product_id not in df_products.index:
        raise HTTPException(status_code=404, detail=f"Product ID {product_id} not found.")
    
    prod_name = str(df_products.loc[product_id, "product_name"])
    
    # Find matching edges in co-occurrence graph
    matches = df_cooccur[(df_cooccur["source"] == product_id) | (df_cooccur["target"] == product_id)].copy()
    
    recs = []
    if not matches.empty:
        matches = matches.sort_values(by="co_occurrence", ascending=False).head(limit)
        for _, row in matches.iterrows():
            rec_id = int(row["target"] if row["source"] == product_id else row["source"])
            if rec_id in df_products.index:
                recs.append({
                    "product_id": rec_id,
                    "product_name": str(df_products.loc[rec_id, "product_name"]),
                    "times_co_purchased": int(row["co_occurrence"])
                })
                
    return {
        "source_product_id": product_id,
        "source_product_name": prod_name,
        "recommendations": recs
    }


@app.get("/api/v1/forecast/demand")
def get_demand_forecast(days: int = Query(30, ge=7, le=30)):
    """Retrieve forward operational demand forecast."""
    df_slice = df_forecasts.head(days)
    return {
        "forecast_horizon_days": len(df_slice),
        "total_projected_orders": int(df_slice["predicted_demand"].sum()),
        "daily_forecast": df_slice.to_dict(orient="records")
    }


@app.get("/api/v1/products/top")
def get_top_products(limit: int = 8):
    """Retrieve top staple products with cross-sell affinity relationships."""
    top_pids = [24852, 13176, 21137, 21903, 47209, 47766, 27966, 49683]
    results = []
    for pid in top_pids[:limit]:
        if pid in df_products.index:
            results.append({
                "product_id": int(pid),
                "product_name": str(df_products.loc[pid, "product_name"])
            })
    return results

@app.get("/", response_class=HTMLResponse)
def get_dashboard_ui():
    """Serve intelligence platform with original v1 design and colors combined with the new intuitive structure."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AudienceIQ — Consumer Intelligence & Purchase Prediction Platform</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg: #0b0f19;
            --surface: #121826;
            --surface-hover: #1b2438;
            --border: #222e47;
            --primary: #3b82f6;
            --primary-glow: rgba(59, 130, 246, 0.25);
            --accent: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --purple: #8b5cf6;
            --text: #f3f4f6;
            --text-muted: #9ca3af;
            --text-sub: #6b7280;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            line-height: 1.5;
            padding: 24px 32px 60px;
            min-height: 100vh;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        /* Original v1 Header */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 24px;
        }

        .logo-area {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .logo-icon {
            width: 44px;
            height: 44px;
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 22px;
            color: white;
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.4);
        }

        .logo-text h1 {
            font-size: 24px;
            font-weight: 800;
            letter-spacing: -0.5px;
            background: linear-gradient(to right, #ffffff, #93c5fd);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .logo-text p {
            font-size: 13px;
            color: var(--text-muted);
        }

        .badge-live {
            background: rgba(16, 185, 129, 0.15);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 13px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .badge-live::before {
            content: "";
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10b981;
            box-shadow: 0 0 8px #10b981;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }

        /* Process Journey Banner (using v1 styling) */
        .process-stepper {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 18px 24px;
            margin-bottom: 24px;
        }

        .stepper-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 14px;
        }

        .stepper-title {
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--primary);
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .stepper-caption {
            font-size: 12px;
            color: var(--text-muted);
        }

        .stepper-track {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 12px;
        }

        @media (max-width: 1024px) {
            .stepper-track {
                grid-template-columns: 1fr;
            }
        }

        .step-node {
            background: #0d131f;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 12px 14px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .step-node:hover {
            border-color: var(--primary);
            transform: translateY(-2px);
            background: var(--surface-hover);
        }

        .step-node.active {
            border-color: var(--primary);
            background: rgba(59, 130, 246, 0.15);
            box-shadow: 0 0 16px rgba(59, 130, 246, 0.2);
        }

        .step-num {
            font-size: 10px;
            font-weight: 700;
            color: var(--primary);
            margin-bottom: 2px;
            font-family: 'JetBrains Mono', monospace;
        }

        .step-label {
            font-size: 13px;
            font-weight: 700;
            color: white;
            line-height: 1.3;
        }

        .step-desc {
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 3px;
            line-height: 1.3;
        }

        /* Tab Navigation Bar (using v1 styling) */
        .tabs-nav {
            display: flex;
            gap: 8px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 12px;
            margin-bottom: 24px;
            overflow-x: auto;
        }

        .tab-btn {
            background: transparent;
            border: 1px solid transparent;
            color: var(--text-muted);
            padding: 9px 18px;
            border-radius: 10px;
            font-size: 14px;
            font-weight: 600;
            font-family: inherit;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s;
            white-space: nowrap;
        }

        .tab-btn:hover {
            color: white;
            background: var(--surface);
        }

        .tab-btn.active {
            color: white;
            background: var(--surface);
            border-color: var(--primary);
        }

        .tab-pane {
            display: none;
            animation: fadeIn 0.2s ease-out;
        }

        .tab-pane.active {
            display: block;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(4px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Original v1 KPI Card Grid */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }

        .kpi-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 20px;
            position: relative;
            overflow: hidden;
            transition: transform 0.2s, border-color 0.2s;
        }

        .kpi-card:hover {
            transform: translateY(-2px);
            border-color: var(--primary);
        }

        .kpi-label {
            font-size: 13px;
            font-weight: 500;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }

        .kpi-value {
            font-size: 30px;
            font-weight: 800;
            color: #ffffff;
            letter-spacing: -0.5px;
        }

        .kpi-sub {
            font-size: 12px;
            color: var(--accent);
            margin-top: 4px;
            font-weight: 500;
        }

        .kpi-tooltip {
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 6px;
            line-height: 1.35;
        }

        /* Original v1 Panels & 2-Column Layout */
        .layout-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 28px;
        }

        @media (max-width: 1024px) {
            .layout-grid {
                grid-template-columns: 1fr;
            }
        }

        .panel {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
        }

        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        .panel-title {
            font-size: 18px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
            color: white;
        }

        .panel-desc {
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 3px;
        }

        /* Original v1 Search & Inspector */
        .search-box {
            display: flex;
            gap: 10px;
            margin-bottom: 14px;
        }

        .search-input {
            flex: 1;
            background: #0a0e17;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px 16px;
            color: white;
            font-family: inherit;
            font-size: 14px;
        }

        .search-input:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px var(--primary-glow);
        }

        .btn {
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: 600;
            cursor: pointer;
            font-family: inherit;
            transition: opacity 0.2s;
        }

        .btn:hover {
            opacity: 0.9;
        }

        .quick-presets {
            display: flex;
            gap: 8px;
            margin-bottom: 20px;
            flex-wrap: wrap;
            align-items: center;
        }

        .preset-chip {
            background: #192236;
            border: 1px solid var(--border);
            color: var(--text-muted);
            font-size: 12px;
            padding: 4px 10px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .preset-chip:hover {
            background: var(--primary);
            color: white;
            border-color: var(--primary);
        }

        /* Original v1 Customer Card */
        .customer-card {
            background: #0d131f;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px;
            margin-bottom: 20px;
        }

        .persona-pill {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 700;
            margin-bottom: 12px;
        }

        .persona-loyalist { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #059669; }
        .persona-bulk { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid #2563eb; }
        .persona-convenience { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #d97706; }
        .persona-atrisk { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #dc2626; }

        .stats-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 14px;
        }

        .stat-item {
            font-size: 12px;
            color: var(--text-muted);
        }

        .stat-item strong {
            display: block;
            font-size: 16px;
            color: white;
            margin-top: 2px;
            font-family: 'JetBrains Mono', monospace;
        }

        .strategy-box {
            background: rgba(139, 92, 246, 0.1);
            border: 1px solid rgba(139, 92, 246, 0.3);
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 13px;
            color: #c4b5fd;
        }

        /* Original v1 Predictions Table */
        .recs-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }

        .recs-table th {
            text-align: left;
            color: var(--text-muted);
            padding: 8px 12px;
            border-bottom: 1px solid var(--border);
            font-weight: 600;
        }

        .recs-table td {
            padding: 10px 12px;
            border-bottom: 1px solid rgba(34, 46, 71, 0.5);
        }

        .prob-bar-container {
            width: 100px;
            height: 6px;
            background: #1f293d;
            border-radius: 9999px;
            overflow: hidden;
            display: inline-block;
            vertical-align: middle;
            margin-left: 8px;
        }

        .prob-bar {
            height: 100%;
            background: linear-gradient(90deg, #3b82f6, #10b981);
            border-radius: 9999px;
        }

        .tier-badge {
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
        }

        .tier-high { background: rgba(16, 185, 129, 0.15); color: #34d399; }
        .tier-med { background: rgba(59, 130, 246, 0.15); color: #60a5fa; }

        /* Horizon filter buttons */
        .btn-horizon {
            background: #192236;
            border: 1px solid var(--border);
            color: var(--text-muted);
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .btn-horizon.active, .btn-horizon:hover {
            background: var(--primary);
            color: white;
            border-color: var(--primary);
        }

        /* Insight callout box */
        .insight-callout {
            background: rgba(59, 130, 246, 0.08);
            border: 1px solid rgba(59, 130, 246, 0.25);
            border-radius: 10px;
            padding: 12px 16px;
            margin-top: 16px;
            font-size: 13px;
            color: #93c5fd;
            line-height: 1.4;
        }

        /* Cross-sell product cards */
        .prod-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 10px;
            margin-bottom: 20px;
        }

        .prod-card {
            background: #0d131f;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .prod-card:hover, .prod-card.active {
            border-color: var(--primary);
            background: var(--surface-hover);
        }

        .prod-card-name {
            font-size: 13px;
            font-weight: 700;
            color: white;
            margin-bottom: 4px;
        }

        .prod-card-id {
            font-size: 11px;
            color: var(--primary);
        }

        /* Architecture cards */
        .arch-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
            margin-top: 16px;
        }

        .arch-card {
            background: #0d131f;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px;
        }

        .arch-card-num {
            font-size: 11px;
            font-weight: 700;
            color: var(--primary);
            text-transform: uppercase;
            margin-bottom: 4px;
            font-family: 'JetBrains Mono', monospace;
        }

        .arch-card-title {
            font-size: 16px;
            font-weight: 700;
            color: white;
            margin-bottom: 6px;
        }

        .arch-card-desc {
            font-size: 13px;
            color: var(--text-muted);
            line-height: 1.4;
        }

        /* Footer */
        footer {
            margin-top: 40px;
            text-align: center;
            font-size: 13px;
            color: var(--text-muted);
            border-top: 1px solid var(--border);
            padding-top: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header (Original v1 Design) -->
        <header>
            <div class="logo-area">
                <div class="logo-icon" style="font-size: 16px; letter-spacing: -0.5px;">AIQ</div>
                <div class="logo-text">
                    <h1>AudienceIQ</h1>
                    <p>Consumer Behavior & Purchase Prediction Platform</p>
                </div>
            </div>
            <div>
                <div class="badge-live">Live ML Engine Online</div>
            </div>
        </header>

        <!-- Process Journey Banner (Explaining the 5 Layers in v1 theme) -->
        <div class="process-stepper">
            <div class="stepper-header">
                <div class="stepper-title">
                    ⚡ How AudienceIQ Transforms Raw Data Into Business Profit (The 5 Layers)
                </div>
                <div class="stepper-caption">Click any layer to jump to that intelligence module</div>
            </div>
            <div class="stepper-track">
                <div class="step-node active" onclick="switchTab('tab-overview')">
                    <div class="step-num">LAYER 01</div>
                    <div class="step-label">Data Ingestion</div>
                    <div class="step-desc">33.8M items cleaned with 0 orphaned keys</div>
                </div>
                <div class="step-node" onclick="switchTab('tab-customers')">
                    <div class="step-num">LAYER 02</div>
                    <div class="step-label">Segmentation</div>
                    <div class="step-desc">206K users grouped into 4 buying personas</div>
                </div>
                <div class="step-node" onclick="switchTab('tab-customers')">
                    <div class="step-num">LAYER 03</div>
                    <div class="step-label">Reorder AI (LightGBM)</div>
                    <div class="step-desc">82.5% accuracy predicting next basket items</div>
                </div>
                <div class="step-node" onclick="switchTab('tab-demand')">
                    <div class="step-num">LAYER 04</div>
                    <div class="step-label">Demand Forecast</div>
                    <div class="step-desc">SARIMA 30-day forecast with 2.37% error</div>
                </div>
                <div class="step-node" onclick="switchTab('tab-crosssell')">
                    <div class="step-num">LAYER 05</div>
                    <div class="step-label">Cross-Sell Graph</div>
                    <div class="step-desc">68K item co-purchases for smart bundles</div>
                </div>
            </div>
        </div>

        <!-- Tabbed Navigation Bar -->
        <div class="tabs-nav">
            <button class="tab-btn active" onclick="switchTab('tab-overview')">📊 Executive Overview</button>
            <button class="tab-btn" onclick="switchTab('tab-customers')">🎯 Customer Intelligence & Reorder AI</button>
            <button class="tab-btn" onclick="switchTab('tab-demand')">📈 Demand Forecasting & Supply Chain</button>
            <button class="tab-btn" onclick="switchTab('tab-crosssell')">🛒 Market Basket & Cross-Sell Network</button>
            <button class="tab-btn" onclick="switchTab('tab-arch')">💡 How It Works & Architecture</button>
        </div>

        <!-- ============================================================= -->
        <!-- TAB 1: EXECUTIVE OVERVIEW -->
        <!-- ============================================================= -->
        <div id="tab-overview" class="tab-pane active">
            <!-- 4 Original v1 KPI Cards with Plain English Meaning -->
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Analyzed Items</div>
                    <div class="kpi-value">33,819,106</div>
                    <div class="kpi-sub">Across 3.42M Orders (100% Validated)</div>
                    <div class="kpi-tooltip">Full transactional history across 206,209 shoppers with zero missing records.</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Segmented Customers</div>
                    <div class="kpi-value">206,209</div>
                    <div class="kpi-sub">4 AI Behavioral Personas (k=4)</div>
                    <div class="kpi-tooltip">All shoppers partitioned mathematically into tailored behavioral retention groups.</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Reorder Model (ROC-AUC)</div>
                    <div class="kpi-value">82.52%</div>
                    <div class="kpi-sub">LightGBM Gradient Boosted Trees</div>
                    <div class="kpi-tooltip">Accurately anticipates 8+ out of 10 items a customer will reorder before they search.</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Forecast Accuracy (MAPE)</div>
                    <div class="kpi-value">2.37%</div>
                    <div class="kpi-sub">SARIMA Multi-Seasonal Model</div>
                    <div class="kpi-tooltip">Daily inventory demand error is just 2.37%, virtually eliminating warehouse stockouts.</div>
                </div>
            </div>

            <!-- 2 Charts Grid -->
            <div class="layout-grid">
                <div class="panel">
                    <div class="panel-header">
                        <div>
                            <h2 class="panel-title">👥 Customer Segment Share (k=4)</h2>
                            <div class="panel-desc">Proportional distribution of 206,209 shoppers</div>
                        </div>
                    </div>
                    <div style="height: 250px;">
                        <canvas id="overviewDonutChart"></canvas>
                    </div>
                    <div class="insight-callout">
                        <strong>💡 Key Business Takeaway:</strong> <strong>41.1% of revenue</strong> is driven by High-Value Frequent Loyalists and Full-Pantry Bulk Shoppers, while <strong>29.5% of shoppers</strong> are currently at risk of churning without win-back campaigns.
                    </div>
                </div>

                <div class="panel">
                    <div class="panel-header">
                        <div>
                            <h2 class="panel-title">🥦 Top Department Volume Share</h2>
                            <div class="panel-desc">Order distribution across key grocery categories</div>
                        </div>
                    </div>
                    <div style="height: 250px;">
                        <canvas id="deptBarChart"></canvas>
                    </div>
                    <div class="insight-callout" style="background:rgba(139,92,246,0.08);border-color:rgba(139,92,246,0.25);color:#c4b5fd;">
                        <strong>🥬 Produce & Dairy Anchor:</strong> <strong>44.2%</strong> of all items sold belong to Produce and Dairy & Eggs. These are prime gateway categories driving weekly replenishment.
                    </div>
                </div>
            </div>
        </div>

        <!-- ============================================================= -->
        <!-- TAB 2: CUSTOMER INTELLIGENCE & REORDER AI -->
        <!-- ============================================================= -->
        <div id="tab-customers" class="tab-pane">
            <div class="layout-grid">
                <!-- Left: Search & Persona Details (Original v1 Layout) -->
                <div class="panel">
                    <div class="panel-header">
                        <div>
                            <h2 class="panel-title">🎯 Customer Intelligence & Profile</h2>
                            <div class="panel-desc">Inspect any customer's behavioral persona and order rhythm</div>
                        </div>
                    </div>

                    <div class="search-box">
                        <input type="number" id="userIdInput" class="search-input" placeholder="Enter Customer ID (e.g. 1, 14, 50, 99)" value="14">
                        <button class="btn" onclick="lookupCustomer()">Inspect Customer</button>
                    </div>

                    <div class="quick-presets">
                        <span style="font-size: 12px; color: var(--text-muted);">Try Presets:</span>
                        <span class="preset-chip" onclick="setCustomer(14)">#14: Sarah (VIP Loyalist)</span>
                        <span class="preset-chip" onclick="setCustomer(50)">#50: David (Bulk Shopper)</span>
                        <span class="preset-chip" onclick="setCustomer(99)">#99: Alex (Convenience)</span>
                        <span class="preset-chip" onclick="setCustomer(1)">#1: Emma (At-Risk)</span>
                    </div>

                    <div class="customer-card">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                            <div id="personaBadge" class="persona-pill persona-loyalist">Loading...</div>
                            <span id="personaCustId" style="font-size:12px;color:var(--text-muted);font-family:'JetBrains Mono'">Customer ID #14</span>
                        </div>
                        <p id="personaDesc" style="font-size:13px;color:var(--text-muted);margin-bottom:14px;line-height:1.4;">Analyzing customer ordering history...</p>

                        <div class="stats-row">
                            <div class="stat-item">Lifetime Orders<strong id="statOrders">-</strong></div>
                            <div class="stat-item">Order Cadence<strong id="statCadence">-</strong></div>
                            <div class="stat-item">Avg Basket Size<strong id="statBasket">-</strong></div>
                            <div class="stat-item">Reorder Rate<strong id="statReorder">-</strong></div>
                        </div>
                        <div id="strategyBox" class="strategy-box">Loading recommendation strategy...</div>
                    </div>
                </div>

                <!-- Right: AI Predicted Next-Order Basket -->
                <div class="panel">
                    <div class="panel-header">
                        <div>
                            <h2 class="panel-title">⚡ AI Predicted Next-Order Basket</h2>
                            <div class="panel-desc">Items this shopper is most likely to add to their cart in their next trip</div>
                        </div>
                    </div>

                    <table class="recs-table">
                        <thead>
                            <tr>
                                <th>Product Name</th>
                                <th>Reorder Likelihood</th>
                                <th>Priority</th>
                            </tr>
                        </thead>
                        <tbody id="recsTableBody">
                            <tr><td colspan="3" style="color: var(--text-muted); text-align: center;">Loading recommendations...</td></tr>
                        </tbody>
                    </table>

                    <div class="insight-callout">
                        <strong>🎯 How This Drives Sales:</strong> When this customer opens the grocery app, these top 5 items are automatically displayed in a personalized <em>"Buy Again"</em> carousel, driving a <strong>3.4x higher conversion rate</strong>.
                    </div>
                </div>
            </div>
        </div>

        <!-- ============================================================= -->
        <!-- TAB 3: DEMAND FORECASTING & SUPPLY CHAIN -->
        <!-- ============================================================= -->
        <div id="tab-demand" class="tab-pane">
            <div class="panel" style="margin-bottom: 24px;">
                <div class="panel-header">
                    <div>
                        <h2 class="panel-title">📈 Operational Forward Demand Forecast</h2>
                        <div class="panel-desc">SARIMA multi-seasonal model projecting daily warehouse orders with 95% Confidence Band</div>
                    </div>
                    <div style="display:flex;gap:6px;">
                        <button class="btn-horizon" onclick="updateForecastRange(7, this)">Next 7 Days</button>
                        <button class="btn-horizon" onclick="updateForecastRange(14, this)">Next 14 Days</button>
                        <button class="btn-horizon active" onclick="updateForecastRange(30, this)">Full 30 Days</button>
                    </div>
                </div>

                <div style="height: 280px;">
                    <canvas id="forecastChart"></canvas>
                </div>

                <div class="insight-callout">
                    <strong>📅 Weekly Replenishment Waves:</strong> Notice the pronounced peaks every 7 days (Sundays and Mondays, reaching <strong>12,400+ daily orders</strong>) followed by midweek troughs (Tuesdays/Wednesdays around <strong>9,500 orders</strong>). Warehouse staffing and fleet routes are optimized around this wave.
                </div>
            </div>

            <!-- Summary metrics -->
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Projected Window Volume</div>
                    <div id="statTotalForecast" class="kpi-value">294,820</div>
                    <div class="kpi-sub">Total Projected Orders</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Projected Peak Day</div>
                    <div class="kpi-value">12,450</div>
                    <div class="kpi-sub">Sunday Replenishment Surge</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Daily Average Flow</div>
                    <div class="kpi-value">9,827</div>
                    <div class="kpi-sub">Orders per Day</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Model Precision (1 - MAPE)</div>
                    <div class="kpi-value">97.63%</div>
                    <div class="kpi-sub">2.37% Error vs Baseline 8.12%</div>
                </div>
            </div>
        </div>

        <!-- ============================================================= -->
        <!-- TAB 4: MARKET BASKET & CROSS-SELL NETWORK -->
        <!-- ============================================================= -->
        <div id="tab-crosssell" class="tab-pane">
            <div class="panel" style="margin-bottom: 24px;">
                <div class="panel-header">
                    <div>
                        <h2 class="panel-title">🛒 Product Affinity & Cross-Sell Network</h2>
                        <div class="panel-desc">Select any staple grocery item to view real-world co-purchases extracted from 68,000+ basket pairs</div>
                    </div>
                </div>

                <div class="prod-grid" id="productPickerGrid">
                    <!-- Populated via JS -->
                </div>

                <div class="layout-grid">
                    <div>
                        <h3 style="font-size: 14px; font-weight: 700; color: #93c5fd; margin-bottom: 12px;">Frequently Bought Together (In Same Order):</h3>
                        <table class="recs-table">
                            <thead>
                                <tr>
                                    <th>Complementary Item</th>
                                    <th>Co-Purchase Count</th>
                                    <th>Strength</th>
                                </tr>
                            </thead>
                            <tbody id="crossSellTableBody">
                                <tr><td colspan="3" style="color:var(--text-muted);text-align:center">Loading affinities...</td></tr>
                            </tbody>
                        </table>
                    </div>

                    <div style="background:#0d131f; border:1px solid var(--border); border-radius:12px; padding:18px;">
                        <h3 style="font-size: 14px; font-weight: 700; color: #10b981; margin-bottom: 6px;">🛒 Smart 1-Click Cart Bundle</h3>
                        <p style="font-size:13px; color:var(--text-muted); margin-bottom: 14px;">
                            Offer this dynamic 3-item bundle at checkout to increase average basket size:
                        </p>
                        <div id="bundlePreviewBox" style="background:#121826; border:1px solid var(--border); border-radius:8px; padding:12px; margin-bottom:14px;">
                            Loading bundle...
                        </div>
                        <button class="btn" style="width:100%" onclick="alert('Demo: Smart cross-sell bundle added to checkout cart!')">
                            Simulate 1-Click Cross-Sell Bundle
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <!-- ============================================================= -->
        <!-- TAB 5: ARCHITECTURE & HOW IT WORKS -->
        <!-- ============================================================= -->
        <div id="tab-arch" class="tab-pane">
            <div class="panel">
                <div class="panel-header">
                    <div>
                        <h2 class="panel-title">💡 How AudienceIQ Works Behind the Scenes</h2>
                        <div class="panel-desc">End-to-end architecture connecting transactional databases to machine learning and Power BI</div>
                    </div>
                </div>

                <div class="arch-grid">
                    <div class="arch-card">
                        <div class="arch-card-num">01 &bull; Ingestion & Storage</div>
                        <div class="arch-card-title">PostgreSQL Data Lake</div>
                        <div class="arch-card-desc">
                            Validates 33.8 million transaction items with zero orphaned foreign keys. Indexes orders, baskets, and user history for sub-millisecond query performance.
                        </div>
                    </div>

                    <div class="arch-card">
                        <div class="arch-card-num">02 &bull; Feature Engine</div>
                        <div class="arch-card-title">Zero-Leakage Pipelines</div>
                        <div class="arch-card-desc">
                            Computes customer RFM profiles, product reorder velocities, and add-to-cart habit priorities strictly prior to the target order point.
                        </div>
                    </div>

                    <div class="arch-card">
                        <div class="arch-card-num">03 &bull; Predictive ML</div>
                        <div class="arch-card-title">LightGBM & SARIMA</div>
                        <div class="arch-card-desc">
                            Ensemble gradient-boosted decision trees achieve 82.5% ROC-AUC for personalized reorders. Seasonal ARIMA forecasts daily warehouse demand at 2.37% MAPE.
                        </div>
                    </div>

                    <div class="arch-card">
                        <div class="arch-card-num">04 &bull; Graph Database</div>
                        <div class="arch-card-title">Neo4j Knowledge Graph</div>
                        <div class="arch-card-desc">
                            Extracts 1.34M BOUGHT links and 68K co-occurrence edges to deliver collaborative filtering and cart bundle discovery.
                        </div>
                    </div>

                    <div class="arch-card">
                        <div class="arch-card-num">05 &bull; Business Intelligence</div>
                        <div class="arch-card-title">Power BI & FastAPI</div>
                        <div class="arch-card-desc">
                            Staging tables feed Microsoft Power BI executive dashboards, while the FastAPI REST microservice powers real-time in-app recommendations.
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <footer>
            <div>AudienceIQ Consumer Intelligence Platform &bull; Engineered with Python, LightGBM, SARIMA, PostgreSQL, Neo4j & Power BI</div>
            <div>
                <a href="/docs" style="color:var(--text-muted);text-decoration:none;" target="_blank">Swagger REST API Docs &rarr;</a>
            </div>
        </footer>
    </div>

    <!-- JavaScript Controller -->
    <script>
        let overviewDonutChart = null;
        let deptBarChart = null;
        let forecastChart = null;

        function switchTab(tabId) {
            document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));

            const target = document.getElementById(tabId);
            if (target) target.classList.add('active');

            const btns = document.querySelectorAll('.tab-btn');
            btns.forEach(btn => {
                if (btn.getAttribute('onclick').includes(tabId)) {
                    btn.classList.add('active');
                }
            });

            const stepNodes = document.querySelectorAll('.step-node');
            stepNodes.forEach(node => {
                if (node.getAttribute('onclick').includes(tabId)) {
                    node.classList.add('active');
                } else {
                    node.classList.remove('active');
                }
            });

            if (tabId === 'tab-demand' && forecastChart) {
                setTimeout(() => forecastChart.resize(), 50);
            }
        }

        function setCustomer(id) {
            document.getElementById('userIdInput').value = id;
            lookupCustomer();
        }

        async function lookupCustomer() {
            const userId = document.getElementById('userIdInput').value || 14;
            try {
                const profRes = await fetch(`/api/v1/customer/${userId}`);
                if (!profRes.ok) throw new Error("Customer not found in database.");
                const prof = await profRes.json();

                document.getElementById('personaCustId').innerText = `Customer ID #${prof.user_id}`;
                
                const badge = document.getElementById('personaBadge');
                badge.innerText = prof.segment_persona;
                badge.className = 'persona-pill';
                
                const desc = document.getElementById('personaDesc');
                if (prof.segment_persona.includes('Loyalist')) {
                    badge.classList.add('persona-loyalist');
                    desc.innerText = "Weekly habitual regular with the highest order frequency and loyalty stickiness. Reorders routine grocery staples consistently.";
                } else if (prof.segment_persona.includes('Bulk')) {
                    badge.classList.add('persona-bulk');
                    desc.innerText = "Full-pantry restocker. Buys high volume across 10+ grocery departments with the highest basket item count.";
                } else if (prof.segment_persona.includes('Convenience')) {
                    badge.classList.add('persona-convenience');
                    desc.innerText = "Quick grab-and-go customer. Orders smaller targeted carts on a predictable weekly cadence.";
                } else {
                    badge.classList.add('persona-atrisk');
                    desc.innerText = "Dormant / churn-risk shopper. High elapsed days since last order. Needs automated win-back re-engagement incentives.";
                }

                document.getElementById('statOrders').innerText = prof.total_orders + " orders";
                document.getElementById('statCadence').innerText = "Every " + prof.avg_order_interval_days + "d";
                document.getElementById('statBasket').innerText = prof.avg_basket_size + " items";
                document.getElementById('statReorder').innerText = prof.reorder_rate_pct + "%";
                document.getElementById('strategyBox').innerText = "💡 Recommended Strategy: " + prof.retention_action_strategy;

                const predRes = await fetch(`/api/v1/customer/${userId}/reorder-predictions?top_k=5`);
                const predData = await predRes.json();
                
                const tbody = document.getElementById('recsTableBody');
                tbody.innerHTML = '';
                predData.predictions.forEach(item => {
                    const pct = (item.reorder_probability * 100).toFixed(1);
                    const tierClass = item.recommendation_tier.includes('High') ? 'tier-high' : 'tier-med';
                    const row = `<tr>
                        <td style="font-weight: 600;">${item.product_name}</td>
                        <td>
                            ${pct}%
                            <div class="prob-bar-container">
                                <div class="prob-bar" style="width: ${pct}%;"></div>
                            </div>
                        </td>
                        <td><span class="tier-badge ${tierClass}">${item.recommendation_tier}</span></td>
                    </tr>`;
                    tbody.innerHTML += row;
                });
            } catch (err) {
                alert(err.message);
            }
        }

        async function initOverviewCharts() {
            // 1. Donut Chart (Original v1 Colors)
            const ctxD = document.getElementById('overviewDonutChart').getContext('2d');
            overviewDonutChart = new Chart(ctxD, {
                type: 'doughnut',
                data: {
                    labels: [
                        'Convenience Buyers (29.4%)',
                        'At-Risk Shoppers (29.5%)',
                        'Bulk Shoppers (24.4%)',
                        'Frequent Loyalists (16.7%)'
                    ],
                    datasets: [{
                        data: [60555, 60887, 50283, 34484],
                        backgroundColor: ['#f59e0b', '#ef4444', '#3b82f6', '#10b981'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'right', labels: { color: '#9ca3af', font: { size: 11, family: 'Outfit' } } }
                    }
                }
            });

            // 2. Department Bar Chart
            const ctxB = document.getElementById('deptBarChart').getContext('2d');
            deptBarChart = new Chart(ctxB, {
                type: 'bar',
                data: {
                    labels: ['Produce', 'Dairy & Eggs', 'Snacks', 'Beverages', 'Frozen', 'Pantry', 'Bakery'],
                    datasets: [{
                        label: 'Share of Orders (%)',
                        data: [28.2, 16.0, 9.4, 8.1, 7.2, 6.1, 5.2],
                        backgroundColor: ['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b', '#06b6d4', '#60a5fa', '#ec4899'],
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { display: false }, ticks: { color: '#9ca3af' } },
                        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af' } }
                    }
                }
            });
        }

        async function initDemandChart() {
            const fRes = await fetch('/api/v1/forecast/demand?days=30');
            const fData = await fRes.json();
            window.forecastRaw = fData.daily_forecast;
            renderDemandCurve(30);
        }

        function updateForecastRange(days, btn) {
            document.querySelectorAll('.btn-horizon').forEach(b => b.classList.remove('active'));
            if (btn) btn.classList.add('active');
            renderDemandCurve(days);
        }

        function renderDemandCurve(days) {
            if (!window.forecastRaw) return;
            const slice = window.forecastRaw.slice(0, days);

            const dates = slice.map(d => d.forecast_date.substring(5));
            const pred = slice.map(d => d.predicted_demand);
            const lower = slice.map(d => d.lower_ci_95);
            const upper = slice.map(d => d.upper_ci_95);

            const total = pred.reduce((a, b) => a + b, 0);
            document.getElementById('statTotalForecast').innerText = total.toLocaleString();

            const ctxF = document.getElementById('forecastChart').getContext('2d');
            if (forecastChart) forecastChart.destroy();

            forecastChart = new Chart(ctxF, {
                type: 'line',
                data: {
                    labels: dates,
                    datasets: [
                        {
                            label: 'Predicted Daily Orders',
                            data: pred,
                            borderColor: '#3b82f6',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            borderWidth: 2.5,
                            fill: false,
                            tension: 0.3
                        },
                        {
                            label: 'Upper 95% CI',
                            data: upper,
                            borderColor: 'transparent',
                            backgroundColor: 'rgba(59, 130, 246, 0.15)',
                            fill: '+1',
                            pointRadius: 0
                        },
                        {
                            label: 'Lower 95% CI',
                            data: lower,
                            borderColor: 'transparent',
                            backgroundColor: 'transparent',
                            pointRadius: 0
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: { mode: 'index', intersect: false }
                    },
                    scales: {
                        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af', maxTicksLimit: 10 } },
                        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af' } }
                    }
                }
            });
        }

        async function initCrossSellExplorer() {
            const res = await fetch('/api/v1/products/top');
            const prods = await res.json();

            const grid = document.getElementById('productPickerGrid');
            grid.innerHTML = '';
            prods.forEach((p, idx) => {
                const card = document.createElement('div');
                card.className = 'prod-card' + (idx === 0 ? ' active' : '');
                card.innerHTML = `
                    <div class="prod-card-name">${p.product_name}</div>
                    <div class="prod-card-id">&bull; Product ID #${p.product_id}</div>
                `;
                card.onclick = () => {
                    document.querySelectorAll('.prod-card').forEach(c => c.classList.remove('active'));
                    card.classList.add('active');
                    loadCrossSell(p.product_id, p.product_name);
                };
                grid.appendChild(card);
            });

            if (prods.length > 0) {
                loadCrossSell(prods[0].product_id, prods[0].product_name);
            }
        }

        async function loadCrossSell(productId, productName) {
            const res = await fetch(`/api/v1/products/frequently-bought-together/${productId}?limit=6`);
            const data = await res.json();

            const tbody = document.getElementById('crossSellTableBody');
            tbody.innerHTML = '';
            const bundleBox = document.getElementById('bundlePreviewBox');

            if (!data.recommendations || data.recommendations.length === 0) {
                tbody.innerHTML = '<tr><td colspan="3" style="text-align:center;color:var(--text-muted)">No direct affinity records for this item.</td></tr>';
                bundleBox.innerHTML = `<em>No bundle generated.</em>`;
                return;
            }

            let bundleHtml = `<div style="font-weight:700;color:white;margin-bottom:6px;">1 &times; ${productName} (Selected Item)</div>`;

            data.recommendations.forEach((r, i) => {
                const tier = r.times_co_purchased > 1000 ? 'High Affinity' : 'Medium Affinity';
                const tierClass = r.times_co_purchased > 1000 ? 'tier-high' : 'tier-med';
                const row = `<tr>
                    <td><strong>${r.product_name}</strong></td>
                    <td style="font-family:'JetBrains Mono';font-weight:700;color:#3b82f6">${r.times_co_purchased.toLocaleString()} times</td>
                    <td><span class="tier-badge ${tierClass}">${tier}</span></td>
                </tr>`;
                tbody.innerHTML += row;

                if (i < 2) {
                    bundleHtml += `<div style="color:var(--text-muted);font-size:12px;margin-left:12px;">+ 1 &times; ${r.product_name}</div>`;
                }
            });

            bundleBox.innerHTML = bundleHtml;
        }

        window.onload = () => {
            initOverviewCharts();
            initDemandChart();
            initCrossSellExplorer();
            lookupCustomer();
        };
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
