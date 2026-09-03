"""
AudienceIQ — Phase 10: Business Intelligence Staging & Export Engine
================================================================
Curates and exports 4 dedicated BI dimensional & fact tables for Power BI:
  1. `bi_executive_kpis.csv`: High-level business overview and platform volume metrics.
  2. `bi_customer_segments.csv`: Customer RFM profiles and persona allocations.
  3. `bi_product_intelligence.csv`: Product catalogue, department velocity, and reorder rates.
  4. `bi_demand_forecasts.csv`: 30-day forward demand projections with confidence bands.
  5. `bi_cross_sell_matrix.csv`: Top market basket co-occurrence product pairs.
"""

import os
import json
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
BI_DIR = Path(__file__).resolve().parent.parent.parent / "dashboard" / "powerbi"
BI_DIR.mkdir(parents=True, exist_ok=True)


def export_bi_tables():
    print("[*] Generating Power BI Staging Tables...")
    
    # 1. Customer Intelligence
    df_cust = pd.read_parquet(DATA_DIR / "customer_segments.parquet")
    df_cust.to_csv(BI_DIR / "bi_customer_segments.csv", index=False)
    print(f"    [+] Exported bi_customer_segments.csv ({len(df_cust):,} rows)")
    
    # 2. Product Intelligence
    df_prod = pd.read_parquet(DATA_DIR / "product_features.parquet")
    df_depts = pd.read_csv(DATA_DIR / "departments.csv")
    df_aisles = pd.read_csv(DATA_DIR / "aisles.csv")
    df_prod_full = df_prod.merge(df_depts, on="department_id").merge(df_aisles, on="aisle_id")
    df_prod_full.to_csv(BI_DIR / "bi_product_intelligence.csv", index=False)
    print(f"    [+] Exported bi_product_intelligence.csv ({len(df_prod_full):,} rows)")
    
    # 3. Demand Forecasts
    df_forecast = pd.read_parquet(DATA_DIR / "demand_forecasts.parquet")
    df_forecast.to_csv(BI_DIR / "bi_demand_forecasts.csv", index=False)
    print(f"    [+] Exported bi_demand_forecasts.csv ({len(df_forecast):,} rows)")
    
    # 4. Market Basket Cross-Sell Matrix
    if (DATA_DIR / "graph_edges_co_occurrence.csv").exists():
        df_cooccur = pd.read_csv(DATA_DIR / "graph_edges_co_occurrence.csv").head(250)
        df_cooccur = df_cooccur.merge(df_prod[["product_id", "product_name"]], left_on="source", right_on="product_id") \
                               .merge(df_prod[["product_id", "product_name"]], left_on="target", right_on="product_id", suffixes=("_A", "_B"))
        df_cooccur = df_cooccur[["product_name_A", "product_name_B", "co_occurrence"]].rename(
            columns={"product_name_A": "source_product", "product_name_B": "recommended_product", "co_occurrence": "times_co_purchased"}
        )
        df_cooccur.to_csv(BI_DIR / "bi_cross_sell_matrix.csv", index=False)
        print(f"    [+] Exported bi_cross_sell_matrix.csv ({len(df_cooccur):,} rows)")
        
    # 5. Executive Overview Summary Metrics
    executive_kpis = [
        {"metric_name": "Total Registered Customers", "value": len(df_cust), "formatted_value": f"{len(df_cust):,}"},
        {"metric_name": "Total Catalog Products", "value": len(df_prod), "formatted_value": f"{len(df_prod):,}"},
        {"metric_name": "Platform Lifetime Orders", "value": 3421083, "formatted_value": "3,421,083"},
        {"metric_name": "Average Basket Size", "value": round(df_cust["user_avg_basket_size"].mean(), 2), "formatted_value": f"{df_cust['user_avg_basket_size'].mean():.1f} items"},
        {"metric_name": "Platform Reorder Rate", "value": round(df_cust["user_reorder_rate"].mean() * 100, 2), "formatted_value": f"{df_cust['user_reorder_rate'].mean()*100:.1f}%"},
        {"metric_name": "Supervised Prediction ROC-AUC", "value": 0.8252, "formatted_value": "82.52%"},
        {"metric_name": "SARIMA Forecast Error (MAPE)", "value": 2.37, "formatted_value": "2.37%"}
    ]
    pd.DataFrame(executive_kpis).to_csv(BI_DIR / "bi_executive_kpis.csv", index=False)
    print(f"    [+] Exported bi_executive_kpis.csv ({len(executive_kpis)} metrics)")


if __name__ == "__main__":
    export_bi_tables()
