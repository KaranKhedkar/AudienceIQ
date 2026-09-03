"""
Nexora — Phase 4: EDA Statistical Summary Runner
=================================================
Runs the statistical analysis across Customer, Product, and Temporal dimensions
and prints key findings for Phase 4 reporting.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

print("=" * 70)
print(" Nexora — Phase 4: Exploratory Data Analysis Key Metrics")
print("=" * 70)

df_orders = pd.read_csv(DATA_DIR / "orders.csv")
df_customers = pd.read_csv(DATA_DIR / "customers.csv")
df_products = pd.read_csv(DATA_DIR / "products.csv")
df_depts = pd.read_csv(DATA_DIR / "departments.csv")
df_aisles = pd.read_csv(DATA_DIR / "aisles.csv")
df_op_sample = pd.read_csv(DATA_DIR / "order_products.csv", nrows=3000000)

print("\n--- 1. Temporal & Ordering Patterns ---")
peak_day = df_orders['order_dow'].mode()[0]
peak_hour = df_orders['order_hour_of_day'].mode()[0]
dow_names = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
print(f"Peak Day of Week:  {dow_names[peak_day]} ({peak_day})")
print(f"Peak Hour of Day:  {peak_hour}:00 - {peak_hour+1}:00")

repeat_orders = df_orders[df_orders['is_first_order'] == 0]
mean_interval = repeat_orders['days_since_prior_order'].mean()
median_interval = repeat_orders['days_since_prior_order'].median()
cycle_7 = (repeat_orders['days_since_prior_order'] == 7).mean() * 100
cycle_30 = (repeat_orders['days_since_prior_order'] == 30).mean() * 100
print(f"Mean Repurchase Interval:   {mean_interval:.2f} days")
print(f"Median Repurchase Interval: {median_interval:.1f} days")
print(f"Weekly Cycle (7 days):      {cycle_7:.2f}% of reorders")
print(f"Monthly Cycle / Cap (30d):  {cycle_30:.2f}% of reorders")

print("\n--- 2. Basket Size & Reorder Dynamics ---")
basket_sizes = df_op_sample.groupby("order_id").size()
overall_reorder_rate = df_op_sample['reordered'].mean()
print(f"Mean Basket Size:           {basket_sizes.mean():.2f} items")
print(f"Median Basket Size:         {basket_sizes.median():.0f} items")
print(f"Overall Reorder Rate:       {overall_reorder_rate * 100:.2f}%")

print("\n--- 3. Customer Frequency & Retention ---")
print(f"Total Customer Count:       {len(df_customers):,}")
print(f"Mean Lifetime Orders:       {df_customers['total_orders'].mean():.2f}")
print(f"Median Lifetime Orders:     {df_customers['total_orders'].median():.0f}")
print(f"Max Lifetime Orders:        {df_customers['total_orders'].max()}")

print("\n--- 4. Top 10 Bestselling Products ---")
df_prod_full = df_products.merge(df_depts, on='department_id').merge(df_aisles, on='aisle_id')
prod_stats = df_op_sample.groupby('product_id').agg(
    total_orders=('reordered', 'count'),
    reorder_rate=('reordered', 'mean')
).reset_index().merge(df_prod_full, on='product_id').sort_values(by='total_orders', ascending=False)

for i, r in prod_stats.head(10).reset_index().iterrows():
    print(f"  {i+1:2d}. {r['product_name']:<40} | Dept: {r['department']:<12} | Volume: {r['total_orders']:,} | Reorder: {r['reorder_rate']*100:.1f}%")

print("\n--- 5. Department Breakdown ---")
dept_stats = prod_stats.groupby('department').agg(
    volume=('total_orders', 'sum'),
    reorder_rate=('reorder_rate', 'mean')
).reset_index().sort_values(by='volume', ascending=False)

for i, r in dept_stats.head(5).reset_index().iterrows():
    print(f"  {i+1}. {r['department']:<15} | Volume: {r['volume']:,} | Mean Reorder Rate: {r['reorder_rate']*100:.1f}%")

print("\n" + "=" * 70)
