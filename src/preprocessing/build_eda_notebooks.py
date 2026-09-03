"""
Nexora — Phase 4: Automated EDA Notebook Builder & Runner
=========================================================
Builds production-quality, beautifully documented Jupyter notebooks for:
  - 01_data_exploration.ipynb
  - 02_customer_eda.ipynb
  - 03_product_eda.ipynb
Each notebook answers stated business questions with visualizations and insights.
"""

import os
import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent.parent / "notebooks"


def create_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.13"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [s + "\n" for s in source.split("\n")]
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [s + "\n" for s in source.split("\n")]
    }


def build_01_data_exploration():
    cells = [
        md_cell("""# 🛒 Nexora — Exploratory Data Analysis: Global Dynamics & Temporal Patterns
**Phase 4: Multi-Level EDA | Notebook 01**

### Objective
Understand global order volume dynamics, temporal purchasing patterns (hour of day, day of week), repurchase intervals, and shopping basket distributions across 3.4M+ orders."""),

        code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Styling setup
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.size"] = 11

DATA_DIR = Path("../data/processed")
print("Loading cleaned dataset tables...")
df_orders = pd.read_csv(DATA_DIR / "orders.csv")
df_products = pd.read_csv(DATA_DIR / "products.csv")
print(f"Loaded {len(df_orders):,} orders and {len(df_products):,} products.")"""),

        md_cell("""---
## ❓ Business Question 1: What are the peak ordering days and hours for consumers?
*Understanding peak demand periods is critical for server load balancing, delivery logistics scheduling, and targeted ad bidding.*"""),

        code_cell("""# 1. Day of week and Hour of day analysis
dow_map = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
df_orders['dow_name'] = df_orders['order_dow'].map(dow_map)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Day of week volume
dow_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
sns.countplot(data=df_orders, x='dow_name', order=dow_order, ax=axes[0], color='#2b5c8f')
axes[0].set_title("Order Volume by Day of Week", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Day of Week")
axes[0].set_ylabel("Total Orders")

# Hour of day volume
sns.countplot(data=df_orders, x='order_hour_of_day', ax=axes[1], color='#e26d5c')
axes[1].set_title("Order Volume by Hour of Day", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Hour of Day (0-23)")
axes[1].set_ylabel("Total Orders")

plt.tight_layout()
plt.show()"""),

        code_cell("""# 2. Day-Hour Heatmap
pivot_time = df_orders.pivot_table(index='order_dow', columns='order_hour_of_day', values='order_id', aggfunc='count')
pivot_time.index = [dow_map[i] for i in pivot_time.index]

plt.figure(figsize=(12, 5))
sns.heatmap(pivot_time, cmap="YlGnBu", cbar_kws={'label': 'Order Count'})
plt.title("Consumer Order Intensity: Day of Week vs. Hour of Day", fontsize=14, fontweight='bold')
plt.xlabel("Hour of Day")
plt.ylabel("Day of Week")
plt.show()"""),

        md_cell("""---
## ❓ Business Question 2: What is the repurchase cadence and replenishment cycle?
*Identifying customer replenishment intervals (e.g. 7-day, 14-day, 30-day spikes) enables automated reorder reminders and smart cart push notifications.*"""),

        code_cell("""# Repurchase interval distribution (excluding initial first orders where days=0 & is_first_order=1)
repeat_orders = df_orders[df_orders['is_first_order'] == 0]

plt.figure(figsize=(12, 5))
sns.histplot(repeat_orders['days_since_prior_order'], bins=31, color='#38b000', discrete=True, kde=False)
plt.title("Distribution of Days Between Consecutive Purchases", fontsize=14, fontweight='bold')
plt.xlabel("Days Since Prior Order")
plt.ylabel("Number of Orders")
plt.axvline(x=7, color='red', linestyle='--', label='Weekly Cycle (Day 7)')
plt.axvline(x=14, color='orange', linestyle='--', label='Bi-Weekly Cycle (Day 14)')
plt.axvline(x=30, color='purple', linestyle='--', label='Monthly Cap / Cap Spike (Day 30)')
plt.legend()
plt.show()"""),

        md_cell("""---
## ❓ Business Question 3: How large are shopping baskets across transactions?
*Understanding basket size distribution informs minimum order thresholds, free shipping tiers, and bundling opportunities.*"""),

        code_cell("""# Sample order products to inspect basket size distribution
df_op_sample = pd.read_csv(DATA_DIR / "order_products.csv", nrows=3000000)
basket_sizes = df_op_sample.groupby("order_id").size()

print("--- Basket Size Summary Statistics ---")
print(f"Mean Basket Size:   {basket_sizes.mean():.2f} items")
print(f"Median Basket Size: {basket_sizes.median():.0f} items")
print(f"75th Percentile:    {basket_sizes.quantile(0.75):.0f} items")
print(f"95th Percentile:    {basket_sizes.quantile(0.95):.0f} items")

plt.figure(figsize=(11, 4))
sns.histplot(basket_sizes, bins=40, color='#6a040f', binrange=(1, 40))
plt.title("Distribution of Basket Sizes (Items per Order)", fontsize=14, fontweight='bold')
plt.xlabel("Number of Items in Basket")
plt.ylabel("Frequency")
plt.axvline(x=basket_sizes.median(), color='blue', linestyle='--', label=f'Median = {basket_sizes.median():.0f} items')
plt.legend()
plt.show()"""),

        md_cell("""---
## ❓ Business Question 4: What is the overall reorder ratio across transactions?
*Reorder ratio reflects brand stickiness, consumable necessity vs novelty, and platform retention.*"""),

        code_cell("""# Reorder proportion
reorder_rate = df_op_sample['reordered'].mean()
print(f"Overall Product Reorder Rate: {reorder_rate * 100:.2f}%")

plt.figure(figsize=(6, 4))
sns.barplot(x=['First-Time Purchase', 'Reordered Item'], y=[1 - reorder_rate, reorder_rate], palette=['#457b9d', '#e63946'])
plt.title("Item Purchase Type Breakdown", fontsize=13, fontweight='bold')
plt.ylabel("Proportion")
plt.ylim(0, 1.0)
for i, v in enumerate([1 - reorder_rate, reorder_rate]):
    plt.text(i, v + 0.03, f"{v*100:.1f}%", ha='center', fontweight='bold')
plt.show()""")
    ]
    return create_nb(cells)


def build_02_customer_eda():
    cells = [
        md_cell("""# 👥 Nexora — Customer Intelligence & Behavioral EDA
**Phase 4: Multi-Level EDA | Notebook 02**

### Objective
Deep dive into customer-level behavior, purchase frequencies, retention curves, category diversity, and lifetime ordering trajectories across 206,209 customers."""),

        code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 5)

DATA_DIR = Path("../data/processed")
df_customers = pd.read_csv(DATA_DIR / "customers.csv")
df_orders = pd.read_csv(DATA_DIR / "orders.csv")
print(f"Loaded {len(df_customers):,} customers and {len(df_orders):,} orders.")"""),

        md_cell("""---
## ❓ Business Question 1: What is the lifetime order frequency per customer?
*Helps segment one-time/infrequent buyers from high-frequency brand loyalists.*"""),

        code_cell("""plt.figure(figsize=(12, 5))
sns.histplot(df_customers['total_orders'], bins=50, color='#1d3557', kde=True)
plt.title("Customer Distribution by Total Lifetime Orders", fontsize=14, fontweight='bold')
plt.xlabel("Total Orders per Customer")
plt.ylabel("Number of Customers")
plt.axvline(df_customers['total_orders'].median(), color='red', linestyle='--', label=f"Median Orders: {df_customers['total_orders'].median():.0f}")
plt.legend()
plt.show()"""),

        md_cell("""---
## ❓ Business Question 2: Does customer reorder rate increase with order tenure?
*Reveals the customer lifecycle curve: do customers develop predictable grocery habits as they place more orders?*"""),

        code_cell("""# Calculate average reorder rate as order_number increases
df_op_sample = pd.read_csv(DATA_DIR / "order_products.csv", nrows=4000000)
df_merged = df_op_sample.merge(df_orders[['order_id', 'order_number']], on='order_id')

order_progression = df_merged.groupby('order_number')['reordered'].mean().reset_index()

plt.figure(figsize=(12, 5))
sns.lineplot(data=order_progression[order_progression['order_number'] <= 60], x='order_number', y='reordered', color='#2a9d8f', linewidth=2.5)
plt.title("Reorder Rate Progression Across Customer Lifetime Orders", fontsize=14, fontweight='bold')
plt.xlabel("Order Sequence Number")
plt.ylabel("Average Reorder Rate")
plt.ylim(0, 1.0)
plt.show()"""),

        md_cell("""---
## ❓ Business Question 3: How consistent are customers in their order intervals?
*Customers with low variance in order interval are habitual shoppers; high-variance shoppers are opportunistic.*"""),

        code_cell("""user_intervals = df_orders[df_orders['is_first_order'] == 0].groupby('user_id')['days_since_prior_order'].agg(
    mean_interval='mean',
    std_interval='std',
    order_count='count'
).dropna()

plt.figure(figsize=(10, 5))
sns.scatterplot(data=user_intervals.sample(5000, random_state=42), x='mean_interval', y='std_interval', alpha=0.3, color='#023047')
plt.title("Customer Purchase Cadence: Mean Interval vs. Interval Standard Deviation", fontsize=13, fontweight='bold')
plt.xlabel("Mean Days Between Orders")
plt.ylabel("Standard Deviation of Order Interval")
plt.show()"""),

        md_cell("""---
## ❓ Business Question 4: What is the distribution of department diversity per customer?
*Assessing how many departments a customer orders from indicates full grocery adoption vs. single-niche utility.*"""),

        code_cell("""df_prod = pd.read_csv(DATA_DIR / "products.csv")
df_merged_prod = df_op_sample.merge(df_prod[['product_id', 'department_id']], on='product_id').merge(df_orders[['order_id', 'user_id']], on='order_id')

user_dept_diversity = df_merged_prod.groupby('user_id')['department_id'].nunique()

plt.figure(figsize=(10, 4))
sns.countplot(x=user_dept_diversity, color='#f4a261')
plt.title("Department Diversity Count per Customer (Sample)", fontsize=13, fontweight='bold')
plt.xlabel("Number of Unique Departments Explored")
plt.ylabel("Number of Customers")
plt.show()""")
    ]
    return create_nb(cells)


def build_03_product_eda():
    cells = [
        md_cell("""# 🥦 Nexora — Product Intelligence & Association EDA
**Phase 4: Multi-Level EDA | Notebook 03**

### Objective
Examine bestselling products, department velocity, reorder stickiness, add-to-cart position dynamics, and itemset co-occurrence associations across catalog products."""),

        code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.figsize"] = (10, 5)

DATA_DIR = Path("../data/processed")
df_products = pd.read_csv(DATA_DIR / "products.csv")
df_depts = pd.read_csv(DATA_DIR / "departments.csv")
df_aisles = pd.read_csv(DATA_DIR / "aisles.csv")
df_op_sample = pd.read_csv(DATA_DIR / "order_products.csv", nrows=4000000)

df_prod_full = df_products.merge(df_depts, on='department_id').merge(df_aisles, on='aisle_id')
print(f"Catalog contains {len(df_products):,} products across {len(df_depts):,} departments and {len(df_aisles):,} aisles.")"""),

        md_cell("""---
## ❓ Business Question 1: What are the top 20 bestselling products and their reorder rates?
*Identifies core anchor items that drive customer loyalty and high reorder velocity.*"""),

        code_cell("""prod_stats = df_op_sample.groupby('product_id').agg(
    total_purchases=('reordered', 'count'),
    total_reorders=('reordered', 'sum'),
    reorder_rate=('reordered', 'mean')
).reset_index().merge(df_prod_full, on='product_id')

top20 = prod_stats.sort_values(by='total_purchases', ascending=False).head(20)

plt.figure(figsize=(12, 6))
sns.barplot(data=top20, y='product_name', x='total_purchases', hue='reorder_rate', palette='viridis')
plt.title("Top 20 Most Purchased Products (Color = Reorder Rate)", fontsize=14, fontweight='bold')
plt.xlabel("Total Order Count (in sample)")
plt.ylabel("Product Name")
plt.show()"""),

        md_cell("""---
## ❓ Business Question 2: Which departments generate the highest volume and highest retention?
*Identifies revenue drivers vs. high-loyalty staple categories.*"""),

        code_cell("""dept_stats = prod_stats.groupby('department').agg(
    total_volume=('total_purchases', 'sum'),
    mean_reorder_rate=('reorder_rate', 'mean')
).reset_index().sort_values(by='total_volume', ascending=False)

fig, ax1 = plt.subplots(figsize=(14, 5))

color = '#1f77b4'
ax1.set_xlabel('Department', fontweight='bold')
ax1.set_ylabel('Total Volume', color=color, fontweight='bold')
sns.barplot(data=dept_stats, x='department', y='total_volume', ax=ax1, color=color, alpha=0.8)
ax1.tick_params(axis='x', rotation=45)

ax2 = ax1.twinx()
color = '#d62728'
ax2.set_ylabel('Reorder Rate', color=color, fontweight='bold')
sns.lineplot(data=dept_stats, x='department', y='mean_reorder_rate', ax=ax2, color=color, marker='o', linewidth=2.5)

plt.title("Department Volume vs. Department Reorder Rate", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""),

        md_cell("""---
## ❓ Business Question 3: How does add-to-cart position influence reorder probability?
*Top-of-mind staple items (milk, bananas) are added first into carts, exhibiting higher reorder probabilities.*"""),

        code_cell("""cart_reorder = df_op_sample.groupby('add_to_cart_order')['reordered'].mean().reset_index()

plt.figure(figsize=(11, 4))
sns.lineplot(data=cart_reorder[cart_reorder['add_to_cart_order'] <= 25], x='add_to_cart_order', y='reordered', marker='o', color='#9b5de5', linewidth=2.5)
plt.title("Reorder Probability vs. Add-to-Cart Sequence Position", fontsize=13, fontweight='bold')
plt.xlabel("Add to Cart Order (Position in Basket)")
plt.ylabel("Reorder Probability")
plt.show()"""),

        md_cell("""---
## ❓ Business Question 4: Which product pairs co-occur most frequently in the same basket?
*Market basket analysis for recommendation cross-selling and knowledge graph relationships.*"""),

        code_cell("""# High-frequency co-occurrence analysis on popular products
top_50_pids = set(prod_stats.sort_values(by='total_purchases', ascending=False).head(50)['product_id'])
df_top_op = df_op_sample[df_op_sample['product_id'].isin(top_50_pids)]

# Self-join on order_id to find pairs
order_pairs = df_top_op.merge(df_top_op, on='order_id')
order_pairs = order_pairs[order_pairs['product_id_x'] < order_pairs['product_id_y']]

pair_counts = order_pairs.groupby(['product_id_x', 'product_id_y']).size().reset_index(name='co_occurrence')
pair_counts = pair_counts.merge(df_products[['product_id', 'product_name']], left_on='product_id_x', right_on='product_id') \\
                         .merge(df_products[['product_id', 'product_name']], left_on='product_id_y', right_on='product_id', suffixes=('_A', '_B'))

top_pairs = pair_counts.sort_values(by='co_occurrence', ascending=False).head(10)
top_pairs['pair_label'] = top_pairs['product_name_A'] + " + " + top_pairs['product_name_B']

plt.figure(figsize=(12, 5))
sns.barplot(data=top_pairs, y='pair_label', x='co_occurrence', color='#00b4d8')
plt.title("Top 10 Most Frequently Co-Purchased Product Pairs", fontsize=14, fontweight='bold')
plt.xlabel("Co-Occurrence Count in Basket Sample")
plt.ylabel("Product Pair")
plt.show()""")
    ]
    return create_nb(cells)


def main():
    print("=" * 70)
    print(" Nexora — Phase 4: EDA Notebook Generation")
    print("=" * 70)
    
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Notebook 01
    nb1 = build_01_data_exploration()
    with open(NOTEBOOKS_DIR / "01_data_exploration.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb1, f, indent=2)
    print("    [+] Generated notebooks/01_data_exploration.ipynb")
    
    # 2. Notebook 02
    nb2 = build_02_customer_eda()
    with open(NOTEBOOKS_DIR / "02_customer_eda.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb2, f, indent=2)
    print("    [+] Generated notebooks/02_customer_eda.ipynb")
    
    # 3. Notebook 03
    nb3 = build_03_product_eda()
    with open(NOTEBOOKS_DIR / "03_product_eda.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb3, f, indent=2)
    print("    [+] Generated notebooks/03_product_eda.ipynb")
    
    print("-" * 70)
    print("[+] All Phase 4 EDA Notebooks constructed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
