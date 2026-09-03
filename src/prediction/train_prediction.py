"""
AudienceIQ — Phase 7: Purchase / Reorder Prediction Engine
======================================================
Trains and evaluates supervised binary classification models:
  1. Logistic Regression (Explainable Baseline)
  2. Random Forest Classifier
  3. LightGBM Classifier

Features:
  - User-product interaction features (personal reorder rate, order lag, cart priority)
  - Customer features (lifetime frequency, cadence, loyalty)
  - Product features (global reorder rate, popularity)

Evaluation:
  - Temporal user-cohort train/val split (80/20)
  - ROC-AUC, PR-AUC, Precision, Recall, F1
  - Precision/Recall trade-off analysis across decision thresholds
  - Model serialization to `models/reorder_prediction_model.joblib`
"""

import os
import sys
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import lightgbm as lgb

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    # Interaction features
    "up_total_orders",
    "up_order_rate",
    "up_orders_since_last",
    "up_avg_add_to_cart",
    # User features
    "user_total_orders",
    "user_avg_order_interval",
    "user_avg_basket_size",
    "user_reorder_rate",
    "user_unique_departments",
    # Product features
    "prod_total_orders",
    "prod_reorder_rate",
    "prod_avg_add_to_cart"
]


def load_dataset(sample_rows: int = 500000):
    """Load and merge user-product, customer, and product feature matrices."""
    print("[1/5] Loading and merging feature matrices for supervised modeling...")
    start = time.time()
    
    df_up = pd.read_parquet(DATA_DIR / "user_product_features.parquet")
    if sample_rows and len(df_up) > sample_rows:
        print(f"    Subsampling to {sample_rows:,} candidate pairs for fast training...")
        df_up = df_up.sample(sample_rows, random_state=42).reset_index(drop=True)
        
    df_cust = pd.read_parquet(DATA_DIR / "customer_features.parquet")
    df_prod = pd.read_parquet(DATA_DIR / "product_features.parquet")
    
    # Merge features
    df_full = df_up.merge(
        df_cust[["user_id", "user_total_orders", "user_avg_order_interval", "user_avg_basket_size", "user_reorder_rate", "user_unique_departments"]],
        on="user_id",
        how="left"
    ).merge(
        df_prod[["product_id", "prod_total_orders", "prod_reorder_rate", "prod_avg_add_to_cart"]],
        on="product_id",
        how="left"
    )
    
    print(f"    [+] Merged dataset shape: {df_full.shape[0]:,} rows x {df_full.shape[1]} cols in {time.time()-start:.1f}s")
    print(f"    Positive target rate: {df_full['target'].mean()*100:.2f}% (Class balance)")
    return df_full


def split_train_val(df_full, train_ratio=0.80):
    """Temporal / user-cohort split to strictly avoid data leakage across customer baskets."""
    print("[2/5] Creating user-cohort train / validation split (80% / 20%)...")
    unique_users = df_full["user_id"].unique()
    np.random.seed(42)
    np.random.shuffle(unique_users)
    
    n_train = int(len(unique_users) * train_ratio)
    train_users = set(unique_users[:n_train])
    val_users = set(unique_users[n_train:])
    
    train_mask = df_full["user_id"].isin(train_users)
    val_mask = df_full["user_id"].isin(val_users)
    
    X_train = df_full.loc[train_mask, FEATURE_COLS]
    y_train = df_full.loc[train_mask, "target"]
    
    X_val = df_full.loc[val_mask, FEATURE_COLS]
    y_val = df_full.loc[val_mask, "target"]
    
    print(f"    Training set:   {len(X_train):,} samples ({y_train.mean()*100:.2f}% positive)")
    print(f"    Validation set: {len(X_val):,} samples ({y_val.mean()*100:.2f}% positive)")
    return X_train, y_train, X_val, y_val


def evaluate_model(model_name: str, y_true, y_prob, threshold: float = 0.20):
    """Compute comprehensive evaluation metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    return {
        "model": model_name,
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "threshold": threshold,
        "confusion_matrix": cm
    }


def train_models(X_train, y_train, X_val, y_val):
    """Train Baseline (Logistic Regression), Random Forest, and LightGBM."""
    print("[3/5] Benchmarking models from explainable baseline to gradient boosted trees...")
    metrics_list = []
    
    # 1. Baseline: Logistic Regression (Standardized)
    print("    -> [1/3] Training Logistic Regression Baseline...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_train_scaled, y_train)
    lr_probs = lr.predict_proba(X_val_scaled)[:, 1]
    lr_metrics = evaluate_model("Logistic Regression (Baseline)", y_val, lr_probs, threshold=0.50)
    metrics_list.append(lr_metrics)
    
    # 2. Challenger 1: Random Forest Classifier
    print("    -> [2/3] Training Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=80, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_val)[:, 1]
    rf_metrics = evaluate_model("Random Forest", y_val, rf_probs, threshold=0.20)
    metrics_list.append(rf_metrics)
    
    # 3. Challenger 2: LightGBM Classifier
    print("    -> [3/3] Training LightGBM Classifier...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_probs = lgb_model.predict_proba(X_val)[:, 1]
    lgb_metrics = evaluate_model("LightGBM (Promoted Model)", y_val, lgb_probs, threshold=0.20)
    metrics_list.append(lgb_metrics)
    
    return metrics_list, lgb_model, lgb_probs, y_val


def analyze_thresholds(y_val, y_probs):
    """Evaluate business trade-offs across different probability thresholds."""
    print("[4/5] Computing Precision / Recall Trade-off across Decision Thresholds...")
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    th_records = []
    
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        p = precision_score(y_val, y_pred, zero_division=0)
        r = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        th_records.append({
            "threshold": t,
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1_score": round(float(f1), 4)
        })
    return th_records


def main():
    start_total = time.time()
    print("=" * 70)
    print(" AudienceIQ — Phase 7: Purchase / Reorder Prediction Model Engine")
    print("=" * 70)
    
    df_full = load_dataset(sample_rows=500000)
    X_train, y_train, X_val, y_val = split_train_val(df_full)
    
    metrics_list, best_model, best_probs, y_val = train_models(X_train, y_train, X_val, y_val)
    threshold_analysis = analyze_thresholds(y_val, best_probs)
    
    # 5. Serialization
    print("[5/5] Exporting best model artifact and evaluation report...")
    joblib.dump(best_model, MODELS_DIR / "reorder_prediction_model.joblib")
    
    report_output = {
        "models_benchmark": metrics_list,
        "threshold_analysis": threshold_analysis,
        "feature_importances": {
            col: round(float(imp), 4)
            for col, imp in zip(FEATURE_COLS, best_model.feature_importances_)
        }
    }
    
    with open(DATA_DIR / "model_evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report_output, f, indent=2)
        
    print("\n" + "=" * 70)
    print(" AudienceIQ — Model Evaluation Benchmark Summary")
    print("=" * 70)
    print(f"{'Model':<30} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Precision':<10} | {'Recall':<8} | {'F1':<6}")
    print("-" * 78)
    for m in metrics_list:
        print(f"{m['model']:<30} | {m['roc_auc']:<8.4f} | {m['pr_auc']:<8.4f} | {m['precision']:<10.4f} | {m['recall']:<8.4f} | {m['f1_score']:<6.4f}")
        
    print("\n--- Precision-Recall Tradeoff on LightGBM ---")
    print(f"{'Threshold':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 46)
    for th in threshold_analysis:
        print(f"{th['threshold']:<12.2f} | {th['precision']:<10.4f} | {th['recall']:<10.4f} | {th['f1_score']:<10.4f}")
        
    print("=" * 70)
    print(f"[+] Phase 7 Model Training & Validation completed in {time.time()-start_total:.1f}s.")
    print("=" * 70)


if __name__ == "__main__":
    main()
