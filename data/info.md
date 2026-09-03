# AudienceIQ — Data Quality & Schema Validation Report

**Phase 1 Deliverable**: Raw Data Inspection, Quality Audit, and Referential Integrity Profiling.

## 1. Summary of Raw Files

| File Name | Size (MB) | Total Rows | Columns | Duplicates / Key Check |
| :--- | :--- | :--- | :--- | :--- |
| `aisles.csv` | 0.0 MB | 134 | 2 | 0 |
| `departments.csv` | 0.0 MB | 21 | 2 | 0 |
| `order_products__prior.csv` | 550.8 MB | 32,434,489 | 4 | Checked on sampled / primary keys |
| `order_products__train.csv` | 23.54 MB | 1,384,617 | 4 | 0 |
| `orders.csv` | 103.92 MB | 3,421,083 | 7 | Checked on sampled / primary keys |
| `products.csv` | 2.07 MB | 49,688 | 4 | 0 |

## 2. Table Schemas, Data Types & Missingness

### Table: `aisles.csv`
- **Total Records**: 134
- **Columns**: `aisle_id`, `aisle`

| Column | Data Type | Null Count | Null % | Min Value | Max Value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `aisle_id` | `int64` | 0 | 0.00% | 1 | 134 |
| `aisle` | `str` | 0 | 0.00% | N/A | N/A |

### Table: `departments.csv`
- **Total Records**: 21
- **Columns**: `department_id`, `department`

| Column | Data Type | Null Count | Null % | Min Value | Max Value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `department_id` | `int64` | 0 | 0.00% | 1 | 21 |
| `department` | `str` | 0 | 0.00% | N/A | N/A |

### Table: `order_products__prior.csv`
- **Total Records**: 32,434,489
- **Columns**: `order_id`, `product_id`, `add_to_cart_order`, `reordered`

| Column | Data Type | Null Count | Null % | Min Value | Max Value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `order_id` | `int64` | 0 | 0.00% | 2 | 3421083 |
| `product_id` | `int64` | 0 | 0.00% | 1 | 49688 |
| `add_to_cart_order` | `int64` | 0 | 0.00% | 1 | 145 |
| `reordered` | `int64` | 0 | 0.00% | 0 | 1 |

### Table: `order_products__train.csv`
- **Total Records**: 1,384,617
- **Columns**: `order_id`, `product_id`, `add_to_cart_order`, `reordered`

| Column | Data Type | Null Count | Null % | Min Value | Max Value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `order_id` | `int64` | 0 | 0.00% | 1 | 3421070 |
| `product_id` | `int64` | 0 | 0.00% | 1 | 49688 |
| `add_to_cart_order` | `int64` | 0 | 0.00% | 1 | 80 |
| `reordered` | `int64` | 0 | 0.00% | 0 | 1 |

### Table: `orders.csv`
- **Total Records**: 3,421,083
- **Columns**: `order_id`, `user_id`, `eval_set`, `order_number`, `order_dow`, `order_hour_of_day`, `days_since_prior_order`

| Column | Data Type | Null Count | Null % | Min Value | Max Value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `order_id` | `int64` | 0 | 0.00% | 1 | 3421083 |
| `user_id` | `int64` | 0 | 0.00% | 1 | 206209 |
| `eval_set` | `str` | 0 | 0.00% | N/A | N/A |
| `order_number` | `int64` | 0 | 0.00% | 1 | 100 |
| `order_dow` | `int64` | 0 | 0.00% | 0 | 6 |
| `order_hour_of_day` | `int64` | 0 | 0.00% | 0 | 23 |
| `days_since_prior_order` | `float64` | 206,209 | 6.03% | 0.0 | 30.0 |

### Table: `products.csv`
- **Total Records**: 49,688
- **Columns**: `product_id`, `product_name`, `aisle_id`, `department_id`

| Column | Data Type | Null Count | Null % | Min Value | Max Value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `product_id` | `int64` | 0 | 0.00% | 1 | 49688 |
| `product_name` | `str` | 0 | 0.00% | N/A | N/A |
| `aisle_id` | `int64` | 0 | 0.00% | 1 | 134 |
| `department_id` | `int64` | 0 | 0.00% | 1 | 21 |

## 3. Referential Integrity Checks

| Foreign Key Relationship | Status | Orphan Records | Validation Details |
| :--- | :---: | :---: | :--- |
| `products.aisle_id -> aisles.aisle_id` | **✅ PASS** | 0 | All product aisle references are valid. |
| `products.department_id -> departments.department_id` | **✅ PASS** | 0 | All product department references are valid. |
| `order_products__prior.order_id -> orders.order_id (prior set)` | **✅ PASS** | 0 | Checked 32,434,489 items across prior orders. |
| `order_products__prior.product_id -> products.product_id` | **✅ PASS** | 0 | Checked 32,434,489 items against products catalog. |
| `order_products__train.order_id -> orders.order_id (train set)` | **✅ PASS** | 0 | Checked 1,384,617 items across train orders. |
| `order_products__train.product_id -> products.product_id` | **✅ PASS** | 0 | Checked 1,384,617 items against products catalog. |

## 4. Key Observations & Ingestion Insights

1. **Missing Values in `orders.csv` (`days_since_prior_order`)**:
   - Missing exclusively for `order_number = 1` for each customer (first order has no prior order).
   - This is structurally expected and should be handled cleanly during Phase 2 (e.g. 0 or separate initial order flag).
2. **Referential Integrity**:
   - All order references in `order_products__prior.csv` and `order_products__train.csv` match their corresponding `eval_set` partitions in `orders.csv` with zero orphans.
   - All product IDs map 100% to `products.csv`.
   - All `aisle_id` and `department_id` references map 100% to `aisles.csv` and `departments.csv`.
3. **Value Domains**:
   - `order_dow` ranges from 0 to 6 (7 days of the week).
   - `order_hour_of_day` ranges from 0 to 23 (24 hours).
   - `reordered` is strictly binary (0 or 1).
   - `add_to_cart_order` starts at 1.

---
*Report generated by `src/ingestion/validate_raw.py`*