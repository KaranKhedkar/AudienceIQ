-- AudienceIQ — Phase 3: PostgreSQL Schema Definition
-- Base relational schema for e-commerce / Instacart market basket transactions

CREATE SCHEMA IF NOT EXISTS audienceiq;

-- 1. Departments Table
CREATE TABLE IF NOT EXISTS audienceiq.departments (
    department_id SMALLINT PRIMARY KEY,
    department VARCHAR(100) NOT NULL
);

-- 2. Aisles Table
CREATE TABLE IF NOT EXISTS audienceiq.aisles (
    aisle_id SMALLINT PRIMARY KEY,
    aisle VARCHAR(100) NOT NULL
);

-- 3. Products Table
CREATE TABLE IF NOT EXISTS audienceiq.products (
    product_id INT PRIMARY KEY,
    aisle_id SMALLINT REFERENCES audienceiq.aisles(aisle_id),
    department_id SMALLINT REFERENCES audienceiq.departments(department_id),
    product_name VARCHAR(255) NOT NULL
);

-- 4. Customers Table (Extracted entity)
CREATE TABLE IF NOT EXISTS audienceiq.customers (
    user_id INT PRIMARY KEY,
    total_orders INT NOT NULL,
    has_train_order BOOLEAN NOT NULL DEFAULT FALSE,
    has_test_order BOOLEAN NOT NULL DEFAULT FALSE
);

-- 5. Orders Table
CREATE TABLE IF NOT EXISTS audienceiq.orders (
    order_id INT PRIMARY KEY,
    user_id INT REFERENCES audienceiq.customers(user_id),
    eval_set VARCHAR(10) NOT NULL,
    order_number INT NOT NULL,
    order_dow SMALLINT NOT NULL,
    order_hour_of_day SMALLINT NOT NULL,
    days_since_prior_order FLOAT NOT NULL,
    is_first_order SMALLINT NOT NULL DEFAULT 0
);

-- 6. Order Products Table (Combined prior + train)
CREATE TABLE IF NOT EXISTS audienceiq.order_products (
    order_id INT REFERENCES audienceiq.orders(order_id),
    product_id INT REFERENCES audienceiq.products(product_id),
    add_to_cart_order SMALLINT NOT NULL,
    reordered SMALLINT NOT NULL,
    PRIMARY KEY (order_id, product_id)
);

-- Indexes for lightning fast analytics
CREATE INDEX IF NOT EXISTS idx_orders_user_id ON audienceiq.orders(user_id);
CREATE INDEX IF NOT EXISTS idx_orders_eval_set ON audienceiq.orders(eval_set);
CREATE INDEX IF NOT EXISTS idx_op_product_id ON audienceiq.order_products(product_id);
CREATE INDEX IF NOT EXISTS idx_op_order_id ON audienceiq.order_products(order_id);
CREATE INDEX IF NOT EXISTS idx_products_dept_aisle ON audienceiq.products(department_id, aisle_id);
