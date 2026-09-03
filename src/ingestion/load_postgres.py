"""
Nexora — Phase 3: PostgreSQL Data Loader & SQL Pipeline Runner
==============================================================
This script automates:
  1. Connecting to PostgreSQL using environment variables or connection parameters.
  2. Executing `sql/schema.sql` DDL to create tables and indexes.
  3. Bulk loading cleaned CSV data from `data/processed/` using high-speed PostgreSQL `COPY`.
  4. Executing `sql/transformations.sql` to build analytical views.
  5. Running verification queries to confirm row counts and integrity.
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Load optional .env file
load_dotenv()

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
SQL_DIR = Path(__file__).resolve().parent.parent.parent / "sql"

# Connection parameters
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = os.getenv("POSTGRES_PORT", "5432")
PG_DB = os.getenv("POSTGRES_DB", "audienceiq_db")
PG_USER = os.getenv("POSTGRES_USER", "postgres")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")


def get_connection():
    """Create and return a psycopg2 connection."""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DB,
            user=PG_USER,
            password=PG_PASSWORD
        )
        conn.autocommit = True
        return conn
    except Exception as e:
        print(f"[!] PostgreSQL Connection Failed: {e}")
        print("    Ensure PostgreSQL is running and credentials in .env or environment variables are valid.")
        return None


def execute_sql_file(conn, file_path: Path):
    """Execute all SQL statements in a file."""
    print(f"[*] Executing SQL file: {file_path.name}...")
    with open(file_path, "r", encoding="utf-8") as f:
        sql_content = f.read()
    with conn.cursor() as cur:
        cur.execute(sql_content)
    print(f"    [+] Successfully applied {file_path.name}")


def bulk_copy_csv(conn, table_name: str, csv_path: Path, columns: list):
    """Load CSV into Postgres table using COPY for maximum throughput."""
    print(f"[*] Bulk loading {csv_path.name} -> {table_name}...")
    cols_str = f"({', '.join(columns)})" if columns else ""
    copy_sql = f"""
        COPY {table_name} {cols_str}
        FROM STDIN
        WITH (FORMAT csv, HEADER true, DELIMITER ',');
    """
    with conn.cursor() as cur:
        with open(csv_path, "r", encoding="utf-8") as f:
            cur.copy_expert(sql=copy_sql, file=f)
    print(f"    [+] Loaded {csv_path.name} into {table_name}")


def main():
    print("=" * 70)
    print(" AudienceIQ — Phase 3: PostgreSQL Data Ingestion & SQL Modeling Engine")
    print("=" * 70)
    
    conn = get_connection()
    if conn is None:
        print("[!] Aborting PostgreSQL ingestion. Database connection could not be established.")
        sys.exit(1)
        
    start_time = time.time()
    
    # 1. Execute Schema DDL
    execute_sql_file(conn, SQL_DIR / "schema.sql")
    
    # 2. Bulk Copy data in foreign key dependency order
    bulk_copy_csv(conn, "audienceiq.aisles", PROCESSED_DIR / "aisles.csv", ["aisle_id", "aisle"])
    bulk_copy_csv(conn, "audienceiq.departments", PROCESSED_DIR / "departments.csv", ["department_id", "department"])
    bulk_copy_csv(conn, "audienceiq.products", PROCESSED_DIR / "products.csv", ["product_id", "product_name", "aisle_id", "department_id"])
    bulk_copy_csv(conn, "audienceiq.customers", PROCESSED_DIR / "customers.csv", ["user_id", "total_orders", "has_train_order", "has_test_order"])
    bulk_copy_csv(conn, "audienceiq.orders", PROCESSED_DIR / "orders.csv", ["order_id", "user_id", "eval_set", "order_number", "order_dow", "order_hour_of_day", "days_since_prior_order", "is_first_order"])
    bulk_copy_csv(conn, "audienceiq.order_products", PROCESSED_DIR / "order_products.csv", ["order_id", "product_id", "add_to_cart_order", "reordered"])
    
    # 3. Apply Transformations
    execute_sql_file(conn, SQL_DIR / "transformations.sql")
    
    # 4. Verify Row Counts
    print("[*] Verifying PostgreSQL table row counts...")
    tables = ["departments", "aisles", "products", "customers", "orders", "order_products"]
    with conn.cursor() as cur:
        for t in tables:
            cur.execute(f"SELECT COUNT(*) FROM audienceiq.{t};")
            cnt = cur.fetchone()[0]
            print(f"    - `audienceiq.{t}`: {cnt:,} rows")
            
    conn.close()
    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"[+] PostgreSQL Pipeline completed successfully in {elapsed:.1f}s.")
    print("=" * 70)


if __name__ == "__main__":
    main()
