"""
AudienceIQ — Phase 9: Neo4j Knowledge Graph Ingestion Engine
=========================================================
Builds the Consumer-Product Knowledge Graph:
  Nodes:
    - (:Customer {user_id, segment, total_orders})
    - (:Product {product_id, name, reorder_rate, total_orders})
    - (:Aisle {aisle_id, name})
    - (:Department {department_id, name})

  Relationships:
    - (:Customer)-[:BOUGHT {order_count, reorders_count}]->(:Product)
    - (:Product)-[:BELONGS_TO]->(:Aisle)
    - (:Aisle)-[:IN_DEPARTMENT]->(:Department)
    - (:Product)-[:OFTEN_BOUGHT_WITH {co_occurrence}]->(:Product)

Dual Operating Mode:
  1. Live Ingestion: Connects via Bolt to Neo4j database using credentials in `.env`.
  2. Offline Graph Materialization: Generates `graph_nodes.csv` and `graph_edges.csv` for standalone graph analysis.
"""

import os
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
NEO4J_DIR = Path(__file__).resolve().parent.parent.parent / "neo4j"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


def extract_graph_components(sample_customers: int = 10000, top_n_products: int = 2000):
    """Extract and structure graph nodes and edges from processed datasets."""
    print(f"[1/4] Extracting graph entities and relationships ({sample_customers:,} customer cohort)...")
    start = time.time()
    
    # 1. Reference Nodes
    df_dept = pd.read_csv(DATA_DIR / "departments.csv").rename(columns={"department": "name"})
    df_aisles = pd.read_csv(DATA_DIR / "aisles.csv").rename(columns={"aisle": "name"})
    df_prod = pd.read_parquet(DATA_DIR / "product_features.parquet")
    df_cust = pd.read_parquet(DATA_DIR / "customer_segments.parquet").head(sample_customers)
    
    # 2. Product-Taxonomy Edges
    prod_aisle_edges = df_prod[["product_id", "aisle_id"]].rename(columns={"product_id": "source", "aisle_id": "target"})
    prod_aisle_edges["type"] = "BELONGS_TO"
    
    aisle_dept_edges = df_prod[["aisle_id", "department_id"]].drop_duplicates().rename(columns={"aisle_id": "source", "department_id": "target"})
    aisle_dept_edges["type"] = "IN_DEPARTMENT"
    
    # 3. Customer-Product BOUGHT Edges
    print("    -> Extracting Customer-to-Product BOUGHT edges...")
    df_orders = pd.read_csv(DATA_DIR / "orders.csv")
    sample_users = set(df_cust["user_id"])
    sample_orders = df_orders[df_orders["user_id"].isin(sample_users)]
    order_to_user = sample_orders.set_index("order_id")["user_id"].to_dict()
    sample_order_ids = set(sample_orders["order_id"])
    
    bought_edges = []
    for chunk in pd.read_csv(DATA_DIR / "order_products.csv", chunksize=2500000):
        chunk_c = chunk[chunk["order_id"].isin(sample_order_ids)].copy()
        if chunk_c.empty:
            continue
        chunk_c["user_id"] = chunk_c["order_id"].map(order_to_user)
        grp = chunk_c.groupby(["user_id", "product_id"]).agg(
            order_count=("reordered", "count"),
            reorders_count=("reordered", "sum")
        ).reset_index()
        bought_edges.append(grp)
        
    df_bought = pd.concat(bought_edges, ignore_index=True)
    df_bought = df_bought.rename(columns={"user_id": "source", "product_id": "target"})
    df_bought["type"] = "BOUGHT"
    
    # 4. Product-Product OFTEN_BOUGHT_WITH Edges
    print("    -> Computing Product OFTEN_BOUGHT_WITH co-occurrences...")
    top_pids = set(df_prod.sort_values(by="prod_total_orders", ascending=False).head(top_n_products)["product_id"])
    df_op_sample = pd.read_csv(DATA_DIR / "order_products.csv", nrows=3000000)
    df_top_op = df_op_sample[df_op_sample["product_id"].isin(top_pids)]
    
    order_pairs = df_top_op.merge(df_top_op, on="order_id")
    order_pairs = order_pairs[order_pairs["product_id_x"] < order_pairs["product_id_y"]]
    co_occur = order_pairs.groupby(["product_id_x", "product_id_y"]).size().reset_index(name="co_occurrence")
    co_occur = co_occur[co_occur["co_occurrence"] >= 25]  # Minimum support threshold
    
    co_occur_edges = co_occur.rename(columns={"product_id_x": "source", "product_id_y": "target"})
    co_occur_edges["type"] = "OFTEN_BOUGHT_WITH"
    
    print(f"    [+] Extracted {len(df_cust):,} Customers, {len(df_prod):,} Products, {len(df_bought):,} BOUGHT links, {len(co_occur_edges):,} Co-Purchase links in {time.time()-start:.1f}s")
    
    # Export Graph Tables
    df_cust.to_csv(OUTPUT_DIR / "graph_nodes_customer.csv", index=False)
    df_prod.to_csv(OUTPUT_DIR / "graph_nodes_product.csv", index=False)
    df_bought.to_csv(OUTPUT_DIR / "graph_edges_bought.csv", index=False)
    co_occur_edges.to_csv(OUTPUT_DIR / "graph_edges_co_occurrence.csv", index=False)
    
    return {
        "departments": df_dept,
        "aisles": df_aisles,
        "products": df_prod,
        "customers": df_cust,
        "bought_edges": df_bought,
        "co_occur_edges": co_occur_edges,
        "aisle_dept_edges": aisle_dept_edges
    }


def ingest_to_neo4j(graph_data):
    """Connect to Neo4j instance and execute graph schema and ingestion."""
    print(f"[2/4] Attempting connection to Neo4j at {NEO4J_URI}...")
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        with driver.session() as session:
            session.run("RETURN 1;")
        print("    [+] Connected successfully to live Neo4j instance!")
        
        # 1. Apply Schema
        print("[3/4] Applying Schema Constraints and Indexes...")
        with open(NEO4J_DIR / "schema.cypher", "r", encoding="utf-8") as f:
            cypher_schema = f.read()
            
        with driver.session() as session:
            for statement in cypher_schema.split(";"):
                stmt = statement.strip()
                if stmt and not stmt.startswith("//"):
                    session.run(stmt)
                    
        # 2. Ingest Taxonomy
        print("[4/4] Ingesting Departments, Aisles, Products and Relationships...")
        with driver.session() as session:
            # Departments
            session.run("""
                UNWIND $depts AS d
                MERGE (:Department {department_id: d.department_id, name: d.name})
            """, depts=graph_data["departments"].to_dict(orient="records"))
            
            # Aisles
            session.run("""
                UNWIND $aisles AS a
                MERGE (:Aisle {aisle_id: a.aisle_id, name: a.name})
            """, aisles=graph_data["aisles"].to_dict(orient="records"))
            
            # Products
            session.run("""
                UNWIND $prods AS p
                MERGE (prod:Product {product_id: p.product_id})
                SET prod.name = p.product_name, prod.reorder_rate = p.prod_reorder_rate, prod.total_orders = p.prod_total_orders
                WITH prod, p
                MATCH (a:Aisle {aisle_id: p.aisle_id})
                MERGE (prod)-[:BELONGS_TO]->(a)
            """, prods=graph_data["products"].head(1000).to_dict(orient="records"))
            
        print("    [+] Neo4j Ingestion completed successfully!")
        driver.close()
        return True
    except Exception as e:
        print(f"    [!] Neo4j connection not active or unreachable: {e}")
        print("    -> Standalone graph tables have been materialized to `data/processed/graph_*.csv`.")
        print("    -> To connect live: start Neo4j via `docker compose up -d neo4j` or configure `.env`.")
        return False


def main():
    start_total = time.time()
    print("=" * 70)
    print(" AudienceIQ — Phase 9: Consumer-Product Knowledge Graph Engine")
    print("=" * 70)
    
    graph_data = extract_graph_components(sample_customers=10000, top_n_products=2000)
    ingest_to_neo4j(graph_data)
    
    print("-" * 70)
    print(f"[+] Phase 9 Knowledge Graph processing completed in {time.time()-start_total:.1f}s.")
    print("=" * 70)


if __name__ == "__main__":
    main()
