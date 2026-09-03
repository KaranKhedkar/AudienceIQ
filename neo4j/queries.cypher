// Nexora — Phase 9: Neo4j Analytical & Recommendation Queries

// ====================================================================
// 1. Cross-Sell Intelligence: Frequently Bought Together Products
// ====================================================================
MATCH (p1:Product {name: $product_name})-[r:OFTEN_BOUGHT_WITH]->(p2:Product)
RETURN p1.name AS source_product, p2.name AS recommended_product, r.co_occurrence AS times_co_purchased
ORDER BY r.co_occurrence DESC
LIMIT 10;

// ====================================================================
// 2. Customer Collaborative Filtering: Similar Customer Recommendations
// ====================================================================
MATCH (target:Customer {user_id: $target_user_id})-[:BOUGHT]->(p:Product)<-[:BOUGHT]-(other:Customer)
WHERE target <> other
WITH other, count(p) AS common_products_count
ORDER BY common_products_count DESC
LIMIT 15
MATCH (other)-[:BOUGHT]->(rec:Product)
WHERE NOT (target)-[:BOUGHT]->(rec)
RETURN rec.name AS recommended_product, count(other) AS affinity_score, rec.reorder_rate AS product_reorder_rate
ORDER BY affinity_score DESC, product_reorder_rate DESC
LIMIT 10;

// ====================================================================
// 3. Segment Affinity: Top Category Breakdown by Customer Segment
// ====================================================================
MATCH (c:Customer {segment: $segment_name})-[:BOUGHT]->(p:Product)-[:BELONGS_TO]->(d:Department)
RETURN d.name AS department, count(p) AS total_purchases
ORDER BY total_purchases DESC
LIMIT 8;

// ====================================================================
// 4. Product Graph Neighborhood Discovery
// ====================================================================
MATCH (p:Product {name: $product_name})-[:BELONGS_TO]->(a:Aisle)-[:IN_DEPARTMENT]->(d:Department)
OPTIONAL MATCH (p)-[r:OFTEN_BOUGHT_WITH]->(related:Product)
RETURN p.name AS product, a.name AS aisle, d.name AS department, collect(related.name)[..5] AS top_affinity_items;
