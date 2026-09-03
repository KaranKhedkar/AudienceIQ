// Nexora — Phase 9: Neo4j Knowledge Graph Schema
// Node Key Constraints & Indexes

// 1. Uniqueness Constraints
CREATE CONSTRAINT IF NOT EXISTS FOR (c:Customer) REQUIRE c.user_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (p:Product) REQUIRE p.product_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (a:Aisle) REQUIRE a.aisle_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (d:Department) REQUIRE d.department_id IS UNIQUE;

// 2. Performance Indexes
CREATE INDEX IF NOT EXISTS FOR (c:Customer) ON (c.segment);
CREATE INDEX IF NOT EXISTS FOR (p:Product) ON (p.name);
CREATE INDEX IF NOT EXISTS FOR (p:Product) ON (p.reorder_rate);
CREATE INDEX IF NOT EXISTS FOR (d:Department) ON (d.name);
CREATE INDEX IF NOT EXISTS FOR (a:Aisle) ON (a.name);
