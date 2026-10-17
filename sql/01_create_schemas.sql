-- 01_create_schemas.sql
-- Create database namespaces/schemas for E-Commerce Data Warehouse
-- Supports both PostgreSQL and SQLite (as table prefixes if schemas unsupported)

CREATE TABLE IF NOT EXISTS meta_schema_version (
    version VARCHAR(20) PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

INSERT OR IGNORE INTO meta_schema_version (version, description)
VALUES ('1.0.0', 'Initial E-Commerce Star Schema and OLAP structures');
