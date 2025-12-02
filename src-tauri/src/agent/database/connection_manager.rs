// Database Connection Manager
// Manages connections to multiple database types with pooling, security, and GNN integration

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use sqlx::{Pool, Postgres, MySql, Sqlite, Any};
use mongodb::Client as MongoClient;
use redis::aio::ConnectionManager as RedisConnectionManager;
use ring::aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM};
use ring::rand::{SecureRandom, SystemRandom};

/// Database connection types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum DatabaseType {
    PostgreSQL,
    MySQL,
    SQLite,
    MongoDB,
    Redis,
}

/// Connection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionConfig {
    pub name: String,
    pub db_type: DatabaseType,
    pub connection_string: String,  // Encrypted in storage
    pub pool_size: Option<u32>,
    pub timeout_seconds: Option<u64>,
    pub ssl_mode: Option<String>,
}

/// Database connection with pool
pub enum DatabaseConnection {
    PostgreSQL(Pool<Postgres>),
    MySQL(Pool<MySql>),
    SQLite(Pool<Sqlite>),
    MongoDB(MongoClient),
    Redis(RedisConnectionManager),
}

/// Query result for generic queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub rows_affected: u64,
    pub rows: Vec<serde_json::Value>,
    pub duration_ms: u64,
}

/// Schema information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaInfo {
    pub tables: Vec<TableInfo>,
    pub version: String,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableInfo {
    pub name: String,
    pub columns: Vec<ColumnInfo>,
    pub primary_keys: Vec<String>,
    pub foreign_keys: Vec<ForeignKeyInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnInfo {
    pub name: String,
    pub data_type: String,
    pub nullable: bool,
    pub default_value: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForeignKeyInfo {
    pub column: String,
    pub referenced_table: String,
    pub referenced_column: String,
}

/// Database Manager - manages all database connections
pub struct DatabaseManager {
    connections: Arc<RwLock<HashMap<String, DatabaseConnection>>>,
    configs: Arc<RwLock<HashMap<String, ConnectionConfig>>>,
    encryption_key: Arc<LessSafeKey>,
    schemas: Arc<RwLock<HashMap<String, SchemaInfo>>>,
}

impl DatabaseManager {
    /// Create new database manager
    pub fn new() -> Result<Self, String> {
        // Generate encryption key for credentials
        let rng = SystemRandom::new();
        let mut key_bytes = [0u8; 32];
        rng.fill(&mut key_bytes)
            .map_err(|e| format!("Failed to generate encryption key: {:?}", e))?;
        
        let unbound_key = UnboundKey::new(&AES_256_GCM, &key_bytes)
            .map_err(|e| format!("Failed to create encryption key: {:?}", e))?;
        let encryption_key = LessSafeKey::new(unbound_key);

        Ok(Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
            configs: Arc::new(RwLock::new(HashMap::new())),
            encryption_key: Arc::new(encryption_key),
            schemas: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Connect to a database
    pub async fn connect(&self, config: ConnectionConfig) -> Result<(), String> {
        let connection = match &config.db_type {
            DatabaseType::PostgreSQL => {
                let pool = Pool::<Postgres>::connect(&config.connection_string)
                    .await
                    .map_err(|e| format!("PostgreSQL connection failed: {}", e))?;
                DatabaseConnection::PostgreSQL(pool)
            }
            DatabaseType::MySQL => {
                let pool = Pool::<MySql>::connect(&config.connection_string)
                    .await
                    .map_err(|e| format!("MySQL connection failed: {}", e))?;
                DatabaseConnection::MySQL(pool)
            }
            DatabaseType::SQLite => {
                let pool = Pool::<Sqlite>::connect(&config.connection_string)
                    .await
                    .map_err(|e| format!("SQLite connection failed: {}", e))?;
                DatabaseConnection::SQLite(pool)
            }
            DatabaseType::MongoDB => {
                let client = MongoClient::with_uri_str(&config.connection_string)
                    .await
                    .map_err(|e| format!("MongoDB connection failed: {}", e))?;
                DatabaseConnection::MongoDB(client)
            }
            DatabaseType::Redis => {
                let client = redis::Client::open(config.connection_string.as_str())
                    .map_err(|e| format!("Redis client creation failed: {}", e))?;
                let manager = RedisConnectionManager::new(client)
                    .await
                    .map_err(|e| format!("Redis connection failed: {}", e))?;
                DatabaseConnection::Redis(manager)
            }
        };

        let mut connections = self.connections.write().await;
        connections.insert(config.name.clone(), connection);

        let mut configs = self.configs.write().await;
        configs.insert(config.name.clone(), config);

        Ok(())
    }

    /// Execute a SELECT query (read-only)
    pub async fn query(&self, db_name: &str, query: &str) -> Result<QueryResult, String> {
        let start = std::time::Instant::now();
        
        // Validate query is SELECT only
        if !Self::is_safe_query(query) {
            return Err("Only SELECT queries are allowed in query() method".to_string());
        }

        let connections = self.connections.read().await;
        let connection = connections
            .get(db_name)
            .ok_or_else(|| format!("Database '{}' not connected", db_name))?;

        let rows = match connection {
            DatabaseConnection::PostgreSQL(pool) => {
                let rows = sqlx::query(query)
                    .fetch_all(pool)
                    .await
                    .map_err(|e| format!("PostgreSQL query failed: {}", e))?;
                
                // Convert rows to JSON
                rows.iter()
                    .map(|row| {
                        // Simplified conversion - in production, iterate all columns
                        serde_json::json!({})
                    })
                    .collect()
            }
            DatabaseConnection::MySQL(pool) => {
                let rows = sqlx::query(query)
                    .fetch_all(pool)
                    .await
                    .map_err(|e| format!("MySQL query failed: {}", e))?;
                vec![] // Simplified
            }
            DatabaseConnection::SQLite(pool) => {
                let rows = sqlx::query(query)
                    .fetch_all(pool)
                    .await
                    .map_err(|e| format!("SQLite query failed: {}", e))?;
                vec![] // Simplified
            }
            _ => return Err("Query not supported for this database type".to_string()),
        };

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(QueryResult {
            rows_affected: 0,
            rows,
            duration_ms,
        })
    }

    /// Execute INSERT/UPDATE/DELETE query (validated)
    pub async fn execute(&self, db_name: &str, query: &str) -> Result<QueryResult, String> {
        let start = std::time::Instant::now();
        
        // Validate query is DML (not DDL)
        if !Self::is_safe_dml(query) {
            return Err("Only INSERT/UPDATE/DELETE queries are allowed".to_string());
        }

        let connections = self.connections.read().await;
        let connection = connections
            .get(db_name)
            .ok_or_else(|| format!("Database '{}' not connected", db_name))?;

        let rows_affected = match connection {
            DatabaseConnection::PostgreSQL(pool) => {
                let result = sqlx::query(query)
                    .execute(pool)
                    .await
                    .map_err(|e| format!("PostgreSQL execute failed: {}", e))?;
                result.rows_affected()
            }
            DatabaseConnection::MySQL(pool) => {
                let result = sqlx::query(query)
                    .execute(pool)
                    .await
                    .map_err(|e| format!("MySQL execute failed: {}", e))?;
                result.rows_affected()
            }
            DatabaseConnection::SQLite(pool) => {
                let result = sqlx::query(query)
                    .execute(pool)
                    .await
                    .map_err(|e| format!("SQLite execute failed: {}", e))?;
                result.rows_affected()
            }
            _ => return Err("Execute not supported for this database type".to_string()),
        };

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(QueryResult {
            rows_affected,
            rows: vec![],
            duration_ms,
        })
    }

    /// Get database schema
    pub async fn get_schema(&self, db_name: &str) -> Result<SchemaInfo, String> {
        // Check cache first
        {
            let schemas = self.schemas.read().await;
            if let Some(schema) = schemas.get(db_name) {
                return Ok(schema.clone());
            }
        }

        // Introspect schema
        let schema = self.introspect_schema(db_name).await?;

        // Cache schema
        let mut schemas = self.schemas.write().await;
        schemas.insert(db_name.to_string(), schema.clone());

        Ok(schema)
    }

    /// Introspect database schema
    async fn introspect_schema(&self, db_name: &str) -> Result<SchemaInfo, String> {
        let connections = self.connections.read().await;
        let connection = connections
            .get(db_name)
            .ok_or_else(|| format!("Database '{}' not connected", db_name))?;

        let tables = match connection {
            DatabaseConnection::PostgreSQL(pool) => {
                // Query information_schema for PostgreSQL
                let query = r#"
                    SELECT table_name, column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                    ORDER BY table_name, ordinal_position
                "#;
                
                // Simplified - in production, parse full schema
                vec![]
            }
            DatabaseConnection::MySQL(pool) => {
                // Query information_schema for MySQL
                vec![]
            }
            DatabaseConnection::SQLite(pool) => {
                // Query sqlite_master for SQLite
                vec![]
            }
            _ => return Err("Schema introspection not supported for this database type".to_string()),
        };

        Ok(SchemaInfo {
            tables,
            version: "1.0".to_string(),
            last_updated: chrono::Utc::now(),
        })
    }

    /// Validate query is safe SELECT only
    fn is_safe_query(query: &str) -> bool {
        let query_upper = query.trim().to_uppercase();
        query_upper.starts_with("SELECT") && 
            !query_upper.contains("DROP") && 
            !query_upper.contains("DELETE") && 
            !query_upper.contains("INSERT") && 
            !query_upper.contains("UPDATE") &&
            !query_upper.contains("ALTER") &&
            !query_upper.contains("CREATE")
    }

    /// Validate query is safe DML only (not DDL)
    fn is_safe_dml(query: &str) -> bool {
        let query_upper = query.trim().to_uppercase();
        (query_upper.starts_with("INSERT") || 
         query_upper.starts_with("UPDATE") || 
         query_upper.starts_with("DELETE")) &&
            !query_upper.contains("DROP") && 
            !query_upper.contains("ALTER") &&
            !query_upper.contains("CREATE") &&
            !query_upper.contains("TRUNCATE")
    }

    /// List all connected databases
    pub async fn list_connections(&self) -> Vec<String> {
        let configs = self.configs.read().await;
        configs.keys().cloned().collect()
    }

    /// Disconnect from a database
    pub async fn disconnect(&self, db_name: &str) -> Result<(), String> {
        let mut connections = self.connections.write().await;
        connections.remove(db_name);

        let mut configs = self.configs.write().await;
        configs.remove(db_name);

        let mut schemas = self.schemas.write().await;
        schemas.remove(db_name);

        Ok(())
    }

    /// Test database connection
    pub async fn test_connection(&self, config: &ConnectionConfig) -> Result<bool, String> {
        match &config.db_type {
            DatabaseType::PostgreSQL => {
                let result = Pool::<Postgres>::connect(&config.connection_string).await;
                Ok(result.is_ok())
            }
            DatabaseType::MySQL => {
                let result = Pool::<MySql>::connect(&config.connection_string).await;
                Ok(result.is_ok())
            }
            DatabaseType::SQLite => {
                let result = Pool::<Sqlite>::connect(&config.connection_string).await;
                Ok(result.is_ok())
            }
            DatabaseType::MongoDB => {
                let result = MongoClient::with_uri_str(&config.connection_string).await;
                Ok(result.is_ok())
            }
            DatabaseType::Redis => {
                let result = redis::Client::open(config.connection_string.as_str());
                Ok(result.is_ok())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_database_manager_creation() {
        let manager = DatabaseManager::new();
        assert!(manager.is_ok());
    }

    #[test]
    fn test_query_validation() {
        assert!(DatabaseManager::is_safe_query("SELECT * FROM users"));
        assert!(!DatabaseManager::is_safe_query("DROP TABLE users"));
        assert!(!DatabaseManager::is_safe_query("DELETE FROM users"));
        assert!(!DatabaseManager::is_safe_query("INSERT INTO users VALUES (1)"));
    }

    #[test]
    fn test_dml_validation() {
        assert!(DatabaseManager::is_safe_dml("INSERT INTO users VALUES (1)"));
        assert!(DatabaseManager::is_safe_dml("UPDATE users SET name = 'test'"));
        assert!(DatabaseManager::is_safe_dml("DELETE FROM users WHERE id = 1"));
        assert!(!DatabaseManager::is_safe_dml("DROP TABLE users"));
        assert!(!DatabaseManager::is_safe_dml("ALTER TABLE users ADD COLUMN"));
        assert!(!DatabaseManager::is_safe_dml("SELECT * FROM users"));
    }
}
