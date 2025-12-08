// Database module - connection management, schema tracking, and migrations
// Provides unified database access across PostgreSQL, MySQL, SQLite, MongoDB, and Redis

pub mod connection_manager;
pub mod schema_tracker;
pub mod migration_manager;

pub use connection_manager::{
    DatabaseManager, DatabaseType, ConnectionConfig, DatabaseConnection,
    QueryResult, SchemaInfo, TableInfo, ColumnInfo, ForeignKeyInfo,
};
pub use schema_tracker::{SchemaTracker, SchemaChange, SchemaVersion};
pub use migration_manager::{MigrationManager, Migration, MigrationDirection, MigrationResult};
