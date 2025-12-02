// Database Migration Manager
// Generates and applies database migrations with safety checks and rollback support

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::connection_manager::DatabaseManager;
use super::schema_tracker::{SchemaChange, SchemaTracker};

/// Migration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Migration {
    pub id: String,
    pub name: String,
    pub version: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub up_sql: String,
    pub down_sql: String,
    pub dependencies: Vec<String>,
    pub checksum: String,
    pub applied: bool,
    pub applied_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Migration status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MigrationStatus {
    Pending,
    Applied,
    Failed { error: String },
    RolledBack,
}

/// Migration result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationResult {
    pub migration_id: String,
    pub status: MigrationStatus,
    pub duration_ms: u64,
    pub message: String,
}

/// Migration Manager - generates and applies migrations
pub struct MigrationManager {
    db_manager: Arc<DatabaseManager>,
    schema_tracker: Arc<SchemaTracker>,
    migrations: Arc<RwLock<HashMap<String, Vec<Migration>>>>,
}

impl MigrationManager {
    /// Create new migration manager
    pub fn new(db_manager: Arc<DatabaseManager>, schema_tracker: Arc<SchemaTracker>) -> Self {
        Self {
            db_manager,
            schema_tracker,
            migrations: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Generate migration from schema changes
    pub async fn generate_migration(
        &self,
        db_name: &str,
        name: &str,
        changes: &[SchemaChange],
    ) -> Result<Migration, String> {
        let mut up_sql = String::new();
        let mut down_sql = String::new();

        for change in changes {
            let (up, down) = self.generate_sql_for_change(change)?;
            up_sql.push_str(&up);
            up_sql.push_str(";\n");
            down_sql.push_str(&down);
            down_sql.push_str(";\n");
        }

        let migration = Migration {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.to_string(),
            version: format!("v{}", chrono::Utc::now().timestamp()),
            timestamp: chrono::Utc::now(),
            up_sql,
            down_sql,
            dependencies: vec![],
            checksum: self.calculate_checksum(&up_sql),
            applied: false,
            applied_at: None,
        };

        Ok(migration)
    }

    /// Generate SQL for a schema change
    fn generate_sql_for_change(&self, change: &SchemaChange) -> Result<(String, String), String> {
        match change {
            SchemaChange::AddTable { name } => {
                let up = format!("CREATE TABLE {} (id INTEGER PRIMARY KEY)", name);
                let down = format!("DROP TABLE {}", name);
                Ok((up, down))
            }
            SchemaChange::DropTable { name } => {
                let up = format!("DROP TABLE {}", name);
                let down = format!("-- Cannot automatically recreate table {}", name);
                Ok((up, down))
            }
            SchemaChange::AddColumn { table, column } => {
                let nullable = if column.nullable { "NULL" } else { "NOT NULL" };
                let default = column
                    .default_value
                    .as_ref()
                    .map(|v| format!(" DEFAULT {}", v))
                    .unwrap_or_default();
                let up = format!(
                    "ALTER TABLE {} ADD COLUMN {} {} {}{}",
                    table, column.name, column.data_type, nullable, default
                );
                let down = format!("ALTER TABLE {} DROP COLUMN {}", table, column.name);
                Ok((up, down))
            }
            SchemaChange::DropColumn { table, column } => {
                let up = format!("ALTER TABLE {} DROP COLUMN {}", table, column);
                let down = format!("-- Cannot automatically recreate column {}.{}", table, column);
                Ok((up, down))
            }
            SchemaChange::ModifyColumn { table, old_column, new_column } => {
                let nullable = if new_column.nullable { "NULL" } else { "NOT NULL" };
                let up = format!(
                    "ALTER TABLE {} ALTER COLUMN {} TYPE {} {}",
                    table, new_column.name, new_column.data_type, nullable
                );
                let down = format!(
                    "ALTER TABLE {} ALTER COLUMN {} TYPE {} {}",
                    table, old_column.name, old_column.data_type,
                    if old_column.nullable { "NULL" } else { "NOT NULL" }
                );
                Ok((up, down))
            }
            SchemaChange::AddForeignKey { table, column, referenced_table } => {
                let constraint_name = format!("fk_{}_{}", table, column);
                let up = format!(
                    "ALTER TABLE {} ADD CONSTRAINT {} FOREIGN KEY ({}) REFERENCES {}(id)",
                    table, constraint_name, column, referenced_table
                );
                let down = format!("ALTER TABLE {} DROP CONSTRAINT {}", table, constraint_name);
                Ok((up, down))
            }
            SchemaChange::DropForeignKey { table, column } => {
                let constraint_name = format!("fk_{}_{}", table, column);
                let up = format!("ALTER TABLE {} DROP CONSTRAINT {}", table, constraint_name);
                let down = format!("-- Cannot automatically recreate foreign key");
                Ok((up, down))
            }
        }
    }

    /// Apply migration with safety checks
    pub async fn apply_migration(
        &self,
        db_name: &str,
        migration: &Migration,
    ) -> Result<MigrationResult, String> {
        let start = std::time::Instant::now();

        // Validate migration hasn't been applied
        if migration.applied {
            return Err("Migration already applied".to_string());
        }

        // Check dependencies
        self.check_dependencies(db_name, migration).await?;

        // Validate migration with GNN
        self.validate_with_gnn(db_name, migration).await?;

        // Start transaction (conceptual - actual implementation would use DB transactions)
        // Execute migration
        match self.execute_migration_sql(db_name, &migration.up_sql).await {
            Ok(_) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                
                // Mark migration as applied
                let mut applied_migration = migration.clone();
                applied_migration.applied = true;
                applied_migration.applied_at = Some(chrono::Utc::now());
                
                // Store migration
                self.store_migration(db_name, applied_migration).await?;

                Ok(MigrationResult {
                    migration_id: migration.id.clone(),
                    status: MigrationStatus::Applied,
                    duration_ms,
                    message: "Migration applied successfully".to_string(),
                })
            }
            Err(e) => {
                // Rollback on error
                let rollback_result = self.rollback_migration(db_name, migration).await;
                
                let duration_ms = start.elapsed().as_millis() as u64;
                Ok(MigrationResult {
                    migration_id: migration.id.clone(),
                    status: MigrationStatus::Failed {
                        error: format!("{} (rollback: {:?})", e, rollback_result),
                    },
                    duration_ms,
                    message: format!("Migration failed: {}", e),
                })
            }
        }
    }

    /// Execute migration SQL
    async fn execute_migration_sql(&self, db_name: &str, sql: &str) -> Result<(), String> {
        // Split SQL into individual statements
        let statements: Vec<&str> = sql.split(';').filter(|s| !s.trim().is_empty()).collect();

        for statement in statements {
            self.db_manager.execute(db_name, statement).await?;
        }

        Ok(())
    }

    /// Rollback migration
    pub async fn rollback_migration(
        &self,
        db_name: &str,
        migration: &Migration,
    ) -> Result<MigrationResult, String> {
        let start = std::time::Instant::now();

        if !migration.applied {
            return Err("Migration not applied, cannot rollback".to_string());
        }

        match self.execute_migration_sql(db_name, &migration.down_sql).await {
            Ok(_) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                
                // Mark migration as rolled back
                self.remove_migration(db_name, &migration.id).await?;

                Ok(MigrationResult {
                    migration_id: migration.id.clone(),
                    status: MigrationStatus::RolledBack,
                    duration_ms,
                    message: "Migration rolled back successfully".to_string(),
                })
            }
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                Ok(MigrationResult {
                    migration_id: migration.id.clone(),
                    status: MigrationStatus::Failed {
                        error: e.clone(),
                    },
                    duration_ms,
                    message: format!("Rollback failed: {}", e),
                })
            }
        }
    }

    /// Validate migration with GNN
    async fn validate_with_gnn(&self, db_name: &str, migration: &Migration) -> Result<(), String> {
        // TODO: Integrate with GNN
        // Check if any code references tables/columns being dropped
        // Analyze impact of schema changes on application code
        // Return error if unsafe changes detected
        
        println!("GNN validation for migration {}", migration.id);
        Ok(())
    }

    /// Check migration dependencies
    async fn check_dependencies(&self, db_name: &str, migration: &Migration) -> Result<(), String> {
        let migrations = self.migrations.read().await;
        let db_migrations = migrations.get(db_name).ok_or("No migrations found")?;

        for dep_id in &migration.dependencies {
            let dep_applied = db_migrations.iter().any(|m| &m.id == dep_id && m.applied);
            if !dep_applied {
                return Err(format!("Dependency migration {} not applied", dep_id));
            }
        }

        Ok(())
    }

    /// Store applied migration
    async fn store_migration(&self, db_name: &str, migration: Migration) -> Result<(), String> {
        let mut migrations = self.migrations.write().await;
        migrations
            .entry(db_name.to_string())
            .or_insert_with(Vec::new)
            .push(migration);
        Ok(())
    }

    /// Remove migration (after rollback)
    async fn remove_migration(&self, db_name: &str, migration_id: &str) -> Result<(), String> {
        let mut migrations = self.migrations.write().await;
        if let Some(db_migrations) = migrations.get_mut(db_name) {
            db_migrations.retain(|m| m.id != migration_id);
        }
        Ok(())
    }

    /// Get migration history
    pub async fn get_history(&self, db_name: &str) -> Vec<Migration> {
        let migrations = self.migrations.read().await;
        migrations.get(db_name).cloned().unwrap_or_default()
    }

    /// Get pending migrations
    pub async fn get_pending(&self, db_name: &str) -> Vec<Migration> {
        let migrations = self.migrations.read().await;
        migrations
            .get(db_name)
            .map(|migs| migs.iter().filter(|m| !m.applied).cloned().collect())
            .unwrap_or_default()
    }

    /// Calculate checksum for SQL
    fn calculate_checksum(&self, sql: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        sql.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_generate_add_table_migration() {
        let db_manager = Arc::new(DatabaseManager::new().unwrap());
        let schema_tracker = Arc::new(SchemaTracker::new(db_manager.clone()));
        let migration_manager = MigrationManager::new(db_manager, schema_tracker);

        let changes = vec![SchemaChange::AddTable {
            name: "users".to_string(),
        }];

        let migration = migration_manager
            .generate_migration("test_db", "create_users_table", &changes)
            .await
            .unwrap();

        assert!(migration.up_sql.contains("CREATE TABLE users"));
        assert!(migration.down_sql.contains("DROP TABLE users"));
        assert!(!migration.applied);
    }

    #[tokio::test]
    async fn test_migration_checksum() {
        let db_manager = Arc::new(DatabaseManager::new().unwrap());
        let schema_tracker = Arc::new(SchemaTracker::new(db_manager.clone()));
        let migration_manager = MigrationManager::new(db_manager, schema_tracker);

        let checksum1 = migration_manager.calculate_checksum("SELECT * FROM users");
        let checksum2 = migration_manager.calculate_checksum("SELECT * FROM users");
        let checksum3 = migration_manager.calculate_checksum("SELECT * FROM posts");

        assert_eq!(checksum1, checksum2);
        assert_ne!(checksum1, checksum3);
    }
}
