// Database Schema Tracker
// Tracks schema changes and integrates with GNN for dependency analysis

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::connection_manager::{DatabaseManager, SchemaInfo, TableInfo, ColumnInfo};

/// Schema change types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SchemaChange {
    AddTable { name: String },
    DropTable { name: String },
    AddColumn { table: String, column: ColumnInfo },
    DropColumn { table: String, column: String },
    ModifyColumn { table: String, old_column: ColumnInfo, new_column: ColumnInfo },
    AddForeignKey { table: String, column: String, referenced_table: String },
    DropForeignKey { table: String, column: String },
}

/// Schema version with changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaVersion {
    pub version: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub changes: Vec<SchemaChange>,
    pub schema: SchemaInfo,
}

/// Schema tracker - detects and tracks schema changes
pub struct SchemaTracker {
    db_manager: Arc<DatabaseManager>,
    schema_history: Arc<RwLock<HashMap<String, Vec<SchemaVersion>>>>,
}

impl SchemaTracker {
    /// Create new schema tracker
    pub fn new(db_manager: Arc<DatabaseManager>) -> Self {
        Self {
            db_manager,
            schema_history: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Introspect current database schema
    pub async fn introspect_schema(&self, db_name: &str) -> Result<SchemaInfo, String> {
        self.db_manager.get_schema(db_name).await
    }

    /// Detect changes between two schemas
    pub fn detect_changes(
        &self,
        old_schema: &SchemaInfo,
        new_schema: &SchemaInfo,
    ) -> Vec<SchemaChange> {
        let mut changes = Vec::new();

        // Build table maps for easy lookup
        let old_tables: HashMap<_, _> = old_schema
            .tables
            .iter()
            .map(|t| (t.name.clone(), t))
            .collect();
        let new_tables: HashMap<_, _> = new_schema
            .tables
            .iter()
            .map(|t| (t.name.clone(), t))
            .collect();

        // Detect new tables
        for (name, _) in &new_tables {
            if !old_tables.contains_key(name) {
                changes.push(SchemaChange::AddTable {
                    name: name.clone(),
                });
            }
        }

        // Detect dropped tables
        for (name, _) in &old_tables {
            if !new_tables.contains_key(name) {
                changes.push(SchemaChange::DropTable {
                    name: name.clone(),
                });
            }
        }

        // Detect column changes in existing tables
        for (name, new_table) in &new_tables {
            if let Some(old_table) = old_tables.get(name) {
                let column_changes = self.detect_column_changes(old_table, new_table);
                changes.extend(column_changes);
            }
        }

        changes
    }

    /// Detect column changes within a table
    fn detect_column_changes(&self, old_table: &TableInfo, new_table: &TableInfo) -> Vec<SchemaChange> {
        let mut changes = Vec::new();

        let old_columns: HashMap<_, _> = old_table
            .columns
            .iter()
            .map(|c| (c.name.clone(), c))
            .collect();
        let new_columns: HashMap<_, _> = new_table
            .columns
            .iter()
            .map(|c| (c.name.clone(), c))
            .collect();

        // Detect new columns
        for (col_name, col_info) in &new_columns {
            if !old_columns.contains_key(col_name) {
                changes.push(SchemaChange::AddColumn {
                    table: new_table.name.clone(),
                    column: (*col_info).clone(),
                });
            }
        }

        // Detect dropped columns
        for (col_name, _) in &old_columns {
            if !new_columns.contains_key(col_name) {
                changes.push(SchemaChange::DropColumn {
                    table: new_table.name.clone(),
                    column: col_name.clone(),
                });
            }
        }

        // Detect modified columns
        for (col_name, new_col) in &new_columns {
            if let Some(old_col) = old_columns.get(col_name) {
                if old_col.data_type != new_col.data_type
                    || old_col.nullable != new_col.nullable
                    || old_col.default_value != new_col.default_value
                {
                    changes.push(SchemaChange::ModifyColumn {
                        table: new_table.name.clone(),
                        old_column: (*old_col).clone(),
                        new_column: (*new_col).clone(),
                    });
                }
            }
        }

        changes
    }

    /// Update GNN with schema changes
    pub async fn update_gnn(&self, db_name: &str, changes: &[SchemaChange]) -> Result<(), String> {
        // TODO: Integrate with GNN engine
        // For each table:
        // - Add table node to GNN
        // - Add column nodes
        // - Add foreign key edges
        // - Track dependencies between tables
        
        for change in changes {
            match change {
                SchemaChange::AddTable { name } => {
                    println!("GNN: Add table node: {}", name);
                    // gnn.add_node(NodeType::DatabaseTable, name, metadata);
                }
                SchemaChange::AddForeignKey { table, column, referenced_table } => {
                    println!("GNN: Add foreign key edge: {}.{} -> {}", table, column, referenced_table);
                    // gnn.add_edge(table, referenced_table, EdgeType::ForeignKey);
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Track schema version
    pub async fn track_version(
        &self,
        db_name: &str,
        schema: SchemaInfo,
        changes: Vec<SchemaChange>,
    ) -> Result<(), String> {
        let version = SchemaVersion {
            version: schema.version.clone(),
            timestamp: chrono::Utc::now(),
            changes,
            schema,
        };

        let mut history = self.schema_history.write().await;
        history
            .entry(db_name.to_string())
            .or_insert_with(Vec::new)
            .push(version);

        Ok(())
    }

    /// Get schema history for a database
    pub async fn get_history(&self, db_name: &str) -> Vec<SchemaVersion> {
        let history = self.schema_history.read().await;
        history.get(db_name).cloned().unwrap_or_default()
    }

    /// Get latest schema version
    pub async fn get_latest_version(&self, db_name: &str) -> Option<SchemaVersion> {
        let history = self.schema_history.read().await;
        history.get(db_name)?.last().cloned()
    }

    /// Analyze impact of schema changes on code
    pub async fn analyze_impact(&self, db_name: &str, changes: &[SchemaChange]) -> Result<Vec<String>, String> {
        let mut impacted_files = Vec::new();

        // TODO: Integrate with GNN to find code that uses affected tables/columns
        // For each change:
        // - Query GNN for code references to table/column
        // - Find all functions/classes that query this table
        // - Return list of files that need to be updated

        for change in changes {
            match change {
                SchemaChange::DropColumn { table, column } => {
                    println!("Analyzing impact of dropping {}.{}", table, column);
                    // Query GNN: find all SQL queries that reference this column
                    // impacted_files.extend(gnn.find_references(table, column));
                }
                SchemaChange::ModifyColumn { table, old_column, new_column } => {
                    if old_column.data_type != new_column.data_type {
                        println!("Type change detected for {}.{}", table, old_column.name);
                        // Query GNN: find all code that reads/writes this column
                    }
                }
                _ => {}
            }
        }

        Ok(impacted_files)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::database::connection_manager::ForeignKeyInfo;

    fn create_test_schema(tables: Vec<TableInfo>) -> SchemaInfo {
        SchemaInfo {
            tables,
            version: "1.0".to_string(),
            last_updated: chrono::Utc::now(),
        }
    }

    fn create_test_table(name: &str, columns: Vec<ColumnInfo>) -> TableInfo {
        TableInfo {
            name: name.to_string(),
            columns,
            primary_keys: vec![],
            foreign_keys: vec![],
        }
    }

    fn create_test_column(name: &str, data_type: &str, nullable: bool) -> ColumnInfo {
        ColumnInfo {
            name: name.to_string(),
            data_type: data_type.to_string(),
            nullable,
            default_value: None,
        }
    }

    #[test]
    fn test_detect_new_table() {
        let old_schema = create_test_schema(vec![]);
        let new_schema = create_test_schema(vec![
            create_test_table("users", vec![]),
        ]);

        let db_manager = Arc::new(DatabaseManager::new().unwrap());
        let tracker = SchemaTracker::new(db_manager);
        let changes = tracker.detect_changes(&old_schema, &new_schema);

        assert_eq!(changes.len(), 1);
        match &changes[0] {
            SchemaChange::AddTable { name } => assert_eq!(name, "users"),
            _ => panic!("Expected AddTable change"),
        }
    }

    #[test]
    fn test_detect_dropped_table() {
        let old_schema = create_test_schema(vec![
            create_test_table("users", vec![]),
        ]);
        let new_schema = create_test_schema(vec![]);

        let db_manager = Arc::new(DatabaseManager::new().unwrap());
        let tracker = SchemaTracker::new(db_manager);
        let changes = tracker.detect_changes(&old_schema, &new_schema);

        assert_eq!(changes.len(), 1);
        match &changes[0] {
            SchemaChange::DropTable { name } => assert_eq!(name, "users"),
            _ => panic!("Expected DropTable change"),
        }
    }

    #[test]
    fn test_detect_new_column() {
        let old_schema = create_test_schema(vec![
            create_test_table("users", vec![
                create_test_column("id", "INTEGER", false),
            ]),
        ]);
        let new_schema = create_test_schema(vec![
            create_test_table("users", vec![
                create_test_column("id", "INTEGER", false),
                create_test_column("email", "VARCHAR", false),
            ]),
        ]);

        let db_manager = Arc::new(DatabaseManager::new().unwrap());
        let tracker = SchemaTracker::new(db_manager);
        let changes = tracker.detect_changes(&old_schema, &new_schema);

        assert_eq!(changes.len(), 1);
        match &changes[0] {
            SchemaChange::AddColumn { table, column } => {
                assert_eq!(table, "users");
                assert_eq!(column.name, "email");
            }
            _ => panic!("Expected AddColumn change"),
        }
    }

    #[test]
    fn test_detect_modified_column() {
        let old_schema = create_test_schema(vec![
            create_test_table("users", vec![
                create_test_column("age", "INTEGER", false),
            ]),
        ]);
        let new_schema = create_test_schema(vec![
            create_test_table("users", vec![
                create_test_column("age", "VARCHAR", false),
            ]),
        ]);

        let db_manager = Arc::new(DatabaseManager::new().unwrap());
        let tracker = SchemaTracker::new(db_manager);
        let changes = tracker.detect_changes(&old_schema, &new_schema);

        assert_eq!(changes.len(), 1);
        match &changes[0] {
            SchemaChange::ModifyColumn { table, old_column, new_column } => {
                assert_eq!(table, "users");
                assert_eq!(old_column.data_type, "INTEGER");
                assert_eq!(new_column.data_type, "VARCHAR");
            }
            _ => panic!("Expected ModifyColumn change"),
        }
    }
}
