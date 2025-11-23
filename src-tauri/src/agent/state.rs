// File: src-tauri/src/agent/state.rs
// Purpose: Agent state machine with SQLite persistence for crash recovery
// Last Updated: November 21, 2025

use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension, Result as SqliteResult};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Agent execution phases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentPhase {
    /// Assembling context from GNN
    ContextAssembly,
    /// Generating code with LLM
    CodeGeneration,
    /// Validating dependencies via GNN
    DependencyValidation,
    /// Setting up execution environment (venv, env vars)
    EnvironmentSetup,
    /// Installing missing dependencies
    DependencyInstallation,
    /// Executing generated script
    ScriptExecution,
    /// Validating runtime behavior
    RuntimeValidation,
    /// Profiling performance
    PerformanceProfiling,
    /// Running unit tests
    UnitTesting,
    /// Running integration tests
    IntegrationTesting,
    /// Scanning for security vulnerabilities
    SecurityScanning,
    /// Validating in browser runtime
    BrowserValidation,
    /// Analyzing failures and applying fixes
    FixingIssues,
    /// Committing to git
    GitCommit,
    /// Successfully completed
    Complete,
    /// Failed (human intervention needed)
    Failed,
}

impl AgentPhase {
    /// Convert to string for database storage
    pub fn to_string(&self) -> String {
        match self {
            AgentPhase::ContextAssembly => "ContextAssembly".to_string(),
            AgentPhase::CodeGeneration => "CodeGeneration".to_string(),
            AgentPhase::DependencyValidation => "DependencyValidation".to_string(),
            AgentPhase::EnvironmentSetup => "EnvironmentSetup".to_string(),
            AgentPhase::DependencyInstallation => "DependencyInstallation".to_string(),
            AgentPhase::ScriptExecution => "ScriptExecution".to_string(),
            AgentPhase::RuntimeValidation => "RuntimeValidation".to_string(),
            AgentPhase::PerformanceProfiling => "PerformanceProfiling".to_string(),
            AgentPhase::UnitTesting => "UnitTesting".to_string(),
            AgentPhase::IntegrationTesting => "IntegrationTesting".to_string(),
            AgentPhase::SecurityScanning => "SecurityScanning".to_string(),
            AgentPhase::BrowserValidation => "BrowserValidation".to_string(),
            AgentPhase::FixingIssues => "FixingIssues".to_string(),
            AgentPhase::GitCommit => "GitCommit".to_string(),
            AgentPhase::Complete => "Complete".to_string(),
            AgentPhase::Failed => "Failed".to_string(),
        }
    }

    /// Parse from string
    pub fn from_string(s: &str) -> Option<Self> {
        match s {
            "ContextAssembly" => Some(AgentPhase::ContextAssembly),
            "CodeGeneration" => Some(AgentPhase::CodeGeneration),
            "DependencyValidation" => Some(AgentPhase::DependencyValidation),
            "EnvironmentSetup" => Some(AgentPhase::EnvironmentSetup),
            "DependencyInstallation" => Some(AgentPhase::DependencyInstallation),
            "ScriptExecution" => Some(AgentPhase::ScriptExecution),
            "RuntimeValidation" => Some(AgentPhase::RuntimeValidation),
            "PerformanceProfiling" => Some(AgentPhase::PerformanceProfiling),
            "UnitTesting" => Some(AgentPhase::UnitTesting),
            "IntegrationTesting" => Some(AgentPhase::IntegrationTesting),
            "SecurityScanning" => Some(AgentPhase::SecurityScanning),
            "BrowserValidation" => Some(AgentPhase::BrowserValidation),
            "FixingIssues" => Some(AgentPhase::FixingIssues),
            "GitCommit" => Some(AgentPhase::GitCommit),
            "Complete" => Some(AgentPhase::Complete),
            "Failed" => Some(AgentPhase::Failed),
            _ => None,
        }
    }
}

/// Agent state with persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    /// Unique session ID
    pub session_id: String,
    /// Current execution phase
    pub current_phase: AgentPhase,
    /// Number of retry attempts
    pub attempt_count: u32,
    /// Overall confidence score (0.0-1.0)
    pub confidence_score: f32,
    /// User's original task/intent
    pub user_task: String,
    /// Generated code (if any)
    pub generated_code: Option<String>,
    /// Validation errors (if any)
    pub validation_errors: Vec<String>,
    /// Session creation time
    pub created_at: DateTime<Utc>,
    /// Last update time
    pub updated_at: DateTime<Utc>,
}

impl AgentState {
    /// Create new agent state for a task
    pub fn new(user_task: String) -> Self {
        let now = Utc::now();
        Self {
            session_id: Uuid::new_v4().to_string(),
            current_phase: AgentPhase::ContextAssembly,
            attempt_count: 0,
            confidence_score: 1.0,
            user_task,
            generated_code: None,
            validation_errors: Vec::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Transition to next phase
    pub fn transition_to(&mut self, phase: AgentPhase) {
        self.current_phase = phase;
        self.updated_at = Utc::now();
    }

    /// Increment retry attempt
    pub fn increment_attempt(&mut self) {
        self.attempt_count += 1;
        self.updated_at = Utc::now();
    }

    /// Update confidence score
    pub fn set_confidence(&mut self, score: f32) {
        self.confidence_score = score.clamp(0.0, 1.0);
        self.updated_at = Utc::now();
    }

    /// Add validation error
    pub fn add_error(&mut self, error: String) {
        self.validation_errors.push(error);
        self.updated_at = Utc::now();
    }

    /// Set generated code
    pub fn set_generated_code(&mut self, code: String) {
        self.generated_code = Some(code);
        self.updated_at = Utc::now();
    }

    /// Check if should retry based on attempts and confidence
    pub fn should_retry(&self) -> bool {
        self.attempt_count < 3 && self.confidence_score >= 0.5
    }

    /// Check if should escalate to human
    pub fn should_escalate(&self) -> bool {
        self.confidence_score < 0.5 || self.attempt_count >= 3
    }
}

/// Manager for agent state persistence
pub struct AgentStateManager {
    db_path: String,
}

impl AgentStateManager {
    /// Create new state manager
    pub fn new(db_path: String) -> Result<Self, String> {
        let manager = Self { db_path };
        manager.init_database()?;
        Ok(manager)
    }

    /// Initialize database schema
    fn init_database(&self) -> Result<(), String> {
        let conn = Connection::open(&self.db_path)
            .map_err(|e| format!("Failed to open database: {}", e))?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS agent_states (
                session_id TEXT PRIMARY KEY,
                current_phase TEXT NOT NULL,
                attempt_count INTEGER NOT NULL,
                confidence_score REAL NOT NULL,
                user_task TEXT NOT NULL,
                generated_code TEXT,
                validation_errors TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )",
            [],
        )
        .map_err(|e| format!("Failed to create table: {}", e))?;

        Ok(())
    }

    /// Save agent state to database
    pub fn save_state(&self, state: &AgentState) -> Result<(), String> {
        let conn = Connection::open(&self.db_path)
            .map_err(|e| format!("Failed to open database: {}", e))?;

        let errors_json = serde_json::to_string(&state.validation_errors)
            .map_err(|e| format!("Failed to serialize errors: {}", e))?;

        conn.execute(
            "INSERT OR REPLACE INTO agent_states 
             (session_id, current_phase, attempt_count, confidence_score, 
              user_task, generated_code, validation_errors, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                &state.session_id,
                state.current_phase.to_string(),
                state.attempt_count,
                state.confidence_score,
                &state.user_task,
                &state.generated_code,
                errors_json,
                state.created_at.to_rfc3339(),
                state.updated_at.to_rfc3339(),
            ],
        )
        .map_err(|e| format!("Failed to save state: {}", e))?;

        Ok(())
    }

    /// Load agent state from database
    pub fn load_state(&self, session_id: &str) -> Result<Option<AgentState>, String> {
        let conn = Connection::open(&self.db_path)
            .map_err(|e| format!("Failed to open database: {}", e))?;

        let mut stmt = conn
            .prepare(
                "SELECT session_id, current_phase, attempt_count, confidence_score,
                        user_task, generated_code, validation_errors, created_at, updated_at
                 FROM agent_states WHERE session_id = ?1",
            )
            .map_err(|e| format!("Failed to prepare query: {}", e))?;

        let result = stmt
            .query_row(params![session_id], |row| {
                let phase_str: String = row.get(1)?;
                let phase = AgentPhase::from_string(&phase_str)
                    .ok_or_else(|| rusqlite::Error::InvalidQuery)?;

                let errors_json: String = row.get(6)?;
                let validation_errors: Vec<String> = serde_json::from_str(&errors_json)
                    .map_err(|_| rusqlite::Error::InvalidQuery)?;

                let created_str: String = row.get(7)?;
                let updated_str: String = row.get(8)?;

                Ok(AgentState {
                    session_id: row.get(0)?,
                    current_phase: phase,
                    attempt_count: row.get(2)?,
                    confidence_score: row.get(3)?,
                    user_task: row.get(4)?,
                    generated_code: row.get(5)?,
                    validation_errors,
                    created_at: DateTime::parse_from_rfc3339(&created_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .map_err(|_| rusqlite::Error::InvalidQuery)?,
                    updated_at: DateTime::parse_from_rfc3339(&updated_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .map_err(|_| rusqlite::Error::InvalidQuery)?,
                })
            })
            .optional()
            .map_err(|e| format!("Failed to load state: {}", e))?;

        Ok(result)
    }

    /// Delete agent state
    pub fn delete_state(&self, session_id: &str) -> Result<(), String> {
        let conn = Connection::open(&self.db_path)
            .map_err(|e| format!("Failed to open database: {}", e))?;

        conn.execute("DELETE FROM agent_states WHERE session_id = ?1", params![session_id])
            .map_err(|e| format!("Failed to delete state: {}", e))?;

        Ok(())
    }

    /// Get all active sessions
    pub fn get_active_sessions(&self) -> Result<Vec<AgentState>, String> {
        let conn = Connection::open(&self.db_path)
            .map_err(|e| format!("Failed to open database: {}", e))?;

        let mut stmt = conn
            .prepare(
                "SELECT session_id, current_phase, attempt_count, confidence_score,
                        user_task, generated_code, validation_errors, created_at, updated_at
                 FROM agent_states 
                 WHERE current_phase NOT IN ('Complete', 'Failed')
                 ORDER BY updated_at DESC",
            )
            .map_err(|e| format!("Failed to prepare query: {}", e))?;

        let rows = stmt
            .query_map([], |row| {
                let phase_str: String = row.get(1)?;
                let phase = AgentPhase::from_string(&phase_str)
                    .ok_or_else(|| rusqlite::Error::InvalidQuery)?;

                let errors_json: String = row.get(6)?;
                let validation_errors: Vec<String> = serde_json::from_str(&errors_json)
                    .map_err(|_| rusqlite::Error::InvalidQuery)?;

                let created_str: String = row.get(7)?;
                let updated_str: String = row.get(8)?;

                Ok(AgentState {
                    session_id: row.get(0)?,
                    current_phase: phase,
                    attempt_count: row.get(2)?,
                    confidence_score: row.get(3)?,
                    user_task: row.get(4)?,
                    generated_code: row.get(5)?,
                    validation_errors,
                    created_at: DateTime::parse_from_rfc3339(&created_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .map_err(|_| rusqlite::Error::InvalidQuery)?,
                    updated_at: DateTime::parse_from_rfc3339(&updated_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .map_err(|_| rusqlite::Error::InvalidQuery)?,
                })
            })
            .map_err(|e| format!("Failed to query active sessions: {}", e))?;

        rows.collect::<SqliteResult<Vec<_>>>()
            .map_err(|e| format!("Failed to collect sessions: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_agent_phase_serialization() {
        let phase = AgentPhase::CodeGeneration;
        let serialized = phase.to_string();
        assert_eq!(serialized, "CodeGeneration");

        let deserialized = AgentPhase::from_string(&serialized);
        assert_eq!(deserialized, Some(phase));
    }

    #[test]
    fn test_agent_state_creation() {
        let state = AgentState::new("Create authentication API".to_string());
        assert_eq!(state.current_phase, AgentPhase::ContextAssembly);
        assert_eq!(state.attempt_count, 0);
        assert_eq!(state.confidence_score, 1.0);
        assert!(state.generated_code.is_none());
    }

    #[test]
    fn test_agent_state_transitions() {
        let mut state = AgentState::new("Test task".to_string());
        
        state.transition_to(AgentPhase::CodeGeneration);
        assert_eq!(state.current_phase, AgentPhase::CodeGeneration);

        state.increment_attempt();
        assert_eq!(state.attempt_count, 1);

        state.set_confidence(0.75);
        assert_eq!(state.confidence_score, 0.75);
    }

    #[test]
    fn test_retry_logic() {
        let mut state = AgentState::new("Test task".to_string());
        
        // Should retry with high confidence
        assert!(state.should_retry());
        assert!(!state.should_escalate());

        // After 3 attempts, should not retry
        state.attempt_count = 3;
        assert!(!state.should_retry());
        assert!(state.should_escalate());

        // Low confidence should escalate
        state.attempt_count = 1;
        state.set_confidence(0.3);
        assert!(!state.should_retry());
        assert!(state.should_escalate());
    }

    #[test]
    fn test_state_persistence() {
        let temp_file = NamedTempFile::new().unwrap();
        let db_path = temp_file.path().to_str().unwrap().to_string();

        let manager = AgentStateManager::new(db_path).unwrap();

        let mut state = AgentState::new("Test persistence".to_string());
        state.transition_to(AgentPhase::CodeGeneration);
        state.set_confidence(0.85);
        state.add_error("Test error".to_string());

        // Save state
        manager.save_state(&state).unwrap();

        // Load state
        let loaded = manager.load_state(&state.session_id).unwrap();
        assert!(loaded.is_some());

        let loaded = loaded.unwrap();
        assert_eq!(loaded.session_id, state.session_id);
        assert_eq!(loaded.current_phase, AgentPhase::CodeGeneration);
        assert_eq!(loaded.confidence_score, 0.85);
        assert_eq!(loaded.validation_errors.len(), 1);
    }

    #[test]
    fn test_active_sessions() {
        let temp_file = NamedTempFile::new().unwrap();
        let db_path = temp_file.path().to_str().unwrap().to_string();

        let manager = AgentStateManager::new(db_path).unwrap();

        // Create multiple states
        let state1 = AgentState::new("Task 1".to_string());
        let mut state2 = AgentState::new("Task 2".to_string());
        state2.transition_to(AgentPhase::Complete);

        manager.save_state(&state1).unwrap();
        manager.save_state(&state2).unwrap();

        // Get active sessions (should only include state1)
        let active = manager.get_active_sessions().unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].user_task, "Task 1");
    }
}
