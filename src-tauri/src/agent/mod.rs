// File: src-tauri/src/agent/mod.rs
// Purpose: Agentic validation pipeline for autonomous code generation
// Last Updated: November 21, 2025
//
// This module implements the fully agentic capabilities of Yantra:
// - Agent state machine with crash recovery
// - Confidence scoring for auto-retry decisions
// - Known issues database for learning from failures
// - Auto-retry logic with intelligent escalation
// - Dependency validation via GNN
//
// The agent autonomously validates generated code through multiple stages:
// 1. Dependency validation (GNN-based)
// 2. Unit test execution
// 3. Integration test execution  
// 4. Security scanning
// 5. Browser validation
//
// If validation fails, the agent:
// - Analyzes the failure
// - Checks known issues database for similar patterns
// - Applies known fix if confidence >0.8
// - Retries validation (up to 3 attempts)
// - Escalates to human if confidence <0.5

pub mod state;
pub mod confidence;
pub mod validation;
pub mod orchestrator;
pub mod terminal;

// Re-export key types
pub use state::{AgentPhase, AgentState, AgentStateManager};
pub use confidence::{ConfidenceScore, ConfidenceFactor};
pub use validation::{ValidationResult, ValidationError, validate_dependencies};
pub use orchestrator::{orchestrate_code_generation, OrchestrationResult};
pub use terminal::{TerminalExecutor, TerminalOutput, ExecutionResult};
