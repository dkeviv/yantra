// File: src-tauri/src/agent/mod.rs
// Purpose: Agentic validation pipeline for autonomous code generation
// Last Updated: November 22, 2025
//
// This module implements the fully agentic capabilities of Yantra:
// - Agent state machine with crash recovery
// - Confidence scoring for auto-retry decisions
// - Known issues database for learning from failures
// - Auto-retry logic with intelligent escalation
// - Dependency validation via GNN
// - Terminal command execution with security
// - Package building and deployment automation
// - Production monitoring and self-healing
//
// The agent autonomously validates generated code through multiple stages:
// 1. Dependency validation (GNN-based)
// 2. Unit test execution
// 3. Integration test execution  
// 4. Security scanning
// 5. Browser validation
// 6. Runtime execution validation
// 7. Package building
// 8. Deployment to staging/production
// 9. Production monitoring and self-healing
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
pub mod project_orchestrator;
pub mod project_initializer;
pub mod terminal;
pub mod dependencies;
pub mod execution;
pub mod packaging;
pub mod deployment;
pub mod monitoring;
pub mod task_queue;
pub mod database;
pub mod http_client;
pub mod file_editor;
pub mod file_ops;
pub mod command_classifier;
pub mod status_emitter;
pub mod api_manager;

// Re-export key types (many not yet used but part of public API)
#[allow(unused_imports)]
pub use state::{AgentPhase, AgentState, AgentStateManager};
#[allow(unused_imports)]
pub use confidence::{ConfidenceScore, ConfidenceFactor};
#[allow(unused_imports)]
pub use task_queue::{Task, TaskQueue, TaskStatus, TaskPriority, TaskStats};
#[allow(unused_imports)]
pub use validation::{ValidationResult, ValidationError, validate_dependencies};
#[allow(unused_imports)]
pub use orchestrator::{orchestrate_code_generation, OrchestrationResult};
#[allow(unused_imports)]
pub use project_orchestrator::{ProjectOrchestrator, ProjectPlan, ProjectResult, ProjectTemplate, FileToGenerate, TestSummary};
#[allow(unused_imports)]
pub use project_initializer::{ProjectInitializer, InitializationResult, ApprovalResult, ArchitectureSource, ProjectAnalysisReport};
#[allow(unused_imports)]
pub use terminal::{TerminalExecutor, TerminalOutput, ExecutionResult};
#[allow(unused_imports)]
pub use dependencies::{DependencyInstaller, ProjectType, InstallationResult};
#[allow(unused_imports)]
pub use execution::{ScriptExecutor, ScriptExecutionResult, ErrorType};
#[allow(unused_imports)]
pub use packaging::{PackageBuilder, PackageType, PackageConfig, PackageBuildResult};
#[allow(unused_imports)]
pub use deployment::{DeploymentManager, DeploymentTarget, DeploymentConfig, DeploymentResult, Environment};
#[allow(unused_imports)]
pub use monitoring::{MonitoringManager, Alert, Severity, PerformanceMetrics, HealingAction};
