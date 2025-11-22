// File: src-tauri/src/agent/orchestrator.rs
// Purpose: Auto-retry orchestration loop for autonomous code generation
// Last Updated: December 21, 2025
//
// This orchestrator implements the core agentic loop:
// 1. Create agent session
// 2. Assemble hierarchical context from GNN
// 3. Generate code with LLM
// 4. Validate dependencies with GNN
// 5. If validation fails and should_retry():
//    - Analyze errors
//    - Apply known fixes (if available)
//    - Update confidence score
//    - Repeat (up to 3 attempts)
// 6. If should_escalate(): return errors for human review
// 7. Otherwise: commit and return success

use super::confidence::ConfidenceScore;
use super::state::{AgentPhase, AgentState, AgentStateManager};
use super::validation::{validate_dependencies, ValidationResult};
use crate::gnn::GNNEngine;
use crate::llm::context::{assemble_hierarchical_context, HierarchicalContext};
use crate::llm::orchestrator::LLMOrchestrator;
use crate::llm::{CodeGenerationRequest, CodeGenerationResponse};
use serde::{Deserialize, Serialize};

/// Result of code generation orchestration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrchestrationResult {
    /// Code generation succeeded, ready to commit
    Success {
        code: String,
        language: String,
        confidence: f32,
        attempt_count: u32,
        session_id: String,
    },
    /// Code generation failed after retries, needs human intervention
    Escalated {
        errors: Vec<String>,
        last_code: Option<String>,
        confidence: f32,
        attempt_count: u32,
        session_id: String,
    },
    /// Critical error (GNN unavailable, LLM error, etc.)
    Error {
        message: String,
        session_id: Option<String>,
    },
}

/// Orchestrate code generation with auto-retry
/// 
/// This is the main entry point for autonomous code generation.
/// It manages the entire lifecycle from context assembly to validated code.
/// 
/// # Arguments
/// * `gnn` - GNN engine for context and validation
/// * `llm` - LLM orchestrator for code generation
/// * `state_manager` - Agent state manager for persistence
/// * `user_task` - User's natural language intent
/// * `file_path` - Target file path for generated code
/// * `target_node` - Optional specific function/class to modify
/// 
/// # Returns
/// OrchestrationResult with success, escalation, or error
pub async fn orchestrate_code_generation(
    gnn: &GNNEngine,
    llm: &LLMOrchestrator,
    state_manager: &AgentStateManager,
    user_task: String,
    file_path: String,
    target_node: Option<String>,
) -> OrchestrationResult {
    // Create agent state for this session
    let mut state = AgentState::new(user_task.clone());
    let session_id = state.session_id.clone();

    // Save initial state
    if let Err(e) = state_manager.save_state(&state) {
        return OrchestrationResult::Error {
            message: format!("Failed to save initial state: {}", e),
            session_id: Some(session_id),
        };
    }

    // Phase 1: Context Assembly
    state.transition_to(AgentPhase::ContextAssembly);
    if let Err(e) = state_manager.save_state(&state) {
        return OrchestrationResult::Error {
            message: format!("Failed to save state: {}", e),
            session_id: Some(session_id),
        };
    }

    let context = match assemble_hierarchical_context(
        gnn,
        target_node.as_deref(),
        Some(&file_path),
        160_000, // Claude's 160K token limit
    ) {
        Ok(ctx) => ctx,
        Err(e) => {
            state.transition_to(AgentPhase::Failed);
            state.add_error(format!("Context assembly failed: {}", e));
            let _ = state_manager.save_state(&state);
            return OrchestrationResult::Error {
                message: format!("Context assembly failed: {}", e),
                session_id: Some(session_id),
            };
        }
    };

    // Auto-retry loop (up to 3 attempts)
    loop {
        state.increment_attempt();

        // Phase 2: Code Generation
        state.transition_to(AgentPhase::CodeGeneration);
        if let Err(e) = state_manager.save_state(&state) {
            return OrchestrationResult::Error {
                message: format!("Failed to save state: {}", e),
                session_id: Some(session_id),
            };
        }

        let code_result = generate_code_with_context(llm, &user_task, &context, &file_path).await;

        let response = match code_result {
            Ok(resp) => resp,
            Err(e) => {
                state.add_error(format!("Code generation failed: {}", e));
                
                // Check if should retry or escalate
                if state.should_retry() {
                    continue; // Retry with same context
                } else {
                    state.transition_to(AgentPhase::Failed);
                    let _ = state_manager.save_state(&state);
                    
                    return OrchestrationResult::Escalated {
                        errors: state.validation_errors.clone(),
                        last_code: state.generated_code.clone(),
                        confidence: state.confidence_score,
                        attempt_count: state.attempt_count,
                        session_id,
                    };
                }
            }
        };

        state.set_generated_code(response.code.clone());
        let _ = state_manager.save_state(&state);

        // Phase 3: Dependency Validation
        state.transition_to(AgentPhase::DependencyValidation);
        if let Err(e) = state_manager.save_state(&state) {
            return OrchestrationResult::Error {
                message: format!("Failed to save state: {}", e),
                session_id: Some(session_id),
            };
        }

        let validation_result = validate_dependencies(gnn, &response.code, &file_path);

        match validation_result {
            Ok(ValidationResult::Success) => {
                // Validation passed! Calculate final confidence and return success
                let mut confidence = ConfidenceScore::new();
                
                // Set LLM confidence (if available from response metadata)
                confidence.set_llm_confidence(0.9); // Default high confidence for successful generation
                
                // No validation errors, so test pass rate is 100% (no tests run yet)
                confidence.set_test_pass_rate(0, 0); // Will be updated after test execution
                
                // Calculate code complexity (simple heuristic: lines of code)
                let loc = response.code.lines().count();
                let complexity = (loc / 10).min(10); // 1 LOC = complexity 1, capped at 10
                confidence.set_code_complexity(complexity);
                
                // Dependency impact: count files affected (just the target file for now)
                confidence.set_dependency_impact(1);
                
                state.set_confidence(confidence.overall());
                state.transition_to(AgentPhase::Complete);
                let _ = state_manager.save_state(&state);

                return OrchestrationResult::Success {
                    code: response.code,
                    language: response.language,
                    confidence: confidence.overall(),
                    attempt_count: state.attempt_count,
                    session_id,
                };
            }
            Ok(ValidationResult::Failed(errors)) => {
                // Validation failed - analyze and decide whether to retry
                for error in &errors {
                    state.add_error(error.message.clone());
                }

                // Calculate confidence based on error severity
                let mut confidence = ConfidenceScore::new();
                confidence.set_llm_confidence(0.6); // Lower confidence due to validation failure
                
                // Calculate test pass rate based on validation errors
                // Treat validation errors as "tests" that failed
                let error_count = errors.len();
                let pass_rate = if error_count > 5 { 0.0 } else { 0.5 };
                confidence.set_test_pass_rate(if pass_rate > 0.0 { 1 } else { 0 }, 2);
                
                // Check if errors match known patterns (placeholder - will be enhanced)
                confidence.set_known_failure_match(0.0); // No known patterns yet
                
                state.set_confidence(confidence.overall());
                let _ = state_manager.save_state(&state);

                // Check if should retry
                if state.should_retry() {
                    // Transition to fixing phase
                    state.transition_to(AgentPhase::FixingIssues);
                    let _ = state_manager.save_state(&state);
                    
                    // TODO: In future, analyze errors and apply known fixes here
                    // For now, just retry with the same context
                    continue;
                } else {
                    // Should escalate to human
                    state.transition_to(AgentPhase::Failed);
                    let _ = state_manager.save_state(&state);

                    return OrchestrationResult::Escalated {
                        errors: state.validation_errors.clone(),
                        last_code: Some(response.code),
                        confidence: state.confidence_score,
                        attempt_count: state.attempt_count,
                        session_id,
                    };
                }
            }
            Err(e) => {
                // Validation error (not a validation failure, but a system error)
                state.add_error(format!("Validation error: {}", e));
                state.transition_to(AgentPhase::Failed);
                let _ = state_manager.save_state(&state);

                return OrchestrationResult::Error {
                    message: format!("Validation error: {}", e),
                    session_id: Some(session_id),
                };
            }
        }
    }
}

/// Generate code with LLM using hierarchical context
async fn generate_code_with_context(
    llm: &LLMOrchestrator,
    user_task: &str,
    context: &HierarchicalContext,
    file_path: &str,
) -> Result<CodeGenerationResponse, String> {
    // Build code generation request
    let request = CodeGenerationRequest {
        intent: user_task.to_string(),
        file_path: Some(file_path.to_string()),
        context: vec![context.to_string()],
        dependencies: vec![], // Dependencies are in the context already
    };

    // Generate code with LLM
    llm.generate_code(&request)
        .await
        .map_err(|e| e.message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::LLMConfig;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_orchestration_error_on_empty_gnn() {
        // Create empty GNN (no nodes)
        let temp_gnn = NamedTempFile::new().unwrap();
        let gnn = GNNEngine::new(temp_gnn.path()).unwrap();

        // Create state manager
        let temp_state = NamedTempFile::new().unwrap();
        let state_manager = AgentStateManager::new(temp_state.path().to_str().unwrap().to_string()).unwrap();

        // Create LLM orchestrator (no API keys - will fail on actual call)
        let llm_config = LLMConfig::default();
        let llm = LLMOrchestrator::new(llm_config);

        // Try to generate code
        let result = orchestrate_code_generation(
            &gnn,
            &llm,
            &state_manager,
            "Create a function".to_string(),
            "test.py".to_string(),
            None,
        )
        .await;

        // Should succeed (context assembly works even with empty GNN)
        // But will fail on LLM call (no API key)
        match result {
            OrchestrationResult::Error { .. } => {} // Expected - no LLM API key
            OrchestrationResult::Escalated { .. } => {} // Also acceptable
            OrchestrationResult::Success { .. } => panic!("Should not succeed without LLM API key"),
        }
    }

    #[test]
    fn test_orchestration_result_serialization() {
        let success = OrchestrationResult::Success {
            code: "def foo(): pass".to_string(),
            language: "python".to_string(),
            confidence: 0.95,
            attempt_count: 1,
            session_id: "test-session".to_string(),
        };

        let json = serde_json::to_string(&success).unwrap();
        let deserialized: OrchestrationResult = serde_json::from_str(&json).unwrap();

        match deserialized {
            OrchestrationResult::Success { code, confidence, .. } => {
                assert_eq!(code, "def foo(): pass");
                assert_eq!(confidence, 0.95);
            }
            _ => panic!("Deserialization failed"),
        }
    }
}
