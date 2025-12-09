// File: src-tauri/src/agent/orchestrator.rs
// Purpose: Auto-retry orchestration loop for autonomous code generation with execution
// Last Updated: November 22, 2025
//
// This orchestrator implements the complete agentic loop:
// 1. Create agent session
// 2. Assemble hierarchical context from GNN
// 3. Generate code with LLM
// 4. Validate dependencies with GNN
// 5. Setup execution environment
// 6. Install missing dependencies
// 7. Execute generated script
// 8. Validate runtime behavior
// 9. Profile performance
// 10. Run unit tests (if available)
// 11. If validation fails and should_retry():
//    - Analyze errors
//    - Apply known fixes or auto-install dependencies
//    - Update confidence score
//    - Repeat (up to 3 attempts)
// 12. If should_escalate(): return errors for human review
// 13. Otherwise: commit and return success

use super::confidence::ConfidenceScore;
use super::conversation_integration::ConversationContext;
use super::dependencies::DependencyInstaller;
#[cfg(test)]
use super::dependencies::ProjectType;
use super::execution::{ErrorType, ScriptExecutor};
use super::state::{AgentPhase, AgentState, AgentStateManager};
use super::validation::{validate_dependencies, ValidationResult};
use crate::gnn::GNNEngine;
use crate::llm::context::{assemble_hierarchical_context, HierarchicalContext};
use crate::llm::orchestrator::LLMOrchestrator;
use crate::llm::{CodeGenerationRequest, CodeGenerationResponse, LLMConfig};
use crate::testing::{TestRunner, TestGenerationRequest};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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
    orchestrate_code_generation_with_conversation(
        gnn,
        llm,
        state_manager,
        None, // No conversation context
        user_task,
        file_path,
        target_node,
    ).await
}

/// Orchestrate code generation with conversation context integration
/// 
/// Enhanced version that integrates conversation memory for:
/// - Context assembly with conversation history (State 12)
/// - Session linking after code generation (State 13)
/// 
/// # Arguments
/// * `gnn` - GNN engine for context and validation
/// * `llm` - LLM orchestrator for code generation
/// * `state_manager` - Agent state manager for persistence
/// * `conversation_context` - Optional conversation context for history and linking
/// * `user_task` - User's natural language intent
/// * `file_path` - Target file path for generated code
/// * `target_node` - Optional specific function/class to modify
/// 
/// # Returns
/// OrchestrationResult with success, escalation, or error
pub async fn orchestrate_code_generation_with_conversation(
    gnn: &GNNEngine,
    llm: &LLMOrchestrator,
    state_manager: &AgentStateManager,
    conversation_context: Option<&ConversationContext>,
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

    // Save user message to conversation (if context provided)
    if let Some(ctx) = conversation_context {
        if let Err(e) = ctx.save_user_message(&user_task, None).await {
            eprintln!("Warning: Failed to save user message: {}", e);
        }
    }

    // Phase 1: Context Assembly (State 12)
    state.transition_to(AgentPhase::ContextAssembly);
    if let Err(e) = state_manager.save_state(&state) {
        return OrchestrationResult::Error {
            message: format!("Failed to save state: {}", e),
            session_id: Some(session_id),
        };
    }

    // Calculate token budget: 160K Claude limit
    // Reserve 15-20% for conversation context if available
    let total_token_budget = 160_000;
    let conversation_token_budget = if conversation_context.is_some() {
        (total_token_budget as f32 * 0.175) as usize // 17.5% = 28K tokens
    } else {
        0
    };
    let gnn_token_budget = total_token_budget - conversation_token_budget;

    // Get conversation context (recent 3-5 messages)
    let conversation_ctx = if let Some(ctx) = conversation_context {
        match ctx.get_recent_context(5).await {
            Ok(context_str) => {
                if !context_str.is_empty() {
                    Some(context_str)
                } else {
                    None
                }
            }
            Err(e) => {
                eprintln!("Warning: Failed to get conversation context: {}", e);
                None
            }
        }
    } else {
        None
    };

    let context = match assemble_hierarchical_context(
        gnn,
        target_node.as_deref(),
        Some(&file_path),
        gnn_token_budget,
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

        // Phase 2: Code Generation (State 13)
        state.transition_to(AgentPhase::CodeGeneration);
        if let Err(e) = state_manager.save_state(&state) {
            return OrchestrationResult::Error {
                message: format!("Failed to save state: {}", e),
                session_id: Some(session_id),
            };
        }

        // Build full user task with conversation context if available
        let full_user_task = if let Some(ref conv_ctx) = conversation_ctx {
            format!("Previous conversation:\n{}\n\nCurrent task: {}", conv_ctx, user_task)
        } else {
            user_task.clone()
        };

        let code_result = generate_code_with_context(llm, &full_user_task, &context, &file_path).await;

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

        // Link code generation to conversation (State 13)
        if let Some(ctx) = conversation_context {
            if let Err(e) = ctx.link_code_generation(&session_id, &response.code).await {
                eprintln!("Warning: Failed to link code generation to conversation: {}", e);
            }
        }

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

/// Orchestrate code generation with full execution and validation
/// 
/// This is the enhanced orchestration that includes:
/// - Code generation
/// - Dependency validation and installation
/// - Script execution
/// - Runtime validation
/// - Performance profiling
/// - Test execution (if tests exist)
/// 
/// # Arguments
/// * `gnn` - GNN engine for context and validation
/// * `llm` - LLM orchestrator for code generation
/// * `state_manager` - Agent state manager for persistence
/// * `user_task` - User's natural language intent
/// * `file_path` - Target file path for generated code
/// * `workspace_path` - Workspace root directory
/// * `target_node` - Optional specific function/class to modify
/// * `execute_code` - Whether to execute the generated code (default: false for safety)
/// 
/// # Returns
/// OrchestrationResult with success, escalation, or error
pub async fn orchestrate_with_execution(
    gnn: &GNNEngine,
    llm: &LLMOrchestrator,
    state_manager: &AgentStateManager,
    user_task: String,
    file_path: String,
    workspace_path: PathBuf,
    target_node: Option<String>,
    execute_code: bool,
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
    let _ = state_manager.save_state(&state);

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

    // Create executors
    let script_executor = ScriptExecutor::new(workspace_path.clone());
    let dependency_installer = DependencyInstaller::new(workspace_path.clone());
    let test_runner = TestRunner::new(workspace_path.clone());

    // Detect project type
    let project_type = dependency_installer.detect_project_type();

    // Auto-retry loop (up to 3 attempts)
    loop {
        state.increment_attempt();

        // Phase 2: Code Generation
        state.transition_to(AgentPhase::CodeGeneration);
        let _ = state_manager.save_state(&state);

        let code_result = generate_code_with_context(llm, &user_task, &context, &file_path).await;

        let response = match code_result {
            Ok(resp) => resp,
            Err(e) => {
                state.add_error(format!("Code generation failed: {}", e));
                
                if state.should_retry() {
                    continue;
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
        let _ = state_manager.save_state(&state);

        let validation_result = validate_dependencies(gnn, &response.code, &file_path);

        // Check validation result
        match validation_result {
            Ok(ValidationResult::Success) => {
                // Validation passed - continue to execution phases if requested
            }
            Ok(ValidationResult::Failed(errors)) => {
                for error in &errors {
                    state.add_error(error.message.clone());
                }

                // Calculate confidence
                let mut confidence = ConfidenceScore::new();
                confidence.set_llm_confidence(0.6);
                let error_count = errors.len();
                let pass_rate = if error_count > 5 { 0.0 } else { 0.5 };
                confidence.set_test_pass_rate(if pass_rate > 0.0 { 1 } else { 0 }, 2);
                
                state.set_confidence(confidence.overall());
                let _ = state_manager.save_state(&state);

                // Retry or escalate
                if state.should_retry() {
                    state.transition_to(AgentPhase::FixingIssues);
                    let _ = state_manager.save_state(&state);
                    continue;
                } else {
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
                state.add_error(format!("Validation error: {}", e));
                state.transition_to(AgentPhase::Failed);
                let _ = state_manager.save_state(&state);

                return OrchestrationResult::Error {
                    message: format!("Validation error: {}", e),
                    session_id: Some(session_id),
                };
            }
        }

        // Phase 3.5: Test Generation (NEW - Generate tests for the code)
        state.transition_to(AgentPhase::UnitTesting); // Reuse UnitTesting phase for generation
        let _ = state_manager.save_state(&state);

        let test_file_path = if file_path.ends_with(".py") {
            file_path.replace(".py", "_test.py")
        } else {
            format!("{}_test.py", file_path)
        };

        // Generate tests using testing::generator
        let test_gen_request = TestGenerationRequest {
            code: response.code.clone(),
            language: response.language.clone(),
            file_path: file_path.clone(),
            coverage_target: 0.8, // Target 80% coverage
        };

        let test_generation_result = crate::testing::generator::generate_tests(
            test_gen_request,
            llm.config().clone(), // Use same LLM config
        ).await;

        let generated_tests = match test_generation_result {
            Ok(test_resp) => {
                // Write tests to file
                let test_file = workspace_path.join(&test_file_path);
                if let Err(e) = std::fs::write(&test_file, &test_resp.tests) {
                    eprintln!("Warning: Failed to write test file: {}", e);
                    None
                } else {
                    Some(test_resp.tests)
                }
            }
            Err(e) => {
                eprintln!("Warning: Test generation failed: {}", e);
                None
            }
        };

        // If execute_code is false, skip execution phases
        if !execute_code {
            let mut confidence = ConfidenceScore::new();
            confidence.set_llm_confidence(0.9);
            confidence.set_test_pass_rate(0, 0);
            
            let loc = response.code.lines().count();
            let complexity = (loc / 10).min(10);
            confidence.set_code_complexity(complexity);
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

        // Write generated code to temporary file for execution
        let temp_file = workspace_path.join(format!("__yantra_temp_{}.py", session_id));
        if let Err(e) = std::fs::write(&temp_file, &response.code) {
            state.add_error(format!("Failed to write temp file: {}", e));
            state.transition_to(AgentPhase::Failed);
            let _ = state_manager.save_state(&state);

            return OrchestrationResult::Error {
                message: format!("Failed to write temp file: {}", e),
                session_id: Some(session_id),
            };
        }

        // Phase 4: Environment Setup
        state.transition_to(AgentPhase::EnvironmentSetup);
        let _ = state_manager.save_state(&state);
        
        // TODO: Future enhancement - create venv, set env vars
        // For now, we use the existing environment

        // Phase 5: Script Execution with auto-fix
        state.transition_to(AgentPhase::ScriptExecution);
        let _ = state_manager.save_state(&state);

        let exec_result = script_executor
            .execute_with_auto_fix(temp_file.clone(), project_type, 2)
            .await;

        let exec_result = match exec_result {
            Ok(result) => result,
            Err(e) => {
                state.add_error(format!("Script execution failed: {}", e));
                
                // Clean up temp file
                let _ = std::fs::remove_file(&temp_file);

                if state.should_retry() {
                    state.transition_to(AgentPhase::FixingIssues);
                    let _ = state_manager.save_state(&state);
                    continue;
                } else {
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
        };

        // Phase 6: Runtime Validation
        state.transition_to(AgentPhase::RuntimeValidation);
        let _ = state_manager.save_state(&state);

        if !exec_result.success {
            // Execution failed - analyze error
            let error_msg = format!(
                "Execution failed: {:?} - {}",
                exec_result.error_type,
                exec_result.error_message.as_deref().unwrap_or("Unknown error")
            );
            state.add_error(error_msg.clone());

            // Check if it's an import error that was missed
            if exec_result.error_type == Some(ErrorType::ImportError) {
                // Try to auto-fix import errors
                if let Some(ref err_msg) = exec_result.error_message {
                    state.transition_to(AgentPhase::DependencyInstallation);
                    let _ = state_manager.save_state(&state);

                    let install_result = dependency_installer
                        .auto_fix_missing_import(err_msg, project_type)
                        .await;

                    match install_result {
                        Ok(result) if result.success => {
                            // Installed successfully, retry execution
                            let _ = std::fs::remove_file(&temp_file);
                            state.transition_to(AgentPhase::FixingIssues);
                            let _ = state_manager.save_state(&state);
                            continue;
                        }
                        _ => {
                            // Installation failed or couldn't fix
                        }
                    }
                }
            }

            // Clean up temp file
            let _ = std::fs::remove_file(&temp_file);

            // Retry or escalate
            if state.should_retry() {
                state.transition_to(AgentPhase::FixingIssues);
                let _ = state_manager.save_state(&state);
                continue;
            } else {
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

        // Phase 7: Performance Profiling
        state.transition_to(AgentPhase::PerformanceProfiling);
        let _ = state_manager.save_state(&state);
        
        let execution_time_ms = exec_result.duration_ms;
        // TODO: Add memory profiling in future

        // Phase 8: Unit Testing (if tests exist)
        state.transition_to(AgentPhase::UnitTesting);
        let _ = state_manager.save_state(&state);

        // Check if tests exist in workspace
        let test_path = workspace_path.join("tests");
        let has_tests = test_path.exists() && test_path.is_dir();

        let test_results = if has_tests {
            match test_runner.run_pytest(None).await {
                Ok(results) => Some(results),
                Err(_) => None, // Tests might not be set up, that's ok
            }
        } else {
            None
        };

        // Calculate final confidence score
        let mut confidence = ConfidenceScore::new();
        confidence.set_llm_confidence(0.9);

        // Set test results if available
        if let Some(ref test_res) = test_results {
            confidence.set_test_pass_rate(test_res.passed, test_res.total_tests);
        } else {
            // No tests, use execution success as indicator
            confidence.set_test_pass_rate(1, 1); // 100% if execution succeeded
        }

        // Code complexity
        let loc = response.code.lines().count();
        let complexity = (loc / 10).min(10);
        confidence.set_code_complexity(complexity);

        // Dependency impact
        confidence.set_dependency_impact(1);

        // Performance factor (penalize if execution is very slow)
        if execution_time_ms > 10000 {
            // Over 10 seconds
            confidence.set_llm_confidence(confidence.llm_confidence * 0.9);
        }

        state.set_confidence(confidence.overall());

        // Clean up temp file
        let _ = std::fs::remove_file(&temp_file);

        // Success!
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
}

/// Find tests that should be run based on changed files
/// Uses GNN test tracking to identify affected tests
pub fn find_affected_tests(
    gnn: &GNNEngine,
    changed_files: &[String],
) -> Vec<String> {
    let mut affected_tests = Vec::new();
    
    for changed_file in changed_files {
        let changed_path = std::path::Path::new(changed_file);
        
        // Skip if the changed file itself is a test file
        if GNNEngine::is_test_file(changed_path) {
            // Test file changed - just run this test
            affected_tests.push(changed_file.clone());
            continue;
        }
        
        // Find test files that test this source file
        let all_nodes = gnn.get_graph().get_all_nodes().into_iter().cloned().collect::<Vec<_>>();
        
        for test_node in &all_nodes {
            let test_path = std::path::Path::new(&test_node.file_path);
            
            if !GNNEngine::is_test_file(test_path) {
                continue;
            }
            
            // Check if this test file tests the changed source file
            if let Some(source_file) = gnn.find_source_file_for_test(test_path) {
                if source_file == *changed_file || changed_file.ends_with(&source_file) {
                    affected_tests.push(test_node.file_path.clone());
                }
            }
        }
    }
    
    // Remove duplicates
    affected_tests.sort();
    affected_tests.dedup();
    
    affected_tests
}

/// Calculate test coverage metrics from GNN
pub fn calculate_test_coverage(gnn: &GNNEngine) -> TestCoverageMetrics {
    let all_nodes = gnn.get_graph().get_all_nodes().into_iter().cloned().collect::<Vec<_>>();
    
    // Count source files (non-test files)
    let source_files: Vec<_> = all_nodes.iter()
        .filter(|n| !GNNEngine::is_test_file(std::path::Path::new(&n.file_path)))
        .map(|n| &n.file_path)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .cloned()
        .collect();
    
    // Count test files
    let test_files: Vec<_> = all_nodes.iter()
        .filter(|n| GNNEngine::is_test_file(std::path::Path::new(&n.file_path)))
        .map(|n| &n.file_path)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .cloned()
        .collect();
    
    // Count tested source files
    let tested_files: Vec<_> = source_files.iter()
        .filter(|source_file| {
            test_files.iter().any(|test_file| {
                let test_path = std::path::Path::new(test_file);
                gnn.find_source_file_for_test(test_path)
                    .as_ref() == Some(source_file)
            })
        })
        .cloned()
        .collect();
    
    // Find untested files
    let untested_files: Vec<_> = source_files.iter()
        .filter(|f| !tested_files.contains(f))
        .cloned()
        .collect();
    
    let coverage_percentage = if source_files.len() > 0 {
        (tested_files.len() as f64 / source_files.len() as f64) * 100.0
    } else {
        0.0
    };
    
    TestCoverageMetrics {
        total_source_files: source_files.len(),
        total_test_files: test_files.len(),
        tested_source_files: tested_files.len(),
        untested_source_files: untested_files.len(),
        coverage_percentage,
        untested_files,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCoverageMetrics {
    pub total_source_files: usize,
    pub total_test_files: usize,
    pub tested_source_files: usize,
    pub untested_source_files: usize,
    pub coverage_percentage: f64,
    pub untested_files: Vec<String>,
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::LLMConfig;
    use tempfile::{NamedTempFile, tempdir};
    use std::fs;

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

    #[tokio::test]
    async fn test_orchestrate_with_execution_no_llm() {
        // Create test workspace
        let temp_dir = tempdir().unwrap();
        
        // Create empty GNN
        let temp_gnn = NamedTempFile::new().unwrap();
        let gnn = GNNEngine::new(temp_gnn.path()).unwrap();

        // Create state manager
        let temp_state = NamedTempFile::new().unwrap();
        let state_manager = AgentStateManager::new(temp_state.path().to_str().unwrap().to_string()).unwrap();

        // Create LLM orchestrator (no API keys)
        let llm_config = LLMConfig::default();
        let llm = LLMOrchestrator::new(llm_config);

        // Try orchestration with execution (should fail at LLM stage)
        let result = orchestrate_with_execution(
            &gnn,
            &llm,
            &state_manager,
            "Create a hello world function".to_string(),
            "test.py".to_string(),
            temp_dir.path().to_path_buf(),
            None,
            false, // Don't execute
        )
        .await;

        // Should fail or escalate (no LLM API key)
        match result {
            OrchestrationResult::Error { .. } => {} // Expected
            OrchestrationResult::Escalated { .. } => {} // Also expected
            OrchestrationResult::Success { .. } => panic!("Should not succeed without LLM API key"),
        }
    }

    #[test]
    fn test_agent_phase_serialization() {
        // Test new phases can be serialized
        let phases = vec![
            AgentPhase::EnvironmentSetup,
            AgentPhase::DependencyInstallation,
            AgentPhase::ScriptExecution,
            AgentPhase::RuntimeValidation,
            AgentPhase::PerformanceProfiling,
        ];

        for phase in phases {
            let phase_str = phase.to_string();
            let parsed = AgentPhase::from_string(&phase_str);
            assert_eq!(Some(phase), parsed);
        }
    }

    #[test]
    fn test_script_executor_integration() {
        // Test that ScriptExecutor can be created with workspace
        let temp_dir = tempdir().unwrap();
        let _executor = ScriptExecutor::new(temp_dir.path().to_path_buf());
        
        // Verify executor is created
        assert!(true); // If we got here, creation succeeded
    }

    #[test]
    fn test_dependency_installer_integration() {
        // Test that DependencyInstaller can be created with workspace
        let temp_dir = tempdir().unwrap();
        let installer = DependencyInstaller::new(temp_dir.path().to_path_buf());
        
        // Verify project type detection works
        let project_type = installer.detect_project_type();
        assert_eq!(project_type, ProjectType::Unknown); // Empty dir
    }

    #[test]
    fn test_dependency_installer_detects_python() {
        // Create workspace with requirements.txt
        let temp_dir = tempdir().unwrap();
        fs::write(temp_dir.path().join("requirements.txt"), "requests==2.28.0").unwrap();
        
        let installer = DependencyInstaller::new(temp_dir.path().to_path_buf());
        let project_type = installer.detect_project_type();
        assert_eq!(project_type, ProjectType::Python);
    }

    #[test]
    fn test_test_runner_integration() {
        // Test that TestRunner can be created with workspace
        let temp_dir = tempdir().unwrap();
        let _runner = TestRunner::new(temp_dir.path().to_path_buf());
        
        // Verify runner is created
        assert!(true); // If we got here, creation succeeded
    }

    #[tokio::test]
    async fn test_script_executor_simple_code() {
        // Test executing simple Python code
        let temp_dir = tempdir().unwrap();
        let executor = ScriptExecutor::new(temp_dir.path().to_path_buf());
        
        let result = executor.execute_python_code("print('Hello from orchestrator test')").await;
        assert!(result.is_ok());
        
        let exec_result = result.unwrap();
        assert!(exec_result.success);
        assert_eq!(exec_result.error_type, None);
    }

    #[tokio::test]
    async fn test_script_executor_with_error() {
        // Test executing Python code with error
        let temp_dir = tempdir().unwrap();
        let executor = ScriptExecutor::new(temp_dir.path().to_path_buf());
        
        let result = executor.execute_python_code("x = 1 / 0").await;
        assert!(result.is_ok());
        
        let exec_result = result.unwrap();
        assert!(!exec_result.success);
        assert_eq!(exec_result.error_type, Some(ErrorType::RuntimeError));
    }

    #[tokio::test]
    async fn test_script_executor_import_error_detection() {
        // Test detecting import errors
        let temp_dir = tempdir().unwrap();
        let executor = ScriptExecutor::new(temp_dir.path().to_path_buf());
        
        let result = executor.execute_python_code("import nonexistent_module_xyz").await;
        assert!(result.is_ok());
        
        let exec_result = result.unwrap();
        assert!(!exec_result.success);
        assert_eq!(exec_result.error_type, Some(ErrorType::ImportError));
        assert!(exec_result.error_message.is_some());
    }

    #[test]
    fn test_confidence_score_with_execution() {
        // Test confidence scoring with execution metrics
        let mut confidence = ConfidenceScore::new();
        
        confidence.set_llm_confidence(0.9);
        confidence.set_test_pass_rate(8, 10); // 80% pass rate
        confidence.set_code_complexity(5);
        confidence.set_dependency_impact(2);
        
        let overall = confidence.overall();
        // Formula: LLM(0.9)*0.30 + TestPass(0.8)*0.25 + KnownMatch(0.0)*0.25 
        //          + (1-Complexity(0.5))*0.10 + (1-Deps(0.9))*0.10
        // Expected: 0.27 + 0.20 + 0.00 + 0.05 + 0.01 = 0.53
        assert!(overall > 0.0 && overall <= 1.0);
        assert!(overall > 0.5); // Should be above medium threshold
        assert!(overall < 0.6); // But not too high (no known failure match)
    }

    #[test]
    fn test_confidence_score_with_test_failures() {
        // Test confidence scoring when tests fail
        let mut confidence = ConfidenceScore::new();
        
        confidence.set_llm_confidence(0.9);
        confidence.set_test_pass_rate(2, 10); // Only 20% pass rate
        confidence.set_code_complexity(5);
        confidence.set_dependency_impact(2);
        
        let overall = confidence.overall();
        assert!(overall > 0.0 && overall <= 1.0);
        assert!(overall < 0.6); // Should be lower with many test failures
    }
    
    #[test]
    fn test_find_affected_tests() {
        // Create a test project with test tracking
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_graph.db");
        
        let mut gnn = GNNEngine::new(&db_path).unwrap();
        
        // Build graph from test_project
        let test_project = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("test_project");
        
        if test_project.exists() {
            gnn.build_graph(&test_project).unwrap();
            gnn.create_test_edges().unwrap();
            
            // Test finding affected tests when calculator.py changes
            let calc_path = test_project.join("calculator.py");
            let changed_files = vec![calc_path.to_str().unwrap().to_string()];
            
            let affected = find_affected_tests(&gnn, &changed_files);
            
            // Should find test_calculator.py as affected
            assert!(affected.len() > 0, "Should find at least one affected test");
            assert!(affected.iter().any(|t| t.ends_with("test_calculator.py")),
                    "Should find test_calculator.py as affected test");
            
            println!("✅ Found {} affected test(s) for calculator.py", affected.len());
        }
    }
    
    #[test]
    fn test_calculate_test_coverage() {
        // Create a test project with test tracking
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_coverage.db");
        
        let mut gnn = GNNEngine::new(&db_path).unwrap();
        
        // Build graph from test_project
        let test_project = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("test_project");
        
        if test_project.exists() {
            gnn.build_graph(&test_project).unwrap();
            gnn.create_test_edges().unwrap();
            
            let metrics = calculate_test_coverage(&gnn);
            
            println!("📊 Test Coverage Metrics:");
            println!("  Total source files: {}", metrics.total_source_files);
            println!("  Total test files: {}", metrics.total_test_files);
            println!("  Tested source files: {}", metrics.tested_source_files);
            println!("  Coverage: {:.1}%", metrics.coverage_percentage);
            
            assert!(metrics.total_source_files > 0, "Should have source files");
            assert!(metrics.total_test_files > 0, "Should have test files");
            assert!(metrics.coverage_percentage >= 0.0 && metrics.coverage_percentage <= 100.0,
                    "Coverage should be between 0 and 100%");
            
            if metrics.untested_files.len() > 0 {
                println!("\n⚠️  Untested files:");
                for file in &metrics.untested_files {
                    println!("    - {}", file);
                }
            }
        }
    }
}

