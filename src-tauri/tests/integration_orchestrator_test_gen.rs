// Integration test: Orchestrator with automatic test generation
// Tests the complete flow: Generate code → Generate tests → Run tests

use yantra::agent::orchestrator::orchestrate_with_execution;
use yantra::agent::state::AgentStateManager;
use yantra::gnn::GNNEngine;
use yantra::llm::orchestrator::LLMOrchestrator;
use yantra::llm::{LLMConfig, LLMProvider};
use tempfile::tempdir;
use std::fs;

#[tokio::test]
async fn test_orchestrator_generates_tests_for_code() {
    // Skip if no API keys (CI environment)
    let claude_key = std::env::var("ANTHROPIC_API_KEY").ok();
    if claude_key.is_none() {
        println!("Skipping test: ANTHROPIC_API_KEY not set");
        return;
    }

    // Setup temp workspace
    let temp_dir = tempdir().unwrap();
    let workspace = temp_dir.path().to_path_buf();
    
    // Create GNN
    let gnn_db = workspace.join("test.db");
    let gnn = GNNEngine::new(&gnn_db).unwrap();
    
    // Create state manager
    let state_db = workspace.join("state.db");
    let state_manager = AgentStateManager::new(state_db.to_str().unwrap().to_string()).unwrap();
    
    // Create LLM orchestrator with real API key
    let llm_config = LLMConfig {
        primary_provider: LLMProvider::Claude,
        claude_api_key: claude_key,
        openai_api_key: None,
        openrouter_api_key: None,
        groq_api_key: None,
        gemini_api_key: None,
        timeout_seconds: 30,
        max_retries: 3,
        selected_models: Vec::new(),
    };
    let llm = LLMOrchestrator::new(llm_config);
    
    // Generate simple Python function
    let user_task = "Create a function called add_numbers that takes two numbers and returns their sum".to_string();
    let file_path = "calculator.py".to_string();
    
    let result = orchestrate_with_execution(
        &gnn,
        &llm,
        &state_manager,
        user_task,
        file_path.clone(),
        workspace.clone(),
        None,
        false, // Don't execute yet, just generate
    ).await;
    
    // Check result
    match result {
        yantra::agent::orchestrator::OrchestrationResult::Success { code, .. } => {
            println!("Generated code:\n{}", code);
            
            // Check that code was generated
            assert!(code.contains("def add_numbers"));
            
            // Check that test file was created
            let test_file = workspace.join("calculator_test.py");
            assert!(test_file.exists(), "Test file should be generated");
            
            let test_content = fs::read_to_string(test_file).unwrap();
            println!("Generated tests:\n{}", test_content);
            
            // Verify test content
            assert!(test_content.contains("def test_"), "Should have test functions");
            assert!(test_content.contains("add_numbers"), "Tests should reference the function");
            assert!(test_content.contains("import pytest") || test_content.contains("import unittest"), 
                    "Should have test imports");
        }
        _ => panic!("Orchestration should succeed, got: {:?}", result),
    }
}

#[tokio::test]
async fn test_orchestrator_runs_generated_tests() {
    // Skip if no API keys
    let claude_key = std::env::var("ANTHROPIC_API_KEY").ok();
    if claude_key.is_none() {
        println!("Skipping test: ANTHROPIC_API_KEY not set");
        return;
    }

    // Setup temp workspace
    let temp_dir = tempdir().unwrap();
    let workspace = temp_dir.path().to_path_buf();
    
    // Create tests directory
    fs::create_dir_all(workspace.join("tests")).unwrap();
    
    // Create GNN
    let gnn_db = workspace.join("test.db");
    let gnn = GNNEngine::new(&gnn_db).unwrap();
    
    // Create state manager
    let state_db = workspace.join("state.db");
    let state_manager = AgentStateManager::new(state_db.to_str().unwrap().to_string()).unwrap();
    
    // Create LLM orchestrator
    let llm_config = LLMConfig {
        primary_provider: LLMProvider::Claude,
        claude_api_key: claude_key,
        openai_api_key: None,
        openrouter_api_key: None,
        groq_api_key: None,
        gemini_api_key: None,
        timeout_seconds: 30,
        max_retries: 3,
        selected_models: Vec::new(),
    };
    let llm = LLMOrchestrator::new(llm_config);
    
    // Generate and execute simple function
    let user_task = "Create a function called multiply that takes two numbers and returns their product".to_string();
    let file_path = "math_ops.py".to_string();
    
    let result = orchestrate_with_execution(
        &gnn,
        &llm,
        &state_manager,
        user_task,
        file_path.clone(),
        workspace.clone(),
        None,
        true, // Execute code AND run tests
    ).await;
    
    // Check result
    match result {
        yantra::agent::orchestrator::OrchestrationResult::Success { code, confidence, .. } => {
            println!("Generated code:\n{}", code);
            println!("Confidence: {}", confidence);
            
            // With test execution, confidence should be higher
            assert!(confidence > 0.5, "Confidence should be reasonable with tests");
            
            // Check test file exists
            let test_file = workspace.join("math_ops_test.py");
            assert!(test_file.exists(), "Test file should be generated");
            
            let test_content = fs::read_to_string(test_file).unwrap();
            println!("Generated tests:\n{}", test_content);
            
            // Verify tests ran (check session state)
            // Note: In real implementation, we'd check test results in the state
        }
        yantra::agent::orchestrator::OrchestrationResult::Escalated { errors, .. } => {
            // It's ok if tests fail - we're testing integration, not correctness
            println!("Escalated with errors: {:?}", errors);
            // Still check that test file was created
            let test_file = workspace.join("math_ops_test.py");
            assert!(test_file.exists(), "Test file should be generated even if tests fail");
        }
        _ => panic!("Orchestration should succeed or escalate, got: {:?}", result),
    }
}
