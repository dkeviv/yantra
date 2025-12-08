// File: src-tauri/src/testing/retry.rs
// Purpose: Auto-retry with code regeneration for test failures
// Dependencies: testing/executor, llm/orchestrator, agent/regenerator
// Last Updated: December 3, 2025
//
// Implements autonomous retry logic for "code that never breaks":
// 1. Run tests and detect failures
// 2. Analyze failure patterns
// 3. Regenerate code with fixes
// 4. Retry up to 3 times with progressive improvements
// 5. Learn from successful fixes
//
// Performance targets:
// - Failure analysis: <1s
// - Code regeneration: <5s per attempt
// - Total retry cycle: <30s for 3 attempts

use super::executor::{TestExecutionResult, TestFailureInfo};
use crate::llm::{ChatMessage, CodeGenerationRequest, LLMConfig};
use crate::llm::orchestrator::LLMOrchestrator;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const MAX_RETRY_ATTEMPTS: usize = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryResult {
    pub success: bool,
    pub final_code: String,
    pub attempts_made: usize,
    pub test_results: Vec<TestExecutionResult>,
    pub improvements: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct RetryStrategy {
    max_attempts: usize,
    progressive_fixes: bool,
    learn_from_fixes: bool,
}

impl Default for RetryStrategy {
    fn default() -> Self {
        Self {
            max_attempts: MAX_RETRY_ATTEMPTS,
            progressive_fixes: true,
            learn_from_fixes: true,
        }
    }
}

/// Retry executor with intelligent code regeneration
pub struct RetryExecutor {
    llm_orchestrator: LLMOrchestrator,
    strategy: RetryStrategy,
}

impl RetryExecutor {
    pub fn new(llm_config: LLMConfig) -> Self {
        Self {
            llm_orchestrator: LLMOrchestrator::new(llm_config),
            strategy: RetryStrategy::default(),
        }
    }

    pub fn with_strategy(mut self, strategy: RetryStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Execute code with automatic retry and regeneration
    /// 
    /// # Arguments
    /// * `original_request` - Original code generation request
    /// * `initial_result` - Result from first attempt
    /// * `test_file` - Path to test file
    /// 
    /// # Returns
    /// RetryResult with final code and attempt history
    pub async fn retry_with_regeneration(
        &self,
        original_request: &CodeGenerationRequest,
        initial_result: TestExecutionResult,
        test_file: &PathBuf,
    ) -> Result<RetryResult, String> {
        let mut current_code = String::new(); // Would come from initial generation
        let mut test_results = vec![initial_result.clone()];
        let mut improvements = Vec::new();
        let mut attempts = 1;

        // If initial attempt succeeded, return immediately
        if initial_result.success {
            return Ok(RetryResult {
                success: true,
                final_code: current_code,
                attempts_made: 1,
                test_results,
                improvements,
            });
        }

        // Analyze initial failures
        let failure_analysis = self.analyze_failures(&initial_result.failures);
        improvements.push(format!("Attempt 1: Identified {} failure patterns", failure_analysis.patterns.len()));

        // Retry loop
        while attempts < self.strategy.max_attempts {
            attempts += 1;
            println!("🔄 Retry attempt {}/{}", attempts, self.strategy.max_attempts);

            // Regenerate code with failure analysis
            match self.regenerate_with_fixes(
                original_request,
                &current_code,
                &failure_analysis,
                &test_results,
            ).await {
                Ok(new_code) => {
                    current_code = new_code;
                    
                    // Run tests again
                    let executor = super::executor::PytestExecutor::new(
                        test_file.parent().unwrap_or(test_file.as_path()).to_path_buf()
                    );
                    
                    match executor.execute(test_file.to_str().unwrap(), None).await {
                        Ok(result) => {
                            let improvement = self.calculate_improvement(&test_results.last().unwrap(), &result);
                            improvements.push(format!("Attempt {}: {} (pass rate: {:.1}% → {:.1}%)",
                                attempts,
                                improvement.description,
                                test_results.last().unwrap().pass_rate * 100.0,
                                result.pass_rate * 100.0
                            ));
                            
                            test_results.push(result.clone());
                            
                            // Check if tests now pass
                            if result.success {
                                println!("✅ Tests passed after {} attempts!", attempts);
                                
                                // Learn from successful fix if enabled
                                if self.strategy.learn_from_fixes {
                                    self.learn_from_fix(&current_code, &failure_analysis).await;
                                }
                                
                                return Ok(RetryResult {
                                    success: true,
                                    final_code: current_code,
                                    attempts_made: attempts,
                                    test_results,
                                    improvements,
                                });
                            }
                            
                            // Check if no improvement (stuck)
                            if !improvement.improved && attempts > 1 {
                                println!("⚠️ No improvement detected, stopping retry");
                                break;
                            }
                        }
                        Err(e) => {
                            improvements.push(format!("Attempt {}: Test execution failed: {}", attempts, e));
                            break;
                        }
                    }
                }
                Err(e) => {
                    improvements.push(format!("Attempt {}: Code regeneration failed: {}", attempts, e));
                    break;
                }
            }
        }

        // Max attempts reached or no improvement
        Ok(RetryResult {
            success: false,
            final_code: current_code,
            attempts_made: attempts,
            test_results,
            improvements,
        })
    }

    /// Analyze test failures to identify patterns
    fn analyze_failures(&self, failures: &[TestFailureInfo]) -> FailureAnalysis {
        let mut patterns = Vec::new();
        let mut error_types = std::collections::HashMap::new();

        for failure in failures {
            // Count error types
            *error_types.entry(failure.error_type.clone()).or_insert(0) += 1;

            // Identify common patterns
            if failure.error_message.contains("AttributeError") {
                patterns.push(FailurePattern {
                    pattern_type: "missing_attribute".to_string(),
                    description: "Missing attribute or method".to_string(),
                    test_names: vec![failure.test_name.clone()],
                    suggested_fix: "Add missing attribute/method to class".to_string(),
                });
            } else if failure.error_message.contains("TypeError") {
                patterns.push(FailurePattern {
                    pattern_type: "type_mismatch".to_string(),
                    description: "Type mismatch in function call".to_string(),
                    test_names: vec![failure.test_name.clone()],
                    suggested_fix: "Fix function signature or type conversion".to_string(),
                });
            } else if failure.error_message.contains("ImportError") || failure.error_message.contains("ModuleNotFoundError") {
                patterns.push(FailurePattern {
                    pattern_type: "import_error".to_string(),
                    description: "Missing import or module".to_string(),
                    test_names: vec![failure.test_name.clone()],
                    suggested_fix: "Add missing import statement".to_string(),
                });
            } else if failure.error_message.contains("AssertionError") {
                patterns.push(FailurePattern {
                    pattern_type: "assertion_failure".to_string(),
                    description: "Test assertion failed".to_string(),
                    test_names: vec![failure.test_name.clone()],
                    suggested_fix: "Fix logic to match expected behavior".to_string(),
                });
            }
        }

        FailureAnalysis {
            total_failures: failures.len(),
            error_types,
            patterns,
        }
    }

    /// Regenerate code with fixes based on failure analysis
    async fn regenerate_with_fixes(
        &self,
        original_request: &CodeGenerationRequest,
        current_code: &str,
        failure_analysis: &FailureAnalysis,
        test_history: &[TestExecutionResult],
    ) -> Result<String, String> {
        // Build context with failure information
        let mut messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are an expert programmer. Fix the code based on test failures.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: format!(
                    "Original request: {}\n\nCurrent code:\n```\n{}\n```\n\n",
                    original_request.intent,
                    current_code
                ),
            },
        ];

        // Add failure analysis
        let failure_context = self.format_failure_analysis(failure_analysis, test_history);
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: failure_context,
        });

        // Add suggested fixes
        let fix_suggestions = self.generate_fix_suggestions(failure_analysis);
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: format!(
                "Suggested fixes:\n{}\n\nPlease regenerate the code with these fixes applied. Return only the fixed code.",
                fix_suggestions
            ),
        });

        // Generate fixed code
        let response = self.llm_orchestrator
            .generate_code_with_context(original_request, &[])
            .await
            .map_err(|e| format!("Code regeneration failed: {}", e))?;

        Ok(response.code)
    }

    /// Format failure analysis for LLM context
    fn format_failure_analysis(
        &self,
        analysis: &FailureAnalysis,
        test_history: &[TestExecutionResult],
    ) -> String {
        let mut output = format!("Test Failures Analysis:\n");
        output.push_str(&format!("Total failures: {}\n\n", analysis.total_failures));

        output.push_str("Error types:\n");
        for (error_type, count) in &analysis.error_types {
            output.push_str(&format!("  - {}: {} occurrences\n", error_type, count));
        }

        output.push_str("\nIdentified patterns:\n");
        for (i, pattern) in analysis.patterns.iter().enumerate() {
            output.push_str(&format!("{}. {} ({})\n", i + 1, pattern.description, pattern.pattern_type));
            output.push_str(&format!("   Affected tests: {}\n", pattern.test_names.join(", ")));
        }

        output.push_str("\nTest history:\n");
        for (i, result) in test_history.iter().enumerate() {
            output.push_str(&format!("Attempt {}: {}/{} passed ({:.1}%)\n",
                i + 1,
                result.passed,
                result.total,
                result.pass_rate * 100.0
            ));
        }

        output
    }

    /// Generate fix suggestions based on patterns
    fn generate_fix_suggestions(&self, analysis: &FailureAnalysis) -> String {
        let mut suggestions = Vec::new();

        for pattern in &analysis.patterns {
            suggestions.push(format!("- {}: {}", pattern.description, pattern.suggested_fix));
        }

        suggestions.join("\n")
    }

    /// Calculate improvement between test runs
    fn calculate_improvement(&self, previous: &TestExecutionResult, current: &TestExecutionResult) -> Improvement {
        let pass_rate_delta = current.pass_rate - previous.pass_rate;
        let fixed_tests = (current.passed as i32) - (previous.passed as i32);

        let improved = pass_rate_delta > 0.0 || fixed_tests > 0;
        let description = if improved {
            if fixed_tests > 0 {
                format!("Fixed {} test(s)", fixed_tests)
            } else {
                format!("Pass rate improved by {:.1}%", pass_rate_delta * 100.0)
            }
        } else if pass_rate_delta < 0.0 {
            "Regression detected".to_string()
        } else {
            "No change".to_string()
        };

        Improvement {
            improved,
            pass_rate_delta,
            fixed_tests,
            description,
        }
    }

    /// Learn from successful fix (store pattern for future use)
    async fn learn_from_fix(&self, _fixed_code: &str, _failure_analysis: &FailureAnalysis) {
        // TODO: Implement learning storage
        // Store (failure_pattern, fix_pattern) pair in database or ChromaDB
        // This will enable faster fixes in the future
        println!("📚 Learning from successful fix (not yet implemented)");
    }
}

#[derive(Debug, Clone)]
struct FailureAnalysis {
    total_failures: usize,
    error_types: std::collections::HashMap<String, usize>,
    patterns: Vec<FailurePattern>,
}

#[derive(Debug, Clone)]
struct FailurePattern {
    pattern_type: String,
    description: String,
    test_names: Vec<String>,
    suggested_fix: String,
}

#[derive(Debug, Clone)]
struct Improvement {
    improved: bool,
    pass_rate_delta: f64,
    fixed_tests: i32,
    description: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_failure_analysis() {
        let failures = vec![
            TestFailureInfo {
                test_name: "test_add".to_string(),
                error_type: "AttributeError".to_string(),
                error_message: "AttributeError: 'Calculator' object has no attribute 'add'".to_string(),
            },
            TestFailureInfo {
                test_name: "test_subtract".to_string(),
                error_type: "TypeError".to_string(),
                error_message: "TypeError: unsupported operand type(s) for -: 'str' and 'int'".to_string(),
            },
        ];

        let llm_config = LLMConfig::default();
        let executor = RetryExecutor::new(llm_config);
        let analysis = executor.analyze_failures(&failures);

        assert_eq!(analysis.total_failures, 2);
        assert_eq!(analysis.patterns.len(), 2);
    }
}
