// File: src-tauri/src/testing/generator.rs
// Purpose: Generate pytest tests from code using LLM
// Last Updated: November 21, 2025

use crate::llm::{CodeGenerationRequest, LLMConfig};
use crate::llm::orchestrator::LLMOrchestrator;
use crate::testing::{TestGenerationRequest, TestGenerationResponse};

/// Generate pytest tests for given code using LLM
pub async fn generate_tests(
    request: TestGenerationRequest,
    llm_config: LLMConfig,
) -> Result<TestGenerationResponse, String> {
    // Create prompt for test generation
    let test_prompt = build_test_generation_prompt(&request);
    
    // Create LLM request
    let llm_request = CodeGenerationRequest {
        intent: test_prompt,
        file_path: Some(format!("test_{}", request.file_path)),
        context: vec![
            format!("# Code to test:\n{}", request.code),
            format!("# Target coverage: {:.0}%", request.coverage_target * 100.0),
        ],
        dependencies: Vec::new(),
    };
    
    // Generate tests using LLM orchestrator
    let orchestrator = LLMOrchestrator::new(llm_config);
    let response = orchestrator.generate_code(&llm_request).await
        .map_err(|e| e.message)?;
    
    // Use the code field which contains the generated tests
    let test_code = &response.code;
    
    // Parse the generated tests
    let test_count = count_test_functions(test_code);
    let fixtures = extract_fixtures(test_code);
    
    Ok(TestGenerationResponse {
        tests: test_code.clone(),
        test_count,
        estimated_coverage: estimate_coverage(test_count, &request.code),
        fixtures,
    })
}

/// Build prompt for test generation
fn build_test_generation_prompt(request: &TestGenerationRequest) -> String {
    format!(
        r#"Generate comprehensive pytest tests for the following {} code.

Requirements:
1. Achieve at least {:.0}% code coverage
2. Include happy path tests
3. Include edge case tests (empty inputs, null values, boundary conditions)
4. Include error condition tests
5. Use pytest fixtures where appropriate
6. Add clear docstrings for each test
7. Mock external dependencies
8. Follow pytest best practices

Code to test:
```{}
{}
```

Generate ONLY the test code, with proper imports and fixtures."#,
        request.language,
        request.coverage_target * 100.0,
        request.language,
        request.code
    )
}

/// Count number of test functions in generated tests
fn count_test_functions(test_code: &str) -> usize {
    test_code.lines()
        .filter(|line| line.trim().starts_with("def test_") || line.trim().starts_with("async def test_"))
        .count()
}

/// Extract pytest fixtures from test code
fn extract_fixtures(test_code: &str) -> Vec<String> {
    let mut fixtures = Vec::new();
    let lines: Vec<&str> = test_code.lines().collect();
    
    for (i, line) in lines.iter().enumerate() {
        if line.contains("@pytest.fixture") {
            // Get the next line which should be the function definition
            if i + 1 < lines.len() {
                if let Some(func_name) = lines[i + 1].trim()
                    .strip_prefix("def ")
                    .and_then(|s| s.split('(').next()) {
                    fixtures.push(func_name.to_string());
                }
            }
        }
    }
    
    fixtures
}

/// Estimate coverage based on test count and code complexity
fn estimate_coverage(test_count: usize, code: &str) -> f32 {
    // Simple heuristic: count functions/methods in code
    let function_count = code.lines()
        .filter(|line| {
            let trimmed = line.trim();
            trimmed.starts_with("def ") || trimmed.starts_with("async def ")
        })
        .count()
        .max(1); // Avoid division by zero
    
    // Assume each test covers one function on average
    // Cap at 1.0 (100%)
    let coverage = (test_count as f32) / (function_count as f32);
    coverage.min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_count_test_functions() {
        let test_code = r#"
def test_addition():
    assert 1 + 1 == 2

def test_subtraction():
    assert 2 - 1 == 1

async def test_async_function():
    result = await some_function()
    assert result == expected
"#;
        assert_eq!(count_test_functions(test_code), 3);
    }
    
    #[test]
    fn test_extract_fixtures() {
        let test_code = r#"
@pytest.fixture
def sample_data():
    return {"key": "value"}

@pytest.fixture
def mock_client():
    return MockClient()
"#;
        let fixtures = extract_fixtures(test_code);
        assert_eq!(fixtures.len(), 2);
        assert!(fixtures.contains(&"sample_data".to_string()));
        assert!(fixtures.contains(&"mock_client".to_string()));
    }
    
    #[test]
    fn test_estimate_coverage() {
        let code = r#"
def function1():
    pass

def function2():
    pass

def function3():
    pass
"#;
        // 3 functions, 2 tests = ~67% coverage
        let coverage = estimate_coverage(2, code);
        assert!(coverage >= 0.6 && coverage <= 0.7);
        
        // 3 functions, 3 tests = 100% coverage
        let coverage = estimate_coverage(3, code);
        assert!(coverage >= 0.99 && coverage <= 1.0);
    }
    
    #[test]
    fn test_build_test_generation_prompt() {
        let request = TestGenerationRequest {
            code: "def add(a, b): return a + b".to_string(),
            language: "python".to_string(),
            file_path: "calculator.py".to_string(),
            coverage_target: 0.9,
        };
        
        let prompt = build_test_generation_prompt(&request);
        assert!(prompt.contains("90%"));
        assert!(prompt.contains("pytest"));
        assert!(prompt.contains("def add(a, b)"));
    }
}
