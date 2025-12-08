// File: src-tauri/src/testing/generator_unified.rs
// Purpose: Unified test generator that routes to language-specific generators
// Last Updated: December 4, 2025
//
// Supports all 13 implemented languages:
// - Python (pytest)
// - JavaScript/TypeScript (Jest)
// - Rust (cargo test)
// - Go (go test)
// - Java (JUnit 5)
// - C (Unity/CUnit)
// - C++ (Google Test)
// - Ruby (RSpec)
// - PHP (PHPUnit)
// - Swift (XCTest)
// - Kotlin (JUnit 5)

use crate::llm::{CodeGenerationRequest, LLMConfig};
use crate::llm::orchestrator::LLMOrchestrator;
use crate::testing::{TestGenerationRequest, TestGenerationResponse};

/// Supported test frameworks by language
#[derive(Debug, Clone)]
pub enum TestFramework {
    Pytest,           // Python
    Jest,             // JavaScript/TypeScript
    CargoTest,        // Rust
    GoTest,           // Go
    JUnit5,           // Java/Kotlin
    Unity,            // C
    GoogleTest,       // C++
    RSpec,            // Ruby
    PHPUnit,          // PHP
    XCTest,           // Swift
}

/// Detect test framework from language
fn detect_framework(language: &str) -> Result<TestFramework, String> {
    match language.to_lowercase().as_str() {
        "python" | "py" => Ok(TestFramework::Pytest),
        "javascript" | "js" | "jsx" | "typescript" | "ts" | "tsx" => Ok(TestFramework::Jest),
        "rust" | "rs" => Ok(TestFramework::CargoTest),
        "go" => Ok(TestFramework::GoTest),
        "java" => Ok(TestFramework::JUnit5),
        "kotlin" | "kt" => Ok(TestFramework::JUnit5),
        "c" => Ok(TestFramework::Unity),
        "cpp" | "c++" | "cc" | "cxx" => Ok(TestFramework::GoogleTest),
        "ruby" | "rb" => Ok(TestFramework::RSpec),
        "php" => Ok(TestFramework::PHPUnit),
        "swift" => Ok(TestFramework::XCTest),
        _ => Err(format!("Unsupported language for test generation: {}", language)),
    }
}

/// Generate tests for any supported language
pub async fn generate_tests_unified(
    request: TestGenerationRequest,
    llm_config: LLMConfig,
) -> Result<TestGenerationResponse, String> {
    let framework = detect_framework(&request.language)?;
    
    match framework {
        TestFramework::Pytest => {
            crate::testing::generator::generate_tests(request, llm_config).await
        }
        TestFramework::Jest => {
            crate::testing::generator_js::generate_jest_tests(request, llm_config).await
        }
        TestFramework::CargoTest => {
            generate_rust_tests(request, llm_config).await
        }
        TestFramework::GoTest => {
            generate_go_tests(request, llm_config).await
        }
        TestFramework::JUnit5 => {
            generate_junit_tests(request, llm_config).await
        }
        TestFramework::Unity => {
            generate_c_tests(request, llm_config).await
        }
        TestFramework::GoogleTest => {
            generate_cpp_tests(request, llm_config).await
        }
        TestFramework::RSpec => {
            generate_rspec_tests(request, llm_config).await
        }
        TestFramework::PHPUnit => {
            generate_phpunit_tests(request, llm_config).await
        }
        TestFramework::XCTest => {
            generate_xctest_tests(request, llm_config).await
        }
    }
}

/// Generate Rust tests using cargo test
async fn generate_rust_tests(
    request: TestGenerationRequest,
    llm_config: LLMConfig,
) -> Result<TestGenerationResponse, String> {
    let test_prompt = build_rust_test_prompt(&request);
    
    let llm_request = CodeGenerationRequest {
        intent: test_prompt,
        file_path: Some(format!("tests/test_{}.rs", request.file_path.replace(".rs", ""))),
        context: vec![
            format!("# Rust code to test:\n{}", request.code),
            format!("# Target coverage: {:.0}%", request.coverage_target * 100.0),
        ],
        dependencies: Vec::new(),
    };
    
    let orchestrator = LLMOrchestrator::new(llm_config);
    let response = orchestrator.generate_code(&llm_request).await
        .map_err(|e| e.message)?;
    
    let test_code = &response.code;
    let test_count = count_rust_test_functions(test_code);
    
    Ok(TestGenerationResponse {
        tests: test_code.clone(),
        test_count,
        estimated_coverage: estimate_coverage(test_count, &request.code),
        fixtures: Vec::new(),
    })
}

/// Generate Go tests using go test
async fn generate_go_tests(
    request: TestGenerationRequest,
    llm_config: LLMConfig,
) -> Result<TestGenerationResponse, String> {
    let test_prompt = build_go_test_prompt(&request);
    
    let llm_request = CodeGenerationRequest {
        intent: test_prompt,
        file_path: Some(format!("{}_test.go", request.file_path.replace(".go", ""))),
        context: vec![
            format!("# Go code to test:\n{}", request.code),
            format!("# Target coverage: {:.0}%", request.coverage_target * 100.0),
        ],
        dependencies: Vec::new(),
    };
    
    let orchestrator = LLMOrchestrator::new(llm_config);
    let response = orchestrator.generate_code(&llm_request).await
        .map_err(|e| e.message)?;
    
    let test_code = &response.code;
    let test_count = count_go_test_functions(test_code);
    
    Ok(TestGenerationResponse {
        tests: test_code.clone(),
        test_count,
        estimated_coverage: estimate_coverage(test_count, &request.code),
        fixtures: Vec::new(),
    })
}

/// Generate Java/Kotlin tests using JUnit 5
async fn generate_junit_tests(
    request: TestGenerationRequest,
    llm_config: LLMConfig,
) -> Result<TestGenerationResponse, String> {
    let test_prompt = build_junit_test_prompt(&request);
    
    let extension = if request.language.to_lowercase().contains("kotlin") { "kt" } else { "java" };
    
    let llm_request = CodeGenerationRequest {
        intent: test_prompt,
        file_path: Some(format!("src/test/{}/{}Test.{}", 
            request.language.to_lowercase(), 
            request.file_path.replace(&format!(".{}", extension), ""),
            extension)),
        context: vec![
            format!("# {} code to test:\n{}", request.language, request.code),
            format!("# Target coverage: {:.0}%", request.coverage_target * 100.0),
        ],
        dependencies: Vec::new(),
    };
    
    let orchestrator = LLMOrchestrator::new(llm_config);
    let response = orchestrator.generate_code(&llm_request).await
        .map_err(|e| e.message)?;
    
    let test_code = &response.code;
    let test_count = count_junit_test_methods(test_code);
    
    Ok(TestGenerationResponse {
        tests: test_code.clone(),
        test_count,
        estimated_coverage: estimate_coverage(test_count, &request.code),
        fixtures: Vec::new(),
    })
}

/// Generate C tests using Unity framework
async fn generate_c_tests(
    request: TestGenerationRequest,
    llm_config: LLMConfig,
) -> Result<TestGenerationResponse, String> {
    let test_prompt = build_c_test_prompt(&request);
    
    let llm_request = CodeGenerationRequest {
        intent: test_prompt,
        file_path: Some(format!("test_{}", request.file_path)),
        context: vec![
            format!("# C code to test:\n{}", request.code),
            format!("# Target coverage: {:.0}%", request.coverage_target * 100.0),
        ],
        dependencies: Vec::new(),
    };
    
    let orchestrator = LLMOrchestrator::new(llm_config);
    let response = orchestrator.generate_code(&llm_request).await
        .map_err(|e| e.message)?;
    
    let test_code = &response.code;
    let test_count = count_c_test_functions(test_code);
    
    Ok(TestGenerationResponse {
        tests: test_code.clone(),
        test_count,
        estimated_coverage: estimate_coverage(test_count, &request.code),
        fixtures: Vec::new(),
    })
}

/// Generate C++ tests using Google Test
async fn generate_cpp_tests(
    request: TestGenerationRequest,
    llm_config: LLMConfig,
) -> Result<TestGenerationResponse, String> {
    let test_prompt = build_cpp_test_prompt(&request);
    
    let llm_request = CodeGenerationRequest {
        intent: test_prompt,
        file_path: Some(format!("tests/test_{}", request.file_path)),
        context: vec![
            format!("# C++ code to test:\n{}", request.code),
            format!("# Target coverage: {:.0}%", request.coverage_target * 100.0),
        ],
        dependencies: Vec::new(),
    };
    
    let orchestrator = LLMOrchestrator::new(llm_config);
    let response = orchestrator.generate_code(&llm_request).await
        .map_err(|e| e.message)?;
    
    let test_code = &response.code;
    let test_count = count_gtest_tests(test_code);
    
    Ok(TestGenerationResponse {
        tests: test_code.clone(),
        test_count,
        estimated_coverage: estimate_coverage(test_count, &request.code),
        fixtures: Vec::new(),
    })
}

/// Generate Ruby tests using RSpec
async fn generate_rspec_tests(
    request: TestGenerationRequest,
    llm_config: LLMConfig,
) -> Result<TestGenerationResponse, String> {
    let test_prompt = build_rspec_test_prompt(&request);
    
    let llm_request = CodeGenerationRequest {
        intent: test_prompt,
        file_path: Some(format!("spec/{}_spec.rb", request.file_path.replace(".rb", ""))),
        context: vec![
            format!("# Ruby code to test:\n{}", request.code),
            format!("# Target coverage: {:.0}%", request.coverage_target * 100.0),
        ],
        dependencies: Vec::new(),
    };
    
    let orchestrator = LLMOrchestrator::new(llm_config);
    let response = orchestrator.generate_code(&llm_request).await
        .map_err(|e| e.message)?;
    
    let test_code = &response.code;
    let test_count = count_rspec_tests(test_code);
    
    Ok(TestGenerationResponse {
        tests: test_code.clone(),
        test_count,
        estimated_coverage: estimate_coverage(test_count, &request.code),
        fixtures: Vec::new(),
    })
}

/// Generate PHP tests using PHPUnit
async fn generate_phpunit_tests(
    request: TestGenerationRequest,
    llm_config: LLMConfig,
) -> Result<TestGenerationResponse, String> {
    let test_prompt = build_phpunit_test_prompt(&request);
    
    let llm_request = CodeGenerationRequest {
        intent: test_prompt,
        file_path: Some(format!("tests/{}Test.php", request.file_path.replace(".php", ""))),
        context: vec![
            format!("# PHP code to test:\n{}", request.code),
            format!("# Target coverage: {:.0}%", request.coverage_target * 100.0),
        ],
        dependencies: Vec::new(),
    };
    
    let orchestrator = LLMOrchestrator::new(llm_config);
    let response = orchestrator.generate_code(&llm_request).await
        .map_err(|e| e.message)?;
    
    let test_code = &response.code;
    let test_count = count_phpunit_tests(test_code);
    
    Ok(TestGenerationResponse {
        tests: test_code.clone(),
        test_count,
        estimated_coverage: estimate_coverage(test_count, &request.code),
        fixtures: Vec::new(),
    })
}

/// Generate Swift tests using XCTest
async fn generate_xctest_tests(
    request: TestGenerationRequest,
    llm_config: LLMConfig,
) -> Result<TestGenerationResponse, String> {
    let test_prompt = build_xctest_test_prompt(&request);
    
    let llm_request = CodeGenerationRequest {
        intent: test_prompt,
        file_path: Some(format!("Tests/{}Tests.swift", request.file_path.replace(".swift", ""))),
        context: vec![
            format!("# Swift code to test:\n{}", request.code),
            format!("# Target coverage: {:.0}%", request.coverage_target * 100.0),
        ],
        dependencies: Vec::new(),
    };
    
    let orchestrator = LLMOrchestrator::new(llm_config);
    let response = orchestrator.generate_code(&llm_request).await
        .map_err(|e| e.message)?;
    
    let test_code = &response.code;
    let test_count = count_xctest_tests(test_code);
    
    Ok(TestGenerationResponse {
        tests: test_code.clone(),
        test_count,
        estimated_coverage: estimate_coverage(test_count, &request.code),
        fixtures: Vec::new(),
    })
}

// ============================================================================
// Prompt Builders
// ============================================================================

fn build_rust_test_prompt(request: &TestGenerationRequest) -> String {
    format!(
        r#"Generate comprehensive Rust tests using cargo test.

Requirements:
1. Achieve at least {:.0}% code coverage
2. Include happy path tests
3. Include edge case tests (empty inputs, None values, boundary conditions)
4. Include error condition tests with Result/Option
5. Use #[test] attribute for test functions
6. Add clear documentation comments
7. Mock external dependencies with test doubles
8. Follow Rust testing best practices

Code to test:
```rust
{}
```

Generate ONLY the test code with proper imports and test module."#,
        request.coverage_target * 100.0,
        request.code
    )
}

fn build_go_test_prompt(request: &TestGenerationRequest) -> String {
    format!(
        r#"Generate comprehensive Go tests using go test framework.

Requirements:
1. Achieve at least {:.0}% code coverage
2. Include happy path tests
3. Include edge case tests (nil, empty, boundary conditions)
4. Include error condition tests
5. Use testing.T for test functions (func TestXxx(t *testing.T))
6. Add clear comments for each test
7. Use table-driven tests where appropriate
8. Follow Go testing conventions

Code to test:
```go
{}
```

Generate ONLY the test code with proper package and imports."#,
        request.coverage_target * 100.0,
        request.code
    )
}

fn build_junit_test_prompt(request: &TestGenerationRequest) -> String {
    format!(
        r#"Generate comprehensive JUnit 5 tests for {}.

Requirements:
1. Achieve at least {:.0}% code coverage
2. Include happy path tests
3. Include edge case tests (null, empty, boundary conditions)
4. Include exception tests with assertThrows
5. Use @Test annotation for test methods
6. Use @BeforeEach/@AfterEach for setup/teardown
7. Use descriptive test method names
8. Follow JUnit 5 best practices

Code to test:
```{}
{}
```

Generate ONLY the test class with proper imports and annotations."#,
        request.language,
        request.coverage_target * 100.0,
        request.language.to_lowercase(),
        request.code
    )
}

fn build_c_test_prompt(request: &TestGenerationRequest) -> String {
    format!(
        r#"Generate comprehensive C tests using Unity test framework.

Requirements:
1. Achieve at least {:.0}% code coverage
2. Include happy path tests
3. Include edge case tests (NULL, 0, boundary conditions)
4. Include error condition tests
5. Use TEST_ASSERT_* macros for assertions
6. Add clear comments for each test
7. Include setUp() and tearDown() functions
8. Follow Unity framework conventions

Code to test:
```c
{}
```

Generate ONLY the test code with proper includes and test functions."#,
        request.coverage_target * 100.0,
        request.code
    )
}

fn build_cpp_test_prompt(request: &TestGenerationRequest) -> String {
    format!(
        r#"Generate comprehensive C++ tests using Google Test framework.

Requirements:
1. Achieve at least {:.0}% code coverage
2. Include happy path tests
3. Include edge case tests (nullptr, empty, boundary conditions)
4. Include exception tests with ASSERT_THROW
5. Use TEST() and TEST_F() macros
6. Use fixtures for test setup/teardown
7. Use descriptive test names
8. Follow Google Test best practices

Code to test:
```cpp
{}
```

Generate ONLY the test code with proper includes and TEST macros."#,
        request.coverage_target * 100.0,
        request.code
    )
}

fn build_rspec_test_prompt(request: &TestGenerationRequest) -> String {
    format!(
        r#"Generate comprehensive RSpec tests for Ruby code.

Requirements:
1. Achieve at least {:.0}% code coverage
2. Include happy path tests
3. Include edge case tests (nil, empty, boundary conditions)
4. Include error condition tests
5. Use describe/context/it blocks
6. Use let for test data
7. Use clear, descriptive test names
8. Follow RSpec best practices

Code to test:
```ruby
{}
```

Generate ONLY the RSpec test code with proper requires and describe blocks."#,
        request.coverage_target * 100.0,
        request.code
    )
}

fn build_phpunit_test_prompt(request: &TestGenerationRequest) -> String {
    format!(
        r#"Generate comprehensive PHPUnit tests for PHP code.

Requirements:
1. Achieve at least {:.0}% code coverage
2. Include happy path tests
3. Include edge case tests (null, empty, boundary conditions)
4. Include exception tests with expectException
5. Extend PHPUnit\Framework\TestCase
6. Use setUp()/tearDown() for test fixtures
7. Use descriptive test method names (testXxx)
8. Follow PHPUnit best practices

Code to test:
```php
{}
```

Generate ONLY the test class with proper namespace and use statements."#,
        request.coverage_target * 100.0,
        request.code
    )
}

fn build_xctest_test_prompt(request: &TestGenerationRequest) -> String {
    format!(
        r#"Generate comprehensive XCTest tests for Swift code.

Requirements:
1. Achieve at least {:.0}% code coverage
2. Include happy path tests
3. Include edge case tests (nil, empty, boundary conditions)
4. Include error condition tests with throws
5. Extend XCTestCase
6. Use setUp()/tearDown() for test fixtures
7. Use descriptive test method names (testXxx)
8. Follow Swift testing best practices

Code to test:
```swift
{}
```

Generate ONLY the test class with proper imports and XCTest setup."#,
        request.coverage_target * 100.0,
        request.code
    )
}

// ============================================================================
// Test Counters
// ============================================================================

fn count_rust_test_functions(test_code: &str) -> usize {
    test_code.lines()
        .filter(|line| line.trim().starts_with("#[test]"))
        .count()
}

fn count_go_test_functions(test_code: &str) -> usize {
    test_code.lines()
        .filter(|line| line.trim().starts_with("func Test"))
        .count()
}

fn count_junit_test_methods(test_code: &str) -> usize {
    test_code.lines()
        .filter(|line| line.trim().starts_with("@Test"))
        .count()
}

fn count_c_test_functions(test_code: &str) -> usize {
    test_code.lines()
        .filter(|line| line.contains("TEST_ASSERT"))
        .count() / 2 // Approximate: each test has multiple assertions
}

fn count_gtest_tests(test_code: &str) -> usize {
    test_code.lines()
        .filter(|line| line.trim().starts_with("TEST(") || line.trim().starts_with("TEST_F("))
        .count()
}

fn count_rspec_tests(test_code: &str) -> usize {
    test_code.lines()
        .filter(|line| line.trim().starts_with("it ") || line.trim().starts_with("it("))
        .count()
}

fn count_phpunit_tests(test_code: &str) -> usize {
    test_code.lines()
        .filter(|line| line.trim().starts_with("public function test") || line.trim().starts_with("/** @test */"))
        .count()
}

fn count_xctest_tests(test_code: &str) -> usize {
    test_code.lines()
        .filter(|line| line.trim().starts_with("func test"))
        .count()
}

fn estimate_coverage(test_count: usize, code: &str) -> f32 {
    let code_lines = code.lines()
        .filter(|line| !line.trim().is_empty() && !line.trim().starts_with("//"))
        .count();
    
    if code_lines == 0 {
        return 0.0;
    }
    
    // Rough estimate: each test covers ~5-10 lines of code
    let estimated_coverage = (test_count as f32 * 7.0) / code_lines as f32;
    estimated_coverage.min(1.0)
}
