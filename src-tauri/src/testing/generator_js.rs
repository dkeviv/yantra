// File: src-tauri/src/testing/generator_js.rs
// Purpose: Jest test generator for JavaScript/TypeScript code
// Dependencies: LLM orchestration, tree-sitter-javascript
// Last Updated: December 4, 2025
//
// Generates comprehensive Jest tests for JavaScript/TypeScript:
// - Unit tests with expect() assertions
// - Mock functions and modules
// - Async/await test patterns
// - Edge cases and error handling
// Type-safe tests for TypeScript
//
// Usage:
// 1. Parse JavaScript/TypeScript source code
// 2. Extract functions/classes to test
// 3. Generate Jest tests with describe/it/expect
// 4. Save to .test.js or .spec.ts files

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use crate::testing::{TestGenerationRequest, TestGenerationResponse};
use crate::llm::{CodeGenerationRequest, LLMConfig};
use crate::llm::orchestrator::LLMOrchestrator;

/// Configuration for Jest test generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JestGeneratorConfig {
    /// Use TypeScript (*.test.ts) instead of JavaScript (*.test.js)
    pub use_typescript: bool,
    /// Include type annotations in tests
    pub include_types: bool,
    /// Generate mock implementations
    pub generate_mocks: bool,
    /// Test async functions
    pub test_async: bool,
    /// Include edge case tests
    pub include_edge_cases: bool,
    /// Code coverage target (0.0 to 1.0)
    pub coverage_target: f64,
}

impl Default for JestGeneratorConfig {
    fn default() -> Self {
        JestGeneratorConfig {
            use_typescript: false,
            include_types: true,
            generate_mocks: true,
            test_async: true,
            include_edge_cases: true,
            coverage_target: 0.9,
        }
    }
}

/// Generate Jest tests using LLM (unified interface)
pub async fn generate_jest_tests(
    request: TestGenerationRequest,
    llm_config: LLMConfig,
) -> Result<TestGenerationResponse, String> {
    let test_prompt = build_jest_test_prompt(&request);
    
    let llm_request = CodeGenerationRequest {
        intent: test_prompt,
        file_path: Some(format!("{}.test.{}", 
            request.file_path.replace(".js", "").replace(".ts", "").replace(".jsx", "").replace(".tsx", ""),
            if request.language.to_lowercase().contains("typescript") { "ts" } else { "js" }
        )),
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
    let test_count = count_jest_tests(test_code);
    
    Ok(TestGenerationResponse {
        tests: test_code.clone(),
        test_count,
        estimated_coverage: estimate_jest_coverage(test_count, &request.code),
        fixtures: Vec::new(),
    })
}

fn build_jest_test_prompt(request: &TestGenerationRequest) -> String {
    let lang = if request.language.to_lowercase().contains("typescript") {
        "TypeScript"
    } else {
        "JavaScript"
    };
    
    format!(
        r#"Generate comprehensive Jest tests for {} code.

Requirements:
1. Achieve at least {:.0}% code coverage
2. Include happy path tests
3. Include edge case tests (undefined, null, empty values, boundary conditions)
4. Include error condition tests
5. Use describe/it blocks with clear names
6. Use expect() assertions
7. Mock external dependencies with jest.mock()
8. Test async functions with async/await
9. Follow Jest best practices

Code to test:
```{}
{}
```

Generate ONLY the test code with proper imports and Jest syntax."#,
        lang,
        request.coverage_target * 100.0,
        lang.to_lowercase(),
        request.code
    )
}

fn count_jest_tests(test_code: &str) -> usize {
    test_code.lines()
        .filter(|line| {
            let trimmed = line.trim();
            trimmed.starts_with("it(") || 
            trimmed.starts_with("it.skip(") || 
            trimmed.starts_with("test(") ||
            trimmed.starts_with("test.skip(")
        })
        .count()
}

fn estimate_jest_coverage(test_count: usize, code: &str) -> f32 {
    let code_lines = code.lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty() && 
            !trimmed.starts_with("//") && 
            !trimmed.starts_with("/*")
        })
        .count();
    
    if code_lines == 0 {
        return 0.0;
    }
    
    // Estimate: each test covers ~5-10 lines of code
    let estimated_coverage = (test_count as f32 * 7.0) / code_lines as f32;
    estimated_coverage.min(1.0)
}

/// Jest test generator
pub struct JestGenerator {
    config: JestGeneratorConfig,
}

impl JestGenerator {
    /// Create new Jest generator with default config
    pub fn new() -> Self {
        JestGenerator {
            config: JestGeneratorConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: JestGeneratorConfig) -> Self {
        JestGenerator { config }
    }

    /// Generate Jest tests for JavaScript/TypeScript file
    /// 
    /// # Arguments
    /// * `source_file` - Path to source file
    /// * `source_code` - Source code content
    /// 
    /// # Returns
    /// Generated test code as string
    pub async fn generate_tests(
        &self,
        source_file: &Path,
        source_code: &str,
    ) -> Result<String, String> {
        // Parse source code to extract testable units
        let units = self.extract_testable_units(source_code)?;

        if units.is_empty() {
            return Ok(String::new());
        }

        // Generate test code
        let mut test_code = String::new();

        // Add imports
        test_code.push_str(&self.generate_imports(source_file));
        test_code.push_str("\n\n");

        // Generate describe blocks for each unit
        for unit in units {
            test_code.push_str(&self.generate_describe_block(&unit)?);
            test_code.push_str("\n\n");
        }

        Ok(test_code)
    }

    /// Extract testable units from source code
    fn extract_testable_units(&self, source_code: &str) -> Result<Vec<TestableUnit>, String> {
        let mut units = Vec::new();

        // Simple regex-based extraction (in production, use tree-sitter)
        
        // Extract functions
        let function_regex = regex::Regex::new(
            r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)"
        ).map_err(|e| format!("Regex error: {}", e))?;

        for cap in function_regex.captures_iter(source_code) {
            let name = cap.get(1).unwrap().as_str().to_string();
            let params = cap.get(2).unwrap().as_str().to_string();
            
            units.push(TestableUnit {
                kind: UnitKind::Function,
                name: name.clone(),
                params: Self::parse_params(&params),
                is_async: source_code.contains(&format!("async function {}", name)),
                is_exported: source_code.contains(&format!("export function {}", name)),
            });
        }

        // Extract arrow functions
        let arrow_regex = regex::Regex::new(
            r"(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>"
        ).map_err(|e| format!("Regex error: {}", e))?;

        for cap in arrow_regex.captures_iter(source_code) {
            let name = cap.get(1).unwrap().as_str().to_string();
            let params = cap.get(2).unwrap().as_str().to_string();
            
            units.push(TestableUnit {
                kind: UnitKind::Function,
                name: name.clone(),
                params: Self::parse_params(&params),
                is_async: source_code.contains(&format!("const {} = async", name)),
                is_exported: source_code.contains(&format!("export const {}", name)),
            });
        }

        // Extract classes
        let class_regex = regex::Regex::new(
            r"(?:export\s+)?class\s+(\w+)"
        ).map_err(|e| format!("Regex error: {}", e))?;

        for cap in class_regex.captures_iter(source_code) {
            let name = cap.get(1).unwrap().as_str().to_string();
            
            units.push(TestableUnit {
                kind: UnitKind::Class,
                name,
                params: Vec::new(),
                is_async: false,
                is_exported: true,
            });
        }

        Ok(units)
    }

    /// Parse function parameters
    fn parse_params(params_str: &str) -> Vec<String> {
        params_str
            .split(',')
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .map(|p| {
                // Remove type annotations for TypeScript
                if let Some(pos) = p.find(':') {
                    p[..pos].trim().to_string()
                } else {
                    p.to_string()
                }
            })
            .collect()
    }

    /// Generate imports for test file
    fn generate_imports(&self, source_file: &Path) -> String {
        let source_name = source_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("module");

        let relative_path = format!("./{}", source_name);

        format!(
            "import {{ {} }} from '{}';\n",
            "/* Import functions/classes to test */",
            relative_path
        )
    }

    /// Generate describe block for a testable unit
    fn generate_describe_block(&self, unit: &TestableUnit) -> Result<String, String> {
        let mut code = String::new();

        match unit.kind {
            UnitKind::Function => {
                code.push_str(&format!("describe('{}', () => {{\n", unit.name));
                
                // Basic test
                code.push_str(&self.generate_basic_test(unit));
                
                // Edge cases
                if self.config.include_edge_cases {
                    code.push_str(&self.generate_edge_case_tests(unit));
                }
                
                // Async tests
                if unit.is_async && self.config.test_async {
                    code.push_str(&self.generate_async_tests(unit));
                }
                
                code.push_str("});");
            }
            UnitKind::Class => {
                code.push_str(&format!("describe('{}', () => {{\n", unit.name));
                code.push_str(&self.generate_class_tests(unit));
                code.push_str("});");
            }
        }

        Ok(code)
    }

    /// Generate basic test case
    fn generate_basic_test(&self, unit: &TestableUnit) -> String {
        let test_name = format!("should work correctly with valid inputs");
        
        let mut code = String::new();
        code.push_str(&format!("  it('{}', ", test_name));
        
        if unit.is_async {
            code.push_str("async ");
        }
        
        code.push_str("() => {\n");
        
        // Generate test body
        if unit.params.is_empty() {
            code.push_str(&format!("    const result = {}();\n", unit.name));
        } else {
            // Generate sample arguments
            let args = unit.params.iter()
                .map(|_| "/* TODO: provide test value */")
                .collect::<Vec<_>>()
                .join(", ");
            
            if unit.is_async {
                code.push_str(&format!("    const result = await {}({});\n", unit.name, args));
            } else {
                code.push_str(&format!("    const result = {}({});\n", unit.name, args));
            }
        }
        
        code.push_str("    expect(result).toBeDefined();\n");
        code.push_str("    // TODO: Add specific assertions\n");
        code.push_str("  });\n\n");
        
        code
    }

    /// Generate edge case tests
    fn generate_edge_case_tests(&self, unit: &TestableUnit) -> String {
        let mut code = String::new();
        
        // Null/undefined test
        code.push_str("  it('should handle null/undefined inputs', ");
        if unit.is_async {
            code.push_str("async ");
        }
        code.push_str("() => {\n");
        code.push_str(&format!("    // TODO: Test {} with null/undefined\n", unit.name));
        code.push_str("  });\n\n");
        
        // Empty input test
        if !unit.params.is_empty() {
            code.push_str("  it('should handle empty inputs', ");
            if unit.is_async {
                code.push_str("async ");
            }
            code.push_str("() => {\n");
            code.push_str(&format!("    // TODO: Test {} with empty values\n", unit.name));
            code.push_str("  });\n\n");
        }
        
        // Error test
        code.push_str("  it('should throw error for invalid inputs', ");
        if unit.is_async {
            code.push_str("async ");
        }
        code.push_str("() => {\n");
        code.push_str("    expect(() => {\n");
        code.push_str(&format!("      {}(/* invalid input */);\n", unit.name));
        code.push_str("    }).toThrow();\n");
        code.push_str("  });\n\n");
        
        code
    }

    /// Generate async-specific tests
    fn generate_async_tests(&self, unit: &TestableUnit) -> String {
        let mut code = String::new();
        
        code.push_str("  it('should resolve successfully', async () => {\n");
        code.push_str(&format!("    await expect({}()).resolves.toBeDefined();\n", unit.name));
        code.push_str("  });\n\n");
        
        code.push_str("  it('should reject on error', async () => {\n");
        code.push_str(&format!("    await expect({}(/* bad input */)).rejects.toThrow();\n", unit.name));
        code.push_str("  });\n\n");
        
        code
    }

    /// Generate class tests
    fn generate_class_tests(&self, unit: &TestableUnit) -> String {
        let mut code = String::new();
        
        code.push_str("  it('should instantiate correctly', () => {\n");
        code.push_str(&format!("    const instance = new {}();\n", unit.name));
        code.push_str("    expect(instance).toBeInstanceOf(");
        code.push_str(&unit.name);
        code.push_str(");\n");
        code.push_str("  });\n\n");
        
        code.push_str("  // TODO: Add method tests\n");
        
        code
    }

    /// Generate test file path for source file
    pub fn get_test_file_path(&self, source_file: &Path) -> PathBuf {
        let extension = if self.config.use_typescript {
            "test.ts"
        } else {
            "test.js"
        };
        
        let file_stem = source_file.file_stem().and_then(|s| s.to_str()).unwrap_or("test");
        let parent = source_file.parent().unwrap_or_else(|| Path::new("."));
        
        parent.join(format!("{}.{}", file_stem, extension))
    }

    /// Generate Jest config file (jest.config.js)
    pub fn generate_jest_config(&self, use_typescript: bool) -> String {
        if use_typescript {
            r#"module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/src'],
  testMatch: ['**/__tests__/**/*.ts', '**/?(*.)+(spec|test).ts'],
  transform: {
    '^.+\\.ts$': 'ts-jest',
  },
  collectCoverageFrom: [
    'src/**/*.ts',
    '!src/**/*.d.ts',
    '!src/**/*.test.ts',
  ],
  coverageThreshold: {
    global: {
      branches: 90,
      functions: 90,
      lines: 90,
      statements: 90,
    },
  },
};
"#.to_string()
        } else {
            r#"module.exports = {
  testEnvironment: 'node',
  roots: ['<rootDir>/src'],
  testMatch: ['**/__tests__/**/*.js', '**/?(*.)+(spec|test).js'],
  collectCoverageFrom: [
    'src/**/*.js',
    '!src/**/*.test.js',
  ],
  coverageThreshold: {
    global: {
      branches: 90,
      functions: 90,
      lines: 90,
      statements: 90,
    },
  },
};
"#.to_string()
        }
    }
}

impl Default for JestGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Testable unit extracted from source code
#[derive(Debug, Clone)]
struct TestableUnit {
    kind: UnitKind,
    name: String,
    params: Vec<String>,
    is_async: bool,
    is_exported: bool,
}

#[derive(Debug, Clone, PartialEq)]
enum UnitKind {
    Function,
    Class,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_creation() {
        let generator = JestGenerator::new();
        assert!(!generator.config.use_typescript);
        assert!(generator.config.generate_mocks);
    }

    #[tokio::test]
    async fn test_generate_function_tests() {
        let generator = JestGenerator::new();
        
        let source = r#"
export function add(a, b) {
    return a + b;
}

export async function fetchData(url) {
    const response = await fetch(url);
    return response.json();
}
        "#;
        
        let result = generator.generate_tests(Path::new("math.js"), source).await;
        assert!(result.is_ok());
        
        let tests = result.unwrap();
        assert!(tests.contains("describe('add'"));
        assert!(tests.contains("describe('fetchData'"));
        assert!(tests.contains("async () =>"));
    }

    #[test]
    fn test_parse_params() {
        let params = "a, b, c";
        let parsed = JestGenerator::parse_params(params);
        assert_eq!(parsed, vec!["a", "b", "c"]);
        
        let typed_params = "a: number, b: string";
        let parsed_typed = JestGenerator::parse_params(typed_params);
        assert_eq!(parsed_typed, vec!["a", "b"]);
    }

    #[test]
    fn test_get_test_file_path() {
        let generator = JestGenerator::new();
        let source = Path::new("src/utils/math.js");
        let test_path = generator.get_test_file_path(source);
        
        assert_eq!(test_path, Path::new("src/utils/math.test.js"));
    }

    #[test]
    fn test_typescript_config() {
        let mut config = JestGeneratorConfig::default();
        config.use_typescript = true;
        
        let generator = JestGenerator::with_config(config);
        let test_path = generator.get_test_file_path(Path::new("src/app.ts"));
        
        assert_eq!(test_path, Path::new("src/app.test.ts"));
    }
}
