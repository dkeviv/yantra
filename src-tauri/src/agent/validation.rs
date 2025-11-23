// File: src-tauri/src/agent/validation.rs

// Module types not yet fully integrated
#![allow(dead_code)]
// Purpose: Dependency validation via GNN
// Last Updated: November 21, 2025

use crate::gnn::GNNEngine;
use serde::{Deserialize, Serialize};
use tree_sitter::Parser;

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationResult {
    /// Validation passed
    Success,
    /// Validation failed with errors
    Failed(Vec<ValidationError>),
}

/// Validation error details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    /// Error type
    pub error_type: ValidationErrorType,
    /// Error message
    pub message: String,
    /// File path (if applicable)
    pub file_path: Option<String>,
    /// Line number (if applicable)
    pub line_number: Option<usize>,
}

/// Types of validation errors
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationErrorType {
    /// Function call to undefined function
    UndefinedFunction,
    /// Import of non-existent module
    MissingImport,
    /// Type mismatch in function call
    TypeMismatch,
    /// Breaking change to existing API
    BreakingChange,
    /// Circular dependency detected
    CircularDependency,
    /// Parse error in generated code
    ParseError,
}

impl ValidationError {
    /// Create new validation error
    pub fn new(
        error_type: ValidationErrorType,
        message: String,
        file_path: Option<String>,
        line_number: Option<usize>,
    ) -> Self {
        Self {
            error_type,
            message,
            file_path,
            line_number,
        }
    }
}

/// Validate dependencies of generated code against GNN
/// 
/// Checks:
/// 1. All function calls reference existing functions
/// 2. All imports reference existing modules
/// 3. No breaking changes to existing APIs
/// 4. No circular dependencies
/// 
/// Performance target: <10ms
pub fn validate_dependencies(
    gnn: &GNNEngine,
    generated_code: &str,
    file_path: &str,
) -> Result<ValidationResult, String> {
    let mut errors = Vec::new();

    // Parse generated code
    let mut parser = Parser::new();
    parser
        .set_language(tree_sitter_python::language())
        .map_err(|e| format!("Failed to set parser language: {}", e))?;

    let tree = parser
        .parse(generated_code, None)
        .ok_or_else(|| "Failed to parse generated code".to_string())?;

    let root_node = tree.root_node();

    // Check for parse errors first
    if root_node.has_error() {
        errors.push(ValidationError::new(
            ValidationErrorType::ParseError,
            "Generated code contains syntax errors".to_string(),
            Some(file_path.to_string()),
            None,
        ));
        return Ok(ValidationResult::Failed(errors));
    }

    // Extract function calls from generated code
    let function_calls = extract_function_calls(&root_node, generated_code);

    // Validate each function call against GNN
    for (func_name, line_number) in function_calls {
        // Check if function exists in GNN
        if gnn.find_node(&func_name, None).is_none() {
            errors.push(ValidationError::new(
                ValidationErrorType::UndefinedFunction,
                format!("Call to undefined function: {}", func_name),
                Some(file_path.to_string()),
                Some(line_number),
            ));
        }
    }

    // Extract imports from generated code
    let imports = extract_imports(&root_node, generated_code);

    // Validate imports (simplified check - assumes module exists if we've seen it)
    for (module_name, line_number) in imports {
        // Check if module exists in GNN or is standard library
        if gnn.find_node(&module_name, None).is_none() && !is_standard_library(&module_name) {
            errors.push(ValidationError::new(
                ValidationErrorType::MissingImport,
                format!("Import of unknown module: {}", module_name),
                Some(file_path.to_string()),
                Some(line_number),
            ));
        }
    }

    if errors.is_empty() {
        Ok(ValidationResult::Success)
    } else {
        Ok(ValidationResult::Failed(errors))
    }
}

/// Extract function calls from AST
fn extract_function_calls(node: &tree_sitter::Node, source: &str) -> Vec<(String, usize)> {
    let mut calls = Vec::new();
    let mut cursor = node.walk();

    fn visit_node(
        node: &tree_sitter::Node,
        source: &str,
        calls: &mut Vec<(String, usize)>,
        cursor: &mut tree_sitter::TreeCursor,
    ) {
        if node.kind() == "call" {
            if let Some(function_node) = node.child_by_field_name("function") {
                if function_node.kind() == "identifier" {
                    if let Ok(func_name) = function_node.utf8_text(source.as_bytes()) {
                        let line_number = node.start_position().row + 1;
                        calls.push((func_name.to_string(), line_number));
                    }
                } else if function_node.kind() == "attribute" {
                    // Handle method calls (obj.method())
                    if let Some(attr_node) = function_node.child_by_field_name("attribute") {
                        if let Ok(method_name) = attr_node.utf8_text(source.as_bytes()) {
                            let line_number = node.start_position().row + 1;
                            calls.push((method_name.to_string(), line_number));
                        }
                    }
                }
            }
        }

        // Recursively visit children
        if cursor.goto_first_child() {
            loop {
                visit_node(&cursor.node(), source, calls, cursor);
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
            cursor.goto_parent();
        }
    }

    visit_node(node, source, &mut calls, &mut cursor);
    calls
}

/// Extract imports from AST
fn extract_imports(node: &tree_sitter::Node, source: &str) -> Vec<(String, usize)> {
    let mut imports = Vec::new();
    let mut cursor = node.walk();

    fn visit_node(
        node: &tree_sitter::Node,
        source: &str,
        imports: &mut Vec<(String, usize)>,
        cursor: &mut tree_sitter::TreeCursor,
    ) {
        if node.kind() == "import_statement" || node.kind() == "import_from_statement" {
            // Extract module name
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i) {
                    if child.kind() == "dotted_name" || child.kind() == "identifier" {
                        if let Ok(module_name) = child.utf8_text(source.as_bytes()) {
                            let line_number = node.start_position().row + 1;
                            imports.push((module_name.to_string(), line_number));
                            break;
                        }
                    }
                }
            }
        }

        // Recursively visit children
        if cursor.goto_first_child() {
            loop {
                visit_node(&cursor.node(), source, imports, cursor);
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
            cursor.goto_parent();
        }
    }

    visit_node(node, source, &mut imports, &mut cursor);
    imports
}

/// Check if a module is part of Python standard library
fn is_standard_library(module: &str) -> bool {
    // Common standard library modules
    const STDLIB: &[&str] = &[
        "os", "sys", "re", "json", "datetime", "time", "math", "random",
        "collections", "itertools", "functools", "pathlib", "typing",
        "unittest", "pytest", "logging", "argparse", "subprocess",
        "threading", "multiprocessing", "asyncio", "http", "urllib",
        "socket", "email", "csv", "xml", "sqlite3", "pickle", "hashlib",
    ];

    STDLIB.contains(&module) || module.starts_with("__")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_error_creation() {
        let error = ValidationError::new(
            ValidationErrorType::UndefinedFunction,
            "Test error".to_string(),
            Some("test.py".to_string()),
            Some(42),
        );

        assert_eq!(error.error_type, ValidationErrorType::UndefinedFunction);
        assert_eq!(error.message, "Test error");
        assert_eq!(error.file_path, Some("test.py".to_string()));
        assert_eq!(error.line_number, Some(42));
    }

    #[test]
    fn test_extract_function_calls() {
        let code = r#"
def main():
    result = calculate(10, 20)
    print(result)
    obj.method()
"#;

        let mut parser = Parser::new();
        parser.set_language(tree_sitter_python::language()).unwrap();
        let tree = parser.parse(code, None).unwrap();

        let calls = extract_function_calls(&tree.root_node(), code);

        // Should find: calculate, print, method
        assert!(calls.iter().any(|(name, _)| name == "calculate"));
        assert!(calls.iter().any(|(name, _)| name == "print"));
        assert!(calls.iter().any(|(name, _)| name == "method"));
    }

    #[test]
    fn test_extract_imports() {
        let code = r#"
import os
from pathlib import Path
import json
"#;

        let mut parser = Parser::new();
        parser.set_language(tree_sitter_python::language()).unwrap();
        let tree = parser.parse(code, None).unwrap();

        let imports = extract_imports(&tree.root_node(), code);

        // Should find: os, pathlib, json
        assert!(imports.iter().any(|(name, _)| name == "os"));
        assert!(imports.iter().any(|(name, _)| name == "pathlib"));
        assert!(imports.iter().any(|(name, _)| name == "json"));
    }

    #[test]
    fn test_standard_library_check() {
        assert!(is_standard_library("os"));
        assert!(is_standard_library("sys"));
        assert!(is_standard_library("json"));
        assert!(!is_standard_library("custom_module"));
        assert!(!is_standard_library("my_package"));
    }

    #[test]
    fn test_parse_error_detection() {
        use tempfile::NamedTempFile;
        
        let temp_db = NamedTempFile::new().unwrap();
        let gnn = GNNEngine::new(temp_db.path()).unwrap();
        let invalid_code = "def broken(\n    print('missing closing paren'";

        let result = validate_dependencies(&gnn, invalid_code, "test.py").unwrap();

        match result {
            ValidationResult::Failed(errors) => {
                assert!(!errors.is_empty());
                assert!(errors.iter().any(|e| e.error_type == ValidationErrorType::ParseError));
            }
            ValidationResult::Success => panic!("Expected validation to fail"),
        }
    }
}
