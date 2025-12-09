// File: src-tauri/src/code_intelligence/mod.rs
// Purpose: Code intelligence primitives - AST parsing, symbol extraction, references, definitions
// Dependencies: tree-sitter
// Last Updated: December 9, 2025

pub mod ast_parser;
pub mod symbol_extractor;
pub mod reference_finder;
pub mod scope_analyzer;

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub start_column: usize,
    pub end_column: usize,
    pub scope: Option<String>,
    pub signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SymbolKind {
    Function,
    Class,
    Method,
    Variable,
    Constant,
    Import,
    Module,
    Interface,
    Type,
    Enum,
    Struct,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reference {
    pub file_path: String,
    pub line: usize,
    pub column: usize,
    pub context: String,
    pub is_definition: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeInfo {
    pub scope_type: ScopeType,
    pub name: Option<String>,
    pub start_line: usize,
    pub end_line: usize,
    pub parent_scope: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScopeType {
    Global,
    Function,
    Class,
    Method,
    Block,
    Module,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTNode {
    pub node_type: String,
    pub start_line: usize,
    pub end_line: usize,
    pub start_column: usize,
    pub end_column: usize,
    pub text: Option<String>,
    pub children: Vec<ASTNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallHierarchy {
    pub function_name: String,
    pub file_path: String,
    pub line: usize,
    pub incoming_calls: Vec<CallSite>,
    pub outgoing_calls: Vec<CallSite>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallSite {
    pub caller_name: String,
    pub file_path: String,
    pub line: usize,
    pub column: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeHierarchy {
    pub type_name: String,
    pub file_path: String,
    pub parent_types: Vec<String>,
    pub child_types: Vec<String>,
}

/// Detect language from file extension
pub fn detect_language(file_path: &str) -> Option<&'static str> {
    let path = Path::new(file_path);
    let extension = path.extension()?.to_str()?;
    
    match extension {
        "rs" => Some("rust"),
        "py" => Some("python"),
        "js" | "jsx" | "mjs" => Some("javascript"),
        "ts" | "tsx" => Some("typescript"),
        "go" => Some("go"),
        "java" => Some("java"),
        "c" => Some("c"),
        "cpp" | "cc" | "cxx" => Some("cpp"),
        "cs" => Some("c_sharp"),
        "rb" => Some("ruby"),
        "php" => Some("php"),
        _ => None,
    }
}

/// Get tree-sitter Language for language name
pub fn get_tree_sitter_language(language: &str) -> Result<tree_sitter::Language, String> {
    match language {
        "rust" => Ok(tree_sitter_rust::language()),
        "python" => {
            // tree-sitter-python 0.23+ uses LANGUAGE constant
            let lang_fn: extern "C" fn() -> *const std::os::raw::c_void = 
                unsafe { std::mem::transmute(tree_sitter_python::LANGUAGE) };
            Ok(unsafe { tree_sitter::Language::from_raw(lang_fn() as *const tree_sitter::ffi::TSLanguage) })
        }
        "javascript" => {
            // tree-sitter-javascript 0.23+ uses LANGUAGE constant
            let lang_fn: extern "C" fn() -> *const std::os::raw::c_void = 
                unsafe { std::mem::transmute(tree_sitter_javascript::LANGUAGE) };
            Ok(unsafe { tree_sitter::Language::from_raw(lang_fn() as *const tree_sitter::ffi::TSLanguage) })
        }
        "typescript" => {
            // tree-sitter-typescript 0.23+ uses LANGUAGE_TYPESCRIPT constant
            let lang_fn: extern "C" fn() -> *const std::os::raw::c_void = 
                unsafe { std::mem::transmute(tree_sitter_typescript::LANGUAGE_TYPESCRIPT) };
            Ok(unsafe { tree_sitter::Language::from_raw(lang_fn() as *const tree_sitter::ffi::TSLanguage) })
        }
        _ => Err(format!("Unsupported language: {}", language)),
    }
}
