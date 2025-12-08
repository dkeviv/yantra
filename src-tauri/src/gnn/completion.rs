// File: src-tauri/src/gnn/completion.rs
// Purpose: Code completion provider using Tree-sitter AST and GNN context
// Dependencies: tree-sitter, gnn
// Last Updated: December 7, 2025

use super::{GNNEngine, CodeNode, NodeType};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tree_sitter::{Parser, Point};

/// Completion item for Monaco Editor
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompletionItem {
    /// The label of this completion item
    pub label: String,
    /// The kind of this completion item (Method, Function, Variable, etc.)
    pub kind: CompletionKind,
    /// A human-readable string with additional information
    pub detail: Option<String>,
    /// Documentation for this completion item
    pub documentation: Option<String>,
    /// The text to insert
    pub insert_text: String,
    /// Snippet format (true if using $1, $2, etc.)
    pub insert_text_as_snippet: bool,
    /// Sort order for this item
    pub sort_text: Option<String>,
    /// Filter text for matching
    pub filter_text: Option<String>,
}

/// Completion kind matching Monaco's CompletionItemKind
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum CompletionKind {
    Method = 0,
    Function = 1,
    Constructor = 2,
    Field = 3,
    Variable = 4,
    Class = 5,
    Struct = 6,
    Interface = 7,
    Module = 8,
    Property = 9,
    Event = 10,
    Operator = 11,
    Unit = 12,
    Value = 13,
    Constant = 14,
    Enum = 15,
    EnumMember = 16,
    Keyword = 17,
    Text = 18,
    Color = 19,
    File = 20,
    Reference = 21,
    Customcolor = 22,
    Folder = 23,
    TypeParameter = 24,
    User = 25,
    Issue = 26,
    Snippet = 27,
}

/// Request for code completions
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompletionRequest {
    /// File path being edited
    pub file_path: String,
    /// Current file content
    pub content: String,
    /// Cursor line (1-indexed)
    pub line: usize,
    /// Cursor column (0-indexed)
    pub column: usize,
    /// Project root path
    pub project_path: String,
    /// Language identifier (python, javascript, rust, etc.)
    pub language: String,
}

/// Code completion provider
pub struct CompletionProvider {
    gnn_engine: Option<GNNEngine>,
}

impl CompletionProvider {
    /// Create a new completion provider
    pub fn new(project_path: Option<&Path>) -> Result<Self, String> {
        let gnn_engine = if let Some(path) = project_path {
            let db_path = path.join(".yantra").join("graph.db");
            if db_path.exists() {
                let mut engine = GNNEngine::new(&db_path)?;
                engine.load().ok();
                Some(engine)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self { gnn_engine })
    }

    /// Get completions for a specific position
    pub fn get_completions(&self, request: &CompletionRequest) -> Result<Vec<CompletionItem>, String> {
        let mut completions = Vec::new();

        // 1. Get Tree-sitter AST-based completions
        let ast_completions = self.get_ast_completions(request)?;
        completions.extend(ast_completions);

        // 2. Get GNN-based context completions (imports, dependencies)
        if let Some(ref engine) = self.gnn_engine {
            let gnn_completions = self.get_gnn_completions(request, engine)?;
            completions.extend(gnn_completions);
        }

        // 3. Get language-specific keyword completions
        let keyword_completions = self.get_keyword_completions(&request.language);
        completions.extend(keyword_completions);

        // Deduplicate by label
        completions.sort_by(|a, b| a.label.cmp(&b.label));
        completions.dedup_by(|a, b| a.label == b.label);

        Ok(completions)
    }

    /// Get completions from Tree-sitter AST (local scope)
    fn get_ast_completions(&self, request: &CompletionRequest) -> Result<Vec<CompletionItem>, String> {
        let mut completions = Vec::new();
        let language = &request.language;

        // Parse the current file with Tree-sitter
        let mut parser = Parser::new();
        let tree_sitter_lang = match language.as_str() {
            "python" => {
                let lang_fn: extern "C" fn() -> *const std::os::raw::c_void = 
                    unsafe { std::mem::transmute(tree_sitter_python::LANGUAGE) };
                unsafe { tree_sitter::Language::from_raw(lang_fn() as *const tree_sitter::ffi::TSLanguage) }
            }
            "javascript" | "typescript" => {
                let lang_fn: extern "C" fn() -> *const std::os::raw::c_void = 
                    unsafe { std::mem::transmute(tree_sitter_javascript::LANGUAGE) };
                unsafe { tree_sitter::Language::from_raw(lang_fn() as *const tree_sitter::ffi::TSLanguage) }
            }
            "rust" => {
                tree_sitter_rust::language()
            }
            _ => return Ok(completions), // Unsupported language for now
        };

        parser.set_language(&tree_sitter_lang)
            .map_err(|e| format!("Failed to set language: {}", e))?;

        let tree = parser.parse(&request.content, None)
            .ok_or_else(|| "Failed to parse code".to_string())?;

        // Extract symbols from AST
        let root = tree.root_node();
        self.extract_symbols_from_ast(&root, &request.content, &mut completions, language)?;

        Ok(completions)
    }

    /// Extract symbols from AST node recursively
    fn extract_symbols_from_ast(
        &self,
        node: &tree_sitter::Node,
        code: &str,
        completions: &mut Vec<CompletionItem>,
        language: &str,
    ) -> Result<(), String> {
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            match (language, child.kind()) {
                ("python", "function_definition") => {
                    if let Some(name_node) = child.child_by_field_name("name") {
                        let name = self.get_node_text(&name_node, code);
                        let params = self.extract_function_params(&child, code, language);
                        
                        completions.push(CompletionItem {
                            label: name.clone(),
                            kind: CompletionKind::Function,
                            detail: Some(format!("def {}({})", name, params)),
                            documentation: self.extract_docstring(&child, code),
                            insert_text: format!("{}($0)", name),
                            insert_text_as_snippet: true,
                            sort_text: Some(format!("0_{}", name)),
                            filter_text: Some(name.clone()),
                        });
                    }
                }
                ("python", "class_definition") => {
                    if let Some(name_node) = child.child_by_field_name("name") {
                        let name = self.get_node_text(&name_node, code);
                        
                        completions.push(CompletionItem {
                            label: name.clone(),
                            kind: CompletionKind::Class,
                            detail: Some(format!("class {}", name)),
                            documentation: self.extract_docstring(&child, code),
                            insert_text: name.clone(),
                            insert_text_as_snippet: false,
                            sort_text: Some(format!("0_{}", name)),
                            filter_text: Some(name.clone()),
                        });
                    }
                }
                ("javascript" | "typescript", "function_declaration") => {
                    if let Some(name_node) = child.child_by_field_name("name") {
                        let name = self.get_node_text(&name_node, code);
                        let params = self.extract_function_params(&child, code, language);
                        
                        completions.push(CompletionItem {
                            label: name.clone(),
                            kind: CompletionKind::Function,
                            detail: Some(format!("function {}({})", name, params)),
                            documentation: None,
                            insert_text: format!("{}($0)", name),
                            insert_text_as_snippet: true,
                            sort_text: Some(format!("0_{}", name)),
                            filter_text: Some(name.clone()),
                        });
                    }
                }
                ("rust", "function_item") => {
                    if let Some(name_node) = child.child_by_field_name("name") {
                        let name = self.get_node_text(&name_node, code);
                        
                        completions.push(CompletionItem {
                            label: name.clone(),
                            kind: CompletionKind::Function,
                            detail: Some(format!("fn {}", name)),
                            documentation: None,
                            insert_text: format!("{}($0)", name),
                            insert_text_as_snippet: true,
                            sort_text: Some(format!("0_{}", name)),
                            filter_text: Some(name.clone()),
                        });
                    }
                }
                _ => {}
            }

            // Recursively process children
            self.extract_symbols_from_ast(&child, code, completions, language)?;
        }

        Ok(())
    }

    /// Get completions from GNN (imported symbols, project-wide context)
    fn get_gnn_completions(
        &self,
        request: &CompletionRequest,
        engine: &GNNEngine,
    ) -> Result<Vec<CompletionItem>, String> {
        let mut completions = Vec::new();

        // Get all nodes in the project
        let all_nodes = engine.get_all_nodes();

        for node in all_nodes.iter().take(100) { // Limit to 100 for performance
            let completion = match node.node_type {
                NodeType::Function => CompletionItem {
                    label: node.name.clone(),
                    kind: CompletionKind::Function,
                    detail: Some(format!("from {}", node.file_path)),
                    documentation: node.docstring.clone(),
                    insert_text: format!("{}($0)", node.name),
                    insert_text_as_snippet: true,
                    sort_text: Some(format!("1_{}", node.name)),
                    filter_text: Some(node.name.clone()),
                },
                NodeType::Class => CompletionItem {
                    label: node.name.clone(),
                    kind: CompletionKind::Class,
                    detail: Some(format!("from {}", node.file_path)),
                    documentation: node.docstring.clone(),
                    insert_text: node.name.clone(),
                    insert_text_as_snippet: false,
                    sort_text: Some(format!("1_{}", node.name)),
                    filter_text: Some(node.name.clone()),
                },
                NodeType::Module => CompletionItem {
                    label: node.name.clone(),
                    kind: CompletionKind::Module,
                    detail: Some(format!("module {}", node.file_path)),
                    documentation: None,
                    insert_text: node.name.clone(),
                    insert_text_as_snippet: false,
                    sort_text: Some(format!("2_{}", node.name)),
                    filter_text: Some(node.name.clone()),
                },
                _ => continue,
            };

            completions.push(completion);
        }

        Ok(completions)
    }

    /// Get language-specific keyword completions
    fn get_keyword_completions(&self, language: &str) -> Vec<CompletionItem> {
        let keywords = match language {
            "python" => vec![
                ("if", "if ${1:condition}:\n\t$0"),
                ("else", "else:\n\t$0"),
                ("elif", "elif ${1:condition}:\n\t$0"),
                ("for", "for ${1:item} in ${2:iterable}:\n\t$0"),
                ("while", "while ${1:condition}:\n\t$0"),
                ("def", "def ${1:function_name}($2):\n\t$0"),
                ("class", "class ${1:ClassName}:\n\t$0"),
                ("try", "try:\n\t$1\nexcept ${2:Exception}:\n\t$0"),
                ("import", "import ${1:module}"),
                ("from", "from ${1:module} import ${2:name}"),
                ("return", "return $0"),
                ("async", "async "),
                ("await", "await "),
            ],
            "javascript" | "typescript" => vec![
                ("if", "if (${1:condition}) {\n\t$0\n}"),
                ("else", "else {\n\t$0\n}"),
                ("for", "for (${1:let i = 0}; ${2:i < length}; ${3:i++}) {\n\t$0\n}"),
                ("while", "while (${1:condition}) {\n\t$0\n}"),
                ("function", "function ${1:name}($2) {\n\t$0\n}"),
                ("const", "const ${1:name} = $0"),
                ("let", "let ${1:name} = $0"),
                ("var", "var ${1:name} = $0"),
                ("return", "return $0"),
                ("async", "async "),
                ("await", "await "),
            ],
            "rust" => vec![
                ("fn", "fn ${1:name}($2) {\n\t$0\n}"),
                ("let", "let ${1:name} = $0;"),
                ("if", "if ${1:condition} {\n\t$0\n}"),
                ("else", "else {\n\t$0\n}"),
                ("for", "for ${1:item} in ${2:iterator} {\n\t$0\n}"),
                ("while", "while ${1:condition} {\n\t$0\n}"),
                ("match", "match ${1:value} {\n\t$0\n}"),
                ("impl", "impl ${1:Type} {\n\t$0\n}"),
            ],
            _ => vec![],
        };

        keywords
            .into_iter()
            .map(|(label, snippet)| CompletionItem {
                label: label.to_string(),
                kind: CompletionKind::Keyword,
                detail: Some(format!("{} keyword", language)),
                documentation: None,
                insert_text: snippet.to_string(),
                insert_text_as_snippet: true,
                sort_text: Some(format!("3_{}", label)),
                filter_text: Some(label.to_string()),
            })
            .collect()
    }

    /// Extract function parameters
    fn extract_function_params(&self, node: &tree_sitter::Node, code: &str, language: &str) -> String {
        if let Some(params_node) = node.child_by_field_name("parameters") {
            self.get_node_text(&params_node, code)
                .trim_start_matches('(')
                .trim_end_matches(')')
                .to_string()
        } else {
            String::new()
        }
    }

    /// Extract docstring from function or class
    fn extract_docstring(&self, node: &tree_sitter::Node, code: &str) -> Option<String> {
        if let Some(body_node) = node.child_by_field_name("body") {
            let mut cursor = body_node.walk();
            for child in body_node.children(&mut cursor) {
                if child.kind() == "expression_statement" {
                    if let Some(string_node) = child.child(0) {
                        if string_node.kind() == "string" {
                            return Some(
                                self.get_node_text(&string_node, code)
                                    .trim_matches('"')
                                    .trim_matches('\'')
                                    .to_string()
                            );
                        }
                    }
                }
            }
        }
        None
    }

    /// Get text content of a node
    fn get_node_text(&self, node: &tree_sitter::Node, code: &str) -> String {
        code[node.byte_range()].to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_completions() {
        let provider = CompletionProvider::new(None).unwrap();
        
        let request = CompletionRequest {
            file_path: "test.py".to_string(),
            content: r#"
def hello_world():
    """Say hello"""
    print("Hello")

class MyClass:
    """A test class"""
    pass

# Complete here: hel
"#.to_string(),
            line: 10,
            column: 18,
            project_path: "/tmp/test".to_string(),
            language: "python".to_string(),
        };

        let completions = provider.get_completions(&request).unwrap();
        
        // Should have function, class, and keyword completions
        assert!(!completions.is_empty());
        assert!(completions.iter().any(|c| c.label == "hello_world"));
        assert!(completions.iter().any(|c| c.label == "MyClass"));
    }

    #[test]
    fn test_keyword_completions() {
        let provider = CompletionProvider::new(None).unwrap();
        let keywords = provider.get_keyword_completions("python");
        
        assert!(keywords.iter().any(|k| k.label == "if"));
        assert!(keywords.iter().any(|k| k.label == "def"));
        assert!(keywords.iter().any(|k| k.label == "class"));
    }
}
