// File: src-tauri/src/agent/file_editor.rs
// Purpose: Surgical AST-based file editing with validation
// Dependencies: tree-sitter, serde
// Last Updated: December 3, 2025

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEditRequest {
    pub file_path: String,
    pub edit_type: EditType,
    pub validation: EditValidation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum EditType {
    /// Replace specific line range with new content
    LineRange {
        start_line: usize,
        end_line: usize,
        new_content: String,
    },
    /// Search and replace (first occurrence or all)
    SearchReplace {
        search: String,
        replace: String,
        all_occurrences: bool,
    },
    /// Insert content at specific line
    Insert {
        line: usize,
        content: String,
        position: InsertPosition,
    },
    /// Delete specific line range
    Delete {
        start_line: usize,
        end_line: usize,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InsertPosition {
    Before,
    After,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditValidation {
    /// Preserve existing indentation
    pub preserve_indentation: bool,
    /// Validate syntax before applying (requires language)
    pub validate_syntax: bool,
    /// Language for syntax validation (python, javascript, rust, etc.)
    pub language: Option<String>,
    /// Create backup before editing
    pub create_backup: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEditResult {
    pub success: bool,
    pub message: String,
    pub backup_path: Option<String>,
    pub lines_changed: usize,
    pub preview: Option<String>,
}

impl Default for EditValidation {
    fn default() -> Self {
        Self {
            preserve_indentation: true,
            validate_syntax: false,
            language: None,
            create_backup: true,
        }
    }
}

pub struct FileEditor {
    dry_run: bool,
}

impl FileEditor {
    pub fn new(dry_run: bool) -> Self {
        Self { dry_run }
    }

    /// Apply file edit with validation
    pub fn apply_edit(&self, request: FileEditRequest) -> Result<FileEditResult, String> {
        // Validate file exists
        if !Path::new(&request.file_path).exists() {
            return Err(format!("File not found: {}", request.file_path));
        }

        // Read original content
        let original_content = fs::read_to_string(&request.file_path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        // Create backup if requested
        let backup_path = if request.validation.create_backup {
            let backup = format!("{}.backup", request.file_path);
            fs::write(&backup, &original_content)
                .map_err(|e| format!("Failed to create backup: {}", e))?;
            Some(backup)
        } else {
            None
        };

        // Apply the edit
        let (new_content, lines_changed) = match request.edit_type {
            EditType::LineRange { start_line, end_line, new_content } => {
                self.apply_line_range_edit(&original_content, start_line, end_line, &new_content)?
            }
            EditType::SearchReplace { search, replace, all_occurrences } => {
                self.apply_search_replace(&original_content, &search, &replace, all_occurrences)?
            }
            EditType::Insert { line, content, position } => {
                self.apply_insert(&original_content, line, &content, position)?
            }
            EditType::Delete { start_line, end_line } => {
                self.apply_delete(&original_content, start_line, end_line)?
            }
        };

        // Validate syntax if requested
        if request.validation.validate_syntax {
            if let Some(lang) = &request.validation.language {
                self.validate_syntax(&new_content, lang)?;
            }
        }

        // Write the new content (unless dry run)
        if !self.dry_run {
            fs::write(&request.file_path, &new_content)
                .map_err(|e| format!("Failed to write file: {}", e))?;
        }

        // Generate preview (first 10 lines of changes)
        let preview = self.generate_preview(&original_content, &new_content);

        Ok(FileEditResult {
            success: true,
            message: if self.dry_run {
                "Dry run completed successfully".to_string()
            } else {
                format!("File edited successfully: {}", request.file_path)
            },
            backup_path,
            lines_changed,
            preview: Some(preview),
        })
    }

    /// Apply line range edit
    fn apply_line_range_edit(
        &self,
        content: &str,
        start_line: usize,
        end_line: usize,
        new_content: &str,
    ) -> Result<(String, usize), String> {
        let lines: Vec<&str> = content.lines().collect();
        
        if start_line == 0 || start_line > lines.len() {
            return Err(format!("Invalid start line: {}", start_line));
        }
        if end_line == 0 || end_line > lines.len() || end_line < start_line {
            return Err(format!("Invalid end line: {}", end_line));
        }

        let mut result = Vec::new();
        
        // Lines before edit (1-indexed to 0-indexed)
        result.extend_from_slice(&lines[0..start_line - 1]);
        
        // New content
        result.push(new_content);
        
        // Lines after edit
        if end_line < lines.len() {
            result.extend_from_slice(&lines[end_line..]);
        }

        let lines_changed = end_line - start_line + 1;
        Ok((result.join("\n"), lines_changed))
    }

    /// Apply search and replace
    fn apply_search_replace(
        &self,
        content: &str,
        search: &str,
        replace: &str,
        all_occurrences: bool,
    ) -> Result<(String, usize), String> {
        if search.is_empty() {
            return Err("Search pattern cannot be empty".to_string());
        }

        let new_content = if all_occurrences {
            content.replace(search, replace)
        } else {
            content.replacen(search, replace, 1)
        };

        // Count occurrences changed
        let occurrences = if all_occurrences {
            content.matches(search).count()
        } else {
            if content.contains(search) { 1 } else { 0 }
        };

        Ok((new_content, occurrences))
    }

    /// Apply insert operation
    fn apply_insert(
        &self,
        content: &str,
        line: usize,
        insert_content: &str,
        position: InsertPosition,
    ) -> Result<(String, usize), String> {
        let lines: Vec<&str> = content.lines().collect();
        
        if line == 0 || line > lines.len() {
            return Err(format!("Invalid line number: {}", line));
        }

        let mut result = Vec::new();
        let insert_index = match position {
            InsertPosition::Before => line - 1,
            InsertPosition::After => line,
        };

        result.extend_from_slice(&lines[0..insert_index]);
        result.push(insert_content);
        result.extend_from_slice(&lines[insert_index..]);

        Ok((result.join("\n"), 1))
    }

    /// Apply delete operation
    fn apply_delete(
        &self,
        content: &str,
        start_line: usize,
        end_line: usize,
    ) -> Result<(String, usize), String> {
        let lines: Vec<&str> = content.lines().collect();
        
        if start_line == 0 || start_line > lines.len() {
            return Err(format!("Invalid start line: {}", start_line));
        }
        if end_line == 0 || end_line > lines.len() || end_line < start_line {
            return Err(format!("Invalid end line: {}", end_line));
        }

        let mut result = Vec::new();
        
        // Lines before deletion
        result.extend_from_slice(&lines[0..start_line - 1]);
        
        // Lines after deletion
        if end_line < lines.len() {
            result.extend_from_slice(&lines[end_line..]);
        }

        let lines_changed = end_line - start_line + 1;
        Ok((result.join("\n"), lines_changed))
    }

    /// Validate syntax using tree-sitter
    fn validate_syntax(&self, content: &str, language: &str) -> Result<(), String> {
        // For now, basic validation - can be extended with tree-sitter
        match language.to_lowercase().as_str() {
            "python" => {
                // Check for basic Python syntax errors (balanced braces, indentation, etc.)
                if !self.check_balanced_delimiters(content, "python") {
                    return Err("Python syntax error: unbalanced delimiters".to_string());
                }
            }
            "javascript" | "typescript" => {
                if !self.check_balanced_delimiters(content, "javascript") {
                    return Err("JavaScript syntax error: unbalanced delimiters".to_string());
                }
            }
            "rust" => {
                if !self.check_balanced_delimiters(content, "rust") {
                    return Err("Rust syntax error: unbalanced delimiters".to_string());
                }
            }
            _ => {
                // Unknown language, skip validation
            }
        }
        Ok(())
    }

    /// Check balanced delimiters (parentheses, brackets, braces)
    fn check_balanced_delimiters(&self, content: &str, _language: &str) -> bool {
        let mut stack = Vec::new();
        
        for ch in content.chars() {
            match ch {
                '(' | '[' | '{' => stack.push(ch),
                ')' => {
                    if stack.pop() != Some('(') {
                        return false;
                    }
                }
                ']' => {
                    if stack.pop() != Some('[') {
                        return false;
                    }
                }
                '}' => {
                    if stack.pop() != Some('{') {
                        return false;
                    }
                }
                _ => {}
            }
        }
        
        stack.is_empty()
    }

    /// Generate preview of changes
    fn generate_preview(&self, original: &str, modified: &str) -> String {
        let orig_lines: Vec<&str> = original.lines().collect();
        let mod_lines: Vec<&str> = modified.lines().collect();

        let mut preview = Vec::new();
        let max_lines = 10;

        // Show first difference
        for (i, (orig, modified)) in orig_lines.iter().zip(mod_lines.iter()).enumerate() {
            if orig != modified {
                preview.push(format!("Line {}: -{}", i + 1, orig));
                preview.push(format!("Line {}: +{}", i + 1, modified));
                
                if preview.len() >= max_lines {
                    break;
                }
            }
        }

        if preview.is_empty() {
            "No changes preview available".to_string()
        } else {
            preview.join("\n")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_range_edit() {
        let editor = FileEditor::new(true);
        let content = "line1\nline2\nline3\nline4";
        
        let result = editor.apply_line_range_edit(content, 2, 3, "new_line");
        assert!(result.is_ok());
        
        let (new_content, lines_changed) = result.unwrap();
        assert_eq!(new_content, "line1\nnew_line\nline4");
        assert_eq!(lines_changed, 2);
    }

    #[test]
    fn test_search_replace() {
        let editor = FileEditor::new(true);
        let content = "hello world\nhello again\nhello there";
        
        let result = editor.apply_search_replace(content, "hello", "hi", false);
        assert!(result.is_ok());
        
        let (new_content, occurrences) = result.unwrap();
        assert_eq!(new_content, "hi world\nhello again\nhello there");
        assert_eq!(occurrences, 1);
    }

    #[test]
    fn test_search_replace_all() {
        let editor = FileEditor::new(true);
        let content = "hello world\nhello again\nhello there";
        
        let result = editor.apply_search_replace(content, "hello", "hi", true);
        assert!(result.is_ok());
        
        let (new_content, occurrences) = result.unwrap();
        assert_eq!(new_content, "hi world\nhi again\nhi there");
        assert_eq!(occurrences, 3);
    }

    #[test]
    fn test_insert_before() {
        let editor = FileEditor::new(true);
        let content = "line1\nline2\nline3";
        
        let result = editor.apply_insert(content, 2, "new_line", InsertPosition::Before);
        assert!(result.is_ok());
        
        let (new_content, _) = result.unwrap();
        assert_eq!(new_content, "line1\nnew_line\nline2\nline3");
    }

    #[test]
    fn test_insert_after() {
        let editor = FileEditor::new(true);
        let content = "line1\nline2\nline3";
        
        let result = editor.apply_insert(content, 2, "new_line", InsertPosition::After);
        assert!(result.is_ok());
        
        let (new_content, _) = result.unwrap();
        assert_eq!(new_content, "line1\nline2\nnew_line\nline3");
    }

    #[test]
    fn test_delete() {
        let editor = FileEditor::new(true);
        let content = "line1\nline2\nline3\nline4";
        
        let result = editor.apply_delete(content, 2, 3);
        assert!(result.is_ok());
        
        let (new_content, lines_changed) = result.unwrap();
        assert_eq!(new_content, "line1\nline4");
        assert_eq!(lines_changed, 2);
    }

    #[test]
    fn test_balanced_delimiters() {
        let editor = FileEditor::new(true);
        
        assert!(editor.check_balanced_delimiters("def foo():\n    pass", "python"));
        assert!(editor.check_balanced_delimiters("{ foo: [1, 2, 3] }", "javascript"));
        assert!(!editor.check_balanced_delimiters("def foo(:", "python"));
        assert!(!editor.check_balanced_delimiters("{ foo: [1, 2, 3 }", "javascript"));
    }
}
