#!/usr/bin/env python3
"""
Add code snippet and docstring extraction to all GNN parsers.
This script adds helper functions for extracting code snippets and docstrings
to each parser file that doesn't have them yet.
"""

import re
import os

# Helper functions to add to each parser
HELPER_FUNCTIONS = r'''
/// Extract code snippet from a node (with reasonable size limit)
fn extract_code_snippet(node: &tree_sitter::Node, code: &str) -> Option<String> {
    const MAX_SNIPPET_LENGTH: usize = 1000; // 1KB max per node
    
    let start_byte = node.start_byte();
    let end_byte = node.end_byte();
    
    if start_byte >= code.len() || end_byte > code.len() || start_byte >= end_byte {
        return None;
    }
    
    let snippet = &code[start_byte..end_byte];
    
    // Truncate if too long
    if snippet.len() > MAX_SNIPPET_LENGTH {
        Some(format!("{}... [truncated]", &snippet[..900]))
    } else if !snippet.is_empty() {
        Some(snippet.to_string())
    } else {
        None
    }
}

/// Extract docstring/comment from a node
fn extract_docstring(node: &tree_sitter::Node, code: &str) -> Option<String> {
    // Look for comment or documentation nodes
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        let kind = child.kind();
        if kind.contains("comment") || kind.contains("doc") || kind.contains("string") {
            let text = match child.utf8_text(code.as_bytes()) {
                Ok(t) => t,
                Err(_) => continue,
            };
            
            // Clean up common doc patterns
            let cleaned = text
                .trim()
                .trim_start_matches("/**")
                .trim_start_matches("/*")
                .trim_start_matches("//")
                .trim_start_matches('#')
                .trim_start_matches("\"\"\"")
                .trim_end_matches("*/")
                .trim_end_matches("\"\"\"")
                .trim()
                .to_string();
            
            if !cleaned.is_empty() {
                return Some(cleaned);
            }
        }
    }
    None
}
'''

# Files to update (excluding parser.rs which we already updated)
PARSER_FILES = [
    'src/gnn/parser_js.rs',
    'src/gnn/parser_rust.rs',
    'src/gnn/parser_go.rs',
    'src/gnn/parser_java.rs',
    'src/gnn/parser_c.rs',
    'src/gnn/parser_cpp.rs',
    'src/gnn/parser_ruby.rs',
    'src/gnn/parser_php.rs',
    'src/gnn/parser_swift.rs',
    'src/gnn/parser_kotlin.rs',
]

def has_helper_functions(content):
    """Check if file already has the helper functions"""
    return 'extract_code_snippet' in content and 'extract_docstring' in content

def add_helper_functions(filepath):
    """Add helper functions before #[cfg(test)] or at end of file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    if has_helper_functions(content):
        print(f"⏭️  {filepath} - already has helpers")
        return False
    
    # Find insertion point (before #[cfg(test)] or at end)
    if '#[cfg(test)]' in content:
        parts = content.split('#[cfg(test)]', 1)
        new_content = parts[0] + HELPER_FUNCTIONS + '\n\n#[cfg(test)]' + parts[1]
    else:
        # Add before last line
        new_content = content.rstrip() + '\n\n' + HELPER_FUNCTIONS + '\n'
    
    with open(filepath, 'w') as f:
        f.write(new_content)
    
    print(f"✅ {filepath} - added helper functions")
    return True

def update_code_node_creation(filepath):
    """Update CodeNode creation to include code_snippet and docstring"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Pattern to find CodeNode { ... } without code_snippet/docstring
    # This is tricky because CodeNode creation is multi-line
    # We'll look for patterns that have ..Default::default() but no code_snippet
    
    # Pattern: CodeNode { fields... ..Default::default() }
    # We need to add before ..Default::default()
    
    # Look for existing code_snippet usage
    if 'code_snippet:' in content and 'extract_code_snippet' in content:
        print(f"⏭️  {filepath} - already extracts snippets")
        return False
    
    # This is complex to do with regex, so we'll just report
    print(f"⚠️  {filepath} - Manual update needed for CodeNode creation")
    return False

def main():
    updated = 0
    manual = 0
    
    for filepath in PARSER_FILES:
        if not os.path.exists(filepath):
            print(f"❌ {filepath} - not found")
            continue
        
        # Add helper functions
        if add_helper_functions(filepath):
            updated += 1
        
        # Note: CodeNode updates need manual intervention due to complexity
        # Each parser has different CodeNode creation patterns
    
    print(f"\n📊 Summary:")
    print(f"  ✅ Updated: {updated} files")
    print(f"  ⚠️  Manual updates needed: {len(PARSER_FILES)} files")
    print(f"\nNext step: Manually add 'code_snippet: extract_code_snippet(&node, code),'")
    print(f"           and 'docstring: extract_docstring(&node, code),' to CodeNode")
    print(f"           creations in each parser.")

if __name__ == '__main__':
    main()
