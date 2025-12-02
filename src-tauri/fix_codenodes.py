#!/usr/bin/env python3
"""
Fix all CodeNode struct initializations by adding ..Default::default()
"""
import re
import sys
from pathlib import Path

def fix_code_node_in_file(filepath):
    """Add ..Default::default() to all CodeNode struct initializations."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Pattern: CodeNode { ... } but WITHOUT ..Default::default()
    # Match CodeNode { ... } where closing } is at same or deeper indentation
    pattern = r'(CodeNode\s*\{[^}]+?)((\n\s+)\}\s*[;,)])'
    
    def replacer(match):
        before = match.group(1)
        closing = match.group(2)
        indent = match.group(3)
        
        # Check if ..Default::default() already exists
        if '..Default::default()' in before:
            return match.group(0)  # Already fixed
        
        # Add ..Default::default() before closing }
        return f"{before},{indent}    ..Default::default(){closing}"
    
    # Apply replacement
    fixed_content = re.sub(pattern, replacer, content)
    
    if fixed_content != content:
        with open(filepath, 'w') as f:
            f.write(fixed_content)
        return True
    return False

def main():
    gnn_dir = Path('src/gnn')
    fixed_count = 0
    
    for rs_file in gnn_dir.glob('*.rs'):
        if fix_code_node_in_file(rs_file):
            print(f"✅ Fixed: {rs_file.name}")
            fixed_count += 1
    
    print(f"\n✨ Fixed {fixed_count} files")

if __name__ == '__main__':
    main()
