import os
import re

def fix_file(filepath):
    """Add ..Default::default() to CodeNode initializations."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Pattern to match CodeNode initialization without Default
    pattern = r'(CodeNode\s*\{[^}]*?)(}\s*;)'
    
    def replace_func(match):
        body = match.group(1)
        closing = match.group(2)
        
        # Check if already has Default
        if '..Default::default()' in body or '..Default' in body:
            return match.group(0)
        
        # Check if has semantic fields
        has_semantic = 'semantic_embedding:' in body or 'code_snippet:' in body or 'docstring:' in body
        
        if has_semantic:
            # Already has semantic fields
            return match.group(0)
        
        # Add Default before closing
        # Remove trailing comma if present
        body = body.rstrip().rstrip(',')
        return f"{body},\n            ..Default::default(){closing}"
    
    new_content = re.sub(pattern, replace_func, content, flags=re.DOTALL)
    
    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        return True
    return False

# Files to fix
files_to_fix = [
    'src/gnn/incremental.rs',
    'src/gnn/features.rs',
    'src/llm/context.rs',
]

fixed_count = 0
for filepath in files_to_fix:
    if os.path.exists(filepath):
        if fix_file(filepath):
            print(f"✅ Fixed: {filepath}")
            fixed_count += 1
        else:
            print(f"⏭️  Skipped (no changes needed): {filepath}")
    else:
        print(f"❌ Not found: {filepath}")

print(f"\nFixed {fixed_count} files")
