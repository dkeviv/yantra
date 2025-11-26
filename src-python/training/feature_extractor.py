"""
Python Feature Extractor for GraphSAGE Training

Extracts 978-dimensional feature vectors from Python code.
This is a Python implementation that mirrors the Rust FeatureExtractor in src-tauri/src/gnn/features.rs

Feature Vector Structure (978 dimensions):
- 50: Node Identity Features (function/class/variable/import/module)
- 100: Structural Features (depth, children, parents, siblings)
- 100: Complexity Features (cyclomatic, cognitive, nesting)
- 150: Dependency Features (imports, calls, usage patterns)
- 100: Context Features (file-level, module-level context)
- 200: Semantic Features (naming conventions, docstrings, comments)
- 100: Quality Features (test coverage, documentation, type hints)
- 50: Temporal Features (recency, change frequency)
- 124: Statistical Features (LOC, token counts, distributions)
- 4: Language Encoding (Python, JavaScript, TypeScript, Other)
"""

import ast
import re
from typing import Optional, List, Dict
import numpy as np


class FeatureExtractor:
    """Extract 978-dim feature vectors from Python code"""
    
    def __init__(self):
        self.file_cache = {}
    
    def extract_features_from_code(self, code: str, language: str = "python") -> np.ndarray:
        """
        Extract 978-dimensional feature vector from source code
        
        Args:
            code: Source code string
            language: Programming language (python, javascript, typescript, other)
        
        Returns:
            978-dimensional numpy array
        """
        features = []
        
        # Parse code to AST (Python only for now)
        tree = None
        if language.lower() == "python":
            try:
                tree = ast.parse(code)
            except SyntaxError:
                # Invalid code, return zero features
                return np.zeros(978, dtype=np.float32)
        
        # Section 1: Node Identity Features (50 dims)
        features.extend(self._extract_node_identity(tree, code))
        
        # Section 2: Structural Features (100 dims)
        features.extend(self._extract_structural_features(tree, code))
        
        # Section 3: Complexity Features (100 dims)
        features.extend(self._extract_complexity_features(tree, code))
        
        # Section 4: Dependency Features (150 dims)
        features.extend(self._extract_dependency_features(tree, code))
        
        # Section 5: Context Features (100 dims)
        features.extend(self._extract_context_features(tree, code))
        
        # Section 6: Semantic Features (200 dims)
        features.extend(self._extract_semantic_features(tree, code))
        
        # Section 7: Quality Features (100 dims)
        features.extend(self._extract_quality_features(tree, code))
        
        # Section 8: Temporal Features (50 dims)
        features.extend(self._extract_temporal_features(tree, code))
        
        # Section 9: Statistical Features (124 dims)
        features.extend(self._extract_statistical_features(tree, code))
        
        # Section 10: Language Encoding (4 dims)
        features.extend(self._extract_language_encoding(language))
        
        assert len(features) == 978, f"Expected 978 features, got {len(features)}"
        return np.array(features, dtype=np.float32)
    
    def _extract_node_identity(self, tree: Optional[ast.AST], code: str) -> List[float]:
        """Section 1: Node Identity Features (50 dims)"""
        features = []
        
        # Node type one-hot (5 dims): function, class, variable, import, module
        if tree is None:
            features.extend([0.0] * 5)
        else:
            has_functions = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
            has_classes = any(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
            has_imports = any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))
            
            features.extend([
                1.0 if has_functions else 0.0,
                1.0 if has_classes else 0.0,
                0.5,  # variable (always present)
                1.0 if has_imports else 0.0,
                1.0,  # module (always present)
            ])
        
        # Padding to 50 dims
        features.extend([0.0] * 45)
        return features
    
    def _extract_structural_features(self, tree: Optional[ast.AST], code: str) -> List[float]:
        """Section 2: Structural Features (100 dims)"""
        features = []
        
        if tree is None:
            return [0.0] * 100
        
        # Count structural elements
        num_functions = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.FunctionDef))
        num_classes = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.ClassDef))
        num_statements = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.stmt))
        
        # Normalized counts
        features.extend([
            min(num_functions / 20.0, 1.0),
            min(num_classes / 10.0, 1.0),
            min(num_statements / 100.0, 1.0),
        ])
        
        # Padding
        features.extend([0.0] * 97)
        return features
    
    def _extract_complexity_features(self, tree: Optional[ast.AST], code: str) -> List[float]:
        """Section 3: Complexity Features (100 dims)"""
        features = []
        
        if tree is None:
            return [0.0] * 100
        
        # Count control flow nodes (if, for, while, try)
        control_flow = sum(1 for node in ast.walk(tree) 
                          if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)))
        
        # Nesting depth
        max_depth = self._calculate_max_depth(tree)
        
        features.extend([
            min(control_flow / 20.0, 1.0),
            min(max_depth / 10.0, 1.0),
        ])
        
        # Padding
        features.extend([0.0] * 98)
        return features
    
    def _extract_dependency_features(self, tree: Optional[ast.AST], code: str) -> List[float]:
        """Section 4: Dependency Features (150 dims)"""
        features = []
        
        if tree is None:
            return [0.0] * 150
        
        # Count imports
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        num_imports = len(imports)
        
        # Common library indicators (top 10 Python libraries)
        common_libs = ['numpy', 'pandas', 'torch', 'tensorflow', 'sklearn', 
                      'matplotlib', 'requests', 'flask', 'django', 'fastapi']
        
        lib_features = []
        for lib in common_libs:
            has_lib = any(lib in ast.unparse(imp) if hasattr(ast, 'unparse') else False 
                         for imp in imports)
            lib_features.append(1.0 if has_lib else 0.0)
        
        features.extend([
            min(num_imports / 20.0, 1.0),
            *lib_features,
        ])
        
        # Padding
        features.extend([0.0] * 139)
        return features
    
    def _extract_context_features(self, tree: Optional[ast.AST], code: str) -> List[float]:
        """Section 5: Context Features (100 dims)"""
        # Simple line count and character count
        lines = code.split('\n')
        num_lines = len(lines)
        num_chars = len(code)
        
        features = [
            min(num_lines / 500.0, 1.0),
            min(num_chars / 10000.0, 1.0),
        ]
        
        # Padding
        features.extend([0.0] * 98)
        return features
    
    def _extract_semantic_features(self, tree: Optional[ast.AST], code: str) -> List[float]:
        """Section 6: Semantic Features (200 dims)"""
        features = []
        
        # Naming conventions
        has_snake_case = bool(re.search(r'[a-z]+_[a-z]+', code))
        has_camel_case = bool(re.search(r'[a-z]+[A-Z][a-z]+', code))
        
        # Docstrings
        has_docstrings = '"""' in code or "'''" in code
        
        # Comments
        num_comments = code.count('#')
        
        features.extend([
            1.0 if has_snake_case else 0.0,
            1.0 if has_camel_case else 0.0,
            1.0 if has_docstrings else 0.0,
            min(num_comments / 20.0, 1.0),
        ])
        
        # Padding
        features.extend([0.0] * 196)
        return features
    
    def _extract_quality_features(self, tree: Optional[ast.AST], code: str) -> List[float]:
        """Section 7: Quality Features (100 dims)"""
        features = []
        
        # Type hints
        has_type_hints = '->' in code or ': ' in code
        
        # Test indicators
        has_tests = 'def test_' in code or 'class Test' in code or 'assert' in code
        
        features.extend([
            1.0 if has_type_hints else 0.0,
            1.0 if has_tests else 0.0,
        ])
        
        # Padding
        features.extend([0.0] * 98)
        return features
    
    def _extract_temporal_features(self, tree: Optional[ast.AST], code: str) -> List[float]:
        """Section 8: Temporal Features (50 dims)"""
        # Placeholder: would require git history
        return [0.0] * 50
    
    def _extract_statistical_features(self, tree: Optional[ast.AST], code: str) -> List[float]:
        """Section 9: Statistical Features (124 dims)"""
        features = []
        
        # Lines of code
        lines = [line.strip() for line in code.split('\n') if line.strip() and not line.strip().startswith('#')]
        loc = len(lines)
        
        # Token counts
        tokens = re.findall(r'\w+', code)
        num_tokens = len(tokens)
        unique_tokens = len(set(tokens))
        
        # Average line length
        avg_line_length = sum(len(line) for line in lines) / max(len(lines), 1)
        
        features.extend([
            min(loc / 200.0, 1.0),
            min(num_tokens / 1000.0, 1.0),
            min(unique_tokens / 500.0, 1.0),
            min(avg_line_length / 80.0, 1.0),
        ])
        
        # Padding
        features.extend([0.0] * 120)
        return features
    
    def _extract_language_encoding(self, language: str) -> List[float]:
        """Section 10: Language Encoding (4 dims)"""
        # One-hot: [python, javascript, typescript, other]
        lang_lower = language.lower()
        if lang_lower == "python":
            return [1.0, 0.0, 0.0, 0.0]
        elif lang_lower in ["javascript", "js"]:
            return [0.0, 1.0, 0.0, 0.0]
        elif lang_lower in ["typescript", "ts"]:
            return [0.0, 0.0, 1.0, 0.0]
        else:
            return [0.0, 0.0, 0.0, 1.0]
    
    def _calculate_max_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth of AST"""
        max_depth = current_depth
        for child in ast.iter_child_nodes(node):
            depth = self._calculate_max_depth(child, current_depth + 1)
            max_depth = max(max_depth, depth)
        return max_depth


# Singleton instance
_feature_extractor = None

def get_feature_extractor() -> FeatureExtractor:
    """Get or create singleton feature extractor"""
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = FeatureExtractor()
    return _feature_extractor
