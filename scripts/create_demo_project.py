#!/usr/bin/env python3
"""
Demo script to test the pytest executor integration
Creates a simple Python project with tests and executes them
"""

import os
import sys
from pathlib import Path

# Create demo project structure
demo_dir = Path(__file__).parent / "demo_project"
demo_dir.mkdir(exist_ok=True)

# Create a simple Python module
module_code = '''"""Simple calculator module for testing"""

def add(a, b):
    """Add two numbers"""
    return a + b

def subtract(a, b):
    """Subtract b from a"""
    return a - b

def multiply(a, b):
    """Multiply two numbers"""
    return a * b

def divide(a, b):
    """Divide a by b"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
'''

(demo_dir / "calculator.py").write_text(module_code)

# Create tests
test_code = '''"""Tests for calculator module"""

import pytest
from calculator import add, subtract, multiply, divide

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0

def test_subtract():
    assert subtract(5, 3) == 2
    assert subtract(10, 10) == 0
    assert subtract(0, 5) == -5

def test_multiply():
    assert multiply(3, 4) == 12
    assert multiply(-2, 5) == -10
    assert multiply(0, 100) == 0

def test_divide():
    assert divide(10, 2) == 5
    assert divide(9, 3) == 3
    assert divide(1, 4) == 0.25

def test_divide_by_zero():
    with pytest.raises(ValueError):
        divide(10, 0)
'''

tests_dir = demo_dir / "tests"
tests_dir.mkdir(exist_ok=True)
(tests_dir / "test_calculator.py").write_text(test_code)

print(f"✅ Created demo project at: {demo_dir}")
print(f"   - calculator.py (4 functions)")
print(f"   - tests/test_calculator.py (5 tests)")
print()
print("To test the pytest executor:")
print(f"1. cd {demo_dir}")
print("2. Run pytest manually: pytest tests/test_calculator.py -v")
print("3. Or use Yantra's executeTests() API from the frontend")
print()
print("Expected result:")
print("  - 5 tests should pass")
print("  - Pass rate: 100%")
print("  - is_learnable(): true (>90% pass rate)")
print("  - quality_score(): 1.0")
