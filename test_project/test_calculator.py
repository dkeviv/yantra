"""Test file for calculator.py"""
import unittest
from calculator import Calculator


class TestCalculator(unittest.TestCase):
    """Test cases for Calculator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calc = Calculator()
    
    def test_add(self):
        """Test addition method"""
        result = self.calc.add(2, 3)
        self.assertEqual(result, "Result: 5")
    
    def test_multiply(self):
        """Test multiplication method"""
        result = self.calc.multiply(4, 5)
        self.assertEqual(result, "Result: 20")
    
    def test_add_negative(self):
        """Test addition with negative numbers"""
        result = self.calc.add(-2, 3)
        self.assertEqual(result, "Result: 1")


if __name__ == '__main__':
    unittest.main()
