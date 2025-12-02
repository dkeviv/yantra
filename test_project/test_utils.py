"""Test file for utils.py"""
import unittest
from utils import calculate_sum, calculate_product, format_result


class TestUtils(unittest.TestCase):
    """Test cases for utility functions"""
    
    def test_calculate_sum(self):
        """Test sum calculation"""
        result = calculate_sum(10, 20)
        self.assertEqual(result, 30)
    
    def test_calculate_product(self):
        """Test product calculation"""
        result = calculate_product(3, 7)
        self.assertEqual(result, 21)
    
    def test_format_result(self):
        """Test result formatting"""
        result = format_result(42)
        self.assertEqual(result, "Result: 42")


if __name__ == '__main__':
    unittest.main()
