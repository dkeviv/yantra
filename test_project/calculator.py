# Test file for GNN - calculator class

from utils import calculate_sum, calculate_product, format_result

class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        """Add two numbers and store in history."""
        result = calculate_sum(a, b)
        self.history.append(format_result(result))
        return result
    
    def multiply(self, a, b):
        """Multiply two numbers and store in history."""
        result = calculate_product(a, b)
        self.history.append(format_result(result))
        return result
    
    def get_history(self):
        """Return calculation history."""
        return self.history
    
    def clear_history(self):
        """Clear calculation history."""
        self.history = []
