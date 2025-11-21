# Test file for GNN - main entry point

from calculator import Calculator

def main():
    """Main function to test calculator."""
    calc = Calculator()
    
    # Perform some calculations
    sum_result = calc.add(5, 3)
    print(f"5 + 3 = {sum_result}")
    
    product_result = calc.multiply(4, 7)
    print(f"4 * 7 = {product_result}")
    
    # Show history
    print("\nHistory:")
    for entry in calc.get_history():
        print(f"  {entry}")
    
    # Clear history
    calc.clear_history()
    print("\nHistory cleared!")

if __name__ == "__main__":
    main()
