#!/usr/bin/env python3
"""
Test script to verify data cleaning logic in the parser.
"""

import json
import tempfile
import os
from parse_guidelines import parse_file, is_valid_product_for_recommendation, PRODUCT_TYPES

def create_test_guideline():
    """Create a test guideline with various item types."""
    return {
        "id": "test_guideline",
        "children": [
            {
                "type": "shelf",
                "children": [
                    # Valid products
                    {"type": "product", "id": "prod1", "name": "Product 1", "brand": "Brand A", "facing": 2},
                    {"type": "tester", "id": "test1", "name": "Tester 1", "brand": "Brand B", "facing": 1},
                    {"type": "shoes", "id": "shoe1", "name": "Shoe 1", "brand": "Brand C", "facing": 1},
                    
                    # Invalid products (missing required fields)
                    {"type": "product", "id": "", "name": "Empty ID Product"},
                    {"type": "product", "id": "no_name", "name": "", "brand": ""},
                    
                    # Display items (should be filtered out)
                    {"type": "visual", "id": "visual1", "name": "Visual 1"},
                    {"type": "object3D", "id": "obj1", "name": "3D Object"},
                    
                    # Infrastructure items (should be filtered out)
                    {"type": "storeComponent", "id": "comp1", "name": "Component 1"},
                    
                    # Valid products
                    {"type": "cloth", "id": "cloth1", "name": "Cloth 1", "brand": "Brand D", "facing": 1},
                    {"type": "textileAccessory", "id": "acc1", "name": "Accessory 1", "brand": "Brand E", "facing": 1},
                ]
            }
        ]
    }

def test_data_cleaning():
    """Test the data cleaning logic."""
    print("Testing data cleaning logic...")
    
    # Test individual item validation
    print("\n1. Testing individual item validation:")
    
    valid_items = [
        {"type": "product", "id": "prod1", "name": "Product 1", "brand": "Brand A"},
        {"type": "tester", "id": "test1", "name": "Tester 1", "code": "CODE1"},
        {"type": "shoes", "id": "shoe1", "brand": "Brand B", "code": "CODE2"},
    ]
    
    invalid_items = [
        {"type": "product", "id": "", "name": "Empty ID"},
        {"type": "product", "id": "no_info", "name": "", "brand": ""},
        {"type": "visual", "id": "visual1", "name": "Visual Item"},
        {"type": "storeComponent", "id": "comp1", "name": "Component"},
        {"type": "accessory", "id": "acc1", "name": "Accessory"},
    ]
    
    print("Valid items:")
    for item in valid_items:
        is_valid = is_valid_product_for_recommendation(item)
        print(f"  {item['type']} - {item['id']}: {is_valid}")
        assert is_valid, f"Expected {item['id']} to be valid"
    
    print("Invalid items:")
    for item in invalid_items:
        is_valid = is_valid_product_for_recommendation(item)
        print(f"  {item['type']} - {item['id']}: {is_valid}")
        assert not is_valid, f"Expected {item['id']} to be invalid"
    
    # Test full file parsing
    print("\n2. Testing full file parsing:")
    
    test_guideline = create_test_guideline()
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_guideline, f)
        temp_file = f.name
    
    try:
        rows, names, metas = parse_file(temp_file)
        
        print(f"Parsed {len(rows)} neighbor relationships")
        print(f"Found {len(names)} unique items")
        
        # Check that only valid products are included
        expected_items = {"prod1", "test1", "shoe1", "cloth1", "acc1"}
        actual_items = set(names.keys())
        
        print(f"Expected items: {expected_items}")
        print(f"Actual items: {actual_items}")
        
        assert actual_items == expected_items, f"Item mismatch: expected {expected_items}, got {actual_items}"
        
        # Check that we have neighbor relationships
        assert len(rows) > 0, "No neighbor relationships found"
        
        print("✅ All tests passed!")
        
    finally:
        os.unlink(temp_file)

if __name__ == "__main__":
    test_data_cleaning()
