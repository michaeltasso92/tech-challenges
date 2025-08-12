import pytest
import json
import os
import tempfile
import sys

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from parse_guidelines import (
    is_item_node, is_shelf_like, expand_facings, 
    iter_sequences, parse_file
)

class TestHelperFunctions:
    """Test helper functions"""
    
    def test_is_item_node_product(self):
        """Test is_item_node with product type"""
        node = {"type": "product"}
        assert is_item_node(node) is True
    
    def test_is_item_node_tester(self):
        """Test is_item_node with tester type"""
        node = {"type": "tester"}
        assert is_item_node(node) is True
    
    def test_is_item_node_accessory(self):
        """Test is_item_node with accessory type"""
        node = {"type": "accessory"}
        assert is_item_node(node) is True
    
    def test_is_item_node_shelf(self):
        """Test is_item_node with shelf type (should be False)"""
        node = {"type": "shelf"}
        assert is_item_node(node) is False
    
    def test_is_item_node_none(self):
        """Test is_item_node with None type"""
        node = {"type": None}
        assert is_item_node(node) is False
    
    def test_is_shelf_like_shelf(self):
        """Test is_shelf_like with shelf type"""
        node = {"type": "shelf"}
        assert is_shelf_like(node) is True
    
    def test_is_shelf_like_bay_header(self):
        """Test is_shelf_like with bayHeader type"""
        node = {"type": "bayHeader"}
        assert is_shelf_like(node) is True
    
    def test_is_shelf_like_product(self):
        """Test is_shelf_like with product type (should be False)"""
        node = {"type": "product"}
        assert is_shelf_like(node) is False
    
    def test_expand_facings_normal(self):
        """Test expand_facings with normal facing count"""
        child = {"id": "item1", "facing": 3}
        result = expand_facings(child)
        assert result == ["item1", "item1", "item1"]
    
    def test_expand_facings_no_facing(self):
        """Test expand_facings without facing field"""
        child = {"id": "item1"}
        result = expand_facings(child)
        assert result == ["item1"]
    
    def test_expand_facings_none_facing(self):
        """Test expand_facings with None facing"""
        child = {"id": "item1", "facing": None}
        result = expand_facings(child)
        assert result == ["item1"]
    
    def test_expand_facings_zero_facing(self):
        """Test expand_facings with zero facing (should default to 1)"""
        child = {"id": "item1", "facing": 0}
        result = expand_facings(child)
        assert result == ["item1"]
    
    def test_expand_facings_no_id(self):
        """Test expand_facings without id"""
        child = {"facing": 3}
        result = expand_facings(child)
        assert result == []
    
    def test_expand_facings_empty_id(self):
        """Test expand_facings with empty id"""
        child = {"id": "", "facing": 3}
        result = expand_facings(child)
        assert result == []

class TestIterSequences:
    """Test the iter_sequences function"""
    
    def test_iter_sequences_shelf_with_items(self):
        """Test iter_sequences with a shelf containing items"""
        node = {
            "type": "shelf",
            "children": [
                {"type": "product", "id": "item1", "facing": 1},
                {"type": "product", "id": "item2", "facing": 1}
            ]
        }
        
        sequences = list(iter_sequences(node))
        assert len(sequences) == 1
        assert len(sequences[0]) == 2
        assert sequences[0][0]["id"] == "item1"
        assert sequences[0][1]["id"] == "item2"
    
    def test_iter_sequences_bay_with_shelves(self):
        """Test iter_sequences with a bay containing shelves"""
        node = {
            "type": "bay",
            "children": [
                {
                    "type": "shelf",
                    "children": [
                        {"type": "product", "id": "item1", "facing": 1},
                        {"type": "product", "id": "item2", "facing": 1}
                    ]
                },
                {
                    "type": "shelf",
                    "children": [
                        {"type": "product", "id": "item3", "facing": 1}
                    ]
                }
            ]
        }
        
        sequences = list(iter_sequences(node))
        assert len(sequences) == 2
        assert len(sequences[0]) == 2
        assert len(sequences[1]) == 1
    
    def test_iter_sequences_no_children(self):
        """Test iter_sequences with no children"""
        node = {"type": "shelf", "children": []}
        sequences = list(iter_sequences(node))
        assert len(sequences) == 0
    
    def test_iter_sequences_none_children(self):
        """Test iter_sequences with None children"""
        node = {"type": "shelf", "children": None}
        sequences = list(iter_sequences(node))
        assert len(sequences) == 0
    
    def test_iter_sequences_not_dict(self):
        """Test iter_sequences with non-dict input"""
        sequences = list(iter_sequences("not a dict"))
        assert len(sequences) == 0

class TestParseFile:
    """Test the parse_file function"""
    
    @pytest.fixture
    def sample_guideline(self):
        """Sample guideline data for testing"""
        return {
            "id": "test-guideline",
            "children": [
                {
                    "type": "bay",
                    "children": [
                        {
                            "type": "shelf",
                            "children": [
                                {
                                    "type": "product",
                                    "id": "item1",
                                    "name": "Product 1",
                                    "facing": 2
                                },
                                {
                                    "type": "product",
                                    "id": "item2",
                                    "name": "Product 2",
                                    "facing": 1
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    
    def test_parse_file_success(self, sample_guideline):
        """Test successful parsing of a guideline file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_guideline, f)
            temp_file = f.name
        
        try:
            rows, names, metas = parse_file(temp_file)
            
            # Check rows
            # item1 has facing=2, item2 has facing=1
            # So we get: ["item1", "item1", "item2"] = 3 slots
            assert len(rows) == 3
            
            # Check that we have left and right neighbors
            left_neighbors = [r for r in rows if r["left_neighbor"] is not None]
            right_neighbors = [r for r in rows if r["right_neighbor"] is not None]
            assert len(left_neighbors) > 0
            assert len(right_neighbors) > 0
            
            # Check names
            assert "item1" in names
            assert "item2" in names
            assert names["item1"] == "Product 1"
            assert names["item2"] == "Product 2"
            
        finally:
            os.unlink(temp_file)
    
    def test_parse_file_invalid_json(self):
        """Test parsing with invalid JSON"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_file = f.name
        
        try:
            result = parse_file(temp_file)
            # parse_file returns empty list on JSON exception
            assert result == []
        except ValueError:
            # If it returns a tuple instead
            rows, names, metas = result
            assert rows == []
            assert names == {}
        finally:
            os.unlink(temp_file)
    
    def test_parse_file_no_id(self, sample_guideline):
        """Test parsing when guideline has no id"""
        del sample_guideline["id"]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_guideline, f)
            temp_file = f.name
        
        try:
            rows, names, metas = parse_file(temp_file)
            # Should use filename as id
            assert len(rows) > 0
            assert all(r["guideline_id"] == os.path.basename(temp_file) for r in rows)
        finally:
            os.unlink(temp_file)
    
    def test_parse_file_single_item(self, sample_guideline):
        """Test parsing with only one item (should be skipped)"""
        # Remove one item so we only have one
        sample_guideline["children"][0]["children"][0]["children"] = [
            sample_guideline["children"][0]["children"][0]["children"][0]
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_guideline, f)
            temp_file = f.name
        
        try:
            rows, names, metas = parse_file(temp_file)
            # With facing=2, we get ["item1", "item1"] = 2 slots, which is enough for neighbors
            assert len(rows) == 2
            assert len(names) == 1  # But should still extract the name
        finally:
            os.unlink(temp_file)
    
    def test_parse_file_with_accessories(self, sample_guideline):
        """Test parsing with accessories"""
        # Add an accessory
        sample_guideline["children"][0]["children"][0]["children"].append({
            "type": "accessory",
            "id": "acc1",
            "name": "Accessory 1"
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_guideline, f)
            temp_file = f.name
        
        try:
            rows, names, metas = parse_file(temp_file)
            assert "acc1" in names
            assert names["acc1"] == "Accessory 1"
        finally:
            os.unlink(temp_file)

class TestIntegration:
    """Integration tests for the parser"""
    
    def test_complete_parsing_workflow(self):
        """Test a complete parsing workflow"""
        guideline = {
            "id": "test-guideline",
            "children": [
                {
                    "type": "bay",
                    "children": [
                        {
                            "type": "shelf",
                            "children": [
                                {
                                    "type": "product",
                                    "id": "item1",
                                    "name": "Product 1",
                                    "facing": 2
                                },
                                {
                                    "type": "product",
                                    "id": "item2",
                                    "name": "Product 2",
                                    "facing": 1
                                },
                                {
                                    "type": "product",
                                    "id": "item3",
                                    "name": "Product 3",
                                    "facing": 1
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(guideline, f)
            temp_file = f.name
        
        try:
            rows, names, metas = parse_file(temp_file)
            
            # Should have 4 rows:
            # item1 has facing=2, item2 has facing=1, item3 has facing=1
            # So we get: ["item1", "item1", "item2", "item3"] = 4 slots
            assert len(rows) == 4
            
            # Check that all items have neighbors
            for row in rows:
                assert row["guideline_id"] == "test-guideline"
                assert row["item_id"] in ["item1", "item2", "item3"]
                # Each item should have either left or right neighbor
                assert row["left_neighbor"] is not None or row["right_neighbor"] is not None
            
            # Check names
            assert len(names) == 3
            assert names["item1"] == "Product 1"
            assert names["item2"] == "Product 2"
            assert names["item3"] == "Product 3"
            
        finally:
            os.unlink(temp_file)

if __name__ == "__main__":
    pytest.main([__file__])
