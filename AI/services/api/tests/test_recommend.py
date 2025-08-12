import pytest
import os
from unittest.mock import patch, mock_open
import sys

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.recommend import Recommender

class TestRecommender:
    """Test the Recommender class"""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing"""
        return {
            "left": {
                "item1": [
                    {"item": "left1", "confidence": 0.8},
                    {"item": "left2", "confidence": 0.6}
                ],
                "item2": [
                    {"item": "left3", "confidence": 0.9}
                ]
            },
            "right": {
                "item1": [
                    {"item": "right1", "confidence": 0.7},
                    {"item": "right2", "confidence": 0.5}
                ],
                "item2": [
                    {"item": "right3", "confidence": 0.8}
                ]
            },
            "fallback": {
                "left": ["fallback1", "fallback2"],
                "right": ["fallback3", "fallback4"]
            }
        }
    
    @pytest.fixture
    def sample_names(self):
        """Sample names data for testing"""
        return {
            "item1": "Product 1",
            "item2": "Product 2",
            "left1": "Left Product 1",
            "right1": "Right Product 1"
        }
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_init_success(self, mock_json_load, mock_open, sample_data, sample_names):
        """Test successful initialization"""
        # Mock json.load to return different data for different files
        def json_load_side_effect(*args, **kwargs):
            if "left.json" in str(args[0]):
                return sample_data["left"]
            elif "right.json" in str(args[0]):
                return sample_data["right"]
            elif "fallback.json" in str(args[0]):
                return sample_data["fallback"]
            elif "item_names.json" in str(args[0]):
                return sample_names
            else:
                return {}
        
        mock_json_load.side_effect = json_load_side_effect
        
        # Mock os.path.exists to return True for all files
        with patch('os.path.exists', return_value=True):
            recommender = Recommender()
        
        assert recommender.left == sample_data["left"]
        assert recommender.right == sample_data["right"]
        assert recommender.fb == sample_data["fallback"]
        assert recommender.names == sample_names
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_init_no_names_file(self, mock_json_load, mock_open, sample_data):
        """Test initialization when names file doesn't exist"""
        # Mock json.load to return data for other files but not names
        def json_load_side_effect(*args, **kwargs):
            if "left.json" in str(args[0]):
                return sample_data["left"]
            elif "right.json" in str(args[0]):
                return sample_data["right"]
            elif "fallback.json" in str(args[0]):
                return sample_data["fallback"]
            else:
                return {}
        
        mock_json_load.side_effect = json_load_side_effect
        
        # Mock os.path.exists to return False for names file
        def exists_side_effect(path):
            return "item_names.json" not in str(path)
        
        with patch('os.path.exists', side_effect=exists_side_effect):
            recommender = Recommender()
        
        assert recommender.names == {}
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_init_json_error(self, mock_json_load, mock_open, sample_data):
        """Test initialization when JSON loading fails"""
        # Mock json.load to raise exception for names file
        def json_load_side_effect(*args, **kwargs):
            if "item_names.json" in str(args[0]):
                raise Exception("JSON error")
            elif "left.json" in str(args[0]):
                return sample_data["left"]
            elif "right.json" in str(args[0]):
                return sample_data["right"]
            elif "fallback.json" in str(args[0]):
                return sample_data["fallback"]
            else:
                return {}
        
        mock_json_load.side_effect = json_load_side_effect
        
        with patch('os.path.exists', return_value=True):
            recommender = Recommender()
        
        assert recommender.names == {}
    
    @patch('app.recommend.load_artifacts')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_name_of_success(self, mock_json_load, mock_open, mock_load_artifacts, sample_data, sample_names):
        """Test successful name lookup"""
        mock_load_artifacts.return_value = (sample_data["left"], sample_data["right"], sample_data["fallback"])
        mock_json_load.return_value = sample_names
        
        with patch('os.path.exists', return_value=True):
            recommender = Recommender()
        
        assert recommender.name_of("item1") == "Product 1"
        assert recommender.name_of("left1") == "Left Product 1"
    
    @patch('app.recommend.load_artifacts')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_name_of_not_found(self, mock_json_load, mock_open, mock_load_artifacts, sample_data, sample_names):
        """Test name lookup for non-existent item"""
        mock_load_artifacts.return_value = (sample_data["left"], sample_data["right"], sample_data["fallback"])
        mock_json_load.return_value = sample_names
        
        with patch('os.path.exists', return_value=True):
            recommender = Recommender()
        
        assert recommender.name_of("nonexistent") is None
    
    @patch('app.recommend.load_artifacts')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_enrich_neighbors(self, mock_json_load, mock_open, mock_load_artifacts, sample_data, sample_names):
        """Test neighbor enrichment with names"""
        mock_load_artifacts.return_value = (sample_data["left"], sample_data["right"], sample_data["fallback"])
        mock_json_load.return_value = sample_names
        
        with patch('os.path.exists', return_value=True):
            recommender = Recommender()
        
        neighbors = [
            {"item": "item1", "confidence": 0.8},
            {"item": "left1", "confidence": 0.6}
        ]
        
        enriched = recommender.enrich_neighbors(neighbors)
        
        assert len(enriched) == 2
        assert enriched[0]["item"] == "item1"
        assert enriched[0]["name"] == "Product 1"
        assert enriched[0]["confidence"] == 0.8
        assert enriched[1]["item"] == "left1"
        assert enriched[1]["name"] == "Left Product 1"
        assert enriched[1]["confidence"] == 0.6
    
    @patch('app.recommend.load_artifacts')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_enrich_neighbors_with_existing_name(self, mock_json_load, mock_open, mock_load_artifacts, sample_data, sample_names):
        """Test neighbor enrichment when name already exists"""
        mock_load_artifacts.return_value = (sample_data["left"], sample_data["right"], sample_data["fallback"])
        mock_json_load.return_value = sample_names
        
        with patch('os.path.exists', return_value=True):
            recommender = Recommender()
        
        neighbors = [
            {"item": "item1", "name": "Custom Name", "confidence": 0.8}
        ]
        
        enriched = recommender.enrich_neighbors(neighbors)
        
        assert len(enriched) == 1
        assert enriched[0]["item"] == "item1"
        assert enriched[0]["name"] == "Custom Name"  # Should use existing name
        assert enriched[0]["confidence"] == 0.8
    
    @patch('app.recommend.load_artifacts')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_get_success(self, mock_json_load, mock_open, mock_load_artifacts, sample_data, sample_names):
        """Test successful recommendation retrieval"""
        mock_load_artifacts.return_value = (sample_data["left"], sample_data["right"], sample_data["fallback"])
        mock_json_load.return_value = sample_names
        
        with patch('os.path.exists', return_value=True):
            recommender = Recommender()
        
        result = recommender.get("item1", k=2)
        
        assert "left" in result
        assert "right" in result
        assert len(result["left"]) == 2
        assert len(result["right"]) == 2
        
        # Check that names are enriched
        assert result["left"][0]["name"] == "Left Product 1"
        assert result["right"][0]["name"] == "Right Product 1"
    
    @patch('app.recommend.load_artifacts')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_get_item_not_found(self, mock_json_load, mock_open, mock_load_artifacts, sample_data, sample_names):
        """Test recommendation for non-existent item (should use fallback)"""
        mock_load_artifacts.return_value = (sample_data["left"], sample_data["right"], sample_data["fallback"])
        mock_json_load.return_value = sample_names
        
        with patch('os.path.exists', return_value=True):
            recommender = Recommender()
        
        result = recommender.get("nonexistent", k=2)
        
        assert "left" in result
        assert "right" in result
        # Should use fallback items with low confidence
        assert len(result["left"]) == 2
        assert len(result["right"]) == 2
        assert result["left"][0]["confidence"] == 0.01
        assert result["right"][0]["confidence"] == 0.01
    
    @patch('app.recommend.load_artifacts')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_get_with_custom_k(self, mock_json_load, mock_open, mock_load_artifacts, sample_data, sample_names):
        """Test recommendation with custom k parameter"""
        mock_load_artifacts.return_value = (sample_data["left"], sample_data["right"], sample_data["fallback"])
        mock_json_load.return_value = sample_names
        
        with patch('os.path.exists', return_value=True):
            recommender = Recommender()
        
        result = recommender.get("item1", k=1)
        
        assert len(result["left"]) == 1
        assert len(result["right"]) == 1
    
    @patch('app.recommend.load_artifacts')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_get_items(self, mock_json_load, mock_open, mock_load_artifacts, sample_data, sample_names):
        """Test getting all available items"""
        mock_load_artifacts.return_value = (sample_data["left"], sample_data["right"], sample_data["fallback"])
        mock_json_load.return_value = sample_names
        
        with patch('os.path.exists', return_value=True):
            recommender = Recommender()
        
        items = recommender.get_items()
        
        # Should return all unique items from left and right
        expected_ids = {"item1", "item2", "left1", "left2", "left3", "right1", "right2", "right3"}
        returned_ids = {item["id"] for item in items}
        
        assert returned_ids == expected_ids
        
        # Check that names are included
        for item in items:
            assert "id" in item
            assert "name" in item
            if item["id"] in sample_names:
                assert item["name"] == sample_names[item["id"]]
            else:
                assert item["name"] == item["id"]  # fallback to ID

if __name__ == "__main__":
    pytest.main([__file__])
