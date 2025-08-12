import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the app functions
from app import (
    norm, seen_badge, fetch_names, fetch_name, 
    label_for, get_recs, to_df
)

class TestHelperFunctions:
    """Test helper functions"""
    
    def test_norm_basic(self):
        """Test basic string normalization"""
        assert norm("Hello World") == "hello world"
        assert norm("") == ""
        assert norm(None) == ""
    
    def test_norm_unicode(self):
        """Test unicode normalization"""
        # Test with accented characters
        assert norm("café") == "cafe"
        assert norm("naïve") == "naive"
        assert norm("résumé") == "resume"
    
    def test_norm_special_chars(self):
        """Test normalization with special characters"""
        assert norm("Product-Name") == "product-name"
        assert norm("Item & More") == "item & more"
        assert norm("Test@123") == "test@123"
    
    def test_seen_badge(self):
        """Test seen badge generation"""
        badge = seen_badge("train")
        assert "train" in badge.upper()
        assert "background" in badge
        assert "color" in badge
        
        # Test with unknown label
        badge = seen_badge("unknown")
        assert "unknown" in badge.upper()
    
    def test_seen_badge_colors(self):
        """Test that different labels get different colors"""
        train_badge = seen_badge("train")
        val_badge = seen_badge("val")
        test_badge = seen_badge("test")
        
        # Should have different background colors
        assert train_badge != val_badge
        assert val_badge != test_badge
        assert train_badge != test_badge

class TestAPIFunctions:
    """Test API-related functions"""
    
    @patch('app.requests.get')
    def test_fetch_names_success(self, mock_get):
        """Test successful name fetching"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "names": {
                "item1": "Product 1",
                "item2": "Product 2"
            }
        }
        mock_get.return_value = mock_response
        
        result = fetch_names()
        
        assert result == {
            "names": {
                "item1": "Product 1",
                "item2": "Product 2"
            }
        }
        mock_get.assert_called_once()
    
    @patch('app.requests.get')
    def test_fetch_names_failure(self, mock_get):
        """Test name fetching failure"""
        mock_get.side_effect = Exception("Network error")
        
        result = fetch_names()
        
        assert result == {}
    
    @patch('app.requests.get')
    def test_fetch_name_success(self, mock_get):
        """Test successful single name fetching"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"name": "Test Product"}
        mock_get.return_value = mock_response
        
        # Mock session state
        with patch('app.st.session_state', {"name_cache": {}}):
            result = fetch_name("item1")
        
        assert result == "Test Product"
        mock_get.assert_called_once()
    
    @patch('app.requests.get')
    def test_fetch_name_failure(self, mock_get):
        """Test single name fetching failure"""
        mock_get.side_effect = Exception("Network error")
        
        # Mock session state
        with patch('app.st.session_state', {"name_cache": {}}):
            result = fetch_name("item1")
        
        assert result is None
    
    @patch('app.requests.get')
    def test_fetch_name_cached(self, mock_get):
        """Test that names are cached"""
        # Mock session state with cached name
        with patch('app.st.session_state', {"name_cache": {"item1": "Cached Product"}}):
            result = fetch_name("item1")
        
        assert result == "Cached Product"
        # Should not make API call
        mock_get.assert_not_called()
    
    def test_label_for_with_name(self):
        """Test label generation with name"""
        with patch('app.fetch_name', return_value="Test Product"):
            result = label_for("item1", {"item1": "Test Product"})
        
        assert result == "Test Product — item1"
    
    def test_label_for_without_name(self):
        """Test label generation without name"""
        with patch('app.fetch_name', return_value=None):
            result = label_for("item1", {})
        
        assert result == "item1"
    
    @patch('app.requests.get')
    def test_get_recs_success(self, mock_get):
        """Test successful recommendation fetching"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "left": [{"item": "left1", "confidence": 0.8}],
            "right": [{"item": "right1", "confidence": 0.7}]
        }
        mock_get.return_value = mock_response
        
        result = get_recs("item1")
        
        assert "left" in result
        assert "right" in result
        assert len(result["left"]) == 1
        assert len(result["right"]) == 1
        mock_get.assert_called_once()

class TestDataProcessing:
    """Test data processing functions"""
    
    def test_to_df_with_names(self):
        """Test dataframe creation with names"""
        items = [
            {"item": "item1", "name": "Product 1", "confidence": 0.8},
            {"item": "item2", "name": "Product 2", "confidence": 0.6}
        ]
        
        with patch('app.names', {"item1": "Product 1", "item2": "Product 2"}):
            df = to_df(items)
        
        assert len(df) == 2
        assert "neighbor" in df.columns
        assert "item" in df.columns
        assert "confidence" in df.columns
        assert "seen" in df.columns
        
        # Check that names are used for neighbor column
        assert df.iloc[0]["neighbor"] == "Product 1"
        assert df.iloc[1]["neighbor"] == "Product 2"
    
    def test_to_df_without_names(self):
        """Test dataframe creation without names"""
        items = [
            {"item": "item1", "confidence": 0.8},
            {"item": "item2", "confidence": 0.6}
        ]
        
        with patch('app.names', {}):
            with patch('app.fetch_name', return_value=None):
                df = to_df(items)
        
        assert len(df) == 2
        # Should use item ID as neighbor when no name available
        assert df.iloc[0]["neighbor"] == "item1"
        assert df.iloc[1]["neighbor"] == "item2"
    
    def test_to_df_empty(self):
        """Test dataframe creation with empty items"""
        df = to_df([])
        
        assert len(df) == 0
        assert "neighbor" in df.columns
        assert "item" in df.columns
        assert "confidence" in df.columns
        assert "seen" in df.columns
    
    def test_to_df_missing_columns(self):
        """Test dataframe creation with missing columns"""
        items = [
            {"item": "item1"},  # Missing confidence and seen
            {"item": "item2", "confidence": 0.6}  # Missing seen
        ]
        
        with patch('app.names', {}):
            with patch('app.fetch_name', return_value=None):
                df = to_df(items)
        
        assert len(df) == 2
        # Should have default values for missing columns
        assert df.iloc[0]["confidence"] == 0.0
        assert df.iloc[0]["seen"] == "unseen"
        assert df.iloc[1]["confidence"] == 0.6
        assert df.iloc[1]["seen"] == "unseen"

class TestSearchFunctionality:
    """Test search functionality"""
    
    def test_search_matching(self):
        """Test search with matching results"""
        names = {
            "item1": "Lipstick Red",
            "item2": "Lipstick Pink",
            "item3": "Foundation",
            "item4": "Black Mascara"
        }
        
        # Test search for "lip"
        search_matches = []
        nq = norm("lip")
        all_ids = list(names.keys())
        
        search_matches = [
            {"id": iid, "name": names[iid]} 
            for iid in all_ids 
            if nq in norm(names.get(iid, ""))
        ]
        
        assert len(search_matches) == 2
        assert any(match["name"] == "Lipstick Red" for match in search_matches)
        assert any(match["name"] == "Lipstick Pink" for match in search_matches)
    
    def test_search_no_matches(self):
        """Test search with no matches"""
        names = {
            "item1": "Lipstick Red",
            "item2": "Foundation"
        }
        
        search_matches = []
        nq = norm("serum")
        all_ids = list(names.keys())
        
        search_matches = [
            {"id": iid, "name": names[iid]} 
            for iid in all_ids 
            if nq in norm(names.get(iid, ""))
        ]
        
        assert len(search_matches) == 0
    
    def test_search_case_insensitive(self):
        """Test that search is case insensitive"""
        names = {
            "item1": "LIPSTICK RED",
            "item2": "lipstick pink",
            "item3": "Foundation"
        }
        
        search_matches = []
        nq = norm("lip")
        all_ids = list(names.keys())
        
        search_matches = [
            {"id": iid, "name": names[iid]} 
            for iid in all_ids 
            if nq in norm(names.get(iid, ""))
        ]
        
        assert len(search_matches) == 2
        assert any(match["name"] == "LIPSTICK RED" for match in search_matches)
        assert any(match["name"] == "lipstick pink" for match in search_matches)

class TestIntegration:
    """Integration tests"""
    
    @patch('app.requests.get')
    def test_complete_workflow(self, mock_get):
        """Test a complete workflow"""
        # Mock API responses
        mock_names_response = MagicMock()
        mock_names_response.status_code = 200
        mock_names_response.json.return_value = {
            "names": {
                "item1": "Product 1",
                "item2": "Product 2"
            }
        }
        
        mock_rec_response = MagicMock()
        mock_rec_response.status_code = 200
        mock_rec_response.json.return_value = {
            "left": [
                {"item": "left1", "name": "Left Product", "confidence": 0.8, "seen": "train"}
            ],
            "right": [
                {"item": "right1", "name": "Right Product", "confidence": 0.7, "seen": "val"}
            ]
        }
        
        mock_get.side_effect = [mock_names_response, mock_rec_response]
        
        # Test the workflow
        names = fetch_names()
        assert "names" in names
        
        recs = get_recs("item1")
        assert "left" in recs
        assert "right" in recs
        
        # Test dataframe creation
        with patch('app.names', names["names"]):
            left_df = to_df(recs["left"])
            right_df = to_df(recs["right"])
        
        assert len(left_df) == 1
        assert len(right_df) == 1
        assert left_df.iloc[0]["neighbor"] == "Left Product"
        assert right_df.iloc[0]["neighbor"] == "Right Product"

if __name__ == "__main__":
    pytest.main([__file__])
