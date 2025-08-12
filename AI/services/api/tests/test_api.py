import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import os

# Import the app after setting up the test environment
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.api import app

@pytest.fixture
def client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)

@pytest.fixture
def mock_recommender():
    """Mock recommender for testing"""
    with patch('app.api.rec') as mock_rec:
        # Mock the recommender instance
        mock_rec.get_items.return_value = [
            {"id": "item1", "name": "Product 1"},
            {"id": "item2", "name": "Product 2"}
        ]
        mock_rec.get.return_value = {
            "left": [
                {"item": "left1", "name": "Left Product 1", "confidence": 0.8},
                {"item": "left2", "name": "Left Product 2", "confidence": 0.6}
            ],
            "right": [
                {"item": "right1", "name": "Right Product 1", "confidence": 0.9},
                {"item": "right2", "name": "Right Product 2", "confidence": 0.7}
            ]
        }
        mock_rec.name_of.return_value = "Test Product"
        yield mock_rec

class TestHealthEndpoint:
    """Test the health check endpoint"""
    
    def test_health_endpoint(self, client):
        """Test that health endpoint returns OK status"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

class TestItemsEndpoint:
    """Test the items endpoint"""
    
    def test_get_items_success(self, client, mock_recommender):
        """Test successful retrieval of items"""
        response = client.get("/items")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["id"] == "item1"
        assert data[0]["name"] == "Product 1"
        assert data[1]["id"] == "item2"
        assert data[1]["name"] == "Product 2"
        
        mock_recommender.get_items.assert_called_once()
    
    def test_get_items_empty(self, client, mock_recommender):
        """Test when no items are available"""
        mock_recommender.get_items.return_value = []
        
        response = client.get("/items")
        assert response.status_code == 200
        assert response.json() == []

class TestNameEndpoint:
    """Test the name endpoint"""
    
    def test_get_name_success(self, client, mock_recommender):
        """Test successful retrieval of item name"""
        response = client.get("/name/item1")
        assert response.status_code == 200
        
        data = response.json()
        assert data["item_id"] == "item1"
        assert data["name"] == "Test Product"
        
        mock_recommender.name_of.assert_called_once_with("item1")
    
    def test_get_name_not_found(self, client, mock_recommender):
        """Test when item name is not found"""
        mock_recommender.name_of.return_value = None
        
        response = client.get("/name/nonexistent")
        assert response.status_code == 200
        
        data = response.json()
        assert data["item_id"] == "nonexistent"
        assert data["name"] is None

class TestNamesEndpoint:
    """Test the names endpoint"""
    
    def test_get_names_success(self, client, mock_recommender):
        """Test successful retrieval of multiple names"""
        response = client.get("/names?ids=item1,item2")
        assert response.status_code == 200
        
        data = response.json()
        assert "names" in data
        assert data["names"]["item1"] == "Test Product"
        assert data["names"]["item2"] == "Test Product"
        
        assert mock_recommender.name_of.call_count == 2
    
    def test_get_names_empty_ids(self, client, mock_recommender):
        """Test with empty IDs parameter"""
        response = client.get("/names?ids=")
        assert response.status_code == 200
        assert response.json() == {"names": {}}
    
    def test_get_names_single_id(self, client, mock_recommender):
        """Test with single ID"""
        response = client.get("/names?ids=item1")
        assert response.status_code == 200
        
        data = response.json()
        assert data["names"]["item1"] == "Test Product"
        mock_recommender.name_of.assert_called_once_with("item1")

class TestRecommendEndpoint:
    """Test the recommend endpoint"""
    
    def test_recommend_success(self, client, mock_recommender):
        """Test successful recommendation"""
        response = client.get("/recommend/item1")
        assert response.status_code == 200
        
        data = response.json()
        assert data["item_id"] == "item1"
        assert "left" in data
        assert "right" in data
        
        # Check left neighbors
        left_neighbors = data["left"]
        assert len(left_neighbors) == 2
        assert left_neighbors[0]["item"] == "left1"
        assert left_neighbors[0]["name"] == "Left Product 1"
        assert left_neighbors[0]["confidence"] == 0.8
        
        # Check right neighbors
        right_neighbors = data["right"]
        assert len(right_neighbors) == 2
        assert right_neighbors[0]["item"] == "right1"
        assert right_neighbors[0]["name"] == "Right Product 1"
        assert right_neighbors[0]["confidence"] == 0.9
        
        mock_recommender.get.assert_called_once_with("item1", 10)
    
    def test_recommend_with_k_parameter(self, client, mock_recommender):
        """Test recommendation with custom k parameter"""
        response = client.get("/recommend/item1?k=5")
        assert response.status_code == 200
        
        mock_recommender.get.assert_called_once_with("item1", 5)
    
    def test_recommend_item_not_found(self, client, mock_recommender):
        """Test recommendation for non-existent item"""
        mock_recommender.get.return_value = {
            "left": [],
            "right": []
        }
        
        response = client.get("/recommend/nonexistent")
        assert response.status_code == 200
        
        data = response.json()
        assert data["item_id"] == "nonexistent"
        assert data["left"] == []
        assert data["right"] == []

class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_recommender_exception(self, client, mock_recommender):
        """Test handling of recommender exceptions"""
        mock_recommender.get.side_effect = Exception("Recommender error")
        
        response = client.get("/recommend/item1")
        assert response.status_code == 500
    
    def test_name_service_exception(self, client, mock_recommender):
        """Test handling of name service exceptions"""
        mock_recommender.name_of.side_effect = Exception("Name service error")
        
        response = client.get("/name/item1")
        assert response.status_code == 500

if __name__ == "__main__":
    pytest.main([__file__])
