import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import os

# Import the app after setting up the test environment
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.api import app

@pytest.fixture
def mock_recommender():
    """Mock recommender for testing"""
    # Create a mock recommender
    mock_rec = type('MockRecommender', (), {
        'names': {
            "item1": "Product 1",
            "item2": "Product 2"
        },
        'get': lambda self, item_id, k=10: {
            "left": [
                {"item": "left1", "name": "Left Product 1", "confidence": 0.8, "seen": "train"},
                {"item": "left2", "name": "Left Product 2", "confidence": 0.6, "seen": "val"}
            ],
            "right": [
                {"item": "right1", "name": "Right Product 1", "confidence": 0.9, "seen": "train"},
                {"item": "right2", "name": "Right Product 2", "confidence": 0.7, "seen": "val"}
            ],
            "query_seen": "train"
        },
        'name_of': lambda self, item_id: "Test Product"
    })()
    
    # Patch the global rec variable
    with patch('app.api.rec', mock_rec):
        yield mock_rec

@pytest.fixture
def client(mock_recommender):
    """Create a test client for the FastAPI app"""
    return TestClient(app)

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
    
    def test_get_items_empty(self, client, mock_recommender):
        """Test when no items are available"""
        # Create a new mock with empty names
        empty_mock = type('MockRecommender', (), {
            'names': {},
            'get': lambda self, item_id, k=10: {"left": [], "right": [], "query_seen": "unseen"},
            'name_of': lambda self, item_id: None
        })()
        
        with patch('app.api.rec', empty_mock):
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
    
    def test_get_names_no_ids(self, client, mock_recommender):
        """Test with no IDs parameter - should return all names"""
        response = client.get("/names")
        assert response.status_code == 200
        assert response.json() == {"names": {"item1": "Product 1", "item2": "Product 2"}}
    
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
        assert data["item_name"] == "Test Product"
        assert "left" in data
        assert "right" in data
        assert "query_seen" in data
        
        # Check left neighbors
        left_neighbors = data["left"]
        assert len(left_neighbors) == 2
        assert left_neighbors[0]["item"] == "left1"
        assert left_neighbors[0]["name"] == "Left Product 1"
        assert left_neighbors[0]["confidence"] == 0.8
        assert left_neighbors[0]["seen"] == "train"
        
        # Check right neighbors
        right_neighbors = data["right"]
        assert len(right_neighbors) == 2
        assert right_neighbors[0]["item"] == "right1"
        assert right_neighbors[0]["name"] == "Right Product 1"
        assert right_neighbors[0]["confidence"] == 0.9
        assert right_neighbors[0]["seen"] == "train"
        
        mock_recommender.get.assert_called_once_with("item1", k=10)
    
    def test_recommend_with_k_parameter(self, client, mock_recommender):
        """Test recommendation with custom k parameter"""
        response = client.get("/recommend/item1?k=5")
        assert response.status_code == 200
        
        mock_recommender.get.assert_called_once_with("item1", k=5)
    
    def test_recommend_item_not_found(self, client, mock_recommender):
        """Test recommendation for non-existent item"""
        mock_recommender.get.return_value = {
            "left": [],
            "right": [],
            "query_seen": "unseen"
        }
        mock_recommender.name_of.return_value = None
        
        response = client.get("/recommend/nonexistent")
        assert response.status_code == 200
        
        data = response.json()
        assert data["item_name"] is None
        assert data["left"] == []
        assert data["right"] == []
        assert data["query_seen"] == "unseen"

class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_recommender_exception(self, client, mock_recommender):
        """Test handling of recommender exceptions"""
        # Create a mock that raises an exception
        exception_mock = type('MockRecommender', (), {
            'names': {"item1": "Product 1"},
            'get': lambda self, item_id, k=10: (_ for _ in ()).throw(Exception("Recommender error")),
            'name_of': lambda self, item_id: "Test Product"
        })()
        
        with patch('app.api.rec', exception_mock):
            response = client.get("/recommend/item1")
            assert response.status_code == 500
    
    def test_name_service_exception(self, client, mock_recommender):
        """Test handling of name service exceptions"""
        # Create a mock that raises an exception
        exception_mock = type('MockRecommender', (), {
            'names': {"item1": "Product 1"},
            'get': lambda self, item_id, k=10: {"left": [], "right": [], "query_seen": "unseen"},
            'name_of': lambda self, item_id: (_ for _ in ()).throw(Exception("Name service error"))
        })()
        
        with patch('app.api.rec', exception_mock):
            response = client.get("/name/item1")
            assert response.status_code == 500

if __name__ == "__main__":
    pytest.main([__file__])
