import pytest
import json
import os
import tempfile
from unittest.mock import patch, mock_open
import sys

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.util import load_artifacts

class TestLoadArtifacts:
    """Test the load_artifacts function"""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing"""
        return {
            "left": {
                "item1": [{"item": "left1", "confidence": 0.8}],
                "item2": [{"item": "left2", "confidence": 0.6}]
            },
            "right": {
                "item1": [{"item": "right1", "confidence": 0.7}],
                "item2": [{"item": "right2", "confidence": 0.5}]
            },
            "fallback": {
                "left": ["fallback1", "fallback2"],
                "right": ["fallback3", "fallback4"]
            }
        }
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_load_artifacts_success(self, mock_json_load, mock_open, sample_data):
        """Test successful loading of artifacts"""
        # Mock json.load to return different data for each file
        def json_load_side_effect(*args, **kwargs):
            if 'left.json' in str(mock_open.call_args):
                return sample_data["left"]
            elif 'right.json' in str(mock_open.call_args):
                return sample_data["right"]
            elif 'fallback.json' in str(mock_open.call_args):
                return sample_data["fallback"]
            else:
                return {}
        
        mock_json_load.side_effect = json_load_side_effect
        
        left, right, fb = load_artifacts("/test/path")
        
        assert left == sample_data["left"]
        assert right == sample_data["right"]
        assert fb == sample_data["fallback"]
        
        # Verify that open was called for each file
        assert mock_open.call_count == 3
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_load_artifacts_empty_data(self, mock_json_load, mock_open):
        """Test loading with empty data"""
        mock_json_load.return_value = {}
        
        left, right, fb = load_artifacts("/test/path")
        
        assert left == {}
        assert right == {}
        assert fb == {}
    
    @patch('builtins.open', new_callable=mock_open)
    def test_load_artifacts_file_not_found(self, mock_open):
        """Test handling of file not found"""
        mock_open.side_effect = FileNotFoundError("File not found")
        
        with pytest.raises(FileNotFoundError):
            load_artifacts("/test/path")
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_load_artifacts_json_error(self, mock_json_load, mock_open):
        """Test handling of JSON parsing error"""
        mock_json_load.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        
        with pytest.raises(json.JSONDecodeError):
            load_artifacts("/test/path")
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_load_artifacts_permission_error(self, mock_json_load, mock_open):
        """Test handling of permission error"""
        mock_open.side_effect = PermissionError("Permission denied")
        
        with pytest.raises(PermissionError):
            load_artifacts("/test/path")
    
    def test_load_artifacts_with_temp_files(self):
        """Test loading artifacts with actual temporary files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            left_data = {"item1": [{"item": "left1", "confidence": 0.8}]}
            right_data = {"item1": [{"item": "right1", "confidence": 0.7}]}
            fallback_data = {"left": ["fallback1"], "right": ["fallback2"]}
            
            # Write test files
            with open(os.path.join(temp_dir, "left.json"), "w") as f:
                json.dump(left_data, f)
            
            with open(os.path.join(temp_dir, "right.json"), "w") as f:
                json.dump(right_data, f)
            
            with open(os.path.join(temp_dir, "fallback.json"), "w") as f:
                json.dump(fallback_data, f)
            
            # Load artifacts
            left, right, fb = load_artifacts(temp_dir)
            
            # Verify results
            assert left == left_data
            assert right == right_data
            assert fb == fallback_data

if __name__ == "__main__":
    pytest.main([__file__])
