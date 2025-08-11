import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
from unittest.mock import patch, mock_open
import sys

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the training functions
from train_model import score_side

class TestScoreSide:
    """Test the score_side function"""
    
    @pytest.fixture
    def sample_pairs(self):
        """Sample pairs data for testing"""
        return pd.DataFrame({
            'item_id': ['item1', 'item1', 'item2', 'item2'],
            'neighbor_id': ['neighbor1', 'neighbor2', 'neighbor1', 'neighbor3'],
            'cnt': [10, 5, 8, 3]
        })
    
    def test_score_side_basic(self, sample_pairs):
        """Test basic scoring functionality"""
        result = score_side(sample_pairs, 'item_id', 'neighbor_id', 'cnt')
        
        # Check that we get results for both items
        assert 'item1' in result
        assert 'item2' in result
        
        # Check structure of results
        for item_id, neighbors in result.items():
            assert isinstance(neighbors, list)
            for neighbor in neighbors:
                assert 'item' in neighbor
                assert 'confidence' in neighbor
                assert isinstance(neighbor['confidence'], float)
                assert 0 <= neighbor['confidence'] <= 1
    
    def test_score_side_single_item(self):
        """Test scoring with single item"""
        pairs = pd.DataFrame({
            'item_id': ['item1', 'item1'],
            'neighbor_id': ['neighbor1', 'neighbor2'],
            'cnt': [5, 3]
        })
        
        result = score_side(pairs, 'item_id', 'neighbor_id', 'cnt')
        
        assert 'item1' in result
        assert len(result['item1']) == 2
        
        # Check that confidences sum to 1 (approximately)
        confidences = [n['confidence'] for n in result['item1']]
        assert abs(sum(confidences) - 1.0) < 1e-6
    
    def test_score_side_empty_dataframe(self):
        """Test scoring with empty dataframe"""
        empty_pairs = pd.DataFrame(columns=['item_id', 'neighbor_id', 'cnt'])
        
        result = score_side(empty_pairs, 'item_id', 'neighbor_id', 'cnt')
        assert result == {}
    
    def test_score_side_different_alpha(self, sample_pairs):
        """Test scoring with different alpha values"""
        result_alpha_1 = score_side(sample_pairs, 'item_id', 'neighbor_id', 'cnt', alpha=1.0)
        result_alpha_10 = score_side(sample_pairs, 'item_id', 'neighbor_id', 'cnt', alpha=10.0)
        
        # Results should be different with different alpha
        assert result_alpha_1 != result_alpha_10
    
    def test_score_side_confidence_ordering(self, sample_pairs):
        """Test that confidences are ordered correctly"""
        result = score_side(sample_pairs, 'item_id', 'neighbor_id', 'cnt')
        
        for item_id, neighbors in result.items():
            if len(neighbors) > 1:
                # Check that confidences are in descending order
                confidences = [n['confidence'] for n in neighbors]
                assert confidences == sorted(confidences, reverse=True)
    
    def test_score_side_zero_counts(self):
        """Test scoring with zero counts"""
        pairs = pd.DataFrame({
            'item_id': ['item1', 'item1'],
            'neighbor_id': ['neighbor1', 'neighbor2'],
            'cnt': [0, 0]
        })
        
        result = score_side(pairs, 'item_id', 'neighbor_id', 'cnt')
        
        assert 'item1' in result
        # With zero counts and alpha, should have equal probabilities
        confidences = [n['confidence'] for n in result['item1']]
        assert abs(confidences[0] - confidences[1]) < 1e-6

class TestTrainingWorkflow:
    """Test the complete training workflow"""
    
    @pytest.fixture
    def sample_parsed_data(self):
        """Sample parsed data for testing"""
        return pd.DataFrame({
            'guideline_id': ['guideline1', 'guideline1', 'guideline2'],
            'group_seq': [1, 1, 1],
            'pos': [0, 1, 0],
            'item_id': ['item1', 'item2', 'item3'],
            'left_neighbor': [None, 'item1', None],
            'right_neighbor': ['item2', None, None]
        })
    
    @pytest.fixture
    def sample_names(self):
        """Sample names data for testing"""
        return pd.DataFrame({
            'id': ['item1', 'item2', 'item3'],
            'name': ['Product 1', 'Product 2', 'Product 3']
        })
    
    def test_training_workflow(self, sample_parsed_data, sample_names):
        """Test the complete training workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create input directory structure
            input_dir = os.path.join(temp_dir, 'input')
            os.makedirs(input_dir)
            
            # Save parsed data
            parsed_path = os.path.join(input_dir, 'parsed.parquet')
            sample_parsed_data.to_parquet(parsed_path)
            
            # Save names data
            names_path = os.path.join(input_dir, 'item_names.parquet')
            sample_names.to_parquet(names_path)
            
            # Create output directory
            output_dir = os.path.join(temp_dir, 'output')
            os.makedirs(output_dir)
            
            # Mock command line arguments
            with patch('sys.argv', ['train_model.py', '--in', input_dir, '--out', output_dir]):
                # Import and run the training script
                import train_model
                
                # Check that output files were created
                assert os.path.exists(os.path.join(output_dir, 'left.json'))
                assert os.path.exists(os.path.join(output_dir, 'right.json'))
                assert os.path.exists(os.path.join(output_dir, 'fallback.json'))
                assert os.path.exists(os.path.join(output_dir, 'item_names.json'))
                
                # Check content of output files
                with open(os.path.join(output_dir, 'left.json'), 'r') as f:
                    left_data = json.load(f)
                with open(os.path.join(output_dir, 'right.json'), 'r') as f:
                    right_data = json.load(f)
                with open(os.path.join(output_dir, 'fallback.json'), 'r') as f:
                    fallback_data = json.load(f)
                with open(os.path.join(output_dir, 'item_names.json'), 'r') as f:
                    names_data = json.load(f)
                
                # Verify data structure
                assert isinstance(left_data, dict)
                assert isinstance(right_data, dict)
                assert isinstance(fallback_data, dict)
                assert isinstance(names_data, dict)
                
                # Check that we have recommendations for items
                assert 'item2' in left_data  # item2 has left neighbor item1
                assert 'item1' in right_data  # item1 has right neighbor item2
    
    def test_training_without_names(self, sample_parsed_data):
        """Test training when names file doesn't exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create input directory structure
            input_dir = os.path.join(temp_dir, 'input')
            os.makedirs(input_dir)
            
            # Save only parsed data (no names)
            parsed_path = os.path.join(input_dir, 'parsed.parquet')
            sample_parsed_data.to_parquet(parsed_path)
            
            # Create output directory
            output_dir = os.path.join(temp_dir, 'output')
            os.makedirs(output_dir)
            
            # Mock command line arguments
            with patch('sys.argv', ['train_model.py', '--in', input_dir, '--out', output_dir]):
                # Import and run the training script
                import train_model
                
                # Check that output files were created
                assert os.path.exists(os.path.join(output_dir, 'left.json'))
                assert os.path.exists(os.path.join(output_dir, 'right.json'))
                assert os.path.exists(os.path.join(output_dir, 'fallback.json'))
                assert os.path.exists(os.path.join(output_dir, 'item_names.json'))
                
                # Check that names file is empty
                with open(os.path.join(output_dir, 'item_names.json'), 'r') as f:
                    names_data = json.load(f)
                assert names_data == {}

class TestDataProcessing:
    """Test data processing steps"""
    
    def test_left_right_grouping(self):
        """Test the grouping of left and right neighbors"""
        # Create sample data
        df = pd.DataFrame({
            'item_id': ['item1', 'item1', 'item2'],
            'left_neighbor': ['left1', 'left2', 'left3'],
            'right_neighbor': ['right1', 'right2', 'right3']
        })
        
        # Test left neighbor grouping
        left = df.dropna(subset=["left_neighbor"]).groupby(
            ["item_id","left_neighbor"]).size().rename("cnt").reset_index()
        
        assert len(left) == 3
        assert 'cnt' in left.columns
        assert all(left['cnt'] == 1)  # Each pair appears once
        
        # Test right neighbor grouping
        right = df.dropna(subset=["right_neighbor"]).groupby(
            ["item_id","right_neighbor"]).size().rename("cnt").reset_index()
        
        assert len(right) == 3
        assert 'cnt' in right.columns
        assert all(right['cnt'] == 1)  # Each pair appears once
    
    def test_duplicate_handling(self):
        """Test handling of duplicate pairs"""
        # Create data with duplicates
        df = pd.DataFrame({
            'item_id': ['item1', 'item1', 'item1'],
            'left_neighbor': ['left1', 'left1', 'left2'],
            'right_neighbor': ['right1', 'right1', 'right2']
        })
        
        # Test left neighbor grouping with duplicates
        left = df.dropna(subset=["left_neighbor"]).groupby(
            ["item_id","left_neighbor"]).size().rename("cnt").reset_index()
        
        # Should have 2 unique pairs (item1-left1 appears twice, item1-left2 once)
        assert len(left) == 2
        item1_left1 = left[(left['item_id'] == 'item1') & (left['left_neighbor'] == 'left1')]
        assert len(item1_left1) == 1
        assert item1_left1.iloc[0]['cnt'] == 2

class TestFallbackGeneration:
    """Test fallback recommendation generation"""
    
    def test_fallback_generation(self):
        """Test generation of fallback recommendations"""
        # Create sample data
        left = pd.DataFrame({
            'left_neighbor': ['left1', 'left2', 'left1', 'left3'],
            'cnt': [10, 5, 8, 3]
        })
        right = pd.DataFrame({
            'right_neighbor': ['right1', 'right2', 'right1', 'right3'],
            'cnt': [7, 4, 6, 2]
        })
        
        # Generate fallback recommendations
        glob_left = (left.groupby("left_neighbor")["cnt"].sum()
                    .sort_values(ascending=False).head(20).index.tolist())
        glob_right = (right.groupby("right_neighbor")["cnt"].sum()
                     .sort_values(ascending=False).head(20).index.tolist())
        
        # Check that we get the most frequent items
        assert 'left1' in glob_left
        assert 'right1' in glob_right
        
        # Check ordering (most frequent first)
        left_counts = left.groupby("left_neighbor")["cnt"].sum()
        right_counts = right.groupby("right_neighbor")["cnt"].sum()
        
        assert left_counts.loc[glob_left[0]] >= left_counts.loc[glob_left[1]]
        assert right_counts.loc[glob_right[0]] >= right_counts.loc[glob_right[1]]

if __name__ == "__main__":
    pytest.main([__file__])
