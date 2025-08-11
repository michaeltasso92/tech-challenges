import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from sentence_transformers import SentenceTransformer, InputExample

# Import the module to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from train_bi import BiEncoderTrainer


class TestBiEncoderTrainer:
    """Test cases for BiEncoderTrainer class"""
    
    @pytest.fixture
    def trainer(self):
        """Create a BiEncoderTrainer instance for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = BiEncoderTrainer(
                base_model="sentence-transformers/all-MiniLM-L6-v2",
                interim_dir=temp_dir,
                artifacts_dir=os.path.join(temp_dir, "artifacts"),
                epochs=1,
                batch_size=32,
                dim=384,
                seed=42
            )
            yield trainer
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return {
            "names_data": pd.DataFrame({
                "name": ["Product A", "Product B", "Product C"],
                "brand": ["Brand1", "Brand2", "Brand3"],
                "folders": [["Category1", "Subcat1"], ["Category2"], ["Category1", "Subcat2"]],
                "type": ["Type1", "Type2", "Type1"],
                "code": ["CODE1", "CODE2", "CODE3"]
            }, index=["item1", "item2", "item3"]),
            
            "meta_data": pd.DataFrame({
                "brand": ["Brand1", "Brand2", "Brand3"],
                "folders": [["Category1", "Subcat1"], ["Category2"], ["Category1", "Subcat2"]],
                "type": ["Type1", "Type2", "Type1"],
                "code": ["CODE1", "CODE2", "CODE3"]
            }, index=["item1", "item2", "item3"]),
            
            "parsed_data": pd.DataFrame({
                "item_id": ["item1", "item2", "item3", "item1"],
                "left_neighbor": ["item2", "item3", "item1", "item3"],
                "right_neighbor": ["item3", "item1", "item2", "item2"],
                "guideline_id": ["guideline1", "guideline1", "guideline1", "guideline2"]
            })
        }
    
    def test_init(self, trainer):
        """Test BiEncoderTrainer initialization"""
        assert trainer.base_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert trainer.epochs == 1
        assert trainer.batch_size == 32
        assert trainer.dim == 384
        assert trainer.seed == 42
        assert trainer.text_map == {}
        assert trainer.item2idx == {}
        assert trainer.items == []
        assert os.path.exists(trainer.artifacts_dir)
    
    def test_row_text_static_method(self):
        """Test _row_text static method with various inputs"""
        # Test with all fields
        result = BiEncoderTrainer._row_text("Product Name", "Brand Name", ["Cat1", "Cat2"], "Type", "CODE123")
        assert result == "Brand Name | Product Name | Cat1 > Cat2 | Type | CODE123"
        
        # Test with missing fields
        result = BiEncoderTrainer._row_text("Product Name", None, None, None, None)
        assert result == "Product Name"
        
        # Test with empty strings
        result = BiEncoderTrainer._row_text("", "", [], "", "")
        assert result == ""
        
        # Test with numpy array folders
        folders_array = np.array(["Cat1", "Cat2"])
        result = BiEncoderTrainer._row_text("Product", "Brand", folders_array, "Type", "CODE")
        assert result == "Brand | Product | Cat1 > Cat2 | Type | CODE"
        
        # Test with string folders
        result = BiEncoderTrainer._row_text("Product", "Brand", "Cat1 > Cat2", "Type", "CODE")
        assert result == "Brand | Product | Cat1 > Cat2 | Type | CODE"
    
    def test_build_text_map(self, trainer, sample_data):
        """Test build_text_map method"""
        # Create temporary files
        names_path = os.path.join(trainer.interim_dir, "item_names.parquet")
        meta_path = os.path.join(trainer.interim_dir, "item_meta.parquet")
        
        sample_data["names_data"].to_parquet(names_path)
        sample_data["meta_data"].to_parquet(meta_path)
        
        # Test building text map
        text_map = trainer.build_text_map()
        
        assert len(text_map) == 3
        assert "item1" in text_map
        assert "item2" in text_map
        assert "item3" in text_map
        assert "Brand1 | Product A | Category1 > Subcat1 | Type1 | CODE1" in text_map.values()
    
    def test_build_text_map_missing_files(self, trainer):
        """Test build_text_map when files don't exist"""
        text_map = trainer.build_text_map()
        assert text_map == {}
    
    def test_mine_pairs_from_df(self, trainer, sample_data):
        """Test mine_pairs_from_df method"""
        parsed_path = os.path.join(trainer.interim_dir, "parsed.parquet")
        sample_data["parsed_data"].to_parquet(parsed_path)
        
        left_df, right_df = trainer.mine_pairs_from_df(parsed_path)
        
        assert len(left_df) == 4  # All pairs
        assert len(right_df) == 4  # All pairs
        assert "item_id" in left_df.columns
        assert "nbr" in left_df.columns
        assert "item_id" in right_df.columns
        assert "nbr" in right_df.columns
    
    def test_mine_pairs_from_df_with_self_loops(self, trainer):
        """Test mine_pairs_from_df filters out self-loops"""
        data = pd.DataFrame({
            "item_id": ["item1", "item2", "item3"],
            "left_neighbor": ["item1", "item2", "item3"],  # Self-loops
            "right_neighbor": ["item2", "item3", "item1"]
        })
        
        parsed_path = os.path.join(trainer.interim_dir, "parsed.parquet")
        data.to_parquet(parsed_path)
        
        left_df, right_df = trainer.mine_pairs_from_df(parsed_path)
        
        # Should filter out self-loops
        assert len(left_df) == 0
        assert len(right_df) == 3  # Only right neighbors remain
    
    def test_to_examples(self, trainer):
        """Test to_examples static method"""
        pairs_df = pd.DataFrame({
            "item_id": ["item1", "item2"],
            "nbr": ["item2", "item3"]
        })
        
        text_map = {
            "item1": "Product 1",
            "item2": "Product 2", 
            "item3": "Product 3"
        }
        
        examples = BiEncoderTrainer.to_examples(pairs_df, text_map)
        
        assert len(examples) == 2
        assert isinstance(examples[0], InputExample)
        assert examples[0].texts == ["Product 1", "Product 2"]
        assert examples[1].texts == ["Product 2", "Product 3"]
    
    def test_to_examples_missing_text(self, trainer):
        """Test to_examples with missing text mappings"""
        pairs_df = pd.DataFrame({
            "item_id": ["item1", "item2"],
            "nbr": ["item2", "item3"]
        })
        
        text_map = {"item1": "Product 1"}  # Missing item2 and item3
        
        examples = BiEncoderTrainer.to_examples(pairs_df, text_map)
        
        assert len(examples) == 1  # Only item1->item2 should work
        assert examples[0].texts == ["Product 1", "item2"]  # item2 falls back to ID
    
    @patch('train_bi.SentenceTransformer')
    @patch('train_bi.DataLoader')
    @patch('train_bi.losses.MultipleNegativesRankingLoss')
    def test_train_model(self, mock_loss, mock_loader, mock_model_class, trainer):
        """Test train_model method"""
        # Mock the model
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        # Mock the loader
        mock_loader_instance = Mock()
        mock_loader.return_value = mock_loader_instance
        
        # Mock the loss
        mock_loss_instance = Mock()
        mock_loss.return_value = mock_loss_instance
        
        # Create sample examples
        examples = [InputExample(texts=["text1", "text2"]) for _ in range(100)]
        
        # Test training
        result = trainer.train_model(examples)
        
        # Verify model was created and trained
        mock_model_class.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        mock_model.max_seq_length = 64
        mock_model.fit.assert_called_once()
        
        # Verify DataLoader was created
        mock_loader.assert_called_once()
        
        # Verify loss was created
        mock_loss.assert_called_once_with(mock_model)
    
    @patch('train_bi.SentenceTransformer')
    def test_encode_all(self, mock_model_class, trainer):
        """Test encode_all method"""
        # Setup
        trainer.text_map = {"item1": "Product 1", "item2": "Product 2"}
        trainer.items = ["item1", "item2"]
        
        # Mock the model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        mock_model_class.return_value = mock_model
        
        # Test encoding
        result = trainer.encode_all(mock_model)
        
        # Verify
        assert result.shape == (2, 2)
        assert result.dtype == np.float32
        mock_model.encode.assert_called_once_with(
            ["Product 1", "Product 2"], 
            batch_size=256, 
            normalize_embeddings=True, 
            show_progress_bar=True, 
            device="cpu"
        )
    
    def test_save_artifacts(self, trainer):
        """Test save_artifacts method"""
        # Setup
        trainer.item2idx = {"item1": 0, "item2": 1}
        emb_left = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        emb_right = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
        
        # Mock faiss
        with patch('train_bi.faiss') as mock_faiss:
            mock_index = Mock()
            mock_faiss.IndexFlatIP.return_value = mock_index
            mock_faiss.write_index = Mock()
            
            # Test saving
            trainer.save_artifacts(emb_left, emb_right)
            
            # Verify files were created
            assert os.path.exists(os.path.join(trainer.artifacts_dir, "embed_left.npy"))
            assert os.path.exists(os.path.join(trainer.artifacts_dir, "embed_right.npy"))
            assert os.path.exists(os.path.join(trainer.artifacts_dir, "item_vocab.json"))
            assert os.path.exists(os.path.join(trainer.artifacts_dir, "left.index"))
            assert os.path.exists(os.path.join(trainer.artifacts_dir, "right.index"))
            
            # Verify faiss was called
            mock_faiss.IndexFlatIP.assert_called()
            mock_faiss.write_index.assert_called()
    
    def test_split_by_guideline(self, trainer):
        """Test split_by_guideline method"""
        # Create test data
        df = pd.DataFrame({
            "guideline_id": ["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4", "g5", "g5"],
            "item_id": ["item1", "item2", "item3", "item4", "item5", "item6", "item7", "item8", "item9", "item10"]
        })
        
        # Test splitting
        train_df, val_df, test_df = trainer.split_by_guideline(df, train_ratio=0.6, val_ratio=0.2, seed=42)
        
        # Verify splits
        assert len(train_df) == 6  # 60% of 10
        assert len(val_df) == 2    # 20% of 10
        assert len(test_df) == 2   # 20% of 10
        
        # Verify no overlap
        train_guids = set(train_df["guideline_id"])
        val_guids = set(val_df["guideline_id"])
        test_guids = set(test_df["guideline_id"])
        
        assert train_guids.isdisjoint(val_guids)
        assert train_guids.isdisjoint(test_guids)
        assert val_guids.isdisjoint(test_guids)
    
    @patch('train_bi.SentenceTransformer')
    @patch('train_bi.faiss')
    def test_run_complete_workflow(self, mock_faiss, mock_model_class, trainer, sample_data):
        """Test the complete run workflow"""
        # Setup mock files
        parsed_path = os.path.join(trainer.interim_dir, "parsed.parquet")
        names_path = os.path.join(trainer.interim_dir, "item_names.parquet")
        meta_path = os.path.join(trainer.interim_dir, "item_meta.parquet")
        
        sample_data["parsed_data"].to_parquet(parsed_path)
        sample_data["names_data"].to_parquet(names_path)
        sample_data["meta_data"].to_parquet(meta_path)
        
        # Mock the model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
        mock_model_class.return_value = mock_model
        
        # Mock faiss
        mock_index = Mock()
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.write_index = Mock()
        
        # Test running the complete workflow
        trainer.run()
        
        # Verify artifacts were created
        assert os.path.exists(os.path.join(trainer.artifacts_dir, "seen_items.json"))
        assert os.path.exists(os.path.join(trainer.artifacts_dir, "bienc_left_model"))
        assert os.path.exists(os.path.join(trainer.artifacts_dir, "bienc_right_model"))
        
        # Verify seen_items.json content
        with open(os.path.join(trainer.artifacts_dir, "seen_items.json"), "r") as f:
            seen_items = json.load(f)
        
        assert "train_items" in seen_items
        assert "val_items" in seen_items
        assert "test_items" in seen_items
    
    def test_run_missing_parsed_file(self, trainer):
        """Test run method when parsed.parquet is missing"""
        with pytest.raises(FileNotFoundError, match="parsed.parquet not found"):
            trainer.run()
    
    def test_run_insufficient_training_pairs(self, trainer, sample_data):
        """Test run method with insufficient training pairs"""
        # Create minimal data that would result in few pairs
        minimal_data = pd.DataFrame({
            "item_id": ["item1"],
            "left_neighbor": ["item2"],
            "right_neighbor": ["item3"],
            "guideline_id": ["guideline1"]
        })
        
        parsed_path = os.path.join(trainer.interim_dir, "parsed.parquet")
        minimal_data.to_parquet(parsed_path)
        
        # Create minimal names data
        names_data = pd.DataFrame({
            "name": ["Product 1", "Product 2", "Product 3"]
        }, index=["item1", "item2", "item3"])
        
        names_path = os.path.join(trainer.interim_dir, "item_names.parquet")
        names_data.to_parquet(names_path)
        
        # This should raise an error due to insufficient pairs
        with pytest.raises(RuntimeError, match="Not enough training pairs"):
            trainer.run()


class TestBiEncoderTrainerIntegration:
    """Integration tests for BiEncoderTrainer"""
    
    def test_end_to_end_with_mocks(self):
        """Test end-to-end workflow with mocked dependencies"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create trainer
            trainer = BiEncoderTrainer(
                base_model="sentence-transformers/all-MiniLM-L6-v2",
                interim_dir=temp_dir,
                artifacts_dir=os.path.join(temp_dir, "artifacts"),
                epochs=1,
                batch_size=32
            )
            
            # Create test data
            parsed_data = pd.DataFrame({
                "item_id": ["item1", "item2", "item3", "item4"],
                "left_neighbor": ["item2", "item3", "item4", "item1"],
                "right_neighbor": ["item3", "item4", "item1", "item2"],
                "guideline_id": ["g1", "g1", "g1", "g1"]
            })
            
            names_data = pd.DataFrame({
                "name": ["Product 1", "Product 2", "Product 3", "Product 4"],
                "brand": ["Brand1", "Brand2", "Brand3", "Brand4"]
            }, index=["item1", "item2", "item3", "item4"])
            
            # Save test data
            parsed_data.to_parquet(os.path.join(temp_dir, "parsed.parquet"))
            names_data.to_parquet(os.path.join(temp_dir, "item_names.parquet"))
            
            # Mock all external dependencies
            with patch('train_bi.SentenceTransformer') as mock_model_class, \
                 patch('train_bi.faiss') as mock_faiss:
                
                # Mock model
                mock_model = Mock()
                mock_model.encode.return_value = np.random.rand(4, 384).astype(np.float32)
                mock_model_class.return_value = mock_model
                
                # Mock faiss
                mock_index = Mock()
                mock_faiss.IndexFlatIP.return_value = mock_index
                mock_faiss.write_index = Mock()
                
                # Run the workflow
                trainer.run()
                
                # Verify the workflow completed successfully
                assert os.path.exists(os.path.join(temp_dir, "artifacts", "seen_items.json"))
                assert os.path.exists(os.path.join(temp_dir, "artifacts", "item_vocab.json"))
                
                # Verify model was called
                mock_model_class.assert_called()
                mock_model.fit.assert_called()
                mock_model.encode.assert_called()
                
                # Verify faiss was called
                mock_faiss.IndexFlatIP.assert_called()
                mock_faiss.write_index.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])
