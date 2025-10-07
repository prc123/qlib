"""
Test cases for NormalizedTSDatasetH and NormalizedTSDataSampler classes
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from qlib.data.dataset import NormalizedTSDatasetH, NormalizedTSDataSampler, TSDatasetH, TSDataSampler


class TestNormalizedTSDataSampler:
    """Test cases for NormalizedTSDataSampler class"""
    
    def test_init(self):
        """Test initialization with different normalization methods"""
        # Create mock DataFrame data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        instruments = [f'stock_{i}' for i in range(10)]
        
        # Create multi-index
        index = pd.MultiIndex.from_product([instruments, dates], names=['instrument', 'datetime'])
        data = pd.DataFrame(np.random.randn(1000, 5), index=index)
        
        # Test zscore normalization
        sampler = NormalizedTSDataSampler(
            data=data,
            start=pd.Timestamp('2020-01-01'),
            end=pd.Timestamp('2020-04-09'),
            step_len=30,
            normalization_method="zscore"
        )
        assert sampler.normalization_method == "zscore"
        
        # Test minmax normalization
        sampler = NormalizedTSDataSampler(
            data=data,
            start=pd.Timestamp('2020-01-01'),
            end=pd.Timestamp('2020-04-09'),
            step_len=30,
            normalization_method="minmax"
        )
        assert sampler.normalization_method == "minmax"
        
        # Test robust normalization
        sampler = NormalizedTSDataSampler(
            data=data,
            start=pd.Timestamp('2020-01-01'),
            end=pd.Timestamp('2020-04-09'),
            step_len=30,
            normalization_method="robust"
        )
        assert sampler.normalization_method == "robust"
    
    def test_fit_normalization(self):
        """Test normalization parameter fitting"""
        # Create test data
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        instruments = [f'stock_{i}' for i in range(5)]
        index = pd.MultiIndex.from_product([instruments, dates], names=['instrument', 'datetime'])
        data = pd.DataFrame(np.random.randn(150, 3), index=index)
        
        # Create sampler instance
        sampler = NormalizedTSDataSampler(
            data=data,
            start=pd.Timestamp('2020-01-01'),
            end=pd.Timestamp('2020-01-30'),
            step_len=10,
            normalization_method="zscore"
        )
        
        # Mock the parent's __getitem__ method to return data
        mock_data = np.random.randn(10, 3)  # Mock returned data
        with patch.object(sampler, '__getitem__', return_value=mock_data):
            sampler._fit_normalization()
            
        # Check that parameters are calculated correctly
        assert sampler.norm_params is not None
        assert "mean" in sampler.norm_params
        assert "std" in sampler.norm_params
    
    def test_normalize_data_2d(self):
        """Test normalization of 2D data (single sample) within each time step"""
        # Create sampler instance (no real data needed)
        sampler = NormalizedTSDataSampler.__new__(NormalizedTSDataSampler)
        sampler.normalization_method = "minmax"
        sampler.step_len = 3  # 3 time steps
        
        # Set up normalization parameters for each time step and feature
        # Parameters shape: (step_len, num_features) = (3, 3)
        sampler.norm_params = {
            "min": np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),  # Same min for each time step
            "max": np.array([[7, 8, 9], [7, 8, 9], [7, 8, 9]])   # Same max for each time step
        }
        
        # Test data: shape (3, 3) - 3 time steps, 3 features
        test_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        
        # Test minmax normalization
        normalized = sampler._normalize_data(test_data)
        
        # Check shape is preserved
        assert normalized.shape == test_data.shape
        
        # Check normalization: (x - min) / (max - min)
        # Each time step should be normalized independently
        expected = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]], dtype=np.float32)
        np.testing.assert_array_almost_equal(normalized, expected)
    
    def test_normalize_data_3d(self):
        """Test normalization of 3D data (batch of samples) within each time step"""
        # Create sampler instance (no real data needed)
        sampler = NormalizedTSDataSampler.__new__(NormalizedTSDataSampler)
        sampler.normalization_method = "zscore"
        sampler.step_len = 3  # 3 time steps
        
        # Set up normalization parameters for each time step and feature
        # Parameters shape: (step_len, num_features) = (3, 4)
        sampler.norm_params = {
            "mean": np.array([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]),
            "std": np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
        }
        
        # Test data: shape (batch_size, step_len, features) = (2, 3, 4)
        test_data = np.random.randn(2, 3, 4)
        
        # Test zscore normalization
        normalized = sampler._normalize_data(test_data)
        
        # Check shape is preserved
        assert normalized.shape == test_data.shape
        
        # Verify that normalization is applied correctly
        # Each time step should be normalized independently
        expected = (test_data - sampler.norm_params["mean"].reshape(1, 3, 4)) / sampler.norm_params["std"].reshape(1, 3, 4)
        np.testing.assert_array_almost_equal(normalized, expected)
    
    def test_getitem_single_index(self):
        """Test __getitem__ with single index"""
        # Create test data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        instruments = [f'stock_{i}' for i in range(10)]
        index = pd.MultiIndex.from_product([instruments, dates], names=['instrument', 'datetime'])
        data = pd.DataFrame(np.random.randn(1000, 5), index=index)
        
        # Create sampler instance
        sampler = NormalizedTSDataSampler(
            data=data,
            start=pd.Timestamp('2020-01-01'),
            end=pd.Timestamp('2020-04-09'),
            step_len=10,
            normalization_method="robust"
        )
        
        # Mock the parent's __getitem__ method to return raw data
        mock_raw_data = np.random.randn(10, 5)
        with patch.object(sampler, '__getitem__', return_value=mock_raw_data):
            # Mock normalization parameter fitting
            sampler._fit_normalization()
            
            # Test single index access
            result = sampler[0]
            
            # Check that data shape is preserved
            assert result.shape == (10, 5)


class TestNormalizedTSDatasetH:
    """Test cases for NormalizedTSDatasetH class"""
    
    def test_init(self):
        """Test initialization"""
        # Mock handler and segments
        mock_handler = Mock()
        segments = {
            'train': ("2020-01-01", "2020-12-31"),
            'test': ("2021-01-01", "2021-12-31")
        }
        
        # Test with different normalization methods
        for method in ["zscore", "minmax", "robust"]:
            dataset = NormalizedTSDatasetH(
                handler=mock_handler,
                segments=segments,
                normalization_method=method
            )
            
            assert dataset.normalization_method == method
            assert dataset.segments == segments
            assert dataset.handler == mock_handler
    
    def test_config(self):
        """Test configuration method"""
        mock_handler = Mock()
        segments = {'train': ("2020-01-01", "2020-12-31")}
        
        dataset = NormalizedTSDatasetH(
            handler=mock_handler,
            segments=segments,
            normalization_method="zscore"
        )
        
        # Test changing normalization method via config
        dataset.config(normalization_method="minmax")
        assert dataset.normalization_method == "minmax"
        
        # Test changing step_len
        dataset.config(step_len=20)
        assert dataset.step_len == 20
    
    def test_prepare_seg(self):
        """Test _prepare_seg method"""
        mock_handler = Mock()
        
        dataset = NormalizedTSDatasetH(
            handler=mock_handler,
            normalization_method="zscore"
        )
        
        # Mock the parent's _prepare_seg method to return a mock TSDataSampler
        mock_ts_sampler = MagicMock()
        mock_ts_sampler.data = pd.DataFrame(np.random.randn(100, 5))
        mock_ts_sampler.start = pd.Timestamp('2020-01-01')
        mock_ts_sampler.end = pd.Timestamp('2020-04-09')
        mock_ts_sampler.step_len = 10
        
        # Use patch to mock the parent's _prepare_seg method
        with patch('qlib.data.dataset.TSDatasetH._prepare_seg', return_value=mock_ts_sampler):
            # Test preparing data segment
            result = dataset._prepare_seg(slice(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-04-09')))
            
            # Verify that NormalizedTSDataSampler instance is returned
            assert isinstance(result, NormalizedTSDataSampler)
            assert result.normalization_method == "zscore"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])