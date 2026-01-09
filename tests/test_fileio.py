#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for deeplabcut.utils.fileio module

This test suite covers:
1. Reading and writing DataFrames in HDF5 format (standard)
2. Polars integration for performance improvements
3. Converting between Pandas and Polars DataFrames
"""

import os
import tempfile
import shutil
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from deeplabcut.utils import fileio


class TestFileIO:
    """Test suite for file I/O utilities with Polars integration."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        np.random.seed(42)
        data = {
            'x': np.random.randn(100),
            'y': np.random.randn(100),
            'likelihood': np.random.rand(100)
        }
        index = [f"frame_{i}" for i in range(100)]
        return pd.DataFrame(data, index=index)
    
    def test_write_read_hdf5(self, temp_dir, sample_dataframe):
        """Test writing and reading HDF5 files (standard format)."""
        filepath = temp_dir / "test_data.h5"
        
        # Write
        fileio.write_hdf_with_polars(sample_dataframe, filepath, optimize_with_polars=False)
        assert filepath.exists()
        
        # Read
        df_read = fileio.read_hdf_with_polars(filepath, use_polars=False)
        pd.testing.assert_frame_equal(sample_dataframe, df_read)
    
    def test_write_read_hdf5_with_polars(self, temp_dir, sample_dataframe):
        """Test HDF5 I/O with Polars optimization."""
        # Skip if polars not available
        if not fileio.HAS_POLARS:
            pytest.skip("Polars not installed")
        
        filepath = temp_dir / "test_data_polars.h5"
        
        # Write with Polars optimization
        fileio.write_hdf_with_polars(sample_dataframe, filepath, optimize_with_polars=True)
        assert filepath.exists()
        
        # Read with Polars
        df_read = fileio.read_hdf_with_polars(filepath, use_polars=True)
        pd.testing.assert_frame_equal(sample_dataframe, df_read)
    
    def test_dataframe_to_polars_conversion(self, sample_dataframe):
        """Test converting Pandas DataFrame to Polars."""
        if not fileio.HAS_POLARS:
            pytest.skip("Polars not installed")
        
        # Convert to Polars
        pl_df = fileio.dataframe_to_polars(sample_dataframe)
        
        # Convert back to Pandas
        df_back = fileio.polars_to_dataframe(pl_df)
        
        # Should be equal
        pd.testing.assert_frame_equal(sample_dataframe, df_back)
    
    def test_process_with_polars(self, sample_dataframe):
        """Test processing DataFrame with Polars operations."""
        if not fileio.HAS_POLARS:
            pytest.skip("Polars not installed")
        
        import polars as pl
        
        # Define a filter operation
        def filter_high_likelihood(pl_df):
            return pl_df.filter(pl.col("likelihood") > 0.5)
        
        # Process with Polars
        result_df = fileio.process_with_polars(sample_dataframe, filter_high_likelihood)
        
        # Verify the result
        assert len(result_df) < len(sample_dataframe)
        assert all(result_df['likelihood'] > 0.5)
    
    def test_multiindex_dataframe(self, temp_dir):
        """Test with MultiIndex DataFrame (DeepLabCut format)."""
        # Create a MultiIndex DataFrame similar to DeepLabCut format
        scorer = "DLC_resnet50"
        bodyparts = ["nose", "tail"]
        coords = ["x", "y", "likelihood"]
        
        iterables = [[scorer], bodyparts, coords]
        columns = pd.MultiIndex.from_product(iterables, names=["scorer", "bodyparts", "coords"])
        
        data = np.random.randn(50, 6)
        index = [f"img_{i:05d}.png" for i in range(50)]
        df = pd.DataFrame(data, columns=columns, index=index)
        
        # Write and read
        filepath = temp_dir / "multiindex_test.h5"
        fileio.write_hdf_with_polars(df, filepath)
        df_read = fileio.read_hdf_with_polars(filepath)
        
        pd.testing.assert_frame_equal(df, df_read)
    
    def test_file_extension_handling(self, temp_dir, sample_dataframe):
        """Test that .h5 extension is enforced."""
        # Try to write without extension
        filepath = temp_dir / "test_data"
        
        with pytest.warns(UserWarning):
            fileio.write_hdf_with_polars(sample_dataframe, filepath)
        
        # Should create .h5 file
        h5_path = temp_dir / "test_data.h5"
        assert h5_path.exists()
    
    def test_polars_not_available_fallback(self, temp_dir, sample_dataframe):
        """Test that operations work even without Polars."""
        filepath = temp_dir / "test_no_polars.h5"
        
        # Should work regardless of Polars availability
        fileio.write_hdf_with_polars(sample_dataframe, filepath, optimize_with_polars=False)
        df_read = fileio.read_hdf_with_polars(filepath, use_polars=False)
        
        pd.testing.assert_frame_equal(sample_dataframe, df_read)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

