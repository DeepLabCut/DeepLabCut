#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for deeplabcut.utils.fileio module

This test suite covers:
1. Reading and writing DataFrames in Parquet format (primary)
2. Optional HDF5 export functionality
3. Converting between formats
4. Polars integration
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
    """Test suite for file I/O utilities with Parquet as primary format."""
    
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
    
    def test_write_read_parquet(self, temp_dir, sample_dataframe):
        """Test writing and reading Parquet files (primary format)."""
        filepath = temp_dir / "test_data.parquet"
        
        # Write
        fileio.write_dataframe(sample_dataframe, filepath, format="parquet")
        assert filepath.exists()
        
        # Read
        df_read = fileio.read_dataframe(filepath, format="parquet")
        pd.testing.assert_frame_equal(sample_dataframe, df_read)
    
    def test_auto_format_detection(self, temp_dir, sample_dataframe):
        """Test automatic format detection."""
        # Write Parquet
        parquet_path = temp_dir / "test_auto.parquet"
        fileio.write_dataframe(sample_dataframe, parquet_path)
        
        # Read with auto-detection
        df_read = fileio.read_dataframe(parquet_path, format="auto")
        pd.testing.assert_frame_equal(sample_dataframe, df_read)
    
    def test_optional_hdf5_export(self, temp_dir, sample_dataframe):
        """Test optional HDF5 export functionality."""
        # Skip if pytables not available
        try:
            import tables
        except ImportError:
            pytest.skip("PyTables not installed (optional dependency)")
        
        h5_path = temp_dir / "test_export.h5"
        
        # Write HDF5 (optional format)
        fileio.write_dataframe(sample_dataframe, h5_path, format="hdf5")
        assert h5_path.exists()
        
        # Read back
        df_read = fileio.read_dataframe(h5_path, format="hdf5")
        pd.testing.assert_frame_equal(sample_dataframe, df_read)
    
    def test_convert_to_parquet(self, temp_dir, sample_dataframe):
        """Test converting HDF5 to Parquet."""
        # Skip if pytables not available
        try:
            import tables
        except ImportError:
            pytest.skip("PyTables not installed (optional dependency)")
        
        # Create HDF5 file
        h5_path = temp_dir / "legacy.h5"
        sample_dataframe.to_hdf(h5_path, key="df_with_missing", mode="w", format="table")
        
        # Convert to Parquet
        parquet_path = fileio.convert_to_parquet(h5_path)
        
        # Verify conversion
        assert parquet_path.exists()
        assert parquet_path.suffix == ".parquet"
        
        # Verify data integrity
        df_read = fileio.read_dataframe(parquet_path)
        pd.testing.assert_frame_equal(sample_dataframe, df_read)
    
    def test_migrate_directory(self, temp_dir, sample_dataframe):
        """Test batch migration of HDF5 files to Parquet."""
        # Skip if pytables not available
        try:
            import tables
        except ImportError:
            pytest.skip("PyTables not installed (optional dependency)")
        
        # Create subdirectories with HDF5 files
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        
        h5_files = [
            temp_dir / "test1.h5",
            subdir / "test2.h5"
        ]
        
        # Create HDF5 files
        for h5_file in h5_files:
            sample_dataframe.to_hdf(h5_file, key="df_with_missing", mode="w", format="table")
        
        # Migrate
        converted_count = fileio.migrate_directory_to_parquet(temp_dir, recursive=True)
        assert converted_count == 2
        
        # Verify Parquet files exist
        for h5_file in h5_files:
            parquet_file = h5_file.with_suffix(".parquet")
            assert parquet_file.exists()
    
    def test_multiindex_dataframe(self, temp_dir):
        """Test with MultiIndex DataFrame (DeepLabCut format)."""
        # Create a MultiIndex DataFrame
        scorer = "DLC_resnet50"
        bodyparts = ["nose", "tail"]
        coords = ["x", "y", "likelihood"]
        
        iterables = [[scorer], bodyparts, coords]
        columns = pd.MultiIndex.from_product(iterables, names=["scorer", "bodyparts", "coords"])
        
        data = np.random.randn(50, 6)
        index = [f"img_{i:05d}.png" for i in range(50)]
        df = pd.DataFrame(data, columns=columns, index=index)
        
        # Write and read Parquet
        filepath = temp_dir / "multiindex_test.parquet"
        fileio.write_dataframe(df, filepath, format="parquet")
        df_read = fileio.read_dataframe(filepath)
        
        pd.testing.assert_frame_equal(df, df_read)
    
    def test_polars_conversion(self, sample_dataframe):
        """Test converting between Pandas and Polars."""
        # Convert to Polars
        pl_df = fileio.dataframe_to_polars(sample_dataframe)
        
        # Convert back to Pandas
        df_back = fileio.polars_to_dataframe(pl_df)
        
        # Should be equal
        pd.testing.assert_frame_equal(sample_dataframe, df_back)
    
    def test_file_extension_handling(self, temp_dir, sample_dataframe):
        """Test automatic file extension handling."""
        # Write without extension
        filepath = temp_dir / "test_data"
        fileio.write_dataframe(sample_dataframe, filepath, format="parquet")
        
        # Should create .parquet file
        parquet_path = temp_dir / "test_data.parquet"
        assert parquet_path.exists()
    
    def test_read_with_alternate_extension(self, temp_dir, sample_dataframe):
        """Test reading file with alternate extension."""
        # Write as Parquet
        parquet_path = temp_dir / "test.parquet"
        fileio.write_dataframe(sample_dataframe, parquet_path)
        
        # Try to read with .h5 extension (should find .parquet)
        h5_path = temp_dir / "test.h5"
        df_read = fileio.read_dataframe(h5_path, format="auto")
        
        pd.testing.assert_frame_equal(sample_dataframe, df_read)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


