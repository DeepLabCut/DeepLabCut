#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for deeplabcut.utils.fileio module

This test suite covers:
1. Reading and writing DataFrames in Parquet format
2. Backward compatibility with HDF5 files
3. Auto-conversion from HDF5 to Parquet
4. Migration utilities
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
    """Test suite for file I/O utilities."""
    
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
        """Test writing and reading Parquet files."""
        filepath = temp_dir / "test_data.parquet"
        
        # Write
        fileio.write_dataframe(sample_dataframe, filepath, format="parquet")
        assert filepath.exists()
        
        # Read
        df_read = fileio.read_dataframe(filepath)
        pd.testing.assert_frame_equal(sample_dataframe, df_read)
    
    def test_write_read_with_h5_extension(self, temp_dir, sample_dataframe):
        """Test that .h5 filepath is auto-converted to .parquet."""
        filepath = temp_dir / "test_data.h5"
        
        # Write (should create .parquet file)
        fileio.write_dataframe(sample_dataframe, filepath, format="parquet")
        
        # Check that parquet file was created
        parquet_path = temp_dir / "test_data.parquet"
        assert parquet_path.exists()
        
        # Read using original .h5 path (should find .parquet)
        df_read = fileio.read_dataframe(filepath)
        pd.testing.assert_frame_equal(sample_dataframe, df_read)
    
    def test_backward_compatibility_hdf5(self, temp_dir, sample_dataframe):
        """Test reading legacy HDF5 files."""
        # Skip test if pytables is not installed
        pytest.importorskip("tables")
        
        h5_path = temp_dir / "test_data.h5"
        
        # Write HDF5 file using pandas directly (simulating legacy data)
        sample_dataframe.to_hdf(h5_path, key="df_with_missing", mode="w", format="table")
        assert h5_path.exists()
        
        # Read using our utility (should work and auto-convert)
        df_read = fileio.read_dataframe(h5_path)
        pd.testing.assert_frame_equal(sample_dataframe, df_read)
        
        # Check that auto-conversion created a parquet file
        parquet_path = temp_dir / "test_data.parquet"
        assert parquet_path.exists()
    
    def test_prefer_parquet_over_h5(self, temp_dir, sample_dataframe):
        """Test that Parquet is preferred when both formats exist."""
        # Skip test if pytables is not installed
        pytest.importorskip("tables")
        
        h5_path = temp_dir / "test_data.h5"
        parquet_path = temp_dir / "test_data.parquet"
        
        # Create slightly different dataframes
        df_h5 = sample_dataframe.copy()
        df_parquet = sample_dataframe.copy()
        df_parquet['x'] = df_parquet['x'] + 1  # Make them different
        
        # Write both formats
        df_h5.to_hdf(h5_path, key="df_with_missing", mode="w", format="table")
        fileio.write_dataframe(df_parquet, parquet_path, format="parquet")
        
        # Read (should prefer parquet)
        df_read = fileio.read_dataframe(h5_path)
        pd.testing.assert_frame_equal(df_parquet, df_read)
    
    def test_get_dataframe_path(self, temp_dir, sample_dataframe):
        """Test getting the actual path to a dataframe file."""
        parquet_path = temp_dir / "test_data.parquet"
        fileio.write_dataframe(sample_dataframe, parquet_path, format="parquet")
        
        # Test with .h5 extension (should find .parquet)
        actual_path = fileio.get_dataframe_path(temp_dir / "test_data.h5")
        assert actual_path == parquet_path
        
        # Test with .parquet extension
        actual_path = fileio.get_dataframe_path(parquet_path)
        assert actual_path == parquet_path
    
    def test_file_not_found(self, temp_dir):
        """Test that appropriate error is raised when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            fileio.read_dataframe(temp_dir / "nonexistent.h5")
        
        with pytest.raises(FileNotFoundError):
            fileio.get_dataframe_path(temp_dir / "nonexistent.parquet")
    
    def test_migrate_h5_to_parquet(self, temp_dir, sample_dataframe):
        """Test batch migration of HDF5 files to Parquet."""
        # Skip test if pytables is not installed
        pytest.importorskip("tables")
        
        # Create subdirectories with HDF5 files
        subdir1 = temp_dir / "subdir1"
        subdir2 = temp_dir / "subdir2"
        subdir1.mkdir()
        subdir2.mkdir()
        
        h5_files = [
            temp_dir / "test1.h5",
            subdir1 / "test2.h5",
            subdir2 / "test3.h5"
        ]
        
        # Create HDF5 files
        for h5_file in h5_files:
            sample_dataframe.to_hdf(h5_file, key="df_with_missing", mode="w", format="table")
        
        # Migrate
        converted_count = fileio.migrate_h5_to_parquet(temp_dir, recursive=True, remove_h5=False)
        assert converted_count == 3
        
        # Check that parquet files exist
        for h5_file in h5_files:
            parquet_file = h5_file.with_suffix(".parquet")
            assert parquet_file.exists()
            # Verify content
            df_read = pd.read_parquet(parquet_file)
            pd.testing.assert_frame_equal(sample_dataframe, df_read)
    
    def test_migrate_with_remove_h5(self, temp_dir, sample_dataframe):
        """Test migration with removal of original HDF5 files."""
        # Skip test if pytables is not installed
        pytest.importorskip("tables")
        
        h5_path = temp_dir / "test_remove.h5"
        sample_dataframe.to_hdf(h5_path, key="df_with_missing", mode="w", format="table")
        
        # Migrate with remove option
        converted_count = fileio.migrate_h5_to_parquet(temp_dir, recursive=False, remove_h5=True)
        assert converted_count == 1
        
        # Check that h5 file is removed
        assert not h5_path.exists()
        
        # Check that parquet file exists
        parquet_path = temp_dir / "test_remove.parquet"
        assert parquet_path.exists()
    
    def test_multiindex_dataframe(self, temp_dir):
        """Test with MultiIndex DataFrame (common in DeepLabCut)."""
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
        filepath = temp_dir / "multiindex_test.parquet"
        fileio.write_dataframe(df, filepath, format="parquet")
        df_read = fileio.read_dataframe(filepath)
        
        pd.testing.assert_frame_equal(df, df_read)
    
    def test_write_hdf5_format(self, temp_dir, sample_dataframe):
        """Test that HDF5 format still works when explicitly requested."""
        # Skip test if pytables is not installed
        pytest.importorskip("tables")
        
        h5_path = temp_dir / "explicit_h5.h5"
        
        # Write in HDF5 format explicitly
        fileio.write_dataframe(sample_dataframe, h5_path, format="hdf5")
        assert h5_path.exists()
        
        # Read it back
        df_read = fileio.read_dataframe(h5_path)
        pd.testing.assert_frame_equal(sample_dataframe, df_read)
    
    def test_csv_compatibility(self, temp_dir, sample_dataframe):
        """Test that CSV files can still be written alongside Parquet."""
        parquet_path = temp_dir / "test_with_csv.parquet"
        csv_path = temp_dir / "test_with_csv.csv"
        
        # Write parquet
        fileio.write_dataframe(sample_dataframe, parquet_path, format="parquet")
        
        # Write CSV manually (as the code does)
        sample_dataframe.to_csv(csv_path)
        
        # Verify both exist
        assert parquet_path.exists()
        assert csv_path.exists()
        
        # Verify parquet content
        df_read = fileio.read_dataframe(parquet_path)
        pd.testing.assert_frame_equal(sample_dataframe, df_read)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
