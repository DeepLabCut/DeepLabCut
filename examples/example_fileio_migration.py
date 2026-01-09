#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script demonstrating the new Parquet-based file I/O in DeepLabCut

This script shows:
1. How to write DataFrames in the new Parquet format
2. How backward compatibility works with existing HDF5 files
3. How to migrate existing HDF5 files to Parquet
"""

import sys
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

# Add deeplabcut to path (for running from examples directory)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the fileio module directly to avoid dependency issues
import warnings
warnings.filterwarnings('ignore')

# Load the fileio module
try:
    from deeplabcut.utils import fileio
    print("✓ Successfully imported deeplabcut.utils.fileio")
except ImportError:
    # Fallback: load module directly
    exec(open(Path(__file__).parent.parent / 'deeplabcut' / 'utils' / 'fileio.py').read())
    print("✓ Loaded fileio module directly")


def create_sample_dlc_dataframe():
    """Create a sample DeepLabCut-style DataFrame with MultiIndex columns."""
    np.random.seed(42)
    
    # DeepLabCut format: MultiIndex with (scorer, bodyparts, coords)
    scorer = "DLC_resnet50_example"
    bodyparts = ["nose", "leftear", "rightear", "tail"]
    coords = ["x", "y", "likelihood"]
    
    # Create MultiIndex columns
    iterables = [[scorer], bodyparts, coords]
    columns = pd.MultiIndex.from_product(
        iterables,
        names=["scorer", "bodyparts", "coords"]
    )
    
    # Generate sample data (100 frames, 4 bodyparts, 3 coords)
    n_frames = 100
    data = np.random.randn(n_frames, len(bodyparts) * len(coords))
    
    # Create DataFrame with image paths as index
    index = [f"img_{i:05d}.png" for i in range(n_frames)]
    df = pd.DataFrame(data, columns=columns, index=index)
    
    # Set likelihood values to be between 0 and 1
    for bodypart in bodyparts:
        df[(scorer, bodypart, "likelihood")] = np.random.rand(n_frames)
    
    return df


def example_1_basic_write_read():
    """Example 1: Basic write and read with Parquet format."""
    print("\n" + "="*60)
    print("Example 1: Basic Write and Read")
    print("="*60)
    
    # Create sample data
    df = create_sample_dlc_dataframe()
    print(f"Created DataFrame with shape: {df.shape}")
    print(f"Columns: {df.columns.names}")
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    try:
        # Write to Parquet
        output_path = temp_dir / "poses.parquet"
        write_dataframe(df, output_path, format="parquet")
        print(f"✓ Written to: {output_path}")
        print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
        
        # Read back
        df_read = read_dataframe(output_path)
        print(f"✓ Read from: {output_path}")
        
        # Verify
        assert df.equals(df_read), "DataFrames don't match!"
        print("✓ Data verified: Original and read DataFrames match")
        
    finally:
        shutil.rmtree(temp_dir)


def example_2_h5_compatibility():
    """Example 2: Backward compatibility with .h5 file paths."""
    print("\n" + "="*60)
    print("Example 2: Backward Compatibility")
    print("="*60)
    
    df = create_sample_dlc_dataframe()
    
    temp_dir = Path(tempfile.mkdtemp())
    try:
        # Use .h5 extension (will create .parquet file)
        h5_path = temp_dir / "poses_analysis.h5"
        write_dataframe(df, h5_path, format="parquet")
        
        # Check what was actually created
        parquet_path = temp_dir / "poses_analysis.parquet"
        print(f"✓ Requested: {h5_path.name}")
        print(f"✓ Created:   {parquet_path.name}")
        assert parquet_path.exists()
        
        # Read using .h5 path (will find .parquet file)
        df_read = read_dataframe(h5_path)
        print(f"✓ Successfully read using .h5 path")
        
        assert df.equals(df_read)
        print("✓ Data verified: Compatible with .h5 file paths")
        
    finally:
        shutil.rmtree(temp_dir)


def example_3_migration():
    """Example 3: Migrating existing HDF5 files to Parquet."""
    print("\n" + "="*60)
    print("Example 3: HDF5 to Parquet Migration")
    print("="*60)
    
    # Check if pytables is available
    try:
        import tables
        has_pytables = True
    except ImportError:
        has_pytables = False
        print("⚠ PyTables not installed - skipping HDF5 creation")
        print("  Install with: pip install tables")
        return
    
    df = create_sample_dlc_dataframe()
    
    temp_dir = Path(tempfile.mkdtemp())
    try:
        # Create subdirectories with HDF5 files (simulating a project structure)
        video_dirs = ["video1", "video2", "video3"]
        for video_dir in video_dirs:
            (temp_dir / video_dir).mkdir()
        
        h5_files = [
            temp_dir / "video1" / "poses.h5",
            temp_dir / "video2" / "poses.h5",
            temp_dir / "video3" / "poses.h5",
        ]
        
        # Create HDF5 files
        for h5_file in h5_files:
            df.to_hdf(h5_file, key="df_with_missing", mode="w", format="table")
            print(f"Created: {h5_file.relative_to(temp_dir)}")
        
        # Migrate all files
        print(f"\nMigrating files in: {temp_dir}")
        converted_count = migrate_h5_to_parquet(
            temp_dir,
            recursive=True,
            remove_h5=False
        )
        
        print(f"✓ Converted {converted_count} files to Parquet")
        
        # Verify migration
        for h5_file in h5_files:
            parquet_file = h5_file.with_suffix(".parquet")
            assert parquet_file.exists(), f"Parquet file not created: {parquet_file}"
            
            # Verify content
            df_read = read_dataframe(parquet_file)
            assert df.equals(df_read), f"Data mismatch in {parquet_file}"
        
        print("✓ All files successfully migrated and verified")
        
    finally:
        shutil.rmtree(temp_dir)


def example_4_performance_comparison():
    """Example 4: Compare file sizes and performance."""
    print("\n" + "="*60)
    print("Example 4: Performance Comparison")
    print("="*60)
    
    # Create larger dataset for meaningful comparison
    np.random.seed(42)
    scorer = "DLC_test"
    bodyparts = [f"bp{i}" for i in range(20)]  # 20 bodyparts
    coords = ["x", "y", "likelihood"]
    
    iterables = [[scorer], bodyparts, coords]
    columns = pd.MultiIndex.from_product(iterables, names=["scorer", "bodyparts", "coords"])
    
    n_frames = 10000  # 10k frames
    data = np.random.randn(n_frames, len(bodyparts) * len(coords))
    index = [f"img_{i:05d}.png" for i in range(n_frames)]
    df = pd.DataFrame(data, columns=columns, index=index)
    
    print(f"Dataset: {n_frames} frames, {len(bodyparts)} bodyparts")
    print(f"Shape: {df.shape}")
    
    temp_dir = Path(tempfile.mkdtemp())
    try:
        import time
        
        # Test Parquet
        parquet_path = temp_dir / "data.parquet"
        t0 = time.time()
        write_dataframe(df, parquet_path, format="parquet")
        write_time_parquet = time.time() - t0
        
        t0 = time.time()
        df_read = read_dataframe(parquet_path)
        read_time_parquet = time.time() - t0
        
        size_parquet = parquet_path.stat().st_size / 1024 / 1024  # MB
        
        print(f"\nParquet Format:")
        print(f"  Write time: {write_time_parquet:.3f}s")
        print(f"  Read time:  {read_time_parquet:.3f}s")
        print(f"  File size:  {size_parquet:.2f} MB")
        
        # Test HDF5 if available
        try:
            import tables
            h5_path = temp_dir / "data.h5"
            
            t0 = time.time()
            df.to_hdf(h5_path, key="df_with_missing", mode="w", format="table")
            write_time_h5 = time.time() - t0
            
            t0 = time.time()
            df_read_h5 = pd.read_hdf(h5_path)
            read_time_h5 = time.time() - t0
            
            size_h5 = h5_path.stat().st_size / 1024 / 1024  # MB
            
            print(f"\nHDF5 Format:")
            print(f"  Write time: {write_time_h5:.3f}s")
            print(f"  Read time:  {read_time_h5:.3f}s")
            print(f"  File size:  {size_h5:.2f} MB")
            
            print(f"\nComparison:")
            print(f"  Parquet is {write_time_h5/write_time_parquet:.1f}x faster for writing")
            print(f"  Parquet is {read_time_h5/read_time_parquet:.1f}x faster for reading")
            print(f"  Parquet is {size_h5/size_parquet:.1f}x smaller in file size")
            
        except ImportError:
            print("\n⚠ PyTables not installed - cannot compare with HDF5")
            print("  Install with: pip install tables")
        
    finally:
        shutil.rmtree(temp_dir)


def main():
    """Run all examples."""
    print("="*60)
    print("DeepLabCut File I/O Examples")
    print("PyTables → Polars/Parquet Migration")
    print("="*60)
    
    try:
        example_1_basic_write_read()
        example_2_h5_compatibility()
        example_3_migration()
        example_4_performance_comparison()
        
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
