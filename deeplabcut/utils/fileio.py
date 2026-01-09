#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepLabCut2.0-3.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.
https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
Licensed under GNU Lesser General Public License v3.0

File I/O utilities with support for both HDF5 (legacy) and Parquet (new) formats.
This module provides backward-compatible functions to read and write dataframes
using Parquet format (via polars/pyarrow) while maintaining compatibility with
existing HDF5 files.
"""

import os
import warnings
from pathlib import Path
from typing import Optional, Union
import pandas as pd


def read_dataframe(
    filepath: Union[str, Path],
    key: Optional[str] = "df_with_missing",
    **kwargs
) -> pd.DataFrame:
    """
    Read a DataFrame from either Parquet or HDF5 format with backward compatibility.
    
    This function first attempts to read from a Parquet file (replacing .h5 extension
    with .parquet). If the Parquet file doesn't exist, it falls back to reading the
    HDF5 file (if it exists) and optionally converts it to Parquet format.
    
    Args:
        filepath: Path to the file (can be .h5 or .parquet)
        key: Key for HDF5 files (default: "df_with_missing")
        **kwargs: Additional arguments to pass to pd.read_parquet or pd.read_hdf
        
    Returns:
        pd.DataFrame: The loaded dataframe
        
    Raises:
        FileNotFoundError: If neither Parquet nor HDF5 file exists
    """
    filepath = Path(filepath)
    
    # Determine both possible file paths
    if filepath.suffix == ".h5":
        h5_path = filepath
        parquet_path = filepath.with_suffix(".parquet")
    elif filepath.suffix == ".parquet":
        parquet_path = filepath
        h5_path = filepath.with_suffix(".h5")
    else:
        # Try to infer format
        h5_path = filepath.with_suffix(".h5")
        parquet_path = filepath.with_suffix(".parquet")
    
    # Try to read Parquet first (new format)
    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path, **kwargs)
            return df
        except Exception as e:
            warnings.warn(
                f"Failed to read Parquet file {parquet_path}: {e}. "
                f"Falling back to HDF5."
            )
    
    # Fall back to HDF5 (legacy format)
    if h5_path.exists():
        try:
            # Note: This requires pytables to be installed as fallback
            # We keep h5py for backward compatibility
            try:
                df = pd.read_hdf(h5_path, key=key, **kwargs)
                
                # Auto-convert to Parquet for future use
                if not parquet_path.exists():
                    try:
                        write_dataframe(df, parquet_path)
                        print(f"Auto-converted {h5_path} to {parquet_path}")
                    except Exception as conv_error:
                        warnings.warn(
                            f"Could not auto-convert to Parquet: {conv_error}"
                        )
                
                return df
            except ImportError:
                raise ImportError(
                    "Reading HDF5 files requires pytables or h5py. "
                    "Please install with: pip install tables h5py"
                )
        except Exception as e:
            raise IOError(f"Failed to read HDF5 file {h5_path}: {e}")
    
    raise FileNotFoundError(
        f"Neither {parquet_path} nor {h5_path} exists"
    )


def write_dataframe(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    key: Optional[str] = "df_with_missing",
    format: str = "parquet",
    mode: str = "w",
    **kwargs
) -> None:
    """
    Write a DataFrame to Parquet format (default) with option for HDF5.
    
    Args:
        df: DataFrame to write
        filepath: Path to the output file
        key: Key for HDF5 files (only used if format="hdf5")
        format: Output format - "parquet" (default) or "hdf5"
        mode: Write mode - "w" for write, "a" for append (HDF5 only)
        **kwargs: Additional arguments to pass to to_parquet or to_hdf
        
    Returns:
        None
    """
    filepath = Path(filepath)
    
    if format == "parquet" or (format != "hdf5" and filepath.suffix == ".parquet"):
        # Ensure we use .parquet extension
        if filepath.suffix != ".parquet":
            filepath = filepath.with_suffix(".parquet")
        
        # Write to Parquet format
        df.to_parquet(filepath, **kwargs)
        
    elif format == "hdf5" or filepath.suffix == ".h5":
        # Ensure we use .h5 extension
        if filepath.suffix != ".h5":
            filepath = filepath.with_suffix(".h5")
        
        # Write to HDF5 format (requires pytables)
        try:
            df.to_hdf(filepath, key=key, mode=mode, format="table", **kwargs)
        except ImportError:
            raise ImportError(
                "Writing HDF5 files requires pytables. "
                "Please install with: pip install tables"
            )
    else:
        raise ValueError(
            f"Unsupported format: {format}. Use 'parquet' or 'hdf5'"
        )


def get_dataframe_path(
    filepath: Union[str, Path],
    prefer_parquet: bool = True
) -> Path:
    """
    Get the actual path to a dataframe file, checking for both formats.
    
    Args:
        filepath: Base filepath (can be .h5 or .parquet)
        prefer_parquet: If True, prefer .parquet over .h5 when both exist
        
    Returns:
        Path: Actual path to the file
        
    Raises:
        FileNotFoundError: If neither format exists
    """
    filepath = Path(filepath)
    
    # Determine both possible paths
    if filepath.suffix == ".h5":
        h5_path = filepath
        parquet_path = filepath.with_suffix(".parquet")
    elif filepath.suffix == ".parquet":
        parquet_path = filepath
        h5_path = filepath.with_suffix(".h5")
    else:
        h5_path = filepath.with_suffix(".h5")
        parquet_path = filepath.with_suffix(".parquet")
    
    # Check existence and prefer based on flag
    parquet_exists = parquet_path.exists()
    h5_exists = h5_path.exists()
    
    if not parquet_exists and not h5_exists:
        raise FileNotFoundError(
            f"Neither {parquet_path} nor {h5_path} exists"
        )
    
    if prefer_parquet and parquet_exists:
        return parquet_path
    elif h5_exists:
        return h5_path
    elif parquet_exists:
        return parquet_path
    else:
        raise FileNotFoundError(
            f"Neither {parquet_path} nor {h5_path} exists"
        )


def migrate_h5_to_parquet(
    directory: Union[str, Path],
    recursive: bool = True,
    remove_h5: bool = False
) -> int:
    """
    Migrate all HDF5 files in a directory to Parquet format.
    
    Args:
        directory: Directory containing .h5 files
        recursive: If True, search subdirectories recursively
        remove_h5: If True, remove .h5 files after successful conversion
        
    Returns:
        int: Number of files converted
    """
    directory = Path(directory)
    pattern = "**/*.h5" if recursive else "*.h5"
    
    converted_count = 0
    for h5_file in directory.glob(pattern):
        parquet_file = h5_file.with_suffix(".parquet")
        
        # Skip if parquet already exists
        if parquet_file.exists():
            print(f"Skipping {h5_file} (Parquet file already exists)")
            continue
        
        try:
            # Read from HDF5
            df = pd.read_hdf(h5_file, key="df_with_missing")
            
            # Write to Parquet
            df.to_parquet(parquet_file)
            
            print(f"Converted: {h5_file} -> {parquet_file}")
            converted_count += 1
            
            # Optionally remove HDF5 file
            if remove_h5:
                h5_file.unlink()
                print(f"Removed: {h5_file}")
                
        except Exception as e:
            warnings.warn(f"Failed to convert {h5_file}: {e}")
    
    return converted_count
