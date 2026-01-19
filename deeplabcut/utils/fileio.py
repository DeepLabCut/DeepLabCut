#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepLabCut2.0-3.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.
https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
Licensed under GNU Lesser General Public License v3.0

File I/O utilities using Parquet as the primary format with optional HDF5/NWB export.
This module uses Polars/Parquet for efficient data storage and provides optional
conversion to HDF5 or NWB formats using pandas.
"""

import os
import warnings
from pathlib import Path
from typing import Optional, Union, Literal
import pandas as pd

# Polars is required for the main functionality
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    raise ImportError(
        "Polars is required. Install with: pip install polars pyarrow"
    )


def read_dataframe(
    filepath: Union[str, Path],
    format: Optional[Literal["parquet", "hdf5", "auto"]] = "auto",
    key: str = "df_with_missing",
    **kwargs
) -> pd.DataFrame:
    """
    Read a DataFrame from Parquet (primary) or HDF5 (legacy) format.
    
    Args:
        filepath: Path to the file
        format: File format - "parquet" (default), "hdf5" (legacy), or "auto" (detect)
        key: Key for HDF5 files (default: "df_with_missing")
        **kwargs: Additional arguments to pass to read function
        
    Returns:
        pd.DataFrame: The loaded dataframe
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        # Try alternate extensions
        if filepath.suffix == ".h5" and filepath.with_suffix(".parquet").exists():
            filepath = filepath.with_suffix(".parquet")
        elif filepath.suffix == ".parquet" and filepath.with_suffix(".h5").exists():
            filepath = filepath.with_suffix(".h5")
        else:
            raise FileNotFoundError(f"File not found: {filepath}")
    
    # Auto-detect format if requested
    if format == "auto":
        if filepath.suffix == ".parquet":
            format = "parquet"
        elif filepath.suffix == ".h5":
            format = "hdf5"
        else:
            # Default to parquet
            format = "parquet"
    
    if format == "parquet":
        # Read Parquet file (primary format)
        return pd.read_parquet(filepath, **kwargs)
    
    elif format == "hdf5":
        # Read HDF5 file (legacy support using pandas without pytables)
        try:
            return pd.read_hdf(filepath, key=key, **kwargs)
        except ImportError as e:
            raise ImportError(
                "Reading HDF5 files requires 'tables' or 'h5py'. "
                "Install with: pip install tables h5py\n"
                "Note: Parquet is now the primary format. Consider converting legacy files."
            ) from e
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def write_dataframe(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    format: Literal["parquet", "hdf5", "nwb"] = "parquet",
    key: str = "df_with_missing",
    mode: str = "w",
    **kwargs
) -> None:
    """
    Write a DataFrame to Parquet (primary) or optionally to HDF5/NWB format.
    
    Args:
        df: DataFrame to write
        filepath: Path to the output file
        format: Output format - "parquet" (default), "hdf5" (optional), or "nwb" (optional)
        key: Key for HDF5 files (default: "df_with_missing")
        mode: Write mode - "w" for write, "a" for append (HDF5 only)
        **kwargs: Additional arguments to pass to write function
        
    Returns:
        None
    """
    filepath = Path(filepath)
    
    if format == "parquet":
        # Write Parquet format (primary)
        if filepath.suffix != ".parquet":
            filepath = filepath.with_suffix(".parquet")
        df.to_parquet(filepath, **kwargs)
    
    elif format == "hdf5":
        # Write HDF5 format (optional, using pandas)
        if filepath.suffix != ".h5":
            filepath = filepath.with_suffix(".h5")
        try:
            df.to_hdf(filepath, key=key, mode=mode, format="table", **kwargs)
        except ImportError as e:
            raise ImportError(
                "Writing HDF5 files requires 'tables'. "
                "Install with: pip install tables\n"
                "Note: Parquet is the primary format. Consider using Parquet instead."
            ) from e
    
    elif format == "nwb":
        # Write NWB format (optional)
        try:
            from pynwb import NWBFile, NWBHDF5IO
            from datetime import datetime
            
            # This is a placeholder - actual NWB writing would need more structure
            warnings.warn(
                "NWB export is experimental. Full NWB support requires additional configuration."
            )
            
            # For now, fall back to HDF5
            if filepath.suffix != ".nwb":
                filepath = filepath.with_suffix(".nwb")
            df.to_hdf(filepath, key=key, mode=mode, format="table", **kwargs)
            
        except ImportError as e:
            raise ImportError(
                "Writing NWB files requires 'pynwb' and 'tables'. "
                "Install with: pip install pynwb tables"
            ) from e
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def convert_to_parquet(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    source_format: Literal["hdf5", "auto"] = "auto",
    key: str = "df_with_missing",
    remove_source: bool = False
) -> Path:
    """
    Convert HDF5 file to Parquet format.
    
    Args:
        input_path: Path to the source file
        output_path: Path to the output Parquet file (optional, auto-generated if None)
        source_format: Source format - "hdf5" or "auto" (detect)
        key: Key for HDF5 files
        remove_source: If True, remove source file after conversion
        
    Returns:
        Path: Path to the created Parquet file
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path.with_suffix(".parquet")
    else:
        output_path = Path(output_path)
    
    # Read from source
    df = read_dataframe(input_path, format=source_format, key=key)
    
    # Write to Parquet
    write_dataframe(df, output_path, format="parquet")
    
    print(f"Converted: {input_path} -> {output_path}")
    
    # Optionally remove source
    if remove_source:
        input_path.unlink()
        print(f"Removed: {input_path}")
    
    return output_path


def migrate_directory_to_parquet(
    directory: Union[str, Path],
    pattern: str = "*.h5",
    recursive: bool = True,
    remove_source: bool = False
) -> int:
    """
    Convert all HDF5 files in a directory to Parquet format.
    
    Args:
        directory: Directory containing files to convert
        pattern: Glob pattern for files to convert (default: "*.h5")
        recursive: If True, search subdirectories recursively
        remove_source: If True, remove source files after conversion
        
    Returns:
        int: Number of files converted
    """
    directory = Path(directory)
    glob_pattern = f"**/{pattern}" if recursive else pattern
    
    converted_count = 0
    for file_path in directory.glob(glob_pattern):
        try:
            parquet_path = file_path.with_suffix(".parquet")
            
            # Skip if parquet already exists
            if parquet_path.exists():
                print(f"Skipping {file_path} (Parquet already exists)")
                continue
            
            # Convert
            convert_to_parquet(file_path, parquet_path, remove_source=remove_source)
            converted_count += 1
            
        except Exception as e:
            warnings.warn(f"Failed to convert {file_path}: {e}")
    
    return converted_count


def dataframe_to_polars(df: pd.DataFrame) -> pl.DataFrame:
    """
    Convert Pandas DataFrame to Polars DataFrame.
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        Polars DataFrame
    """
    return pl.from_pandas(df)


def polars_to_dataframe(pl_df: pl.DataFrame) -> pd.DataFrame:
    """
    Convert Polars DataFrame to Pandas DataFrame.
    
    Args:
        pl_df: Polars DataFrame
        
    Returns:
        Pandas DataFrame
    """
    return pl_df.to_pandas()


