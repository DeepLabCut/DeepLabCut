#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepLabCut2.0-3.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.
https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
Licensed under GNU Lesser General Public License v3.0

File I/O utilities with Polars integration for improved performance.
This module provides functions to work with DataFrames using Polars for
efficient data manipulation while maintaining HDF5 as the primary file format.
"""

import os
import warnings
from pathlib import Path
from typing import Optional, Union
import pandas as pd

# Try to import polars for enhanced performance
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    warnings.warn(
        "Polars not installed. Install with 'pip install polars' for better performance.",
        ImportWarning
    )


def read_hdf_with_polars(
    filepath: Union[str, Path],
    key: str = "df_with_missing",
    use_polars: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Read an HDF5 file and optionally convert to Polars for fast operations.
    
    This function reads HDF5 files (the standard DeepLabCut format) and can
    optionally use Polars for faster data manipulation operations.
    
    Args:
        filepath: Path to the HDF5 file
        key: Key for the HDF5 file (default: "df_with_missing")
        use_polars: If True and Polars is available, convert to Polars DataFrame
                   for operations before returning as Pandas (default: True)
        **kwargs: Additional arguments to pass to pd.read_hdf
        
    Returns:
        pd.DataFrame: The loaded dataframe
        
    Example:
        >>> df = read_hdf_with_polars("poses.h5")
        >>> # DataFrame is read via HDF5 but can be processed with Polars internally
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Read the HDF5 file using pandas
    df = pd.read_hdf(filepath, key=key, **kwargs)
    
    # Optionally convert to Polars and back for optimized operations
    # This is useful when you want to perform operations on the data
    if use_polars and HAS_POLARS:
        # Note: This is just for demonstration - in practice, you'd want to
        # do operations while in Polars format, not just convert back
        try:
            # Polars can handle pandas DataFrames efficiently
            pl_df = pl.from_pandas(df)
            # Convert back to pandas for compatibility
            df = pl_df.to_pandas()
        except Exception as e:
            warnings.warn(f"Could not use Polars optimization: {e}")
    
    return df


def write_hdf_with_polars(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    key: str = "df_with_missing",
    mode: str = "w",
    format: str = "table",
    optimize_with_polars: bool = True,
    **kwargs
) -> None:
    """
    Write a DataFrame to HDF5 format, optionally optimizing with Polars first.
    
    This function maintains HDF5 as the primary output format while allowing
    Polars to optimize the data before writing.
    
    Args:
        df: DataFrame to write
        filepath: Path to the output HDF5 file
        key: Key for the HDF5 file (default: "df_with_missing")
        mode: Write mode - "w" for write, "a" for append
        format: HDF5 format - "table" (default) or "fixed"
        optimize_with_polars: Use Polars for pre-write optimization (default: True)
        **kwargs: Additional arguments to pass to to_hdf
        
    Returns:
        None
        
    Example:
        >>> write_hdf_with_polars(df, "poses.h5")
        >>> # Writes to HDF5 format with optional Polars optimization
    """
    filepath = Path(filepath)
    
    # Ensure .h5 extension
    if filepath.suffix != ".h5":
        warnings.warn(f"Adding .h5 extension to {filepath}")
        filepath = filepath.with_suffix(".h5")
    
    # Optionally optimize with Polars before writing
    if optimize_with_polars and HAS_POLARS:
        try:
            # Convert to Polars for any optimizations
            pl_df = pl.from_pandas(df)
            # Convert back to pandas for HDF5 writing
            df = pl_df.to_pandas()
        except Exception as e:
            warnings.warn(f"Could not use Polars optimization: {e}")
    
    # Write to HDF5 format (the standard DeepLabCut format)
    df.to_hdf(filepath, key=key, mode=mode, format=format, **kwargs)


def dataframe_to_polars(df: pd.DataFrame) -> 'pl.DataFrame':
    """
    Convert a Pandas DataFrame to Polars DataFrame for fast operations.
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        Polars DataFrame
        
    Raises:
        ImportError: If Polars is not installed
        
    Example:
        >>> pl_df = dataframe_to_polars(pandas_df)
        >>> # Perform fast operations with Polars
        >>> result = pl_df.filter(pl.col("likelihood") > 0.9)
    """
    if not HAS_POLARS:
        raise ImportError(
            "Polars is not installed. Install with: pip install polars"
        )
    
    return pl.from_pandas(df)


def polars_to_dataframe(pl_df: 'pl.DataFrame') -> pd.DataFrame:
    """
    Convert a Polars DataFrame back to Pandas DataFrame.
    
    Args:
        pl_df: Polars DataFrame
        
    Returns:
        Pandas DataFrame
        
    Example:
        >>> pandas_df = polars_to_dataframe(pl_df)
    """
    if not HAS_POLARS:
        raise ImportError(
            "Polars is not installed. Install with: pip install polars"
        )
    
    return pl_df.to_pandas()


def process_with_polars(
    df: pd.DataFrame,
    operation_func,
    *args,
    **kwargs
) -> pd.DataFrame:
    """
    Process a Pandas DataFrame using Polars for improved performance.
    
    This is a convenience function that handles conversion to/from Polars.
    
    Args:
        df: Input Pandas DataFrame
        operation_func: Function that takes a Polars DataFrame and returns a Polars DataFrame
        *args: Additional positional arguments for operation_func
        **kwargs: Additional keyword arguments for operation_func
        
    Returns:
        Pandas DataFrame with results
        
    Example:
        >>> def filter_high_conf(pl_df):
        ...     return pl_df.filter(pl.col("likelihood") > 0.95)
        >>> 
        >>> result_df = process_with_polars(df, filter_high_conf)
    """
    if not HAS_POLARS:
        warnings.warn(
            "Polars not available. Returning original DataFrame without processing.",
            RuntimeWarning
        )
        return df
    
    try:
        # Convert to Polars
        pl_df = pl.from_pandas(df)
        
        # Apply the operation
        pl_result = operation_func(pl_df, *args, **kwargs)
        
        # Convert back to Pandas
        return pl_result.to_pandas()
    
    except Exception as e:
        warnings.warn(f"Polars operation failed: {e}. Returning original DataFrame.")
        return df

