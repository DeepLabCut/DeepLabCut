# Using Polars with DeepLabCut

## Overview

This document describes the integration of Polars into DeepLabCut for improved performance while maintaining HDF5 as the primary file format.

## Changes Made

### 1. Dependency Updates

**Files modified:**
- `requirements.txt`
- `setup.py`
- `conda-environments/DEEPLABCUT.yaml`

**Changes:**
- Added: `polars>=0.20.0` (alongside existing `tables` dependency)
- HDF5 format remains the standard output format

### 2. New File I/O Module

**New file:** `deeplabcut/utils/fileio.py`

This module provides:
- `read_hdf_with_polars()` - Read HDF5 files with optional Polars optimization
- `write_hdf_with_polars()` - Write HDF5 files with optional Polars pre-processing
- `dataframe_to_polars()` - Convert Pandas DataFrame to Polars
- `polars_to_dataframe()` - Convert Polars DataFrame to Pandas
- `process_with_polars()` - Process DataFrames using Polars for speed

**Key features:**
- HDF5 remains the primary file format (no breaking changes)
- Polars is used internally for performance improvements
- All functions work without Polars (graceful fallback)
- Compatible with existing DeepLabCut workflows

## Usage

### Standard HDF5 I/O (with optional Polars optimization)

```python
from deeplabcut.utils import fileio
import pandas as pd

# Create a DataFrame
df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})

# Write to HDF5 (standard format)
fileio.write_hdf_with_polars(df, "output.h5")

# Read from HDF5
df = fileio.read_hdf_with_polars("output.h5")
```

### Using Polars for Fast Data Processing

```python
import polars as pl
from deeplabcut.utils import fileio

# Read HDF5 file
df = fileio.read_hdf_with_polars("poses.h5")

# Convert to Polars for fast operations
pl_df = fileio.dataframe_to_polars(df)

# Perform fast filtering with Polars
filtered = pl_df.filter(pl.col("likelihood") > 0.95)

# Convert back to Pandas
result_df = fileio.polars_to_dataframe(filtered)

# Or use the convenience function
def filter_operation(pl_df):
    return pl_df.filter(pl.col("likelihood") > 0.95)

result_df = fileio.process_with_polars(df, filter_operation)
```

## Benefits

1. **Performance**: Polars operations are significantly faster for large datasets
2. **Compatibility**: HDF5 format unchanged, fully backward compatible
3. **Optional**: Polars is optional; code works without it
4. **Zero Breaking Changes**: Existing workflows unaffected

## When to Use Polars

Polars excels at:
- Filtering large DataFrames
- Aggregating data across many frames
- Complex data transformations
- Memory-efficient operations on large datasets

Example use cases in DeepLabCut:
- Filtering poses by likelihood threshold
- Computing statistics across video frames
- Selecting specific bodyparts from multi-animal data
- Downsampling or interpolating tracking data

## Backward Compatibility

The implementation maintains full backward compatibility:

1. **HDF5 remains the output format**
   - No changes to file formats
   - All existing files work without modification

2. **Polars is optional**
   - If Polars is not installed, code falls back to Pandas
   - No breaking changes to existing code

3. **No API changes**
   - Existing functions continue to work
   - New functions are additions, not replacements

## Testing

New test suite: `tests/test_fileio.py`

Run tests:
```bash
pytest tests/test_fileio.py -v
```

Test coverage includes:
- Reading and writing HDF5 files
- Polars integration and fallback
- Converting between Pandas and Polars
- MultiIndex DataFrames (DeepLabCut format)

## Performance Comparison

Polars provides significant speedups for common operations:

```python
import time
import polars as pl
import pandas as pd

# Large DataFrame
df = pd.DataFrame({
    'x': np.random.randn(1000000),
    'likelihood': np.random.rand(1000000)
})

# Pandas filtering
t0 = time.time()
result_pandas = df[df['likelihood'] > 0.95]
pandas_time = time.time() - t0

# Polars filtering
pl_df = pl.from_pandas(df)
t0 = time.time()
result_polars = pl_df.filter(pl.col('likelihood') > 0.95).to_pandas()
polars_time = time.time() - t0

print(f"Pandas: {pandas_time:.3f}s")
print(f"Polars: {polars_time:.3f}s")
print(f"Speedup: {pandas_time/polars_time:.1f}x")
```

## Migration Guide

### For Users

No action required! The codebase continues to use HDF5 format as before.

If you want to use Polars for custom analysis:
```bash
pip install polars
```

### For Developers

When processing large DataFrames in DeepLabCut code:

**Before:**
```python
df = pd.read_hdf(filename, key="df_with_missing")
filtered = df[df['likelihood'] > 0.9]
```

**After (optional optimization):**
```python
from deeplabcut.utils import fileio
import polars as pl

df = fileio.read_hdf_with_polars(filename)
def filter_func(pl_df):
    return pl_df.filter(pl.col('likelihood') > 0.9)
filtered = fileio.process_with_polars(df, filter_func)
```

## Troubleshooting

### Polars Not Installed

If Polars is not installed, you'll see a warning but code will still work:
```
ImportWarning: Polars not installed. Install with 'pip install polars' for better performance.
```

To install:
```bash
pip install polars
```

### Performance Not Improved

Polars shines with:
- Large datasets (>10,000 rows)
- Complex filtering/aggregation operations
- Multiple chained operations

For small datasets or simple operations, overhead may negate benefits.

## Future Work

1. Identify bottlenecks in DeepLabCut that could benefit from Polars
2. Add Polars-optimized versions of common operations
3. Benchmark performance improvements on real-world datasets
4. Consider lazy evaluation for very large video analyses

## Notes

- HDF5 format is maintained for full backward compatibility
- Polars is an additive enhancement, not a replacement
- All existing code continues to work unchanged
- Performance improvements are opt-in

