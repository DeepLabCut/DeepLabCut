# Test Documentation Build

This script tests the documentation build process for the intelligent test selector.

## Usage

```bash
# Test documentation build
make -C docs html

# Alternative: use Sphinx directly
cd docs && sphinx-build -b html . _build/html
```

## Test Coverage

This ensures that:
- Documentation builds without errors
- Markdown files are properly formatted
- Cross-references work correctly