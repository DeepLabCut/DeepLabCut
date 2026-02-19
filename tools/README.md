# Developer tools useful for maintaining the repository

As developer you'll need:

```bash
pip install coverage pytest fnmatch black
```

## Code headers

The code headers can be standardized by running

``` bash
python tools/update_license_headers.py
```

from the repository root.

You can edit the `NOTICE.yml` to update the header.


## Workflow for contributing/checking your code

```bash
black .
```

## Running the tests (locally)

We use the pytest framework. You can just run:

```bash
pytest
```

For coverage run:

```
coverage run -m pytest
coverage report
```

## Intelligent Test Selection

The repository includes an intelligent test selection system that reduces CI runtime by running only relevant tests based on the changes in a pull request.

### Usage

```bash
# Analyze current changes and show what tests would run
python tools/test_selector.py --dry-run

# Run the selected tests
python tools/test_selector.py

# Get JSON output for CI integration
python tools/test_selector.py --output-json

# Validate the test selection system
python tools/validate_test_selection.py

# Test documentation build
python tools/test_docs_build.py
```

### Test Categories

- **Documentation-only**: Runs lightweight docs build (~1-2 minutes)
- **SuperAnimal changes**: Runs SuperAnimal-specific tests (~3-4 minutes)
- **Core component changes**: Runs focused component tests (~2-3 minutes)
- **Complex/mixed changes**: Falls back to full test suite (~5+ minutes)

### Expected Benefits

- 60-80% reduction in CI time for focused changes
- Faster feedback for developers
- More efficient use of CI resources
- Maintains test coverage while reducing runtime
