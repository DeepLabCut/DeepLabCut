# Developer tools useful for maintaining the repository

As developer you'll need:

```bash
pip install -e . --group dev
```

## Code headers

The code headers can be standardized by running

``` bash
python tools/update_license_headers.py
```

from the repository root.

Steering committee members may edit the `NOTICE.yml` to update the header.


## Workflow for contributing/checking your code

Please follow the instructions in `CONTRIBUTING.md` for contributing to the codebase, including running tests and pre-commit checks before opening a pull request.

## Running the tests (locally)

We use the pytest framework. You can just run:

```bash
pytest
```

For coverage run:

```bash
coverage run -m pytest
coverage report
```
