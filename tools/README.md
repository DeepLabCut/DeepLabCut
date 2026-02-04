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
