# Contributing to DeepLabCut

Thanks for your interest in contributing to DeepLabCut! We welcome bug fixes, new features, documentation improvements, tests, and general maintenance contributions.

We especially encourage contributions from people from backgrounds that are underrepresented in open-source software. If you want to discuss an idea before opening a pull request, feel free to start a discussion or open an issue.

If you are new to GitHub, the [GitHub Guides](https://guides.github.com/) are a great place to start.

## Ways to contribute

You can help by:

- Fixing bugs
- Improving documentation
- Adding tests
- Improving examples
- Refactoring or cleaning up code
- Proposing or implementing new features

## Development setup

To work on DeepLabCut locally:

1. [Fork the repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo).
2. Clone your fork:

```bash
git clone https://github.com/<your-username>/DeepLabCut.git
cd DeepLabCut
```

3. Create and activate a Python environment.

We recommend using the project's development dependency group from `pyproject.toml` so you get the tools needed for local development (including formatting, linting, and testing).

For example, with `uv`:

```bash
uv sync --group dev
```

With `pip` (e.g. in a `conda` environment):

```bash
pip install -e . --group dev
```

If you use a different environment manager, install the package in editable/development mode together with the `dev` dependency group defined in `pyproject.toml`.

## Working on the code

Once your environment is ready, your local checkout is what Python will import.

If you want to verify that you are using the local source tree, you can run:

```bash
python -c "import deeplabcut; print(deeplabcut.__file__)"
```

Using `ipython` or Jupyter is completely optional—use whatever workflow you prefer.

If you change packaged resources or otherwise need to refresh the local installation, run:

```bash
./reinstall.sh
```

> [!NOTE]
> This script automatically uninstalls the package, builds a new wheel using `setup.py`, and installs that wheel. It is not a simple `pip install -e .` because some resources are copied during installation and need to be refreshed.

## Code style and pre-commit

We use `pre-commit` to run formatting and other checks before code is committed.

Set it up once in your clone:

```bash
pre-commit install
```

Whenever you commit, `pre-commit` will run the configured checks.

Please run `pre-commit` before opening a pull request. This helps catch formatting, import ordering, whitespace, YAML, and other common issues early and accelerates code review greatly.

## Tests

Pull requests are validated in CI, and contributors are encouraged to run tests locally using:

```bash
pytest tests
```
in the project root before opening a pull request.

> [!IMPORTANT]
> Heavier tests are also run automatically on GitHub, so this is not a strict requirement, > but it can help catch issues early and speed up the review process.

## Pull request guidelines

When submitting a pull request, please:

- Clearly describe what changed and why
- Link any related issue(s)
- Update docstrings and documentation when behavior changes
- Add or update tests when appropriate
- Include a small usage example when it helps reviewers understand and/or test the change

Smaller, focused pull requests are usually much easier to review than very large ones.

## Documentation

Documentation improvements are always welcome.

If your change affects users, please update the relevant docs, examples, or inline docstrings so the behavior is discoverable and easy to understand.

## Code headers and notices

If you need to standardize code headers, run:

```bash
python tools/update_license_headers.py
```

Contributors are requested not to update `NOTICE.yml` or `LICENSE` files.

## Review process

A maintainer will review your pull request. You do not need to supply a specific release timeline in your PR description—contributions are reviewed and merged as capacity allows.

If you have questions about where a change should go or how to structure it, opening a draft pull request is completely fine.

## Need help?

If you are unsure whether something is in scope, open an issue or draft PR and ask.
We'd much rather help early than have you spend time on the wrong thing.
We also welcome "Feature requests" issues if you would like to discuss implementation details or would like preliminary feedback.

## Acknowledgments

DeepLabCut is an open-source project and has benefited from many contributors over time, including:

- The [authors](/AUTHORS)
- Listed [code contributors](https://github.com/DeepLabCut/DeepLabCut/graphs/contributors)
- And many others over the years.

We look forward to your contributions!
