# AGENTS.md

## DeepLabCut

DeepLabCut is a scientific Python toolbox for markerless pose estimation.
Prioritize correctness, preservation of user data, backward compatibility,
reproducibility, and public API compatibility.

DeepLabCut is PyTorch-first. Treat PyTorch as the primary backend for new
functionality. Preserve legacy TensorFlow behavior where the current support
policy still requires it, but do not introduce new TensorFlow-first
functionality.

## Contributing

- State the extent of human involvement at the top of every pull request description and issue you write. When a human has told you in this session that they reviewed the final change and confirmed that they understand why it is suitable for DeepLabCut, write: "Human-reviewed." Otherwise, write "Prepared by an AI agent; Not reviewed by a human." Do not ask for or wait on this confirmation. Do not infer human review from the fact that a human started the task.
- Before starting work, search open issues and pull requests for the same problem or changes to the same file. If work already exists, consider contributing to the existing discussion or implementation rather than creating a duplicate.
- Set up with `uv sync --group dev`;  run `pre-commit run --files <changed>` and the relevant pytest tests before opening a PR. See [CONTRIBUTING.md](CONTRIBUTING.md).
- Reuse existing helpers, fixtures, custom types and exceptions where possible.
- In pull requests, separate what you verified by running it from what you expect to hold. State which tests ran and their outcome, and which did not run and why.
- Test behaviour at the smallest unit that exercises it. Mock only what is expensive or unavailable, and share setup between tests instead of copying it.
- Keep pull request descriptions proportionate: the problem, the change, and how it was verified. Do not restate the diff.

## Guidelines

- Keep patches focused; avoid unrelated refactors and formatting churn.
- Preserve public APIs and user-facing serialized formats unless the task
  explicitly requires a breaking change or migration.
- Treat aliases, deprecations, and default values as intentional user-facing
  behavior. Prefer aliases and explicit deprecation warnings over silently
  renaming or removing inputs.
- Prefer `pathlib.Path`; do not assume POSIX paths or that `Path.resolve()` is
  universally safe across filesystems. Isolate and document intentionally
  platform-specific behavior.
- Preserve optional dependency boundaries. Do not assume that a GUI,
  TensorFlow, CUDA, a GPU, or a display is available.
- Avoid heavy or backend-specific imports at package import time.
- Do not unexpectedly overwrite or destructively modify user-authored data. Preserve existing safeguards, backups, and explicit overwrite controls.
- Never fabricate scientific outputs or benchmark results. Clearly identify
  synthetic test data as synthetic.
- Preserve labels, scorer names, dataframe structure, coordinate conventions,
  identities, and project metadata unless the task explicitly requires
  changing them.
- Use repository tooling where available. Treat repository source, tests, and documentation as primary evidence. When they disagree, identify the inconsistency rather than silently choosing one. When available, use `llms.txt`,
  [published here](https://deeplabcut.github.io/DeepLabCut/llms.txt),
  `knowledge/manifest.json`, `docs.jsonl`, and `api.jsonl` as discovery
  indexes.
- Prefer issue-based bookkeeping for user-visible bugs, regressions, and
  behavior changes when practical. Link related issues from pull requests and
  preserve useful reproduction details, but do not require a new issue for
  trivial or self-contained changes.
- Keep discussion focused on code, behavior, and reproducible conditions. Do
  not assign fault to users, contributors, or other individuals. Describe
  problems neutrally and propose constructive next steps.
- Update docstrings, examples, and relevant user or developer documentation
  when public behavior changes.
- Use Google-style docstrings. Follow the surrounding documentation
  conventions; in plain Markdown, prefer inline code such as `myfunc` over
  Sphinx roles such as `:func:\`myfunc\``.
- Add or update regression tests for bug fixes and behavior changes when practical, ensuring they fail without the corresponding fix.
