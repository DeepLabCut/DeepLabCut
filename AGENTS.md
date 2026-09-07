# AGENTS.md

## DeepLabCut

DeepLabCut is a scientific Python toolbox for markerless pose estimation.
Prioritize correctness, preservation of user data, and public API compatibility.

DeepLabCut is PyTorch-first. TensorFlow is being deprecated, but legacy support
remains. Do not introduce new TensorFlow-first functionality, and preserve
existing TensorFlow compatibility unless a task explicitly changes its support
status.

## Guidelines

- Keep patches focused; avoid unrelated refactors and formatting churn.
- Preserve public APIs and user-facing serialized formats unless the task
  explicitly requires a breaking change or migration.
- Treat aliases, deprecations, and default values as intentional user-facing
  behavior. Prefer aliases and explicit deprecation warnings over silently
  renaming or removing inputs.
- Prefer `pathlib.Path`; do not assume POSIX paths. Isolate and document
  intentionally platform-specific behavior.
- Preserve optional dependency boundaries. Do not assume that a GUI,
  TensorFlow, CUDA, a GPU, or a display is available.
- Avoid heavy or backend-specific imports at package import time.
- Never overwrite or destructively modify user data unless explicitly
  requested.
- Never fabricate scientific outputs or benchmark results. Clearly identify
  synthetic test data as synthetic.
- Preserve labels, scorer names, dataframe structure, coordinate conventions,
  identities, and project metadata unless the task explicitly requires
  changing them.
- Use repository tooling where available. Treat repository source and
  documentation as authoritative. When available, use `llms.txt`,
  `knowledge/manifest.json`, `docs.jsonl`, and `api.jsonl` as discovery indexes.
- Update docstrings, examples, and relevant user or developer documentation
  when public behavior changes.
- Use Google-style docstrings. Follow the surrounding documentation
  conventions; in plain Markdown, prefer inline code such as `myfunc` over
  Sphinx roles such as `:func:\`myfunc\``.
- Add or update regression tests for bug fixes and behavior changes when
  practical.
- Run the smallest relevant tests while iterating. Isolate
  optional-dependency, GUI, and GPU-dependent tests where possible, and report
  relevant checks that were not run.
