# Knowledge indexing

Generates an LLM-friendly knowledge index from the DeepLabCut source and user
documentation. The output is `llms.txt` plus a handful of small JSONL/JSON
files under `knowledge/`, meant to be deployed alongside the docs on gh-pages.

## Usage

```bash
python -m tools.knowledge_indexing
```

| Option | Meaning |
|---|---|
| `--output` | Directory mirroring the gh-pages site root to write into (default: `<repo>/_build/knowledge-index`) |
| `--repo` | Repository root, containing `_toc.yml`, `docs/` and `deeplabcut/` (default: cwd) |
| `--version-label` | Developer-docs version label the API URLs point at, one of the labels mike deploys (default: `main`) |
| `--revision` | Commit the source was read at, recorded in the manifest (default: `HEAD` of `--repo`) |
| `--skip-api` | Don't rebuild `api.jsonl` this run; keep it and its manifest provenance as already on disk |
| `--skip-docs` | Don't rebuild `docs.jsonl`/`llms.txt` this run; keep them and their manifest provenance as already on disk |
| `--delete` | Remove a released version's API index instead of building. `main` cannot be deleted — see "Deleting a version" |

Everything is read from the repository, so no built or deployed documentation is
needed.

```bash
pip install -e ".[knowledge-index]"
```

## Output

```text
_build/knowledge-index/
├── llms.txt                 spec-format entry point (llmstxt.org), only for --version-label main
└── knowledge/
    ├── manifest.json        enumeration: every indexed api version, which one is current
    ├── main/
    │   ├── manifest.json    schema, and independent provenance for its api/docs halves
    │   ├── docs.jsonl       one line per doc page and per section on it
    │   └── api.jsonl        one line per module and per documented symbol on it
    └── 3.0.1/
        ├── manifest.json
        └── api.jsonl        no docs.jsonl -- see "Versioning" below
```

Read `knowledge/manifest.json` first: gh-pages gives no directory listing, so it
is the only way to discover which api versions exist. Each record in `docs.jsonl`
and `api.jsonl` is a self-contained JSON object with a stable `id` and a
`content_hash` (sha256 of its own fields), so an agent can load either file
lazily by scanning for the ids it needs, and a consumer that already has a
record can tell whether a freshly fetched one changed without diffing it.

Each version's `manifest.json` records `api_version_label` (the dev-docs
deploy label its api urls point at, matching mike's own vocabulary), plus
**independent** `api` and `docs` provenance blocks -- each with its own
`revision`/`package_version`/`generated_at`, since `--skip-api`/`--skip-docs`
(see "Deployment") let the two be rebuilt by separate CI runs at different
times. `package_version` is `deeplabcut.__version__` at that block's
`revision`; it usually agrees with `api_version_label` for a tagged release,
but not for `main`, where only `revision` pins the build exactly. `docs` is
absent for a version that has never indexed user docs (every label other
than `main`).

Ids are namespaced by type — `docs:`, `docs:<page>#<anchor>`, `api:` — so a
reference is unambiguous about what it points at. Every `url` is absolute, so a
record carries a working link on its own.

### Versioning — docs and api are not symmetric

The developer-docs API reference is deployed per version by mike, so `api.jsonl`
mirrors that: one directory per version actually built, e.g. `3.0.1/`.

The user docs, by contrast, are only ever deployed as a single rolling build at
the site root — there is no historical snapshot to point at, and indexing raw
markdown from an old git tag would be worse than the live rendered page (and
redundant with git itself). So `docs.jsonl` and `llms.txt` are only ever
written for `--version-label main`; every other version's directory holds
`api.jsonl` alone.

## Where the data comes from

### APIs

[griffe](https://mkdocstrings.github.io/griffe/) is the static analysis library
`mkdocstrings` uses to build the developer docs. Reading the source with the same
tool that renders it means signatures and docstrings match what is published, and
nothing has to be imported — no torch, no GPU, no optional dependencies.

Each symbol gets a `source` (`file.py:lineno`) and a `url` derived from its
dotted path, following the layout `mkdocs-api-autonav` generates. Only documented
symbols are indexed, because `mkdocstrings` omits undocumented members from the
reference pages, so an entry for one would have neither a summary nor a URL that
resolves. A module gets its own `kind: "module"` row only if it has a docstring.

Two constants in `api_index.py` mirror `dev-docs/mkdocs.yml` and must be kept in
sync with it: `EXCLUDED_MODULES` and `API_ROOT_URI`.

### Docs pages

`_toc.yml` defines the scope. It lists the pages that are actually published —
`docs/` holds others that it deliberately leaves out — and it is the only place
the part / chapter / section hierarchy exists, so it also supplies each page's
`section` (its toc part), `parent` and `children`.

Parsing uses `markdown-it-py`, the parser Jupyter Book itself uses, because the
docs nest code fences up to five backticks deep; matching fences with regular
expressions instead lets shell comments inside them parse as headings.

**Headings become sections, not concepts.** A section is a retrievable part of
one document; a concept is a reusable domain entity discussed across documents.
Slugged headings are the former pretending to be the latter, which is why this
tool generates no concepts at all — see "Known gaps".

Each page is one `type: "page"` row in `docs.jsonl`; each heading below its title
is its own `type: "section"` row, carrying an `anchor` and a `url` pointing at
it, plus a `summary`: the first paragraph of prose under the heading. Anchors
come from `docutils.nodes.make_id`, the function docutils uses to build section
ids. A heading whose body is only a subheading, a figure or code has no summary.
A section row denormalises its page's `section` (toc part) so it stands on its
own, but everything else about the page — `status`, `parent`, `children`,
`related_pages`, `labels` — lives only on the page row, reachable via the
section's `page` field.

Admonition-style MyST directives (`{note}`, `{important}`, `{tip}`, …) are fenced
blocks, and their bodies are parsed for prose rather than skipped as code.

Pages link to each other mostly through MyST labels rather than file paths, so
parsing happens in two passes: collect the labels each page defines, then resolve
every link and `{ref}` against them. Each page publishes its own `labels`, which
is what a `{ref}` elsewhere in the docs resolves against.

`status` and `last_verified` are copied from each page's audit frontmatter onto
its page row. `visibility` is not published — it only filters: a page whose
`visibility` is in `HIDDEN_VISIBILITY` (`docs_index.py`) is skipped, same as an
explicit `ignore: true`.

### llms.txt

Written only for `--version-label main`, alongside `docs.jsonl`. Follows the
[llmstxt.org](https://llmstxt.org) convention: an H1 title, a one-line
description, then `##` sections of links.
Links to the docs site, the API reference, and prominently to
`knowledge/manifest.json`, `docs.jsonl` and `api.jsonl` for agents that want
the structured index.

## Deployment

`.github/workflows/deploy-knowledge-index.yml` deploys or deletes. A deploy
publishes by plain copy (`peaceiris/actions-gh-pages`, `keep_files: true`) --
the same mechanism the user docs already use, not mike. Before running the
tool, it checks out the existing `gh-pages` `knowledge/` tree into the output
directory, so `write_top_manifest`'s rescan sees every version published by
earlier runs, and `write_version`'s manifest merge preserves whichever half
of the version it wasn't asked to rebuild.

### Deleting a version

A delete edits and pushes a real `gh-pages` checkout directly with `git`:
`keep_files: true` means the publish action never removes a file absent from
`publish_dir`, so it cannot perform a deletion at all.
**Only released versions can be deleted, and only their API index.** `main`
is refused outright — it tracks the repository's latest state, is redeployed
on every push, and is the only version carrying the user docs.

It has two call sites, kept deliberately separate so the api half and the
user-docs half can never get built by the wrong trigger:

- **`deploy-dev-docs-mike.yml`** calls it (`needs: mike`, so it runs after
  mike's own push) with `skip_docs: true`, for every dev-docs deploy or
  delete. This is the api-only half, and living inside
  `deploy-dev-docs-mike.yml` rather than being called separately by each of
  its own callers is what keeps it in sync with every dev-docs deploy by
  construction: `deploy-docs.yml` (on every push to `main`) and
  `manage-dev-docs.yml` (a release's `version_label` and `git_tag`, so
  `api.jsonl`'s `revision`/`package_version` reflect the tagged source, not
  whatever `HEAD` happens to be) both reach it this way, with no way to
  reach dev-docs deploy without it.
- **`deploy-docs.yml`** also calls it directly, with `version_label: main,
  skip_api: true`, gated on `deploy-main-docs` (transitively, via
  `deploy-dev-docs-main`). This is the only path that ever rebuilds
  `docs.jsonl`/`llms.txt`, and it is unreachable from `manage-dev-docs.yml`
  -- a manually triggered release deploy can never touch the user-docs half.

## Module layout

| File | Responsibility |
|---|---|
| `schemas.py` | The published schema: every record and manifest shape |
| `toc.py` | `_toc.yml` → the set of published pages and their hierarchy |
| `api_index.py` | Source tree → API nodes, via griffe (build-time only) |
| `docs_index.py` | Markdown → docs-page nodes with sections (build-time only) |
| `llms_txt.py` | Renders `llms.txt` |
| `write.py` | Nodes → `docs.jsonl` / `api.jsonl` and both manifests |
| `__main__.py` | CLI |

## Known gaps

- **No concepts or workflows.** These need to be authored rather than derived
  from headings.
- **Notebooks are not indexed.** `_toc.yml` lists notebooks under `examples/`,
  which would need a reader for `.ipynb` markdown cells.
- **Validation is minimal.** Duplicate record ids abort the write, but there is
  no `--check` mode, so anchors, dangling references and unresolvable URLs are
  not caught.
- **`knowledge/manifest.json` is rebuilt by rescanning disk**, not by tracking
  state across separate builds -- see "Deployment" below for how CI seeds that
  rescan. `api.latest` currently always points at `main`; resolving "latest
  stable release" numbering is left for later.
- **`content_hash` is not used for incremental building.** It lets a consumer
  detect that a record changed, but nothing here uses it to skip re-extracting
  or re-writing unchanged records -- every run reads and writes everything.
