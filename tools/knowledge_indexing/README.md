# Knowledge indexing

Generates an LLM-friendly knowledge index from the DeepLabCut source and user
documentation. The output is a directory of small YAML files, one per node, that
an agent can load lazily by id.

## Usage

```bash
python -m tools.knowledge_indexing
```

| Option | Meaning |
|---|---|
| `--output` | Directory to write the index into (default: `<repo>/_build/knowledge-index/<version>`) |
| `--repo` | Repository root, containing `_toc.yml`, `docs/` and `deeplabcut/` (default: cwd) |
| `--version` | Developer-docs version the API URLs point at, one of the versions mike deploys (default: `main`) |

Everything is read from the repository, so no built or deployed documentation is
needed. Both documentation extras are required: `griffe` comes with `dev-docs`,
`docutils` with `docs`, and `markdown-it-py` with either. `pyyaml` is a core
dependency.

```bash
pip install -e ".[docs,dev-docs]"
```

## Output

```text
_build/knowledge-index/main/
├── index.yaml        manifest: schema, provenance, and every node id with its file
├── symbols.yaml      flat symbol id -> module node lookup
├── apis/             one file per published module
└── docs-pages/       one file per published docs page, with its sections
```

Read `index.yaml` first; it maps each node id to its file, which matters because
ids are namespaced and docs-page ids are nested (`docs:pytorch/user_guide`) while
the directories are flat.

Ids are namespaced by type — `docs:`, `docs:<page>#<anchor>`, `api:` — so a
reference is unambiguous about what it points at.

Every `docs_url` is absolute, so a node carries a working link on its own. The two
sites they are built against are recorded under `base_urls` in the manifest: the
user docs are unversioned at the site root, while the developer docs are deployed
per version by mike under `dev/<version>/`, which is the segment `--version`
fills in. Both bases are defined in `__main__.py`.

## Where the data comes from

### APIs

[griffe](https://mkdocstrings.github.io/griffe/) is the static analysis library
`mkdocstrings` uses to build the developer docs. Reading the source with the same
tool that renders it means signatures and docstrings match what is published, and
nothing has to be imported — no torch, no GPU, no optional dependencies.

Each symbol gets a `source` (`file.py:lineno`) and a `docs_url` derived from its
dotted path, following the layout `mkdocs-api-autonav` generates. Only documented
symbols are indexed, because `mkdocstrings` omits undocumented members from the
reference pages, so an entry for one would have neither a summary nor a URL that
resolves.

API nodes are grouped per module, which keeps each file worth loading, so
`symbols.yaml` maps every symbol id to the module node documenting it for lookups
that start from a bare name.

Two constants in `api_index.py` mirror `dev-docs/mkdocs.yml` and must be kept in
sync with it: `EXCLUDED_MODULES` and `API_ROOT_URI`.

### Docs pages

`_toc.yml` defines the scope. It lists the pages that are actually published —
`docs/` holds others that it deliberately leaves out — and it is the only place
the part / chapter / section hierarchy exists, so it also supplies each page's
`part`, `parent` and `children`.

Parsing uses `markdown-it-py`, the parser Jupyter Book itself uses, because the
docs nest code fences up to five backticks deep; matching fences with regular
expressions instead lets shell comments inside them parse as headings.

**Headings become sections, not concepts.** A section is a retrievable part of
one document; a concept is a reusable domain entity discussed across documents.
Slugged headings are the former pretending to be the latter, which is why this
tool generates no concepts at all — see "Known gaps".

Each section carries an `anchor` and a `docs_url` pointing at it, plus an
`excerpt`: the first paragraph of prose under the heading. Anchors come from
`docutils.nodes.make_id`, the function docutils uses to build section ids. A
heading whose body is only a subheading, a figure or code has no excerpt.

Admonition-style MyST directives (`{note}`, `{important}`, `{tip}`, …) are fenced
blocks, and their bodies are parsed for prose rather than skipped as code.

Pages link to each other mostly through MyST labels rather than file paths, so
parsing happens in two passes: collect the labels each page defines, then resolve
every link and `{ref}` against them. Each node publishes its own `labels`, which
is what a `{ref}` elsewhere in the docs resolves against.

`status`, `visibility` and `last_verified` are copied from each page's audit
frontmatter. Nothing is filtered on them; only an explicit `ignore: true` skips a
page.

## Module layout

| File | Responsibility |
|---|---|
| `schemas.py` | The on-disk schema: every published file and node type |
| `toc.py` | `_toc.yml` → the set of published pages and their hierarchy |
| `api_index.py` | Source tree → API nodes, via griffe |
| `docs_index.py` | Markdown → docs-page nodes with sections |
| `write.py` | Nodes → YAML files, symbol table and manifest |
| `__main__.py` | CLI |

## Known gaps

- **No concepts or workflows.** These need to be authored rather than derived
  from headings.
- **Notebooks are not indexed.** `_toc.yml` lists notebooks under `examples/`,
  which would need a reader for `.ipynb` markdown cells.
- **Validation is minimal.** Duplicate node ids abort the write, but there is no
  `--check` mode, so anchors, dangling references and unresolvable URLs are not
  caught.
- **The user docs are not versioned** upstream, so `docs-pages/` is identical
  across index versions and only stamped with the revision it came from.
