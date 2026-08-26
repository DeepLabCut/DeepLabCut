"""Command line entry point.

    python -m tools.knowledge_indexing

See README.md for the output layout.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .api_index import build_api_nodes
from .docs_index import build_docs_nodes
from .llms_txt import build_llms_txt
from .schemas import KNOWLEDGE_DIR, LLMS_TXT
from .toc import TOC_FILE
from .write import write_top_manifest, write_version

PACKAGE = "deeplabcut"

# Default output, relative to --repo, so the index is a build artifact of the
# repository it describes. `_build/` is where the other docs builds already
# write theirs (`_build/html`, `_build/dev`) and is gitignored. Unlike them,
# this mirrors the gh-pages site root: `knowledge/<version>/` and `llms.txt`
# sit directly under it.
DEFAULT_OUTPUT_ROOT = Path("_build/knowledge-index")

# The published sites every url is built against. The user docs are unversioned
# and live at the site root, while the developer docs are deployed per version
# by mike under `dev/` (`plugins.mike` in dev-docs/mkdocs.yml), which is the
# segment --version fills in.
DOCS_BASE_URL = "https://deeplabcut.github.io/DeepLabCut/"
API_BASE_URL = "https://deeplabcut.github.io/DeepLabCut/dev/{version}/"

# The dev-docs version whose build carries docs.jsonl and llms.txt. The user
# docs are only ever deployed as a single rolling build on gh-pages (there is
# no historical snapshot to index), so only this version's build indexes them
# -- see "Versioning" in README.md.
DOCS_VERSION = "main"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m tools.knowledge_indexing",
        description="Generate an LLM-friendly knowledge index from the DeepLabCut docs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Directory mirroring the gh-pages site root to write into -- gets "
            f"'{KNOWLEDGE_DIR}/<version>/' and '{LLMS_TXT}' "
            f"(default: <repo>/{DEFAULT_OUTPUT_ROOT})."
        ),
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(),
        help=f"Repository root, containing {TOC_FILE}, docs/ and {PACKAGE}/ (default: cwd).",
    )
    parser.add_argument(
        "--version",
        default=DOCS_VERSION,
        help=(
            "Developer-docs version the API URLs point at, also recorded in the "
            "manifest. One of the versions mike deploys, e.g. main or 3.0 "
            f"(default: {DOCS_VERSION})."
        ),
    )
    return parser.parse_args(argv)


def _git_revision(repo: Path) -> str:
    """Full commit hash of `repo`, or "" if it is not a git checkout."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo: Path = args.repo

    for required in (repo / TOC_FILE, repo / PACKAGE):
        if not required.exists():
            print(f"Error: {required} not found; is --repo the repository root?", file=sys.stderr)
            return 1

    output: Path = args.output or repo / DEFAULT_OUTPUT_ROOT
    knowledge_dir = output / KNOWLEDGE_DIR
    api_base_url = API_BASE_URL.format(version=args.version)
    include_docs = args.version == DOCS_VERSION

    print(f"Reading API from {repo / PACKAGE} ...")
    apis = build_api_nodes(PACKAGE, repo, api_base_url)
    print(f"  {len(apis)} modules, {sum(len(node.symbols) for node in apis)} documented symbols")

    docs_pages = None
    if include_docs:
        print(f"Reading user docs listed in {repo / TOC_FILE} ...")
        docs_pages = build_docs_nodes(repo, DOCS_BASE_URL)
        print(f"  {len(docs_pages)} pages, {sum(len(page.sections) for page in docs_pages)} sections")
    else:
        print(f"Skipping user docs: only the '{DOCS_VERSION}' build carries {LLMS_TXT} and docs.jsonl")

    api_count, docs_count = write_version(
        knowledge_dir,
        args.version,
        apis,
        docs_pages,
        revision=_git_revision(repo),
    )
    write_top_manifest(knowledge_dir, docs_version=DOCS_VERSION)

    if docs_pages is not None:
        llms_txt = build_llms_txt(
            docs_pages,
            api_base_url=api_base_url,
            knowledge_base_url=f"{DOCS_BASE_URL}{KNOWLEDGE_DIR}/",
            version=args.version,
        )
        (output / LLMS_TXT).write_text(llms_txt, encoding="utf-8")

    print(f"Wrote {api_count} api records and {docs_count} docs records to {knowledge_dir / args.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
