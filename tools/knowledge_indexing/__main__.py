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
from .schemas import API_DIRECTORY, DOCS_DIRECTORY
from .toc import TOC_FILE
from .write import write_index, write_symbol_table

PACKAGE = "deeplabcut"

# Default output, relative to --repo and per version, so the index is a build
# artifact of the repository it describes. `_build/` is where the other docs
# builds already write theirs (`_build/html`, `_build/dev`) and is gitignored.
DEFAULT_OUTPUT_ROOT = Path("_build/knowledge-index")

# The published sites every `docs_url` is built against. The user docs are
# unversioned and live at the site root, while the developer docs are deployed per
# version by mike under `dev/` (`plugins.mike` in dev-docs/mkdocs.yml), which is
# the segment --version fills in.
DOCS_BASE_URL = "https://deeplabcut.github.io/DeepLabCut/"
API_BASE_URL = "https://deeplabcut.github.io/DeepLabCut/dev/{version}/"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m tools.knowledge_indexing",
        description="Generate an LLM-friendly knowledge index from the DeepLabCut docs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=f"Directory to write the index into (default: <repo>/{DEFAULT_OUTPUT_ROOT}/<version>).",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(),
        help=f"Repository root, containing {TOC_FILE}, docs/ and {PACKAGE}/ (default: cwd).",
    )
    parser.add_argument(
        "--version",
        default="main",
        help=(
            "Developer-docs version the API URLs point at, also recorded in the "
            "manifest. One of the versions mike deploys, e.g. main or 3.0 (default: main)."
        ),
    )
    return parser.parse_args(argv)


def _git_revision(repo: Path) -> str:
    """Short commit hash of `repo`, or "" if it is not a git checkout."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
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

    output: Path = args.output or repo / DEFAULT_OUTPUT_ROOT / args.version
    base_urls = {
        API_DIRECTORY: API_BASE_URL.format(version=args.version),
        DOCS_DIRECTORY: DOCS_BASE_URL,
    }

    print(f"Reading API from {repo / PACKAGE} ...")
    apis = build_api_nodes(PACKAGE, repo, base_urls[API_DIRECTORY])
    print(f"  {len(apis)} modules, {sum(len(node.symbols) for node in apis)} documented symbols")

    print(f"Reading user docs listed in {repo / TOC_FILE} ...")
    docs_pages = build_docs_nodes(repo, base_urls[DOCS_DIRECTORY])
    print(f"  {len(docs_pages)} pages, {sum(len(page.sections) for page in docs_pages)} sections")

    groups = {API_DIRECTORY: apis, DOCS_DIRECTORY: docs_pages}
    write_index(
        output,
        groups,
        version=args.version,
        revision=_git_revision(repo),
        base_urls=base_urls,
    )
    symbols = write_symbol_table(output, apis)

    print(f"Wrote {len(apis) + len(docs_pages)} nodes and {symbols} symbol entries to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
