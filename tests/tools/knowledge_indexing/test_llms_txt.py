from __future__ import annotations

from tools.knowledge_indexing.__main__ import API_BASE_URL, API_REFERENCE_URL, DOCS_BASE_URL, PACKAGE, main
from tools.knowledge_indexing.llms_txt import build_llms_txt
from tools.knowledge_indexing.schemas import API_ROOT_URI, KNOWLEDGE_DIR, LATEST_RELEASE_ALIAS, LLMS_TXT

DOCS_URL = "https://example.test/"
KNOWLEDGE_URL = f"{DOCS_URL}{KNOWLEDGE_DIR}/"
REFERENCE_URL = "https://example.test/dev/latest-release/reference/demopkg/"


def _api_reference_line(text: str) -> str:
    (line,) = [line for line in text.splitlines() if line.startswith("- [API reference]")]
    return line


def _build(**overrides: str) -> str:
    kwargs = {
        "docs_base_url": DOCS_URL,
        "api_reference_url": REFERENCE_URL,
        "knowledge_base_url": KNOWLEDGE_URL,
        "version_label": "main",
    }
    return build_llms_txt(**{**kwargs, **overrides})


def test_api_reference_url_is_used_verbatim():
    assert _api_reference_line(_build()) == (
        f"- [API reference]({REFERENCE_URL}): every public module, class and function"
    )


def test_jsonl_links_stay_on_the_run_s_own_version():
    # The machine-readable half is version-scoped even though the human-facing
    # API link is not: those files really do live under the label being built.
    text = _build(version_label="main")

    assert f"{KNOWLEDGE_URL}main/docs.jsonl" in text
    assert f"{KNOWLEDGE_URL}main/api.jsonl" in text


def test_api_reference_url_targets_the_release_and_the_package_root():
    # Two regressions in one line. `llms.txt` is only ever built by the `main`
    # run, so using that run's label would hardcode the unreleased API --
    # exactly what `api.latest` following `latest-release` exists to avoid. And
    # api-autonav publishes no index at `<API_ROOT_URI>/`, so the link has to
    # carry the package segment or it 404s.
    url = API_REFERENCE_URL.format(package=PACKAGE)

    assert url == f"{DOCS_BASE_URL}dev/{LATEST_RELEASE_ALIAS}/{API_ROOT_URI}/{PACKAGE}/"
    assert API_BASE_URL.format(version="main") not in url


def test_cli_writes_the_release_api_link(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "_toc.yml").write_text("format: jb-book\nroot: README\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# DeepLabCut\n\nLead.\n", encoding="utf-8")
    package = tmp_path / PACKAGE
    package.mkdir()
    (package / "__init__.py").write_text('"""Package."""\n', encoding="utf-8")

    output = tmp_path / "out"
    assert main(["--output", str(output), "--repo", str(tmp_path), "--revision", "abc123"]) == 0

    line = _api_reference_line((output / LLMS_TXT).read_text(encoding="utf-8"))
    assert API_REFERENCE_URL.format(package=PACKAGE) in line
    assert API_BASE_URL.format(version="main") not in line
