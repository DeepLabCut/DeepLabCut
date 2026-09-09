from __future__ import annotations

from pathlib import Path

import pytest

from tools.knowledge_indexing.api_index import build_api_nodes

PACKAGE = '''\
"""Demo package."""

from .thing import Thing
'''

MODULE = '''\
"""Thing module."""


class Base:
    """A base class."""

    def inherited(self) -> None:
        """Defined on the base, documented there."""


class Thing(Base):
    """A documented class."""

    attribute: int = 0

    def run(self, times: int = 1) -> str:
        """Run the thing."""
        return "ok"

    @property
    def ready(self) -> bool:
        """Whether it is ready."""
        return True

    def undocumented(self) -> None:
        return None

    def _private(self) -> None:
        """Private, so not published."""
'''


@pytest.fixture
def nodes(tmp_path: Path):
    package = tmp_path / "demopkg"
    package.mkdir()
    (package / "__init__.py").write_text(PACKAGE, encoding="utf-8")
    (package / "thing.py").write_text(MODULE, encoding="utf-8")
    return build_api_nodes("demopkg", tmp_path, "https://example.test/")


def _symbols(nodes, module: str) -> dict[str, object]:
    node = next(n for n in nodes if n.module == module)
    return {symbol.name: symbol for symbol in node.symbols}


def test_public_documented_methods_are_indexed(nodes):
    symbols = _symbols(nodes, "demopkg.thing")

    assert symbols["Thing.run"].kind == "method"
    assert symbols["Thing.run"].signature == "(times: int = 1) -> str"
    assert symbols["Thing.run"].summary == "Run the thing."


def test_methods_anchor_on_their_class_page(nodes):
    # mkdocstrings anchors a member by its dotted path on the module's page.
    symbols = _symbols(nodes, "demopkg.thing")

    assert symbols["Thing.run"].docs_url == ("https://example.test/reference/demopkg/thing/#demopkg.thing.Thing.run")


def test_private_undocumented_and_inherited_methods_are_skipped(nodes):
    symbols = _symbols(nodes, "demopkg.thing")

    assert "Thing._private" not in symbols
    assert "Thing.undocumented" not in symbols
    # `inherited` belongs to Base, which documents it; Thing does not repeat it.
    assert "Thing.inherited" not in symbols
    assert "Base.inherited" in symbols


def test_attributes_and_properties_are_not_methods(nodes):
    symbols = _symbols(nodes, "demopkg.thing")

    assert "Thing.attribute" not in symbols
    # A property has no call signature; `kind: "method"` would misdescribe it.
    assert "Thing.ready" not in symbols


def test_the_class_itself_is_still_recorded(nodes):
    symbols = _symbols(nodes, "demopkg.thing")

    assert symbols["Thing"].kind == "class"


def test_root_reexports_are_indexed_on_the_package(nodes):
    # Public API is re-exported from the package root; URLs must stay on that page.
    symbols = _symbols(nodes, "demopkg")

    assert symbols["Thing"].kind == "class"
    assert symbols["Thing"].docs_url == ("https://example.test/reference/demopkg/#demopkg.Thing")
