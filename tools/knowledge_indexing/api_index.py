"""Extract API nodes from the DeepLabCut source tree using griffe.

griffe is the static-analysis library mkdocstrings uses to build the developer
documentation, so the modules, signatures and docstrings collected here are the
ones on the published reference pages, and each symbol's URL follows from its
dotted path. Loading is static: DeepLabCut is never imported, so torch, a GPU and
the optional dependencies are not needed.

Only documented symbols are indexed, because mkdocstrings omits undocumented
members from the reference pages: such a symbol has no summary and no URL that
resolves.

`EXCLUDED_MODULES` and `API_ROOT_URI` mirror `dev-docs/mkdocs.yml` and have to be
kept in sync with it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import griffe

from .schemas import API_NAMESPACE, ApiNode, Symbol

# griffe models a module member either as the object itself or, when it was
# imported or re-exported, as an alias pointing at it.
Member = griffe.Object | griffe.Alias

# `plugins.api-autonav.exclude` in dev-docs/mkdocs.yml. These modules have no
# published reference page, so no URL may point at one.
EXCLUDED_MODULES: tuple[str, ...] = (
    "deeplabcut.version",
    "deeplabcut.benchmark",
    "deeplabcut.gui",
)

# `plugins.api-autonav.api_root_uri` in dev-docs/mkdocs.yml.
API_ROOT_URI = "reference"

_VARIADIC_PREFIX = {
    griffe.ParameterKind.var_positional: "*",
    griffe.ParameterKind.var_keyword: "**",
}


def load_package(package: str = "deeplabcut", search_path: Path = Path()) -> griffe.Module:
    """Statically load `package` from `search_path` without importing it."""
    # griffe warns about every import it cannot follow. For a package with
    # optional TensorFlow and GUI dependencies that is expected and only noise.
    logging.getLogger("griffe").setLevel(logging.ERROR)
    return griffe.load(
        package,
        search_paths=[str(search_path)],
        resolve_aliases=True,
        allow_inspection=False,
    )


def build_api_nodes(package: str = "deeplabcut", search_path: Path = Path(), base_url: str = "") -> list[ApiNode]:
    """Build one node per published module, listing its documented symbols.

    `base_url` is prepended to every published URL.
    """
    root = load_package(package, search_path)
    nodes = [
        ApiNode(
            id=f"{API_NAMESPACE}:{module.path}",
            module=module.path,
            summary=_summary(module),
            source=_source(module),
            docs_url=_docs_url(module.path, base_url=base_url),
            symbols=_symbols(module, package, base_url),
        )
        for module in _iter_modules(root)
    ]
    # A module with neither a docstring nor a documented symbol adds nothing.
    return sorted(
        (node for node in nodes if node.symbols or node.summary),
        key=lambda node: node.id,
    )


def _iter_modules(module: griffe.Module) -> Iterator[griffe.Module]:
    """Yield `module` and every submodule that has a published reference page."""
    if _is_excluded(module.path) or _is_private(module.path):
        return
    yield module
    for member in module.members.values():
        if _own_kind(member) == "module":
            yield from _iter_modules(member)


def _is_excluded(module_path: str) -> bool:
    """True if the module is left out of the published reference."""
    return any(module_path == excluded or module_path.startswith(f"{excluded}.") for excluded in EXCLUDED_MODULES)


def _is_private(module_path: str) -> bool:
    """True if any part of the dotted path is private (mkdocs sets exclude_private)."""
    return any(part.startswith("_") for part in module_path.split("."))


def _own_kind(member: Member) -> str:
    """Kind of a member defined in the module itself.

    Returns "" for imports and re-exports, which griffe models as aliases.
    """
    if member.is_alias:
        return ""
    if member.is_module:
        return "module"
    if member.is_class:
        return "class"
    if member.is_function:
        return "function"
    return ""


def _symbols(module: griffe.Module, package: str, base_url: str = "") -> tuple[Symbol, ...]:
    """Documented functions and classes belonging to `module`.

    Members defined elsewhere are skipped, except on the package root, where the
    public API consists entirely of re-exports: `analyze_videos` belongs under
    `deeplabcut` rather than under `deeplabcut.compat`.
    """
    is_root = module.path == package
    symbols = []

    for name, member in module.members.items():
        if name.startswith("_"):
            continue

        target, kind = member, _own_kind(member)
        if kind not in ("function", "class"):
            if not (is_root and member.is_alias):
                continue
            target = _resolve(member, package)
            if target is None:
                continue
            kind = "class" if target.is_class else "function"

        if not target.docstring:
            continue

        symbols.append(
            Symbol(
                name=name,
                kind=kind,
                summary=_summary(target),
                signature=_signature(target),
                source=_source(target),
                docs_url=_docs_url(module.path, f"{module.path}.{name}", base_url),
            )
        )

    return tuple(sorted(symbols, key=lambda symbol: symbol.name))


def _resolve(alias: griffe.Alias, package: str) -> griffe.Object | None:
    """Final target of `alias`, if it is a function or class inside `package`."""
    try:
        target = alias.final_target
    except griffe.GriffeError:
        return None
    if not (target.is_function or target.is_class):
        return None
    if not target.canonical_path.startswith(f"{package}."):
        return None
    return target


def _docs_url(module_path: str, symbol_path: str = "", base_url: str = "") -> str:
    """Published URL of a module page, or of a symbol anchored on that page.

    Mirrors the layout mkdocs-api-autonav generates: one directory per module
    under the API root, with every symbol anchored by its dotted path.
    """
    page = f"{base_url}{API_ROOT_URI}/{module_path.replace('.', '/')}/"
    return f"{page}#{symbol_path or module_path}"


def _summary(obj: griffe.Object) -> str:
    """First paragraph of the docstring, collapsed onto a single line."""
    if not obj.docstring:
        return ""
    first_paragraph = obj.docstring.value.strip().split("\n\n", 1)[0]
    return " ".join(first_paragraph.split())


def _signature(obj: griffe.Object) -> str:
    """Call signature, e.g. `(project: str, copy_videos: bool = False) -> Path`.

    Empty for classes that define no explicit `__init__`.
    """
    try:
        parameters = list(obj.parameters)
    except (AttributeError, KeyError, griffe.GriffeError):
        return ""

    rendered = []
    for parameter in parameters:
        if parameter.name in ("self", "cls"):
            continue
        text = _VARIADIC_PREFIX.get(parameter.kind, "") + parameter.name
        if parameter.annotation is not None:
            text += f": {parameter.annotation}"
        if parameter.default is not None:
            # PEP 8 spaces the default only when the parameter is annotated.
            text += f" = {parameter.default}" if parameter.annotation else f"={parameter.default}"
        rendered.append(text)

    signature = f"({', '.join(rendered)})"
    returns = getattr(obj, "returns", None)
    return f"{signature} -> {returns}" if returns is not None else signature


def _source(obj: griffe.Object) -> str:
    """`path/to/file.py:lineno` for `obj`, or "" if griffe has no location."""
    try:
        path = obj.relative_filepath
    except (ValueError, griffe.GriffeError):
        return ""
    return f"{path}:{obj.lineno}" if obj.lineno else str(path)
