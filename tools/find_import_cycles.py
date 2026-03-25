#!/usr/bin/env python3
from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path


def path_to_module(root: Path, file: Path) -> str:
    rel = file.relative_to(root)
    parts = rel.with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((root.name, *parts)) if parts else root.name


def module_to_file_map(root: Path) -> dict[str, Path]:
    mapping = {}
    for file in root.rglob("*.py"):
        mod = path_to_module(root, file)
        mapping[mod] = file
    return mapping


def resolve_relative_import(current_module: str, module: str | None, level: int) -> str | None:
    parts = current_module.split(".")
    if level > len(parts):
        return None
    base = parts[:-level]
    if module:
        return ".".join(base + module.split("."))
    return ".".join(base)


def extract_imports(file: Path, current_module: str) -> set[str]:
    source = file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file))
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and current_module:
                resolved = resolve_relative_import(current_module, node.module, node.level)
                if resolved:
                    imports.add(resolved)
            elif node.module:
                imports.add(node.module)

    return imports


def internal_edges(root: Path) -> dict[str, set[str]]:
    mod_to_file = module_to_file_map(root)
    internal = set(mod_to_file)
    edges: dict[str, set[str]] = defaultdict(set)

    for mod, file in mod_to_file.items():
        for imported in extract_imports(file, mod):
            # Keep only imports that are inside the package
            for candidate in internal:
                if imported == candidate or imported.startswith(candidate + "."):
                    edges[mod].add(candidate)
                    break

    return edges


def find_cycles(edges: dict[str, set[str]]) -> list[list[str]]:
    visited = set()
    stack = []
    on_stack = set()
    cycles = []

    def dfs(node: str):
        visited.add(node)
        stack.append(node)
        on_stack.add(node)

        for neighbor in edges.get(node, ()):
            if neighbor not in visited:
                dfs(neighbor)
            elif neighbor in on_stack:
                idx = stack.index(neighbor)
                cycle = stack[idx:] + [neighbor]
                cycles.append(cycle)

        stack.pop()
        on_stack.remove(node)

    for node in edges:
        if node not in visited:
            dfs(node)

    # Deduplicate roughly
    seen = set()
    unique = []
    for cyc in cycles:
        key = tuple(cyc)
        if key not in seen:
            seen.add(key)
            unique.append(cyc)
    return unique


def main():
    root = Path("deeplabcut")  # change if needed
    edges = internal_edges(root)
    cycles = find_cycles(edges)

    if not cycles:
        print("No cycles found.")
        return

    print("Import cycles found:\n")
    for cyc in cycles:
        print(" -> ".join(cyc))


if __name__ == "__main__":
    main()
