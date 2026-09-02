"""Shared state-dependent operator feasibility helpers."""

from __future__ import annotations

from typing import Any, Iterable


def consecutive_debug_depth(node: Any) -> int:
    """Count explicit consecutive Debug-producing nodes in an ancestry path."""

    depth = 0
    current = node
    visited: set[str] = set()
    while current is not None:
        node_id = str(getattr(current, "id", id(current)))
        if node_id in visited:
            raise ValueError("cycle detected while computing Debug depth")
        visited.add(node_id)
        operators = tuple(getattr(current, "operators_used", None) or ())
        if not operators or operators[0] != "debug":
            break
        depth += 1
        parents = tuple(getattr(current, "parents", None) or ())
        if len(parents) != 1:
            break
        current = parents[0]
    return depth


def is_debuggable(node: Any, max_debug_depth: int) -> bool:
    """Return whether one more Debug action is legal for a buggy leaf."""

    if max_debug_depth < 0:
        raise ValueError("max_debug_depth must be non-negative")
    return (
        bool(node.is_buggy)
        and bool(node.is_leaf)
        and (consecutive_debug_depth(node) < max_debug_depth)
    )


def eligible_debug_nodes(nodes: Iterable[Any], max_debug_depth: int) -> list[Any]:
    """Filter nodes through the shared corrected Debug action mask."""

    return [node for node in nodes if is_debuggable(node, max_debug_depth)]
