#!/usr/bin/env python3
"""Minimal Zep Cloud MCP server -- add_memory, search_memory, list_memory."""

from __future__ import annotations

import logging
import os
import sys

from fastmcp import FastMCP
from zep_cloud.client import Zep

logger = logging.getLogger(__name__)

api_key = os.environ.get("ZEP_API_KEY")
if not api_key:
    print("ZEP_API_KEY required", file=sys.stderr)
    sys.exit(1)

user_id = os.environ.get("ZEP_USER_ID", "default")
client = Zep(api_key=api_key)

# Ensure user exists (idempotent -- ignores "already exists" errors).
try:
    client.user.add(user_id=user_id)
except Exception:
    pass

mcp = FastMCP("zep-memory")


@mcp.tool()
def add_memory(content: str) -> str:
    """Add information to the knowledge graph.

    Use this to store facts, preferences, and information about the user
    that should be remembered across conversations.
    """
    result = client.graph.add(
        user_id=user_id,
        type="text",
        data=content,
    )
    return f"Stored (episode {result.uuid_})"


@mcp.tool()
def search_memory(query: str, limit: int = 10) -> str:
    """Search the knowledge graph for relevant facts.

    Returns facts (relationships between entities) that match the query.
    Uses hybrid search (semantic + keyword + graph traversal).
    """
    results = client.graph.search(
        user_id=user_id,
        query=query,
        limit=min(limit, 50),
        scope="edges",
    )
    if not results.edges:
        return "No results found."
    lines = []
    for i, edge in enumerate(results.edges, 1):
        fact = edge.fact or ""
        if not fact:
            continue
        valid = getattr(edge, "valid_at", None) or ""
        invalid = getattr(edge, "invalid_at", None) or ""
        temporal = ""
        if valid:
            ts = str(valid)[:10]
            temporal = f" (valid: {ts}"
            if invalid:
                temporal += f" - invalid: {str(invalid)[:10]}"
            temporal += ")"
        lines.append(f"{i}. {fact}{temporal}")
    return "\n".join(lines) if lines else "No results found."


@mcp.tool()
def list_memory(limit: int = 50) -> str:
    """List stored memories (episodes) without a specific search query.

    Use when the user asks what you know about them, or to get a broad
    overview of stored knowledge.  Returns the most recent episodes —
    the original text stored via add_memory — which preserves full
    context.  For targeted factual lookups, use search_memory instead.
    """
    episodes = client.graph.episode.get_by_user_id(
        user_id=user_id,
        lastn=min(limit, 1000),
    )
    if not episodes.episodes:
        return "No memories stored yet."
    lines = []
    for i, ep in enumerate(episodes.episodes, 1):
        content = ep.content or ""
        if not content:
            continue
        created = str(ep.created_at)[:10] if ep.created_at else ""
        prefix = f"[{created}] " if created else ""
        lines.append(f"{i}. {prefix}{content}")
    return "\n".join(lines) if lines else "No memories stored yet."


if __name__ == "__main__":
    mcp.run()
