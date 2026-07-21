#!/usr/bin/env python3
"""Minimal Zep Cloud MCP server -- add_memory, search_memory, list_memory."""

from __future__ import annotations

import json
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

DEFAULT_USER_ID = os.environ.get("ZEP_USER_ID", "default")
IDENTITY_HEADER = os.environ.get("ZEP_IDENTITY_HEADER", "X-Spark-Group-Id")
client = Zep(api_key=api_key)

# Users we've ensured exist this process (idempotent guard, per tenant).
_ensured_users: set[str] = set()


def _ensure_user(uid: str) -> None:
    if uid in _ensured_users:
        return
    try:
        client.user.add(user_id=uid)
    except Exception:
        pass
    _ensured_users.add(uid)


def current_user_id() -> str:
    """Resolve the caller's Zep user id.

    Multi-tenant: read the identity header the gateway forwards per
    request. Falls back to ZEP_USER_ID for the single-tenant (stdio /
    personal) deployment — same code path, N=1 is just one id inserted.
    There is no single-tenant special case.
    """
    try:
        from fastmcp.server.dependencies import get_http_headers

        headers = get_http_headers() or {}
        for k, v in headers.items():
            if k.lower() == IDENTITY_HEADER.lower() and v:
                return v
    except Exception:
        pass
    return DEFAULT_USER_ID


mcp = FastMCP("zep-memory")


@mcp.tool()
def add_memory(content: str) -> str:
    """Add information to the knowledge graph.

    Use this to store facts, preferences, and information about the user
    that should be remembered across conversations.
    """
    uid = current_user_id()
    _ensure_user(uid)
    result = client.graph.add(
        user_id=uid,
        type="text",
        data=content,
    )
    return f"Stored (episode {result.uuid_})"


@mcp.tool()
def search_memory(query: str, limit: int = 5) -> str:
    """Search the knowledge graph for relevant facts and relationships.

    Returns extracted facts ranked by cross-encoder relevance scoring.
    Each result includes a relevance score [0,1] for thresholding.
    Use this for targeted recall about specific topics.
    """
    uid = current_user_id()
    _ensure_user(uid)
    results = client.graph.search(
        user_id=uid,
        query=query,
        limit=min(limit, 50),
        scope="edges",
        reranker="cross_encoder",
    )
    if not results.edges:
        return "[]"
    items = []
    for edge in results.edges:
        fact = edge.fact or ""
        if not fact:
            continue
        items.append(
            {
                "text": fact,
                "relevance": edge.relevance
                if edge.relevance is not None
                else edge.score,
                "created_at": str(edge.created_at)[:10] if edge.created_at else "",
            }
        )
    return json.dumps(items) if items else "[]"


@mcp.tool()
def list_memory(limit: int = 50) -> str:
    """List stored memories (episodes) without a specific search query.

    Use when the user asks what you know about them, or to get a broad
    overview of stored knowledge.  Returns the most recent episodes —
    the original text stored via add_memory — which preserves full
    context.  For targeted factual lookups, use search_memory instead.
    """
    uid = current_user_id()
    _ensure_user(uid)
    episodes = client.graph.episode.get_by_user_id(
        user_id=uid,
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


def main():
    """Entry point for console_scripts.

    Transport via env: ZEP_MCP_TRANSPORT=http runs a streamable-HTTP
    server so the gateway can forward per-request identity headers.
    Default stdio keeps the single-user / personal path unchanged.
    """
    transport = os.environ.get("ZEP_MCP_TRANSPORT", "stdio")
    if transport in ("http", "streamable-http"):
        host = os.environ.get("ZEP_MCP_HOST", "0.0.0.0")
        port = int(os.environ.get("ZEP_MCP_PORT", os.environ.get("PORT", "8000")))
        mcp.run(transport="http", host=host, port=port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
