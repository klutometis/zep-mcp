#!/usr/bin/env python3
"""Minimal Zep Cloud MCP server -- add_memory, search_memory, list_memory."""

from __future__ import annotations

import json
import logging
import os
import sys

from fastmcp import FastMCP
from zep_cloud.client import Zep
from zep_cloud.errors import NotFoundError

from zep_mcp.diverse import fetch_size, pick_diverse

logger = logging.getLogger(__name__)

api_key = os.environ.get("ZEP_API_KEY")
if not api_key:
    print("ZEP_API_KEY required", file=sys.stderr)
    sys.exit(1)

DEFAULT_USER_ID = os.environ.get("ZEP_USER_ID", "default")
# Identity arrives as W3C baggage (baggage: <key>=<value>); see the gateway's
# notes/identity-propagation.md. Key is configurable for genericity.
BAGGAGE_KEY = os.environ.get("ZEP_BAGGAGE_KEY", "userId")
client = Zep(api_key=api_key)

# zep-cloud 4 addresses a user's memory by the *graph* it owns, not by user id:
# every graph call takes a graph_uuid. Resolve user id -> graph uuid once per
# process and tenant (creating the user on first sight, as v3's user.add did).
_graph_of: dict[str, str] = {}


def _graph_uuid(uid: str) -> str:
    g = _graph_of.get(uid)
    if g:
        return g
    try:
        user = client.user.lookup(user_id=uid)
    except NotFoundError:
        user = client.user.create(user_id=uid)
    if not user.graph_uuid:
        # A user's graph is provisioned with the user; a fresh one can lag.
        user = client.user.lookup(user_id=uid)
    if not user.graph_uuid:
        raise RuntimeError(f"Zep user {uid!r} has no graph yet")
    _graph_of[uid] = user.graph_uuid
    return user.graph_uuid


def current_user_id() -> str:
    """Resolve the caller's Zep user id from forwarded W3C baggage.

    Multi-tenant: parse ``baggage: <BAGGAGE_KEY>=<value>`` (stamped per
    request by the gateway) and percent-decode it. Falls back to
    ZEP_USER_ID for the single-tenant (stdio / personal) deployment —
    same code path, N=1 is just one id ever flowing through. There is no
    single-tenant special case.
    """
    try:
        from urllib.parse import unquote

        from fastmcp.server.dependencies import get_http_headers

        headers = get_http_headers() or {}
        baggage = next(
            (v for k, v in headers.items() if k.lower() == "baggage"), ""
        )
        for member in baggage.split(","):
            key, _, val = member.strip().partition("=")
            if key.strip() == BAGGAGE_KEY:
                val = val.split(";", 1)[0].strip()  # drop any baggage properties
                if val:
                    return unquote(val)
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
    result = client.graph.episode.add(
        _graph_uuid(current_user_id()),
        type="text",
        data=content,
    )
    return f"Stored (episode {result.episode.uuid_})"


@mcp.tool()
def search_memory(query: str, limit: int = 5) -> str:
    """Search the knowledge graph for relevant facts and relationships.

    Returns extracted facts ranked by cross-encoder relevance scoring, with
    near-duplicate phrasings of the same fact removed.
    Each result includes a relevance score [0,1] for thresholding.
    Use this for targeted recall about specific topics.
    """
    limit = max(1, min(limit, 50))
    # Over-fetch, then keep the first `limit` facts that are not rewordings
    # of one already kept: memory holds many phrasings of the same fact, and
    # the cross-encoder scores them all ~1.0. See zep_mcp/diverse.py.
    # A pager: .items is the first page, which with page_size=N is the top N.
    page = client.graph.search_edges(
        _graph_uuid(current_user_id()),
        query=query,
        page_size=fetch_size(limit),
        reranker="cross_encoder",
    )
    if not page.items:
        return "[]"
    edges = [e for e in page.items if e.fact]
    items = []
    for edge in pick_diverse(edges, limit, text=lambda e: e.fact):
        fact = edge.fact
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
    page = client.graph.episode.list(
        _graph_uuid(current_user_id()),
        page_size=max(1, min(limit, 100)),
        order_by="created_at",
        order="desc",
    )
    if not page.items:
        return "No memories stored yet."
    lines = []
    for i, ep in enumerate(page.items, 1):
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
