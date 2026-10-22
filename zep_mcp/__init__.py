#!/usr/bin/env python3
"""Minimal memory MCP server -- add_memory, search_memory, list_memory.

Three backends, chosen by ZEP_MCP_BACKEND:

    zep  (default)  Zep Cloud, as before.
    own             Our own store: Postgres + pgvector, Voyage embeddings and
                    reranking, MMR, a use log per fact (zep_mcp/store.py).
                    Needs MEMORY_DATABASE_URL and VOYAGE_API_KEY.
    local           A SQLite file on this machine and Vertex AI embeddings in
                    its own Google Cloud project: for a machine whose text
                    may go to Google and nowhere else (zep_mcp/local_store.py).
                    Needs MEMORY_VERTEX_PROJECT (or GOOGLE_CLOUD_PROJECT).

The default is zep, so shipping this changes nothing until the switch is set.
"""

from __future__ import annotations

import json
import logging
import os
import sys

from fastmcp import FastMCP

logger = logging.getLogger(__name__)

BACKEND = os.environ.get("ZEP_MCP_BACKEND", "zep").strip().lower() or "zep"
if BACKEND not in ("zep", "own", "local"):
    print(f"ZEP_MCP_BACKEND must be 'zep', 'own' or 'local', not {BACKEND!r}", file=sys.stderr)
    sys.exit(1)

DEFAULT_USER_ID = os.environ.get("ZEP_USER_ID", "default")
# Identity arrives as W3C baggage (baggage: <key>=<value>); see the gateway's
# notes/identity-propagation.md. Key is configurable for genericity.
BAGGAGE_KEY = os.environ.get("ZEP_BAGGAGE_KEY", "userId")

client = None  # the Zep client (zep backend)
_store = None  # the own or local store, opened on first use

if BACKEND == "zep":
    from zep_cloud.client import Zep
    from zep_cloud.errors import NotFoundError

    api_key = os.environ.get("ZEP_API_KEY")
    if not api_key:
        print("ZEP_API_KEY required", file=sys.stderr)
        sys.exit(1)
    client = Zep(api_key=api_key)
elif BACKEND == "own":
    missing = [k for k in ("MEMORY_DATABASE_URL", "VOYAGE_API_KEY") if not os.environ.get(k)]
    if missing:
        print(f"ZEP_MCP_BACKEND=own requires {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)
elif not (os.environ.get("MEMORY_VERTEX_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")):
    print("ZEP_MCP_BACKEND=local requires MEMORY_VERTEX_PROJECT or GOOGLE_CLOUD_PROJECT", file=sys.stderr)
    sys.exit(1)


def own_store():
    """The own or local backend's store, connected lazily (startup stays instant).

    Not named `store`: importing the submodule zep_mcp.store rebinds that
    name on this package, and the second call would hit a module.
    """
    global _store
    if _store is None and BACKEND == "local":
        from zep_mcp.local_store import LocalStore

        _store = LocalStore.from_env()
    if _store is None:
        from zep_mcp.store import Store

        _store = Store(
            os.environ["MEMORY_DATABASE_URL"],
            schema=os.environ.get("MEMORY_SCHEMA", "memory"),
        )
    return _store


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
    """Resolve the caller's user id from forwarded W3C baggage.

    Multi-tenant: parse ``baggage: <BAGGAGE_KEY>=<value>`` (stamped per
    request by the gateway) and percent-decode it. Falls back to
    ZEP_USER_ID for the single-tenant (stdio / personal) deployment —
    same code path, N=1 is just one id ever flowing through. There is no
    single-tenant special case. Both backends scope memory by this id.
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


# ── provenance (the optional `source` argument) ─────────────────────────

SOURCE_DOC = """
    `source` is optional provenance, a JSON object passed as a string, with
    any of: app, conversation (or session), model, tools (list of tool
    names), turn_time (ISO 8601). For example:
    {"app": "llm-multiplex", "conversation": "…", "model": "…",
     "tools": ["search"], "turn_time": "2026-10-21T09:15:00-07:00"}
"""


def zep_source_kwargs(source: str) -> dict:
    """`source` as episode.add's source_description and metadata.

    Zep's episode metadata wants flat scalar values, so lists are joined and
    nested objects JSON-encoded.
    """
    from zep_mcp.source import normalize_source

    src = normalize_source(source)
    if not src:
        return {}
    meta = {}
    for k, v in src.items():
        if v is None or isinstance(v, (str, int, float, bool)):
            meta[str(k)] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (str, int, float, bool)) for x in v):
            meta[str(k)] = ", ".join(str(x) for x in v)
        else:
            meta[str(k)] = json.dumps(v, ensure_ascii=False)
    desc = "; ".join(f"{k}={v}" for k, v in meta.items() if v not in (None, ""))
    return {"source_description": desc[:1000], "metadata": meta}


# ── tools ────────────────────────────────────────────────────────────────

mcp = FastMCP("zep-memory")

if BACKEND == "zep":
    ADD_DOC = """Add information to the knowledge graph.

    Use this to store facts, preferences, and information about the user
    that should be remembered across conversations.
    """ + SOURCE_DOC
    SEARCH_DOC = """Search the knowledge graph for relevant facts and relationships.

    Returns extracted facts ranked by cross-encoder relevance scoring.
    Each result includes a relevance score [0,1] for thresholding.
    Use this for targeted recall about specific topics.
    """
    LIST_DOC = """List stored memories (episodes) without a specific search query.

    Use when the user asks what you know about them, or to get a broad
    overview of stored knowledge.  Returns the most recent episodes —
    the original text stored via add_memory — which preserves full
    context.  For targeted factual lookups, use search_memory instead.
    """
else:
    ADD_DOC = """Store a fact to remember across conversations.

    Use this to store facts, preferences, and information about the user.
    One concise, standalone statement per call.
    """ + SOURCE_DOC
    SEARCH_DOC = """Search memory for facts relevant to a query.

    Returns facts ranked by relevance, with near-duplicate phrasings of
    a fact already returned left out. Each result has a
    relevance score [0,1] for thresholding, the date it was saved
    (created_at) and, when known, the date it became true (valid_from).
    Use this for targeted recall about specific topics.
    """
    LIST_DOC = """List the most recently saved memories, newest first.

    Use when the user asks what you know about them, or to get a broad
    overview of stored knowledge. For targeted factual lookups, use
    search_memory instead.
    """


@mcp.tool(description=ADD_DOC)
def add_memory(content: str, source: str = "") -> str:
    if BACKEND in ("own", "local"):
        mid = own_store().add(current_user_id(), content, source)
        return f"Stored (memory {mid})"
    graph = _graph_uuid(current_user_id())
    extra = zep_source_kwargs(source)
    try:
        result = client.graph.episode.add(graph, type="text", data=content, **extra)
    except Exception:
        if not extra:
            raise
        # Provenance must never cost a save: if Zep rejects the metadata,
        # store the text without it.
        logger.exception("Zep rejected episode source/metadata; saving without")
        result = client.graph.episode.add(graph, type="text", data=content)
    return f"Stored (episode {result.episode.uuid_})"


@mcp.tool(description=SEARCH_DOC)
def search_memory(query: str, limit: int = 5) -> str:
    limit = max(1, min(limit, 50))
    if BACKEND in ("own", "local"):
        items = own_store().search(current_user_id(), query, limit=limit)
        return json.dumps(items) if items else "[]"
    # A pager: .items is the first page, which with page_size=N is the top N.
    page = client.graph.search_edges(
        _graph_uuid(current_user_id()),
        query=query,
        page_size=limit,
        reranker="cross_encoder",
    )
    if not page.items:
        return "[]"
    items = []
    for edge in page.items:
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


@mcp.tool(description=LIST_DOC)
def list_memory(limit: int = 50) -> str:
    if BACKEND in ("own", "local"):
        rows = own_store().list(current_user_id(), limit=max(1, min(limit, 100)))
        if not rows:
            return "No memories stored yet."
        return "\n".join(f"{i}. [{r['created_at']}] {r['text']}" for i, r in enumerate(rows, 1))
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
