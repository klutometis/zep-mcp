"""The own memory store: facts in Postgres + pgvector, a use log per fact.

    memories     one row per fact; embedding is voyage-4-large (1024, unit length)
    memory_uses  saved | restated | retrieved, one row per use, for ACT-R decay
    zep_episodes the texts originally saved to Zep, kept for provenance

Search: embed the query, take the nearest POOL by cosine (HNSW), score them
with rerank-2.5, pick `limit` by maximal marginal relevance over the
embeddings, log a `retrieved` use for each fact returned. ACT-R base-level
activation is computed for every candidate and recorded with the use, but has
no weight in the ranking yet: forgetting will be a threshold on it, applied at
query time, once there is evidence for where to put one.

Nothing here deletes. A merged fact keeps its row and points at the survivor
through merged_into; search and list skip such rows.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid as uuidlib
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from zep_mcp.rank import mmr_sim
from zep_mcp.source import normalize_source
from zep_mcp.voyage import Voyage

logger = logging.getLogger(__name__)

EMBED_PRICE_PER_MTOK = 0.12  # voyage-4-large, USD per million tokens (2026-10)
RERANK_PRICE_PER_MTOK = 0.05  # rerank-2.5

# MMR trade-off, relevance (rerank-2.5, 0..1) against likeness to a fact
# already picked (cosine). 1.0 is pure rerank order. Chosen on 25 queries
# (scripts/evaluate.py), counting pairs at cosine >= 0.9 in the top 5:
#   lambda      0.3    0.4    0.5    0.6    0.7    1.0
#   rewordings    0      0      1      7     13     22
#   mean rel  0.480  0.511  0.545  0.577  0.589  0.607
# 0.5 is the knee: one rewording left (in one query), and each step down
# costs ~0.03 of relevance for nothing more removed. A rewording pays 0.5 x
# its cosine (~0.45 for cos 0.9), so it loses to any distinct fact within
# ~0.4 of its relevance; a distinct but related fact (cos ~0.5) pays ~0.25.
DEFAULT_LAMBDA = 0.5
DEFAULT_POOL = 50  # nearest neighbours handed to the reranker
ACTR_DECAY = 0.5  # d in B = ln(sum t_j^-d)
# HNSW candidate list, set in nearest(). It must be >= the pool, and it is
# what recall rides on: the reranker reaches deep into the 50, so a neighbour
# HNSW misses near the boundary can be one it would have ranked first. On the
# evaluation queries (recall@50 against an exact scan; queries whose final top
# 5 differs from the exact scan's): ef 100 -> 0.946, 9/25; 200 -> 0.981, 4/25;
# 400 -> 0.994, 3/25, in 8 ms. At 800 the planner prefers a 40 ms seq scan.
EF_SEARCH = 400
MAX_POOL = 200

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS {s};

CREATE TABLE IF NOT EXISTS {s}.memories (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id     text        NOT NULL,
    text        text        NOT NULL,
    embedding   vector(1024),               -- NULL only until a failed embed is retried
    created_at  timestamptz NOT NULL DEFAULT now(),
    valid_from  timestamptz,                -- when the fact became true, if known
    source      jsonb       NOT NULL DEFAULT '{{}}'::jsonb,
    zep         jsonb,                      -- the Zep edge it came from, if any
    zep_uuid    uuid GENERATED ALWAYS AS ((zep->>'uuid')::uuid) STORED UNIQUE,
    merged_into bigint REFERENCES {s}.memories(id)
);
-- Keep vectors in the row. A 1024-d vector is 4 KB, over the TOAST threshold,
-- so by default it is stored out of line; the planner then prices a sequential
-- scan by the small heap alone and never picks the HNSW index (measured: a
-- 65-170 ms scan instead of 1.6 ms). Inline, the heap is honest about its size.
ALTER TABLE {s}.memories ALTER COLUMN embedding SET STORAGE PLAIN;
CREATE INDEX IF NOT EXISTS memories_user_created
    ON {s}.memories (user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS {s}.memory_uses (
    id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    memory_id bigint      NOT NULL REFERENCES {s}.memories(id),
    at        timestamptz NOT NULL DEFAULT now(),
    kind      text        NOT NULL CHECK (kind IN ('saved', 'restated', 'retrieved')),
    context   jsonb       NOT NULL DEFAULT '{{}}'::jsonb
);
CREATE INDEX IF NOT EXISTS memory_uses_memory_at ON {s}.memory_uses (memory_id, at);

-- Nearest neighbours by cosine through the HNSW index. A function so that
-- ef_search (see EF_SEARCH) is set for this call alone: a session SET does
-- not survive Neon's transaction-mode pooler, and SET LOCAL needs a
-- transaction, i.e. extra round trips.
CREATE OR REPLACE FUNCTION {s}.nearest(q vector, uid text, n int)
RETURNS TABLE (id bigint, text text, created_at timestamptz, valid_from timestamptz,
               zep jsonb, embedding vector, dist float8)
LANGUAGE sql STABLE
SET hnsw.ef_search = {ef}
SET search_path = pg_catalog, public
AS $f$
    SELECT m.id, m.text, m.created_at, m.valid_from, m.zep, m.embedding,
           m.embedding <=> q
    FROM {s}.memories m
    WHERE m.user_id = uid AND m.merged_into IS NULL AND m.embedding IS NOT NULL
    ORDER BY m.embedding <=> q
    LIMIT least(n, {max_pool})
$f$;

CREATE TABLE IF NOT EXISTS {s}.zep_episodes (
    uuid       uuid PRIMARY KEY,
    user_id    text        NOT NULL,
    content    text        NOT NULL,
    created_at timestamptz NOT NULL,
    zep        jsonb       NOT NULL DEFAULT '{{}}'::jsonb
);

-- Zep episodes (and the edges extracted from them) that must never be
-- imported: the import skips them, so a rerun cannot embed them again. First
-- use: text saved from danenberg-central, which may go to Google and nowhere
-- else (no Voyage, no Neon), but reached Zep and then this store.
CREATE TABLE IF NOT EXISTS {s}.zep_excluded (
    uuid        uuid PRIMARY KEY,
    kind        text        NOT NULL CHECK (kind IN ('episode', 'edge')),
    reason      text        NOT NULL,
    excluded_at timestamptz NOT NULL DEFAULT now()
);
"""

INDEX_DDL = """
CREATE INDEX IF NOT EXISTS memories_embedding_hnsw
    ON {s}.memories USING hnsw (embedding vector_cosine_ops);
"""


def _date(ts: datetime | None) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d") if ts else ""


@dataclass
class Candidates:
    """The nearest neighbours of one query, scored; input to selection."""

    query: str
    ids: list[int]
    texts: list[str]
    created_at: list[datetime]
    valid_from: list[datetime | None]
    zep: list[dict | None]
    sim: np.ndarray  # n x n cosine between the candidates, computed in SQL
    cosine: list[float]  # to the query
    activation: list[float | None]
    relevance: list[float] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.ids)


class Store:
    def __init__(
        self,
        dsn: str,
        schema: str = "memory",
        voyage: Voyage | None = None,
        lam: float | None = None,
        pool: int | None = None,
    ):
        self.dsn = dsn
        self.schema = schema
        self.s = sql.Identifier(schema)
        self._voyage = voyage
        self.lam = float(lam if lam is not None else os.environ.get("MEMORY_MMR_LAMBDA", DEFAULT_LAMBDA))
        self.pool_size = int(pool if pool is not None else os.environ.get("MEMORY_SEARCH_POOL", DEFAULT_POOL))
        self._pool: ConnectionPool | None = None
        self._lock = threading.Lock()
        self._embed_pending = False
        # The use log is written off the request path (one round trip saved).
        self._bg = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory-log")
        self._pending: list[Future] = []

    # ── plumbing ────────────────────────────────────────────────────────

    @property
    def voyage(self) -> Voyage:
        if self._voyage is None:
            self._voyage = Voyage()
        return self._voyage

    def _q(self, text: str) -> sql.Composed:
        return sql.SQL(text).format(
            s=self.s, ef=sql.Literal(EF_SEARCH), max_pool=sql.Literal(MAX_POOL)
        )

    def _connect_raw(self) -> psycopg.Connection:
        # prepare_threshold=None: Neon's pooler is PgBouncer in transaction
        # mode, where a server-side prepared statement may land on another
        # backend.
        return psycopg.connect(self.dsn, prepare_threshold=None, autocommit=True)

    @property
    def pool(self) -> ConnectionPool:
        # Autocommit: every operation here is one statement (or one pipelined
        # transaction), and an implicit BEGIN/COMMIT would cost two extra
        # round trips, ~75 ms each from the west coast to Neon us-east-1.
        with self._lock:
            if self._pool is None:
                self._pool = ConnectionPool(
                    self.dsn,
                    min_size=1,
                    max_size=4,
                    kwargs={"prepare_threshold": None, "autocommit": True},
                    configure=register_vector,
                    max_idle=240,  # under Neon's 5-minute compute suspend
                    open=True,
                )
        return self._pool

    def _run(self, fn):
        """fn(conn); one retry if the connection had died."""
        for attempt in (0, 1):
            try:
                with self.pool.connection() as conn:
                    return fn(conn)
            except psycopg.OperationalError:
                if attempt:
                    raise
                logger.warning("memory store: connection lost, retrying once")
                self.pool.check()

    def flush(self) -> None:
        """Wait for use-log writes still in flight."""
        pending, self._pending = self._pending, []
        for f in pending:
            f.result()

    def close(self) -> None:
        self.flush()
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    # ── schema ──────────────────────────────────────────────────────────

    def ensure_schema(self) -> None:
        with self._connect_raw() as conn:
            conn.execute(self._q(DDL))

    def ensure_vector_index(self) -> None:
        with self._connect_raw() as conn, conn.transaction():
            # The graph for ~20k x 1024 is ~100 MB; keep the build in memory.
            conn.execute("SET LOCAL maintenance_work_mem = '512MB'")
            conn.execute(self._q(INDEX_DDL))
            conn.execute(self._q("ANALYZE {s}.memories"))

    def drop_schema(self) -> None:
        """For throwaway test schemas only."""
        if self.schema == "memory":
            raise RuntimeError("refusing to drop the production schema")
        with self._connect_raw() as conn:
            conn.execute(self._q("DROP SCHEMA IF EXISTS {s} CASCADE"))

    # ── write ───────────────────────────────────────────────────────────

    def add(self, user_id: str, text: str, source: Any = None) -> int:
        """Embed and insert one fact, with its `saved` use. Returns its id.

        If Voyage fails the fact is stored anyway, without an embedding (so
        invisible to search), and embedded on a later call: losing a save is
        worse than a fact that is searchable a few minutes late.
        """
        text = (text or "").strip()
        if not text:
            raise ValueError("empty memory")
        src = normalize_source(source)
        try:
            vecs, _ = self.voyage.embed([text], "document")
            vec = vecs[0]
        except Exception:
            logger.exception("memory store: embedding failed; storing without one")
            vec = None
            self._embed_pending = True

        def tx(conn):
            row = conn.execute(
                self._q(
                    """
                    WITH m AS (
                        INSERT INTO {s}.memories (user_id, text, embedding, source)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id, created_at)
                    INSERT INTO {s}.memory_uses (memory_id, at, kind)
                    SELECT id, created_at, 'saved' FROM m
                    RETURNING memory_id
                    """
                ),
                (user_id, text, vec, Jsonb(src)),
            ).fetchone()
            return row[0]

        mid = self._run(tx)
        if vec is not None and self._embed_pending:
            self.backfill_embeddings()
        return mid

    def backfill_embeddings(self, limit: int = 500) -> int:
        rows = self._run(
            lambda c: c.execute(
                self._q("SELECT id, text FROM {s}.memories WHERE embedding IS NULL ORDER BY id LIMIT %s"),
                (limit,),
            ).fetchall()
        )
        if not rows:
            self._embed_pending = False
            return 0
        vecs, _ = self.voyage.embed([t for _, t in rows], "document")

        def tx(conn):
            with conn.cursor() as cur:
                cur.executemany(
                    self._q("UPDATE {s}.memories SET embedding = %s WHERE id = %s"),
                    [(v, i) for (i, _), v in zip(rows, vecs)],
                )

        self._run(tx)
        self._embed_pending = len(rows) == limit
        return len(rows)

    # ── read ────────────────────────────────────────────────────────────

    def retrieve(self, user_id: str, query: str, pool: int | None = None, with_zep: bool = False) -> Candidates:
        """Nearest `pool` facts by cosine, each scored by rerank-2.5.

        One statement, one database round trip. The candidates' pairwise
        cosines come back computed in SQL, each row's as a bytea of big-endian
        uint16 in units of 1/32767 (negative cosines floored at 0): 50 x 50 x
        2 bytes = 5 KB, where the vectors would be 200 KB and an int2[] 15 KB
        (4 length bytes per element). The whole reply then fits TCP's initial
        window, ~14 KB after an idle spell, and costs one round trip, 75-90 ms
        from here or from the gateway (europe-west4) to Neon us-east-1.
        """
        n = min(int(pool or self.pool_size), MAX_POOL)
        t0 = time.perf_counter()
        qv, qtok = self.voyage.embed([query], "query")
        t1 = time.perf_counter()
        stmt = self._q(
            """
            WITH c AS MATERIALIZED (
                SELECT * FROM {s}.nearest(%(q)s::vector, %(u)s::text, %(n)s::int))
            SELECT c.id, c.text, c.created_at, c.valid_from,
                   CASE WHEN %(z)s THEN c.zep END,
                   c.dist,
                   (SELECT ln(sum(power(greatest(
                               extract(epoch FROM now() - u.at)::float8, 1.0),
                               -(%(d)s::float8))))
                      FROM {s}.memory_uses u WHERE u.memory_id = c.id) AS activation,
                   (SELECT decode(string_agg(lpad(to_hex(greatest(0, round(
                               32767 * (1 - (c.embedding <=> c2.embedding))))::int), 4, '0'),
                               '' ORDER BY c2.dist, c2.id), 'hex')
                      FROM c c2) AS sims
            FROM c ORDER BY c.dist, c.id
            """
        )
        params = {"q": qv[0], "u": user_id, "n": n, "d": ACTR_DECAY, "z": with_zep}

        def tx(conn):
            with conn.cursor(binary=True) as cur:
                return cur.execute(stmt, params).fetchall()

        rows = self._run(tx)
        t2 = time.perf_counter()
        c = Candidates(
            query=query,
            ids=[r[0] for r in rows],
            texts=[r[1] for r in rows],
            created_at=[r[2] for r in rows],
            valid_from=[r[3] for r in rows],
            zep=[r[4] for r in rows],
            cosine=[1.0 - float(r[5]) for r in rows],
            activation=[float(r[6]) if r[6] is not None else None for r in rows],
            sim=np.array(
                [np.frombuffer(bytes(r[7]), dtype=">u2") for r in rows], dtype=np.float64
            ).reshape(len(rows), len(rows))
            / 32767,
        )
        c.relevance, rtok = self.voyage.rerank(query, c.texts)
        t3 = time.perf_counter()
        c.timings = {"embed": t1 - t0, "db": t2 - t1, "rerank": t3 - t2}
        c.tokens = {"embed": qtok, "rerank": rtok}
        return c

    def select(self, c: Candidates, limit: int, lam: float | None = None) -> list[int]:
        """Indices into c, by MMR. Candidates go in in relevance order so that
        ties (and lam=1) keep the reranker's order."""
        lam = self.lam if lam is None else lam
        order = sorted(range(len(c)), key=lambda i: -c.relevance[i])
        picked = mmr_sim([c.relevance[i] for i in order], c.sim[np.ix_(order, order)], limit, lam)
        return [order[p] for p in picked]

    def search(self, user_id: str, query: str, limit: int = 5, log: bool = True, with_zep: bool = False) -> list[dict]:
        """Top `limit` facts for a query, as dicts for the MCP tool."""
        if self._embed_pending:
            try:
                self.backfill_embeddings()
            except Exception:
                logger.exception("memory store: backfill failed")
        c = self.retrieve(user_id, query, with_zep=with_zep)
        self.last = c  # for callers that want timings (the evaluation)
        if not len(c):
            return []
        t0 = time.perf_counter()
        picks = self.select(c, limit)
        c.timings["mmr"] = time.perf_counter() - t0
        out = []
        for i in picks:
            item = {
                "text": c.texts[i],
                "relevance": round(c.relevance[i], 4),
                "created_at": _date(c.created_at[i]),
            }
            if c.valid_from[i]:
                item["valid_from"] = _date(c.valid_from[i])
            out.append(item)
        if log:
            self._pending = [f for f in self._pending if not f.done()]
            self._pending.append(self._bg.submit(self._log_retrieved_safely, c, picks))
        return out

    def _log_retrieved_safely(self, c: Candidates, picks: list[int]) -> None:
        try:
            self.log_retrieved(c, picks)
        except Exception:
            logger.exception("memory store: could not log retrieved uses")

    def log_retrieved(self, c: Candidates, picks: list[int]) -> None:
        search_id = str(uuidlib.uuid4())
        rows = [
            (
                c.ids[i],
                Jsonb(
                    {
                        "search": search_id,
                        "query": c.query[:500],
                        "rank": r,
                        "relevance": round(c.relevance[i], 4),
                        "cosine": round(c.cosine[i], 4),
                        "activation": None if c.activation[i] is None else round(c.activation[i], 4),
                        "lambda": self.lam,
                        "pool": len(c),
                    }
                ),
            )
            for r, i in enumerate(picks, 1)
        ]
        stmt = self._q(
            """
            INSERT INTO {s}.memory_uses (memory_id, kind, context)
            SELECT m, 'retrieved', c FROM unnest(%s::bigint[], %s::jsonb[]) AS t(m, c)
            """
        )
        self._run(lambda conn: conn.execute(stmt, ([r[0] for r in rows], [r[1] for r in rows])))

    def list(self, user_id: str, limit: int = 50) -> list[dict]:
        rows = self._run(
            lambda conn: conn.execute(
                self._q(
                    """
                    SELECT id, text, created_at, valid_from FROM {s}.memories
                    WHERE user_id = %s AND merged_into IS NULL
                    ORDER BY created_at DESC, id DESC LIMIT %s
                    """
                ),
                (user_id, limit),
            ).fetchall()
        )
        return [
            {"id": r[0], "text": r[1], "created_at": _date(r[2]), "valid_from": _date(r[3])}
            for r in rows
        ]

    # ── migration support ───────────────────────────────────────────────

    def zep_uuids_present(self, uuids: list[str]) -> set[str]:
        """Edge uuids the import must skip: already loaded, or excluded."""
        if not uuids:
            return set()
        rows = self._run(
            lambda c: c.execute(
                self._q(
                    """
                    SELECT zep_uuid::text FROM {s}.memories WHERE zep_uuid = ANY(%s::uuid[])
                    UNION SELECT uuid::text FROM {s}.zep_excluded WHERE uuid = ANY(%s::uuid[])
                    """
                ),
                (uuids, uuids),
            ).fetchall()
        )
        return {r[0] for r in rows}

    def excluded_episodes(self) -> set[str]:
        rows = self._run(
            lambda c: c.execute(
                self._q("SELECT uuid::text FROM {s}.zep_excluded WHERE kind = 'episode'")
            ).fetchall()
        )
        return {r[0] for r in rows}

    def exclude_zep_episodes(self, episode_uuids: list[str], reason: str) -> dict:
        """Remove these Zep episodes, and every fact extracted from them, from
        the store, and record them so no import brings them back.

        Deletes rows (memories, their uses, the episode texts). The store has
        no other delete path; this is for text that must not be here at all.
        """

        def tx(conn):
            with conn.transaction(), conn.cursor() as cur:
                cur.execute(
                    self._q(
                        "SELECT id, zep_uuid::text FROM {s}.memories "
                        "WHERE zep->'episode_uuids' ?| %s::text[]"
                    ),
                    (episode_uuids,),
                )
                hits = cur.fetchall()
                ids = [h[0] for h in hits]
                excl = [(u, "episode", reason) for u in episode_uuids] + [
                    (h[1], "edge", reason) for h in hits if h[1]
                ]
                cur.executemany(
                    self._q(
                        "INSERT INTO {s}.zep_excluded (uuid, kind, reason) VALUES (%s, %s, %s) "
                        "ON CONFLICT (uuid) DO NOTHING"
                    ),
                    excl,
                )
                cur.execute(
                    self._q("UPDATE {s}.memories SET merged_into = NULL WHERE merged_into = ANY(%s)"),
                    (ids,),
                )
                cur.execute(self._q("DELETE FROM {s}.memory_uses WHERE memory_id = ANY(%s)"), (ids,))
                uses = cur.rowcount
                cur.execute(self._q("DELETE FROM {s}.memories WHERE id = ANY(%s)"), (ids,))
                mems = cur.rowcount
                cur.execute(
                    self._q("DELETE FROM {s}.zep_episodes WHERE uuid = ANY(%s::uuid[])"),
                    (episode_uuids,),
                )
                eps = cur.rowcount
            return {"memories_deleted": mems, "uses_deleted": uses, "episodes_deleted": eps,
                    "excluded": len(excl)}

        return self._run(tx)

    def insert_imported(self, user_id: str, rows: list[dict]) -> int:
        """Insert imported facts, each with a `saved` use at its created_at.

        Keyed on zep_uuid: a row already present is skipped, and so is its
        use, because both come from one statement.
        """
        if not rows:
            return 0
        stmt = self._q(
            """
            WITH m AS (
                INSERT INTO {s}.memories
                    (user_id, text, embedding, created_at, valid_from, source, zep)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (zep_uuid) DO NOTHING
                RETURNING id, created_at)
            INSERT INTO {s}.memory_uses (memory_id, at, kind, context)
            SELECT id, created_at, 'saved', '{{"from": "zep-import"}}'::jsonb FROM m
            """
        )
        params = [
            (
                user_id,
                r["text"],
                r["embedding"],
                r["created_at"],
                r.get("valid_from"),
                Jsonb({"imported_from": "zep"}),
                Jsonb(r["zep"]),
            )
            for r in rows
        ]

        def tx(conn):
            with conn.cursor() as cur:
                cur.executemany(stmt, params)
                return cur.rowcount

        return self._run(tx)

    def refresh_zep(self, rows: list[dict]) -> int:
        """Update the `zep` record of facts already imported, where Zep has
        changed it since (retired an edge, attached another episode). Text,
        embedding and uses are left alone."""
        params = [(Jsonb(r["zep"]), r["zep"]["uuid"], Jsonb(r["zep"])) for r in rows]
        stmt = self._q("UPDATE {s}.memories SET zep = %s WHERE zep_uuid = %s AND zep IS DISTINCT FROM %s")

        def tx(conn):
            with conn.cursor() as cur:
                cur.executemany(stmt, params)
                return cur.rowcount

        return self._run(tx) if params else 0

    def insert_zep_episodes(self, user_id: str, episodes: list[dict]) -> int:
        stmt = self._q(
            """
            INSERT INTO {s}.zep_episodes (uuid, user_id, content, created_at, zep)
            SELECT %s, %s, %s, %s, %s
            WHERE NOT EXISTS (SELECT 1 FROM {s}.zep_excluded x WHERE x.uuid = %s::uuid)
            ON CONFLICT (uuid) DO NOTHING
            """
        )
        params = []
        for e in episodes:
            u = e.get("uuid_") or e.get("uuid")
            rest = {k: v for k, v in e.items() if k not in ("uuid_", "uuid", "content", "created_at")}
            params.append((u, user_id, e.get("content") or "", e["created_at"], Jsonb(rest), u))

        def tx(conn):
            with conn.cursor() as cur:
                cur.executemany(stmt, params)
                return cur.rowcount

        return self._run(tx) if params else 0

    def stats(self, user_id: str) -> dict:
        def tx(conn):
            q = lambda t, *a: conn.execute(self._q(t), a).fetchone()
            m = q(
                """SELECT count(*), count(embedding), count(*) FILTER (WHERE zep IS NOT NULL),
                          count(*) FILTER (WHERE zep->>'invalid_at' IS NOT NULL
                                              OR zep->>'expired_at' IS NOT NULL)
                   FROM {s}.memories WHERE user_id = %s""",
                user_id,
            )
            uses = dict(
                conn.execute(
                    self._q("SELECT kind, count(*) FROM {s}.memory_uses GROUP BY kind")
                ).fetchall()
            )
            eps = q("SELECT count(*) FROM {s}.zep_episodes WHERE user_id = %s", user_id)[0]
            sizes = dict(
                conn.execute(
                    """SELECT c.relname, pg_size_pretty(pg_total_relation_size(c.oid))
                       FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                       WHERE n.nspname = %s AND c.relkind IN ('r', 'i')""",
                    (self.schema,),
                ).fetchall()
            )
            schema_bytes = conn.execute(
                """SELECT pg_size_pretty(sum(pg_total_relation_size(c.oid)))
                   FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                   WHERE n.nspname = %s AND c.relkind = 'r'""",
                (self.schema,),
            ).fetchone()[0]
            return {
                "memories": m[0],
                "embedded": m[1],
                "from_zep": m[2],
                "retired_in_zep": m[3],
                "uses": uses,
                "zep_episodes": eps,
                "sizes": sizes,
                "schema_total": schema_bytes,
            }

        return self._run(tx)
