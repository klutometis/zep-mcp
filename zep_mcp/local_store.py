"""Memory for a machine whose text may go to Google and nowhere else.

ZEP_MCP_BACKEND=local. A SQLite file on the machine's own disk, embeddings
from Vertex AI in the machine's own Google Cloud project, and the same
selection as the own backend: maximal marginal relevance over the embeddings
(zep_mcp/rank.py). No Voyage, no Neon, no Zep.

Built for danenberg-central (2026-10-22), whose rule is that corp text goes
to Google and nowhere else. Its memories used to go to Zep, a third party, and
the own backend would send them to Voyage and Neon. Central has saved three
memories ever, so this is deliberately small: a search reads all of a user's
vectors and ranks them in numpy, with no index.

Relevance is the cosine between the query's embedding (task RETRIEVAL_QUERY)
and each memory's (RETRIEVAL_DOCUMENT). There is no reranker: Voyage's is off
limits here, and Vertex's ranking API is more machinery than a handful of rows
need. A row embedded by a different model than the current one is embedded
again on the next search, so changing MEMORY_EMBED_MODEL needs no migration.

Environment:
    MEMORY_VERTEX_PROJECT    Google Cloud project for Vertex AI; falls back to
                             GOOGLE_CLOUD_PROJECT (one of the two is required)
    MEMORY_VERTEX_LOCATION   default "global"
    MEMORY_SQLITE_PATH       default ~/.local/share/zep-mcp/memory.sqlite
    MEMORY_EMBED_MODEL       default gemini-embedding-001, at 768 dimensions
    MEMORY_MMR_LAMBDA        shared with the own backend, default 0.5

Credentials are Application Default Credentials (gcloud auth
application-default login), handed to the client explicitly. That way an API
key in the environment (GOOGLE_API_KEY, GEMINI_API_KEY) can never send the
text to the consumer Gemini API instead of Vertex.

Not backed up off the machine: the file lives with the rest of the machine's
home directory, and copying it elsewhere is subject to the same rule.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from zep_mcp.rank import mmr
from zep_mcp.source import normalize_source

DEFAULT_MODEL = "gemini-embedding-001"
DIMENSIONS = 768
DEFAULT_LAMBDA = 0.5
POOL = 50  # most similar rows that MMR chooses among

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id         INTEGER PRIMARY KEY,
    user_id    TEXT NOT NULL,
    text       TEXT NOT NULL,
    embedding  BLOB NOT NULL,   -- float32, unit length
    model      TEXT NOT NULL,   -- what produced `embedding`
    created_at TEXT NOT NULL,   -- ISO 8601, UTC
    source     TEXT             -- provenance, JSON (see add_memory)
);
CREATE INDEX IF NOT EXISTS memories_user ON memories (user_id, created_at);
CREATE TABLE IF NOT EXISTS memory_uses (
    memory_id INTEGER NOT NULL REFERENCES memories (id),
    at        TEXT NOT NULL,
    kind      TEXT NOT NULL CHECK (kind IN ('saved', 'restated', 'retrieved')),
    context   TEXT
);
CREATE INDEX IF NOT EXISTS memory_uses_memory ON memory_uses (memory_id, at);
"""


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str], task: str) -> np.ndarray:
        """Unit-length float32 rows, one per text. task: RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY."""
        ...


class VertexEmbedder:
    """Vertex AI text embeddings, authenticated with Application Default Credentials."""

    def __init__(self, project: str, location: str = "global", model: str = DEFAULT_MODEL):
        import google.auth
        from google import genai

        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        self.model = model
        self._client = genai.Client(
            vertexai=True, project=project, location=location, credentials=creds
        )

    def embed(self, texts: list[str], task: str) -> np.ndarray:
        from google.genai import types

        out = []
        for i in range(0, len(texts), 100):
            r = self._client.models.embed_content(
                model=self.model,
                contents=texts[i : i + 100],
                config=types.EmbedContentConfig(task_type=task, output_dimensionality=DIMENSIONS),
            )
            out.extend(e.values for e in r.embeddings)
        return _unit(np.asarray(out, dtype=np.float32))


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _iso(ts: Any) -> str:
    if isinstance(ts, datetime):
        return ts.astimezone(timezone.utc).isoformat(timespec="seconds")
    return str(ts)


def vertex_project() -> str | None:
    return os.environ.get("MEMORY_VERTEX_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")


class LocalStore:
    def __init__(self, path: str | os.PathLike, embedder: Embedder, lam: float | None = None):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.lam = float(lam if lam is not None else os.environ.get("MEMORY_MMR_LAMBDA", DEFAULT_LAMBDA))
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(SCHEMA)
        self._db.commit()
        try:
            self.path.chmod(0o600)  # memory text; readable by its owner only
        except OSError:
            pass

    @classmethod
    def from_env(cls) -> "LocalStore":
        project = vertex_project()
        if not project:
            raise RuntimeError("ZEP_MCP_BACKEND=local requires MEMORY_VERTEX_PROJECT or GOOGLE_CLOUD_PROJECT")
        embedder = VertexEmbedder(
            project=project,
            location=os.environ.get("MEMORY_VERTEX_LOCATION", "global"),
            model=os.environ.get("MEMORY_EMBED_MODEL", DEFAULT_MODEL),
        )
        path = os.environ.get("MEMORY_SQLITE_PATH", "~/.local/share/zep-mcp/memory.sqlite")
        return cls(path, embedder)

    def add(self, user_id: str, text: str, source: Any = None, created_at: Any = None) -> int:
        """Store one memory; log it as `saved`. created_at is for imports only."""
        vec = self.embedder.embed([text], "RETRIEVAL_DOCUMENT")[0]
        at = _iso(created_at) if created_at is not None else _now()
        src = normalize_source(source)
        with self._lock, self._db:
            cur = self._db.execute(
                "INSERT INTO memories (user_id, text, embedding, model, created_at, source) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, text, vec.tobytes(), self.embedder.model, at,
                 json.dumps(src) if src else None),
            )
            mid = cur.lastrowid
            self._db.execute(
                "INSERT INTO memory_uses (memory_id, at, kind) VALUES (?, ?, 'saved')", (mid, at)
            )
        return int(mid)

    def _reembed_stale(self, user_id: str) -> None:
        with self._lock:
            stale = self._db.execute(
                "SELECT id, text FROM memories WHERE user_id = ? AND model <> ?",
                (user_id, self.embedder.model),
            ).fetchall()
        if not stale:
            return
        vecs = self.embedder.embed([t for _, t in stale], "RETRIEVAL_DOCUMENT")
        with self._lock, self._db:
            self._db.executemany(
                "UPDATE memories SET embedding = ?, model = ? WHERE id = ?",
                [(v.tobytes(), self.embedder.model, mid) for (mid, _), v in zip(stale, vecs)],
            )

    def search(self, user_id: str, query: str, limit: int = 5, log: bool = True) -> list[dict]:
        """Top `limit` memories by MMR over cosine relevance, as dicts for the MCP tool."""
        self._reembed_stale(user_id)
        with self._lock:
            rows = self._db.execute(
                "SELECT id, text, embedding, created_at FROM memories WHERE user_id = ?", (user_id,)
            ).fetchall()
        if not rows:
            return []
        vecs = np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
        q = self.embedder.embed([query], "RETRIEVAL_QUERY")[0]
        rel = vecs @ q
        pool = np.argsort(-rel, kind="stable")[:POOL]
        picks = [int(pool[i]) for i in mmr(rel[pool].tolist(), vecs[pool], limit, self.lam)]
        out = [
            {"text": rows[i][1], "relevance": round(float(rel[i]), 4), "created_at": rows[i][3][:10]}
            for i in picks
        ]
        if log and picks:
            at = _now()
            ctx = json.dumps({"query": query})
            with self._lock, self._db:
                self._db.executemany(
                    "INSERT INTO memory_uses (memory_id, at, kind, context) VALUES (?, ?, 'retrieved', ?)",
                    [(rows[i][0], at, ctx) for i in picks],
                )
        return out

    def list(self, user_id: str, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, text, created_at FROM memories WHERE user_id = ? "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [{"id": r[0], "text": r[1], "created_at": r[2][:10]} for r in rows]

    def close(self) -> None:
        with self._lock:
            self._db.close()
