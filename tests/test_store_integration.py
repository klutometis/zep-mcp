"""The own store against a real Postgres, in a throwaway schema.

Needs MEMORY_DATABASE_URL (any database with pgvector available) and
VOYAGE_API_KEY; skipped otherwise. Creates memory_test_<random>, drops it
after. Costs a few hundred Voyage tokens.
"""

import os
import uuid

import pytest

pytestmark = pytest.mark.integration

if not (os.environ.get("MEMORY_DATABASE_URL") and os.environ.get("VOYAGE_API_KEY")):
    pytest.skip("MEMORY_DATABASE_URL and VOYAGE_API_KEY needed", allow_module_level=True)

from zep_mcp.store import Store  # noqa: E402
from zep_mcp.voyage import Voyage  # noqa: E402


class FlakyVoyage(Voyage):
    """Real Voyage, except the next `fail` embed calls raise."""

    fail = 0

    def embed(self, texts, input_type):
        if self.fail:
            self.fail -= 1
            raise RuntimeError("voyage down")
        return super().embed(texts, input_type)


@pytest.fixture(scope="module")
def store():
    s = Store(
        os.environ["MEMORY_DATABASE_URL"],
        schema=f"memory_test_{uuid.uuid4().hex[:8]}",
        voyage=FlakyVoyage(),
    )
    s.ensure_schema()
    s.ensure_vector_index()
    yield s
    s.close()
    s.drop_schema()


FACTS = [
    "Peter's daughter takes piano lessons on Tuesdays.",
    "Peter is reading the Aeneid in Latin, currently Book 4.",
    "Peter's son plays pickup soccer on Sunday mornings.",
    "The harness daemon runs on danenberg-desktop.",
    "Peter prefers Voyage embeddings to OpenAI embeddings.",
]


def test_add_search_list(store):
    ids = [store.add("u1", f, source={"app": "test", "model": "none"}) for f in FACTS]
    other = store.add("someone-else", "Peter is reading the Aeneid.")  # another tenant
    assert ids == sorted(ids)

    hits = store.search("u1", "What is Peter reading?", limit=3)
    store.flush()  # the use log is written in the background
    assert 1 <= len(hits) <= 3
    assert hits[0]["text"] == FACTS[1]
    assert set(hits[0]) == {"text", "relevance", "created_at"}  # valid_from unknown
    assert 0.0 <= hits[0]["relevance"] <= 1.0
    assert len(hits[0]["created_at"]) == 10

    # Scoped by user, and one `retrieved` use per fact returned.
    c = store.last
    assert other not in c.ids and len(c) == len(FACTS)
    assert all(a is not None for a in c.activation)  # every fact has its saved use
    with store.pool.connection() as conn:
        n = conn.execute(
            f"SELECT count(*) FROM {store.schema}.memory_uses WHERE kind = 'retrieved'"
        ).fetchone()[0]
        src = conn.execute(f"SELECT source FROM {store.schema}.memories WHERE id = %s", (ids[0],)).fetchone()[0]
    assert n == len(hits)
    assert src == {"app": "test", "model": "none"}

    # log=False (the evaluation) writes nothing.
    store.search("u1", "soccer", limit=2, log=False)
    store.flush()
    with store.pool.connection() as conn:
        n2 = conn.execute(
            f"SELECT count(*) FROM {store.schema}.memory_uses WHERE kind = 'retrieved'"
        ).fetchone()[0]
    assert n2 == n

    rows = store.list("u1", limit=3)
    assert [r["text"] for r in rows] == FACTS[:-4:-1]  # newest first


def test_import_is_idempotent(store):
    zid = str(uuid.uuid4())
    vec, _ = store.voyage.embed(["Peter saw a Noah Kahan concert."], "document")
    row = {
        "text": "Peter saw a Noah Kahan concert.",
        "embedding": vec[0],
        "created_at": "2026-05-01T12:00:00Z",
        "valid_from": "2026-04-30T20:00:00Z",
        "zep": {"uuid": zid, "invalid_at": "2026-05-01T13:00:00Z"},
    }
    assert store.insert_imported("u2", [row]) == 1
    assert store.insert_imported("u2", [row]) == 0
    assert store.zep_uuids_present([zid, str(uuid.uuid4())]) == {zid}
    with store.pool.connection() as conn:
        uses = conn.execute(
            f"""SELECT u.kind, u.at = m.created_at FROM {store.schema}.memory_uses u
                JOIN {store.schema}.memories m ON m.id = u.memory_id WHERE m.zep_uuid = %s""",
            (zid,),
        ).fetchall()
    assert uses == [("saved", True)]
    hit = store.search("u2", "concert", limit=1, log=False)[0]
    assert hit["created_at"] == "2026-05-01" and hit["valid_from"] == "2026-04-30"


def test_save_survives_embedding_failure(store):
    store.voyage.fail = 1
    mid = store.add("u3", "Peter's favourite reranker is rerank-2.5.")
    with store.pool.connection() as conn:
        assert conn.execute(
            f"SELECT embedding IS NULL FROM {store.schema}.memories WHERE id = %s", (mid,)
        ).fetchone()[0]
    store.add("u3", "Peter keeps scratch files in ~/tmp.")  # embeds, then backfills
    with store.pool.connection() as conn:
        assert not conn.execute(
            f"SELECT embedding IS NULL FROM {store.schema}.memories WHERE id = %s", (mid,)
        ).fetchone()[0]
    assert store.search("u3", "reranker", limit=1, log=False)[0]["text"].startswith("Peter's favourite")


def test_exclude_zep_episodes(store):
    """Excluded episodes leave the store with their facts and stay out."""
    ep_bad, ep_ok = str(uuid.uuid4()), str(uuid.uuid4())
    e_bad, e_ok = str(uuid.uuid4()), str(uuid.uuid4())
    vec, _ = store.voyage.embed(["an imported fact"], "document")
    rows = [
        {"text": "a fact from the barred episode", "embedding": vec[0], "created_at": "2026-10-20T00:00:00Z",
         "zep": {"uuid": e_bad, "episode_uuids": [ep_bad]}},
        {"text": "a fact from an ordinary episode", "embedding": vec[0], "created_at": "2026-10-20T00:00:00Z",
         "zep": {"uuid": e_ok, "episode_uuids": [ep_ok]}},
    ]
    assert store.insert_imported("u-x", rows) == 2
    eps = [{"uuid": ep_bad, "content": "barred text", "created_at": "2026-10-20T00:00:00Z"},
           {"uuid": ep_ok, "content": "ordinary text", "created_at": "2026-10-20T00:00:00Z"}]
    assert store.insert_zep_episodes("u-x", eps) == 2

    out = store.exclude_zep_episodes([ep_bad], "test")
    assert out == {"memories_deleted": 1, "uses_deleted": 1, "episodes_deleted": 1, "excluded": 2}
    assert store.excluded_episodes() == {ep_bad}

    # A rerun of the import treats the barred edge as present, so it is not
    # embedded or inserted again, and the barred episode is not reinserted.
    assert store.zep_uuids_present([e_bad, e_ok]) == {e_bad, e_ok}
    assert store.insert_zep_episodes("u-x", eps) == 0
    texts = {r["text"] for r in store.list("u-x", limit=10)}
    assert texts == {"a fact from an ordinary episode"}
