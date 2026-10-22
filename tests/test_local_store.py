"""The local backend's store, with a fake embedder (no network, no Vertex)."""

import json
import sqlite3

import numpy as np
import pytest

from zep_mcp.local_store import LocalStore

# Words a fake embedding is built from: a text's vector is the count of each
# word in it, so texts sharing words are close and rewordings are identical.
VOCAB = ["follow", "up", "schedule", "session", "serve", "port", "browser", "reviewer", "deck", "meeting"]


class FakeEmbedder:
    def __init__(self, model="fake-1"):
        self.model = model
        self.calls = []

    def embed(self, texts, task):
        self.calls.append((task, list(texts)))
        rows = []
        for t in texts:
            words = t.lower().replace("-", " ").split()
            v = np.array([sum(w.startswith(k) for w in words) for k in VOCAB], dtype=np.float32) + 1e-3
            rows.append(v / np.linalg.norm(v))
        return np.stack(rows)


@pytest.fixture
def store(tmp_path):
    s = LocalStore(tmp_path / "m.sqlite", FakeEmbedder(), lam=0.5)
    yield s
    s.close()


def test_add_list_and_dates(store):
    a = store.add("u", "Schedule a follow-up for a session", source='{"app": "harness", "session": "x"}')
    b = store.add("u", "Serve pages on a port for the browser", created_at="2026-10-20T05:07:00+00:00")
    rows = store.list("u")
    assert [r["id"] for r in rows] == [a, b]  # newest first; b keeps its import date
    assert rows[1]["created_at"] == "2026-10-20"
    db = sqlite3.connect(store.path)
    assert json.loads(db.execute("SELECT source FROM memories WHERE id=?", (a,)).fetchone()[0]) == {
        "app": "harness", "session": "x"}
    assert db.execute("SELECT count(*) FROM memory_uses WHERE kind='saved'").fetchone()[0] == 2


def test_search_ranks_and_skips_rewordings(store):
    store.add("u", "Schedule a follow-up for a session")
    store.add("u", "Schedule a follow-up for a session")  # a rewording (identical vector)
    store.add("u", "Serve pages on a port for the browser")
    store.add("u", "The reviewer owns the meeting deck")
    out = store.search("u", "how do I schedule a follow up", limit=2)
    assert out[0]["text"] == "Schedule a follow-up for a session"
    assert out[1]["text"] != out[0]["text"]  # MMR skipped the duplicate
    assert store.embedder.calls[-1][0] == "RETRIEVAL_QUERY"
    db = sqlite3.connect(store.path)
    logged = db.execute("SELECT count(*), max(context) FROM memory_uses WHERE kind='retrieved'").fetchone()
    assert logged[0] == 2 and json.loads(logged[1]) == {"query": "how do I schedule a follow up"}


def test_users_are_separate(store):
    store.add("u", "Schedule a follow-up for a session")
    assert store.search("someone-else", "schedule") == []
    assert store.list("someone-else") == []


def test_model_change_reembeds(tmp_path):
    path = tmp_path / "m.sqlite"
    s1 = LocalStore(path, FakeEmbedder("old"), lam=0.5)
    s1.add("u", "Schedule a follow-up for a session")
    s1.close()
    e2 = FakeEmbedder("new")
    s2 = LocalStore(path, e2, lam=0.5)
    assert s2.search("u", "schedule", limit=1)[0]["text"].startswith("Schedule")
    assert ("RETRIEVAL_DOCUMENT", ["Schedule a follow-up for a session"]) in e2.calls
    db = sqlite3.connect(path)
    assert db.execute("SELECT model FROM memories").fetchone()[0] == "new"
    s2.close()


def test_file_is_private(store):
    assert (store.path.stat().st_mode & 0o077) == 0
