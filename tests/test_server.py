"""The Zep backend's add_memory with the new optional `source` argument."""

import json
from types import SimpleNamespace

import zep_mcp


class FakeEpisodes:
    def __init__(self, reject_metadata=False):
        self.calls = []
        self.reject_metadata = reject_metadata

    def add(self, graph, **kw):
        self.calls.append((graph, kw))
        if self.reject_metadata and "metadata" in kw:
            raise RuntimeError("400 bad metadata")
        return SimpleNamespace(episode=SimpleNamespace(uuid_="ep-1"))


def fake_client(monkeypatch, episodes):
    monkeypatch.setattr(zep_mcp, "client", SimpleNamespace(graph=SimpleNamespace(episode=episodes)))
    monkeypatch.setattr(zep_mcp, "_graph_uuid", lambda uid: "g-1")


def test_default_backend_is_zep():
    assert zep_mcp.BACKEND == "zep"


def test_store_accessor_survives_importing_the_store_module():
    # Importing zep_mcp.store sets the package attribute `store` to the
    # submodule; the accessor must not be called that (it once was, and the
    # own backend's second tool call failed with "'module' is not callable").
    import zep_mcp.store  # noqa: F401

    assert callable(zep_mcp.own_store)


def test_add_without_source_is_unchanged(monkeypatch):
    eps = FakeEpisodes()
    fake_client(monkeypatch, eps)
    assert zep_mcp.add_memory("Peter likes Virgil") == "Stored (episode ep-1)"
    assert eps.calls == [("g-1", {"type": "text", "data": "Peter likes Virgil"})]


def test_add_with_source_passes_description_and_flat_metadata(monkeypatch):
    eps = FakeEpisodes()
    fake_client(monkeypatch, eps)
    src = {"app": "llm-multiplex", "conversation": "c-9", "model": "m", "tools": ["search", "scrape"],
           "turn_time": "2026-10-21T09:00:00-07:00", "extra": {"a": 1}}
    zep_mcp.add_memory("fact", source=json.dumps(src))
    (_, kw), = eps.calls
    assert kw["metadata"]["tools"] == "search, scrape"
    assert kw["metadata"]["extra"] == '{"a": 1}'
    assert kw["metadata"]["app"] == "llm-multiplex"
    assert kw["source_description"].startswith("app=llm-multiplex; conversation=c-9")


def test_add_falls_back_when_zep_rejects_metadata(monkeypatch):
    eps = FakeEpisodes(reject_metadata=True)
    fake_client(monkeypatch, eps)
    assert zep_mcp.add_memory("fact", source='{"app": "harness"}') == "Stored (episode ep-1)"
    assert len(eps.calls) == 2 and "metadata" not in eps.calls[1][1]


def test_source_as_prose():
    assert zep_mcp.zep_source_kwargs("typed by hand")["metadata"] == {"note": "typed by hand"}
    assert zep_mcp.zep_source_kwargs("") == {}
