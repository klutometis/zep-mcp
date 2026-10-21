#!/usr/bin/env python3
"""Migrate a Zep graph into the own memory store, 1:1, without touching Zep.

Two steps, so the slow, read-only half can be kept and replayed:

    export   Zep -> a snapshot directory of JSONL (edges, nodes, episodes).
             Reads only: user.lookup, graph.edge/node/episode.list.
    load     snapshot -> Postgres. Every edge becomes one memory, retired ones
             included, with Zep's dates and ids kept in `zep`; each memory gets
             one `saved` use at its created_at; every episode (the text that
             was actually saved) goes to zep_episodes. Nothing is merged.

`load` is idempotent: rows are keyed on the Zep uuid, only edges not yet in the
table are embedded, and a memory and its seeded use are inserted in the same
statement, so a rerun neither duplicates rows nor pays for embeddings twice.
A fresh export plus `load --refresh-zep` is the catch-up sync: new edges are
added, and Zep's later changes to old ones (retirements) are copied over.

    uv run python scripts/migrate_from_zep.py export --out $TMPDIR/zep-export
    uv run python scripts/migrate_from_zep.py load --snapshot $TMPDIR/zep-export

Env: ZEP_API_KEY (export); MEMORY_DATABASE_URL, VOYAGE_API_KEY (load).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── export ───────────────────────────────────────────────────────────────


def _dump(new_pager, path: Path, label: str, attempts: int = 5) -> int:
    """Write every item of a listing to `path`, one JSON object per line.

    Zep answers 409 `cursor_stale` when the graph changes under a listing
    (a save during the ~2.5 min it takes to page through the edges). The only
    recovery it allows is starting over from the first page, so do that, a
    few times. Export after the writers have moved off Zep and it will not
    happen at all.
    """
    from zep_cloud.errors import ConflictError

    for attempt in range(1, attempts + 1):
        n = 0
        t0 = time.time()
        try:
            with path.open("w") as f:
                for page in new_pager().iter_pages():
                    for item in page.items or []:
                        f.write(json.dumps(item.model_dump(mode="json", by_alias=False)) + "\n")
                        n += 1
                    if n and n % 2000 < 100:
                        print(f"  {label}: {n} ({time.time() - t0:.0f}s)", file=sys.stderr)
        except ConflictError as e:
            if "cursor_stale" not in str(e) or attempt == attempts:
                raise
            print(f"  {label}: cursor went stale after {n}; starting over ({attempt}/{attempts})", file=sys.stderr)
            continue
        print(f"  {label}: {n} done in {time.time() - t0:.0f}s", file=sys.stderr)
        return n
    raise AssertionError("unreachable")


def export(args) -> None:
    from zep_cloud.client import Zep

    c = Zep(api_key=os.environ["ZEP_API_KEY"])
    g = c.user.lookup(user_id=args.user).graph_uuid
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"graph {g} -> {out}", file=sys.stderr)
    jobs = {
        "edges": lambda: c.graph.edge.list(g, page_size=100),
        "nodes": lambda: c.graph.node.list(g, page_size=100),
        "episodes": lambda: c.graph.episode.list(g, page_size=100),
    }
    with ThreadPoolExecutor(3) as ex:
        futs = {k: ex.submit(_dump, f, out / f"{k}.jsonl", k) for k, f in jobs.items()}
        counts = {k: f.result() for k, f in futs.items()}
    meta = {
        "user_id": args.user,
        "graph_uuid": g,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": counts,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))


# ── load ─────────────────────────────────────────────────────────────────


def _read(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _uuid(d: dict) -> str | None:
    return d.get("uuid_") or d.get("uuid")


def edge_to_row(e: dict, names: dict[str, str]) -> dict | None:
    """One Zep edge -> the columns of one memory (embedding added later)."""
    fact = (e.get("fact") or "").strip()
    if not fact:
        return None
    attrs = {k: v for k, v in (e.get("attributes") or {}).items() if k != "fact"}
    src, tgt = e.get("source_node_uuid"), e.get("target_node_uuid")
    zep = {
        "uuid": _uuid(e),
        "name": e.get("name"),
        "valid_at": e.get("valid_at"),
        "invalid_at": e.get("invalid_at"),
        "expired_at": e.get("expired_at"),
        "episode_uuids": e.get("episode_uuids") or [],
        "source_node": {"uuid": src, "name": names.get(src)},
        "target_node": {"uuid": tgt, "name": names.get(tgt)},
        "attributes": attrs,
    }
    return {
        "text": fact,
        "created_at": e["created_at"],
        "valid_from": e.get("valid_at"),
        "zep": zep,
    }


def load(args) -> None:
    from zep_mcp.store import Store, EMBED_PRICE_PER_MTOK

    snap = Path(args.snapshot)
    meta = json.loads((snap / "meta.json").read_text())
    user = args.user or meta["user_id"]
    edges = _read(snap / "edges.jsonl")
    nodes = _read(snap / "nodes.jsonl")
    episodes = _read(snap / "episodes.jsonl")
    names = {_uuid(n): n.get("name") for n in nodes}

    store = Store(os.environ["MEMORY_DATABASE_URL"], schema=args.schema)
    store.ensure_schema()

    rows, empty = [], 0
    for e in edges:
        r = edge_to_row(e, names)
        if r is None:
            empty += 1
        else:
            rows.append(r)
    retired = sum(1 for r in rows if r["zep"]["invalid_at"] or r["zep"]["expired_at"])
    print(
        f"snapshot {meta['exported_at']}: {len(edges)} edges ({empty} without a fact, "
        f"{retired} retired), {len(nodes)} nodes, {len(episodes)} episodes",
        file=sys.stderr,
    )

    have = store.zep_uuids_present([r["zep"]["uuid"] for r in rows])
    todo = [r for r in rows if r["zep"]["uuid"] not in have]
    print(f"{len(have)} already loaded, {len(todo)} to embed and insert", file=sys.stderr)

    # Embed each distinct text once; identical facts share a vector.
    texts = sorted({r["text"] for r in todo})
    t0 = time.time()
    tokens = 0
    inserted = 0
    vec_of: dict[str, object] = {}
    B = args.batch
    for i in range(0, len(texts), B):
        chunk = texts[i : i + B]
        vecs, tok = store.voyage.embed(chunk, "document")
        tokens += tok
        vec_of.update(zip(chunk, vecs))
        # Insert every row whose text is now embedded, a batch per transaction.
        ready = [r for r in todo if r["text"] in vec_of and "embedding" not in r]
        for r in ready:
            r["embedding"] = vec_of[r["text"]]
        inserted += store.insert_imported(user, ready)
        done = min(i + B, len(texts))
        if (i // B) % 10 == 0 or done == len(texts):
            print(
                f"  embedded {done}/{len(texts)} texts, {tokens} tokens, "
                f"{inserted} rows inserted ({time.time() - t0:.0f}s)",
                file=sys.stderr,
            )

    refreshed = store.refresh_zep([r for r in rows if r["zep"]["uuid"] in have]) if args.refresh_zep else 0
    eps = store.insert_zep_episodes(user, episodes)
    t1 = time.time()
    store.ensure_vector_index()
    t_index = time.time() - t1
    stats = store.stats(user)
    report = {
        "snapshot": str(snap),
        "exported_at": meta["exported_at"],
        "schema": args.schema,
        "user_id": user,
        "edges": len(edges),
        "edges_without_fact": empty,
        "retired_in_zep": retired,
        "already_loaded": len(have),
        "inserted": inserted,
        "distinct_texts_embedded": len(texts),
        "embed_tokens": tokens,
        "embed_cost_usd": round(tokens * EMBED_PRICE_PER_MTOK / 1e6, 4),
        "zep_records_refreshed": refreshed,
        "episodes_inserted": eps,
        "index_build_s": round(t_index, 1),
        "embed_insert_s": round(t1 - t0, 1),
        "table": stats,
    }
    print(json.dumps(report, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--user", default=os.environ.get("ZEP_USER_ID", "danenberg"))
    e.add_argument("--out", required=True)
    l = sub.add_parser("load")
    l.add_argument("--snapshot", required=True)
    l.add_argument("--user", help="owner of the memories (default: the snapshot's Zep user)")
    l.add_argument("--schema", default=os.environ.get("MEMORY_SCHEMA", "memory"))
    l.add_argument("--batch", type=int, default=128, help="texts per embedding request")
    l.add_argument(
        "--refresh-zep",
        action="store_true",
        help="also update the zep record (dates, episodes) of facts already loaded, "
        "for a final sync before switching backends",
    )
    a = p.parse_args()
    {"export": export, "load": load}[a.cmd](a)


if __name__ == "__main__":
    main()
