#!/usr/bin/env python3
"""Offline, read-only comparison: Zep's search_memory against the own store.

For each query: Zep's top 5 (search_edges, cross_encoder, exactly what the zep
backend serves) beside the own store's top 5 (voyage-4-large -> nearest 50 ->
rerank-2.5 -> MMR), with latency. Nothing is written anywhere: the own store
is searched with log=False, Zep only searched.

Also measured, from the same candidate pools (no extra API calls):
  - an MMR lambda sweep: relevance given up vs rewordings kept in the top 5;
  - HNSW recall@50 against an exact scan;
  - rerank-2.5's score of Zep's results, so both lists are judged by one model.

Queries: the named ones below, plus the latest user message of the most
recently active llm-multiplex conversations (one per conversation, test
prompts excluded), cut to 400 characters as llm-multiplex's turn-1 search does.

Output is personal data: it goes to --out (default $TMPDIR), never the repo.

Env: MEMORY_DATABASE_URL, VOYAGE_API_KEY, ZEP_API_KEY, LLM_MULTIPLEX_DATABASE_URL.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zep_mcp.store import Store  # noqa: E402

NAMED = [
    "Venus and Cupid conspired to make Dido fall in love with Aeneas",
    "Peter's kids",
    "toga virilis coming of age",
    "flashcards spaced repetition",
    "harness ssh gnubby security key touches",
    "what is Peter reading at the moment",
    "llm.danenberg.ai prompt caching",
]
EXCLUDE_PREFIXES = ("Cache test", "Tool-access test", "Time-zone test")
DUP_COS = 0.90  # cosine at which two facts count as rewordings (the plan's cut)
LAMBDAS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def recent_user_queries(n: int) -> list[str]:
    with psycopg.connect(os.environ["LLM_MULTIPLEX_DATABASE_URL"], prepare_threshold=None) as c:
        rows = c.execute(
            """
            SELECT DISTINCT ON (conversation_id) conversation_id, content, created_at
            FROM messages
            WHERE role = 'user'
              AND length(content) >= 20
              AND content NOT LIKE ALL (%s)
            ORDER BY conversation_id, created_at DESC
            """,
            ([p + "%" for p in EXCLUDE_PREFIXES],),
        ).fetchall()
    rows.sort(key=lambda r: r[2], reverse=True)
    out, seen = [], set()
    for _, content, _ in rows:
        q = content.strip()[:400]
        if q not in seen:
            seen.add(q)
            out.append(q)
        if len(out) == n:
            break
    return out


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def dup_pairs(sim: np.ndarray) -> int:
    """Pairs of facts at cosine >= DUP_COS, given their similarity matrix."""
    if len(sim) < 2:
        return 0
    iu = np.triu_indices(len(sim), 1)
    return int((sim[iu] >= DUP_COS).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="danenberg")
    ap.add_argument("--recent", type=int, default=18)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "own-memory-store"))
    a = ap.parse_args()

    from zep_cloud.client import Zep

    zep = Zep(api_key=os.environ["ZEP_API_KEY"])
    graph = zep.user.lookup(user_id=a.user).graph_uuid
    store = Store(os.environ["MEMORY_DATABASE_URL"])
    queries = [("named", q) for q in NAMED] + [("recent", q) for q in recent_user_queries(a.recent)]

    # Warm both paths once (TLS, pool, Neon compute) so latency is steady-state.
    store.search(a.user, "warm-up", limit=a.k, log=False)
    zep.graph.search_edges(graph, query="warm-up", page_size=a.k, reranker="cross_encoder")

    results = []
    for kind, q in queries:
        # Zep, as the zep backend serves it.
        t0 = time.perf_counter()
        page = zep.graph.search_edges(graph, query=q, page_size=a.k, reranker="cross_encoder")
        zep_s = time.perf_counter() - t0
        zitems = [e for e in (page.items or []) if e.fact]

        # Own store, end to end as search_memory runs it (minus the use log).
        t0 = time.perf_counter()
        own = store.search(a.user, q, limit=a.k, log=False)
        own_s = time.perf_counter() - t0
        c = store.last
        picks = store.select(c, a.k)
        # Zep's record of each candidate (retired or not), fetched apart so it
        # does not weigh on the timed path.
        with store.pool.connection() as conn:
            zrec = dict(
                conn.execute(
                    "SELECT id, zep FROM memory.memories WHERE id = ANY(%s)", (c.ids,)
                ).fetchall()
            )
        c.zep = [zrec.get(i) for i in c.ids]

        # Zep's facts: our embedding (by uuid) for duplicate counting; rerank-2.5 score.
        by_uuid = {}
        with store.pool.connection() as conn:
            for zu, v, inv, exp in conn.execute(
                """SELECT zep_uuid::text, embedding, zep->>'invalid_at', zep->>'expired_at'
                   FROM memory.memories WHERE zep_uuid = ANY(%s::uuid[])""",
                ([e.uuid_ for e in zitems],),
            ).fetchall():
                by_uuid[zu] = (np.asarray(v.to_numpy() if hasattr(v, "to_numpy") else v, dtype=np.float32), bool(inv or exp))
        missing = [e.fact for e in zitems if e.uuid_ not in by_uuid]
        extra = {}
        if missing:  # saved to Zep after the export
            mv, _ = store.voyage.embed(missing, "document")
            extra = dict(zip(missing, mv))
        zvecs = np.array([by_uuid[e.uuid_][0] if e.uuid_ in by_uuid else extra[e.fact] for e in zitems]).reshape(-1, 1024)
        zsim = zvecs @ zvecs.T
        zrr, _ = store.voyage.rerank(q, [e.fact for e in zitems]) if zitems else ([], 0)

        # HNSW recall@pool against an exact scan of the same filter.
        qv, _ = store.voyage.embed([q], "query")
        with store.pool.connection() as conn, conn.transaction():
            conn.execute("SET LOCAL enable_indexscan = off")
            exact = [
                r[0]
                for r in conn.execute(
                    """SELECT id FROM memory.memories WHERE user_id = %s AND merged_into IS NULL
                       AND embedding IS NOT NULL ORDER BY embedding <=> %s LIMIT %s""",
                    (a.user, qv[0], len(c)),
                ).fetchall()
            ]
        recall = len(set(exact) & set(c.ids)) / max(1, len(exact))

        # Lambda sweep on this pool.
        order = sorted(range(len(c)), key=lambda i: -c.relevance[i])
        sweep = {}
        for lam in LAMBDAS:
            p = store.select(c, a.k, lam)
            sweep[lam] = {
                "mean_rel": float(np.mean([c.relevance[i] for i in p])) if p else 0.0,
                "dups": dup_pairs(c.sim[np.ix_(p, p)]),
                "ids": [c.ids[i] for i in p],
            }

        zep_uuid_of = {i: (c.zep[i] or {}).get("uuid") for i in range(len(c))}
        results.append(
            {
                "kind": kind,
                "query": q,
                "zep": {
                    "latency_s": zep_s,
                    "items": [
                        {
                            "text": e.fact,
                            "relevance": e.relevance if e.relevance is not None else e.score,
                            "rerank25": zrr[i] if i < len(zrr) else None,
                            "created_at": str(e.created_at)[:10],
                            "retired": bool(e.invalid_at or e.expired_at),
                            "uuid": e.uuid_,
                        }
                        for i, e in enumerate(zitems)
                    ],
                    "dups": dup_pairs(zsim),
                },
                "own": {
                    "latency_s": own_s,
                    "timings": c.timings,
                    "tokens": c.tokens,
                    "items": [
                        dict(
                            o,
                            retired=bool((c.zep[i] or {}).get("invalid_at") or (c.zep[i] or {}).get("expired_at")),
                            activation=c.activation[i],
                            cosine=c.cosine[i],
                            uuid=zep_uuid_of[i],
                        )
                        for o, i in zip(own, picks)
                    ],
                    "dups": dup_pairs(c.sim[np.ix_(picks, picks)]),
                    "pure_rerank_top": [c.texts[i] for i in order[: a.k]],
                },
                "overlap": len({e.uuid_ for e in zitems} & {zep_uuid_of[i] for i in picks}),
                "hnsw_recall": recall,
                "sweep": sweep,
            }
        )
        print(f"{own_s:5.2f}s own  {zep_s:5.2f}s zep  {q[:70]!r}", file=sys.stderr)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(results, indent=1, default=str))
    (out / "eval.md").write_text(report(results, a.k, store.lam))
    print(out / "eval.md")


def report(results, k, lam) -> str:
    own_lat = [r["own"]["latency_s"] for r in results]
    zep_lat = [r["zep"]["latency_s"] for r in results]
    L = []
    L.append(f"# Own store vs Zep: {len(results)} queries, top {k}, lambda={lam}\n")
    L.append(
        f"Latency (s): own p50 {pct(own_lat, 50):.2f} p95 {pct(own_lat, 95):.2f} max {max(own_lat):.2f}; "
        f"zep p50 {pct(zep_lat, 50):.2f} p95 {pct(zep_lat, 95):.2f} max {max(zep_lat):.2f}\n"
    )
    for part in ("embed", "db", "rerank", "mmr"):
        xs = [r["own"]["timings"].get(part, 0) for r in results]
        L.append(f"- own {part}: p50 {pct(xs, 50) * 1000:.0f} ms, p95 {pct(xs, 95) * 1000:.0f} ms")
    L.append(f"- HNSW recall@50 vs exact: mean {statistics.mean(r['hnsw_recall'] for r in results):.3f}, "
             f"min {min(r['hnsw_recall'] for r in results):.3f}")
    def agg(side, key):
        return sum(r[side][key] for r in results) if key == "dups" else None
    zr = [i["rerank25"] for r in results for i in r["zep"]["items"] if i["rerank25"] is not None]
    orr = [i["relevance"] for r in results for i in r["own"]["items"]]
    L.append(f"- rewording pairs (cos >= {DUP_COS}) in top {k}, summed: zep {agg('zep', 'dups')}, own {agg('own', 'dups')}")
    L.append(f"- retired-in-Zep facts in top {k}: zep {sum(i['retired'] for r in results for i in r['zep']['items'])}, "
             f"own {sum(i['retired'] for r in results for i in r['own']['items'])}")
    L.append(f"- mean rerank-2.5 score of the top {k}: zep {statistics.mean(zr):.3f}, own {statistics.mean(orr):.3f}")
    L.append(f"- results under llm-multiplex's 0.3 relevance floor: zep {sum(1 for r in results for i in r['zep']['items'] if (i['relevance'] or 0) < 0.3)}, "
             f"own {sum(1 for x in orr if x < 0.3)} (own scores are rerank-2.5; zep's are its cross-encoder)")
    L.append(f"- overlap (same Zep edge in both top {k}): mean {statistics.mean(r['overlap'] for r in results):.1f}\n")
    L.append("## Lambda sweep (all queries)\n")
    L.append("| lambda | mean rerank relevance of top 5 | rewording pairs | queries with any |")
    L.append("|---|---|---|---|")
    for lam in LAMBDAS:
        s = [r["sweep"][lam] for r in results]
        L.append(f"| {lam} | {statistics.mean(x['mean_rel'] for x in s):.3f} | {sum(x['dups'] for x in s)} | {sum(1 for x in s if x['dups'])} |")
    L.append("")
    for n, r in enumerate(results, 1):
        L.append(f"## {n}. [{r['kind']}] {r['query'][:300]!r}\n")
        L.append(f"zep {r['zep']['latency_s']:.2f}s, dups {r['zep']['dups']} | own {r['own']['latency_s']:.2f}s, dups {r['own']['dups']} | overlap {r['overlap']} | recall {r['hnsw_recall']:.2f}\n")
        L.append("Zep:")
        for i in r["zep"]["items"]:
            rr = f"{i['rerank25']:.2f}" if i["rerank25"] is not None else "-"
            L.append(f"  - ({i['relevance']:.2f} / rr {rr}) [{i['created_at']}]{' RETIRED' if i['retired'] else ''} {i['text']}")
        L.append("Own:")
        for i in r["own"]["items"]:
            vf = f" valid {i['valid_from']}" if i.get("valid_from") else ""
            L.append(f"  - ({i['relevance']:.2f}) [{i['created_at']}{vf}]{' RETIRED' if i['retired'] else ''} {i['text']}")
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
