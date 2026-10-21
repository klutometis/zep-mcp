# Own memory store, phase 1: store, migration, search

2026-10-21. Status: built and loaded, **not live**. Zep stays the backend until
`ZEP_MCP_BACKEND=own` is set on the gateways. Design: `2026-10-20-own-memory-store.md`.
Numbers are aggregates only; per-query output is personal and stays out of
this (public) repo.

## What exists

- **Where**: Neon project `llm-multiplex` (us-east-1, PG 17), database `neondb`,
  schema `memory`. pgvector 0.8.0 installed (`CREATE EXTENSION vector`, in
  `public`). No existing table touched; drizzle-kit only manages `public`.
- **Tables**: `memories` (+ `user_id`, since the server is multi-tenant by
  W3C baggage, and a generated unique `zep_uuid` for idempotent import),
  `memory_uses`, `zep_episodes`. HNSW on `embedding` (`vector_cosine_ops`,
  m 16, ef_construction 64). Function `nearest(q, uid, n)` does the vector
  search with `hnsw.ef_search = 400` attached.
- **Loaded**: all 21,595 Zep edges as of 07:06Z (6,841 retired, dates in
  `zep`), 21,595 `saved` uses at each edge's created_at, 7,016 episodes.
  Nothing merged. 318,573 tokens for 21,001 distinct texts: $0.038 at
  voyage-4-large's $0.12/M (within the free 200M anyway). 3 min to embed and
  insert, 13 s to build the index. A rerun inserts nothing and embeds nothing.
- **Size**: the schema is 359 MB (heap 169 MB, HNSW 164 MB); the database
  went from 333 MB to 692 MB. The project's logical-size limit is 16 TB
  (Launch, billed per GB-month), so the limit is not the constraint; halfvec
  would halve both if it ever matters.

## Three things that were not obvious

1. **TOAST hid the index from the planner.** A 1024-d vector is 4 KB, over the
   TOAST threshold, so by default it lives out of line. The heap looked tiny,
   a sequential scan looked cheap, and the planner never used HNSW: 65-170 ms
   per search instead of 1.6 ms. `ALTER COLUMN embedding SET STORAGE PLAIN`
   (one row per page, +50 MB) made the cost model honest.
2. **Round trips, not queries, are the latency.** Postgres executes a search
   in 2-8 ms; the gateway (Railway europe-west4) and this machine are each
   ~75-90 ms from Neon us-east-1. The first version spent 536 ms in the
   database: psycopg's implicit BEGIN/COMMIT, a separate `SET LOCAL
   hnsw.ef_search`, and 200 KB of candidate vectors crawling through TCP slow
   start. Now: an autocommit pool, `ef_search` attached to a SQL function
   (works through the transaction-mode pooler), and the candidates' pairwise
   cosines computed in SQL and returned as 100 bytes a row. One round trip.
3. **ef_search governs result quality, not just recall.** The reranker reaches
   deep into the 50 candidates, so a neighbour HNSW misses near the boundary
   can be one it would have ranked first. Queries whose final top 5 differ
   from an exact scan's: 9/25 at ef 100, 4/25 at 200, 3/25 at 400 (recall@50
   0.946 / 0.981 / 0.994). At 800 the planner switches to a 40 ms seq scan.

## Search, measured (25 queries: 7 named, 18 recent llm-multiplex user messages)

| | Zep (cross_encoder) | own |
|---|---|---|
| latency p50 / p95 | 0.17 / 0.20 s | 0.42 / 0.56 s |
| rewording pairs (cos ≥ 0.9) in the 25 top-5s | 28 | 1 |
| mean rerank-2.5 relevance of the top 5 | 0.547 | 0.548 |
| Zep-retired facts in the top 5s | 26 | 28 |

Own latency is embed ~130 ms + database ~120 ms + rerank ~130 ms; the use log
is written off the request path.

By reading the lists side by side: the own store is clearly better on 9
queries, mostly where Zep returned several phrasings of one fact; slightly
better on 3; a wash on 11, mostly queries memory cannot answer, where both
return noise; clearly worse on 2.
rerank-2.5 is better calibrated for llm-multiplex's 0.3 relevance floor: on a
query nothing in memory answers it scored the top results 0.29-0.38, where
Zep's cross-encoder gave 0.72-0.98.

**λ = 0.5.** The sweep (pairs at cos ≥ 0.9 in the 25 top-5s; mean relevance):
0.4 → 0, 0.507; 0.5 → 1, 0.548; 0.6 → 7, 0.579; 1.0 → 21, 0.608. 0.5 is the
knee. Its cost is the two losses: a tight cluster of distinct but similar
relevant facts (several shops' same-day shipping) loses members to unrelated
facts, because raw cosine between topical siblings (0.7-0.85) is not far from
a rewording's (0.9+). λ = 0.6 fixes both and brings back the Dido rewordings
(1 → 6 pairs). A similarity rescaled to penalise only near-duplicates
(`max(0, (cos - c0) / (1 - c0))`) may fix both; it wants a labelled
evaluation set before it is tuned, not these 25 queries.

## Decided here

- `source` is a **string** holding a JSON object, not an object parameter:
  llm-multiplex hands save_memory to Gemini, and one schema Gemini rejects
  400s the whole request. On Zep it becomes source_description + flat
  metadata, with a retry without them if Zep refuses (untested live: phase 1
  writes nothing to Zep).
- The own backend scopes by `user_id`, like Zep's per-user graphs.
- If Voyage is down, `add_memory` stores the fact unembedded and embeds it on
  the next call rather than losing the save.
- The server never runs DDL; the migration does.
- Retired facts are searchable, as on Zep, and are not marked in results.

## Left for phase 2

- **Switching**: the gateways pass only allowlisted env (`env_keys`
  `ZEP_API_KEY, ZEP_USER_ID` in `mcp-gateway-config/deploy/*/servers.json`);
  add `ZEP_MCP_BACKEND, MEMORY_DATABASE_URL, VOYAGE_API_KEY`. Before flipping:
  fresh export + `load --refresh-zep` to catch the edges saved since 07:06Z
  and Zep's later retirements. Consider a least-privilege role for the
  server (today's URL is the owner of the whole llm-multiplex database).
- **Provenance**: llm-multiplex's `runAsyncMemoryWork` and harness to pass
  `source`.
- **Nightly pass**: LLM-judged merging of rewordings (their use logs combine);
  write-time restatement detection; abstractions. Zep already attached a
  second-or-later episode to 7,024 edges (up to 8): restatements, datable
  from `zep_episodes`, that could seed `restated` uses.
- **Activation in ranking**: give it a weight once there is a labelled
  evaluation set; decide whether `retrieved` counts as reinforcement (open
  question 1).
- llm-multiplex's 0.3 floor was set for Zep's scores; recalibrate it for
  rerank-2.5.
