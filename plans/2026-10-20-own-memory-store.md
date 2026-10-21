# Personal memory: Zep, or a store of our own

2026-10-20. Status: proposal, nothing built. Measurements below are from the live
`danenberg` graph on 2026-10-19/20 unless marked otherwise.

## What memory is for, and who uses it

One store, shared by two clients through this repo's three tools (`add_memory`,
`search_memory`, `list_memory`):

- **llm-multiplex** (llm.danenberg.ai) searches before every turn and shows the model the
  top 5 as "relevant memories". After every turn, each model in the turn runs a save step
  (`lib/baby-models.ts` `evaluateForMemory`) and calls `save_memory` (the gateway's name for
  `add_memory`) once per fact. With three models that is up to three extractions per turn.
- **harness** calls `save_memory` only when a model decides to: 45 calls in three months in
  the corpus index.

## What Zep does for us

Zep is Graphiti, hosted (Rasmussen et al., 2025, "Zep: A Temporal Knowledge Graph
Architecture for Agent Memory", arXiv:2501.13956). For every saved text ("episode") it:

1. **Extracts entities** with an LLM, and **resolves** each against existing nodes
   (embedding similarity, then an LLM judgement).
2. **Extracts facts** as edges between entities, and dedups each against similar edges.
3. **Dates facts**: when each became and stopped being true, and when it was recorded and
   superseded.
4. **Retires facts**: for each new edge, an LLM compares it with semantically related
   existing edges and, if it judges a contradiction, sets the old edge's end date.
5. **Searches** hybrid (BM25, cosine, graph breadth-first) with a choice of reranker (RRF,
   MMR, cross-encoder, node distance, episode mentions).
6. Hosts all of it.

## What we have measured it doing

| | |
|---|---|
| Saved texts (episodes) | 7,003 |
| Facts (edges) | 21,557, of which 14,716 not retired |
| Facts on the user node ("danenberg") | 2,625 (18% of live) |
| Other nodes that are also Peter | "User" 131, "Peter Danenberg" 94, "Peter" 76 |
| A node for the assistant | "Assistant", 101 facts |
| Near-duplicate facts | 897 groups, 2,083 facts at embedding cosine ≥ 0.90 (≈1,200 removable) |
| Cross-encoder relevance | 1.00 for all 15 top results of the Dido query |
| Built-in MMR | best match plus four unrelated facts, at every `mmr_lambda` tried |
| Retired facts in search results | 57 of 160 (8 queries); returned unmarked |
| Retirement errors | retired while true: "working on a Western Canon project", "attended the Noah Kahan concert" (same day); live while stale: "currently at movie #226" (since July) |
| Provenance | 0 of 7,003 episodes carry a source or metadata |

What we cannot do on Zep:

- **See or use its embeddings.** Any semantic comparison of our own means embedding again.
- **Record use.** Zep keeps no history of a fact being retrieved or restated, which is what
  every decay model below runs on.
- **Control retirement.** It is an LLM's judgement between single sentences, without
  evidence, and we cannot turn it off or audit it.
- **Control extraction.** It runs on top of llm-multiplex's own extraction, so each saved
  sentence is split into edges again. "Venus and Cupid intervene" and "Cupid and Venus
  intervene" both came out the same day.

## What a decaying memory needs

The model to implement is ACT-R base-level activation (Anderson & Schooler 1991): a
memory's strength is

    B = ln( Σ_j t_j^(−d) ),   d ≈ 0.5

over its past uses j, where t_j is the time since use j. Frequent and recent use keeps it
strong; disuse lets it decay along a power law. Anderson & Schooler showed that the odds an
item is needed again in the environment follow the same law, which makes this the rational
estimate of what will be needed, not only a fit to human forgetting. FSRS, which the srs
tool already uses, is the same idea fitted to review data.

Precedents in LLM memory: Generative Agents (Park et al. 2023) rank by recency, importance
and relevance, refreshing recency when a memory is retrieved; MemoryBank (Zhong et al. 2023)
strengthens a memory on each recall along an Ebbinghaus curve. The offline pass follows
complementary learning systems (McClelland, McNaughton & O'Reilly 1995): a fast episodic
store, replayed offline into a slow semantic one.

What that requires of the store:

1. **A use log per fact**: when it was saved, restated, retrieved.
2. **Write-time merging**: a restatement of an existing fact is a use of it, not a new fact.
   On Zep it becomes a duplicate.
3. **Search that can combine relevance with activation.** ACT-R's own composition is base
   level plus context (spreading activation); the reranker's relevance plays the context part.
4. **Archive, not delete**: a fact below threshold leaves search but stays in the table.
5. **Provenance**, so the offline pass can open the conversation a fact came from.

None of these is possible on Zep without keeping a second store beside it.

## Our own store

Neon (the llm-multiplex database, PostgreSQL 17) has pgvector 0.8.0 available, not yet
installed. `VOYAGE_API_KEY` is set; Voyage is already the embedder in ~/prg/artificial
(`voyage-4-large`, reranking with `rerank-2.5`).

    memories     id, text, embedding vector(1024), created_at, valid_from (when known),
                 source jsonb (app, conversation or session, model, tools used, turn time),
                 merged_into, archived_at
    memory_uses  memory_id, at, kind (saved | restated | retrieved), context jsonb

- **Search**: embed the query (voyage-4-large), take the nearest 50 by cosine, score them
  with rerank-2.5, then pick k by MMR (Carbonell & Goldstein 1998) over the embeddings,
  with activation added to relevance with one weight. Log a `retrieved` use for each
  returned fact.
- **Save**: embed; if the nearest existing fact is a restatement (an LLM decides, on
  candidates above a cosine pre-filter), log a `restated` use on it instead of inserting.
- **Nightly**: merge duplicate groups (their use logs combine, which is what ACT-R implies
  for two traces of one memory); write a few abstractions from the day; archive what has
  decayed below threshold; log every archive and merge to B2. Dry-run reports first.
- **Clients don't change**: the same three MCP tools, same arguments, new backend.

Migration: export the 14,716 live facts with their dates, and the retired ones as archived;
embed them (~300k tokens); seed each fact's use log with one `saved` at its creation time;
merge the near-duplicate groups, so the six Venus/Cupid phrasings become one fact with six
uses. Zep stays read-only until the new store has run for a while.

What we give up:

- **The entity graph** (graph search, node distance, entity resolution). Optional to rebuild:
  most queries are answered by the fact text, and the graph we have is fragmented
  (four Peter nodes).
- **Automatic retirement.** Deliberately: decay replaces it, and dates written into
  changeable facts ("As of 2026-10-19, the user is at movie #226") cover the rest.
- **Community summaries**, which we do not use.
- **Hosting.** It becomes a table and an MCP server we run.

A middle option is self-hosted Graphiti (open source, needs Neo4j or FalkorDB): the box is
open, but the extraction and retirement logic are the same.

## Open questions

1. **What counts as reinforcement.** If being retrieved counts fully, a fact that merely
   matches common queries keeps strengthening itself whether or not it helps. Restatement
   is the safe signal; retrieval could count at a lower weight until measured.
2. **The decay clock.** ACT-R uses wall-clock time. Counting only days the system was used
   would keep a two-week break from weakening everything at once. That is a design choice,
   not something from the literature.
3. **Where extraction happens.** Per turn, by each model (today), or nightly, once, from the
   day's full transcripts (closer to complementary learning systems, and one extraction per
   conversation instead of three per turn). Nightly means a fact is not searchable from
   other conversations until the next morning.
4. **The two weights**: MMR's λ, and how much activation counts against relevance. Both
   want an offline evaluation set: past queries, and which memories would have helped.
