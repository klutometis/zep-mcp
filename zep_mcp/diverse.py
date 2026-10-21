"""Pick search results that are relevant *and* different from each other.

Why this exists
---------------
Zep stores the same fact many times over in different words. Each model in
llm-multiplex saves its own facts after every turn, and Zep's extraction then
splits a sentence into edges ("Venus and Cupid intervene", "Cupid and Venus
intervene"). Asked about Dido on 2026-10-20, `search_memory` returned five
phrasings of "Venus and Cupid conspired to make Dido fall in love with Aeneas",
which is the whole of what llm-multiplex shows a model as relevant memory.

Ranking can't fix it. The cross-encoder scores all fifteen of that query's
top results at 1.00, so the top five are an arbitrary draw from near-identical
facts. Zep's own diversity reranker (`reranker="mmr"`) returned the best match
and then four unrelated facts (a coach, a birthday, a slide renderer) at every
`mmr_lambda` tried, so it is not usable either.

What it does
------------
`search_memory` over-fetches (see `fetch_size`), and `pick_diverse` walks the
results in Zep's order, skipping any fact whose words overlap one already
picked by `threshold` or more. Relevance order is preserved; only repeats go.

Similarity is word overlap (Jaccard over content words, lightly stemmed), not
embeddings. Zep does not return its embeddings, and computing our own would put
an embedding API call and key on every search. The duplicates here are
paraphrases that share nearly all their content words, which overlap catches:

    "Venus conspired with Cupid to make Dido fall in love with Aeneas."
    "Cupid conspired with Venus to make Dido fall in love with Aeneas."   1.00
    "The toga virilis ceremony marked a young Roman's transition to legal
     adulthood." / "A young Roman undergoes transition to legal adulthood
     via the toga virilis ceremony."                                      0.73

while facts that only share a subject do not ("The user prefers tea" /
"The user prefers coffee": 0.33).

Overlap alone misfires on short facts built from the same template:
"The user's flashcards cover Aeneas" and "... cover Dido" overlap 0.67. So a
fact is only a repeat if it also brings no name or number that the kept fact
lacks (`names`: capitalized words and numbers). That keeps "cover Dido", and
also a rewording that adds a detail ("...for the protection of the Trojans"),
which errs on the side of showing more.

Environment: ZEP_MCP_DIVERSE_THRESHOLD (default 0.6; 1 or more turns the
filter off), ZEP_MCP_DIVERSE_FETCH (results fetched per result returned,
default 4).
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable
from typing import TypeVar

T = TypeVar("T")

DEFAULT_THRESHOLD = 0.6
DEFAULT_FETCH_FACTOR = 4

# Words that carry no content of their own. "user" and "peter" are here
# because nearly every fact is about him: two facts that share only "the user"
# are not alike.
_STOP = frozenset(
    """
    a an the and or but nor so yet of to in on at by for from with without into
    onto over under about as than then that this these those there here it its
    is are was were be been being am do does did done has have had having will
    would shall should can could may might must not no also very just only
    who whom whose which what when where why how all any both each few more
    most other some such own same he him his she her hers they them their theirs
    we us our ours you your yours i me my mine user users peter
    """.split()
)

_WORD = re.compile(r"[a-z0-9]+")


def _stem(word: str) -> str:
    """Crude, on purpose: enough that "facts"/"fact" and "named"/"name" meet."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def terms(text: str) -> frozenset[str]:
    """The content words of a fact, lowercased and stemmed."""
    return frozenset(
        _stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1
    )


_TOKEN = re.compile(r"[A-Za-z0-9]+")


def names(text: str) -> frozenset[str]:
    """Capitalized words and numbers: the parts that make a fact this fact."""
    out = set()
    for tok in _TOKEN.findall(text):
        low = tok.lower()
        if low in _STOP or len(tok) < 2 and not tok.isdigit():
            continue
        if tok[0].isupper() or tok.isdigit():
            out.add(_stem(low))
    return frozenset(out)


def similarity(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard overlap of two term sets; 0 when either is empty."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def threshold_from_env() -> float:
    try:
        return float(os.environ.get("ZEP_MCP_DIVERSE_THRESHOLD", DEFAULT_THRESHOLD))
    except ValueError:
        return DEFAULT_THRESHOLD


def fetch_size(limit: int, cap: int = 50) -> int:
    """How many results to ask Zep for, to return `limit` distinct ones."""
    try:
        factor = int(os.environ.get("ZEP_MCP_DIVERSE_FETCH", DEFAULT_FETCH_FACTOR))
    except ValueError:
        factor = DEFAULT_FETCH_FACTOR
    if threshold_from_env() >= 1:
        factor = 1
    return max(1, min(cap, limit * max(1, factor)))


def pick_diverse(
    items: Iterable[T],
    limit: int,
    text: Callable[[T], str],
    threshold: float | None = None,
) -> list[T]:
    """The first `limit` items, in order, none too like one already picked."""
    th = threshold_from_env() if threshold is None else threshold
    picked: list[tuple[T, frozenset[str], frozenset[str]]] = []
    for item in items:
        s = text(item)
        t, n = terms(s), names(s)
        if th < 1 and any(
            similarity(t, pt) >= th and n <= pn for _, pt, pn in picked
        ):
            continue
        picked.append((item, t, n))
        if len(picked) >= limit:
            break
    return [item for item, _, _ in picked]
