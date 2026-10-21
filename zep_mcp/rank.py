"""Ranking primitives: maximal marginal relevance and ACT-R base-level activation.

Pure functions over numbers, no I/O, so they can be tested on fixed vectors.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def mmr(relevance: Sequence[float], vectors: np.ndarray, k: int, lam: float) -> list[int]:
    """Pick k candidates by maximal marginal relevance (Carbonell & Goldstein 1998).

    Each step takes the candidate maximising

        lam * relevance[i]  -  (1 - lam) * max_{j already picked} cos(v_i, v_j)

    so the first pick is the most relevant, and a later candidate pays for its
    likeness to whatever was already picked. lam = 1 is pure relevance order;
    lam = 0 is pure novelty. Ties go to the earlier candidate, so for
    candidates already in relevance order the result is stable.

    `vectors` are rows aligned with `relevance`; they are normalised here, so
    the similarity term is cosine whatever their length. Returns indices into
    the candidates, in pick order.
    """
    v = np.asarray(vectors, dtype=np.float64)
    if v.shape[0] != len(relevance):
        raise ValueError(f"{len(relevance)} relevance scores but {v.shape[0]} vectors")
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    return mmr_sim(relevance, v @ v.T, k, lam)


def mmr_sim(relevance: Sequence[float], sim: np.ndarray, k: int, lam: float) -> list[int]:
    """mmr() over a precomputed similarity matrix, sim[i, j] = cos(v_i, v_j).

    The store computes the candidates' pairwise cosines in SQL, so it never
    has to fetch their vectors.
    """
    n = len(relevance)
    sim = np.asarray(sim, dtype=np.float64)
    if sim.shape != (n, n):
        raise ValueError(f"{n} relevance scores but a {sim.shape} similarity matrix")
    k = max(0, min(k, n))
    if k == 0:
        return []
    rel = np.asarray(relevance, dtype=np.float64)
    picked: list[int] = []
    # Largest similarity of each candidate to anything picked so far; the
    # first pick has nothing to be like, so no penalty.
    max_sim = np.full(n, -np.inf)
    free = np.ones(n, dtype=bool)
    for _ in range(k):
        penalty = np.where(np.isfinite(max_sim), max_sim, 0.0)
        score = lam * rel - (1.0 - lam) * penalty
        score[~free] = -np.inf
        i = int(np.argmax(score))  # argmax takes the first of equal maxima
        picked.append(i)
        free[i] = False
        max_sim = np.maximum(max_sim, sim[i])
    return picked


def base_level_activation(ages_seconds: Sequence[float], d: float = 0.5) -> float | None:
    """ACT-R base-level activation B = ln(sum_j t_j^-d) (Anderson & Schooler 1991).

    t_j is the time since use j, in seconds (ACT-R's unit; with a power law a
    different unit only adds the constant d*ln(unit) to every memory, so the
    ranking it implies does not depend on it). Ages are floored at one second
    so a use logged "now" is finite. None for a memory with no uses.
    """
    if not ages_seconds:
        return None
    return math.log(sum(max(float(t), 1.0) ** -d for t in ages_seconds))
