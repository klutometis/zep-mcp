import math

import numpy as np
import pytest

from zep_mcp.rank import base_level_activation, mmr, mmr_sim

# Fixed vectors: A and A2 are rewordings of one fact (cosine 0.995), B and C
# are distinct facts, orthogonal to A and to each other.
A = [1.0, 0.0, 0.0, 0.0]
A2 = [0.995, 0.0999, 0.0, 0.0]
B = [0.0, 0.0, 1.0, 0.0]
C = [0.0, 0.0, 0.0, 1.0]


def test_duplicate_is_skipped():
    rel = [0.90, 0.88, 0.70, 0.50]  # the rewording is the second most relevant
    vecs = np.array([A, A2, B, C])
    assert mmr(rel, vecs, 3, lam=0.5) == [0, 2, 3]


def test_duplicate_comes_last_when_nothing_else_is_left():
    rel = [0.90, 0.88, 0.70]
    vecs = np.array([A, A2, B])
    assert mmr(rel, vecs, 3, lam=0.5) == [0, 2, 1]


def test_relevance_order_kept_for_distinct_facts():
    rel = [0.2, 0.9, 0.5, 0.7]
    vecs = np.eye(4)  # all orthogonal: no redundancy to penalise
    for lam in (0.3, 0.5, 0.7, 0.9):
        assert mmr(rel, vecs, 4, lam) == [1, 3, 2, 0]


def test_lambda_one_is_pure_relevance():
    rel = [0.90, 0.88, 0.70, 0.50]
    vecs = np.array([A, A2, B, C])
    assert mmr(rel, vecs, 4, lam=1.0) == [0, 1, 2, 3]


def test_lambda_zero_is_pure_novelty_after_the_first():
    # The first pick has nothing to differ from, so it is the most relevant.
    rel = [0.90, 0.88, 0.10]
    vecs = np.array([A, A2, B])
    assert mmr(rel, vecs, 2, lam=0.0)[1] == 2


def test_ties_keep_input_order():
    assert mmr([0.5, 0.5, 0.5], np.eye(3), 3, lam=0.7) == [0, 1, 2]


def test_vectors_need_not_be_normalised():
    rel = [0.90, 0.88, 0.70]
    vecs = np.array([A, [10 * x for x in A2], B])
    assert mmr(rel, vecs, 2, lam=0.5) == [0, 2]


def test_k_bounds():
    assert mmr([0.3, 0.2], np.eye(2), 0, 0.5) == []
    assert mmr([0.3, 0.2], np.eye(2), 5, 0.5) == [0, 1]
    assert mmr([], np.zeros((0, 4)), 5, 0.5) == []


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mmr([0.3, 0.2], np.eye(3), 1, 0.5)
    with pytest.raises(ValueError):
        mmr_sim([0.3, 0.2], np.eye(3), 1, 0.5)


def test_mmr_sim_matches_mmr_on_vectors():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(30, 16))
    v[7] = v[3] + 0.01 * rng.normal(size=16)  # a near-duplicate pair
    rel = rng.uniform(size=30)
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    for lam in (0.0, 0.3, 0.5, 0.8, 1.0):
        assert mmr(rel, v, 10, lam) == mmr_sim(rel, u @ u.T, 10, lam)


def test_activation_single_use():
    assert base_level_activation([100.0]) == pytest.approx(math.log(100**-0.5))


def test_activation_more_and_recent_uses_are_stronger():
    one = base_level_activation([86400.0])
    two = base_level_activation([86400.0, 3600.0])
    old = base_level_activation([30 * 86400.0])
    assert two > one > old


def test_activation_floor_and_empty():
    assert base_level_activation([0.0]) == pytest.approx(0.0)  # floored at 1 s
    assert base_level_activation([]) is None


def test_store_select_feeds_mmr_in_relevance_order():
    # select() reorders candidates (which arrive in cosine order) by rerank
    # relevance before MMR, and maps the picks back to candidate indices.
    from zep_mcp.store import Candidates, Store

    c = Candidates(
        query="q",
        ids=[10, 11, 12, 13],
        texts=["b", "a2", "a", "c"],
        created_at=[None] * 4,
        valid_from=[None] * 4,
        zep=[None] * 4,
        sim=(lambda v: v @ v.T)(np.array([B, A2, A, C]) / np.linalg.norm(np.array([B, A2, A, C]), axis=1, keepdims=True)),
        cosine=[0.9, 0.8, 0.7, 0.6],
        activation=[None] * 4,
        relevance=[0.70, 0.88, 0.90, 0.50],
    )
    s = Store("postgresql://unused", lam=0.5)  # no connection until used
    assert s.select(c, 3) == [2, 0, 3]  # a, b, c: the rewording a2 is skipped
    assert s.select(c, 3, lam=1.0) == [2, 1, 0]  # pure relevance: a, a2, b
