"""zep_mcp.diverse: repeats go, order stays, distinct facts survive.

Facts are real ones from the danenberg graph (2026-10-20).
"""

from zep_mcp.diverse import fetch_size, names, pick_diverse, similarity, terms

DIDO = [
    "Cupid and Venus conspired to make Dido fall in love with Aeneas, intending to torment Dido.",
    "Venus conspired with Cupid to make Dido fall in love with Aeneas.",
    "Virgil gives Dido Bacchic frenzy at 4.300-303, but the poem attributes its source to Venus and Cupid's intervention.",
    "Cupid conspired with Venus to make Dido fall in love with Aeneas.",
    "Venus and Cupid conspired to make Dido fall in love with Aeneas for the protection of the Trojans and to torment Dido.",
    "The user is interested in comparing Virgil's divinely induced love to the wishing-charm mechanism in the 2026 film Obsession.",
    "Amata in Aeneid Book 7 follows the same pattern, with Allecto inducing a Bacchic frenzy at Juno's instigation.",
]


def sim(a: str, b: str) -> float:
    return similarity(terms(a), terms(b))


def test_paraphrases_are_alike():
    assert sim(DIDO[1], DIDO[3]) == 1.0
    assert sim(DIDO[0], DIDO[1]) >= 0.6
    assert sim(
        "The toga virilis ceremony marked a young Roman's transition to legal adulthood.",
        "A young Roman undergoes transition to legal adulthood via the toga virilis ceremony.",
    ) >= 0.6


def test_shared_subject_is_not_alike():
    assert sim("The user prefers tea.", "The user prefers coffee.") < 0.6
    assert sim("Peter lives in San Francisco.", "Peter moved to San Francisco in 2020.") < 0.6
    assert sim(DIDO[0], DIDO[2]) < 0.6


def test_pick_keeps_order_and_drops_repeats():
    picked = pick_diverse(DIDO, 5, text=str, threshold=0.6)
    # 1 and 3 reword 0. 4 rewords it too but adds the Trojans, so it stays.
    assert picked == [DIDO[0], DIDO[2], DIDO[4], DIDO[5], DIDO[6]]


def test_same_template_different_name_is_not_a_repeat():
    facts = [
        "The User's spaced repetition flashcards cover Aeneas.",
        "The User's spaced repetition flashcards cover Dido.",
        "The 22 spaced repetition flashcards cover LLM concepts.",
        "The 9 spaced repetition flashcards cover LLM concepts.",
    ]
    assert pick_diverse(facts, 5, text=str, threshold=0.6) == facts


def test_names():
    assert names("The user's flashcards cover Dido.") == {"dido"}
    assert names("Peter has 22 cards in the Western Canon deck.") == {"22", "western", "canon"}


def test_pick_stops_at_limit():
    assert pick_diverse(DIDO, 2, text=str, threshold=0.6) == [DIDO[0], DIDO[2]]
    assert pick_diverse(DIDO, 3, text=str, threshold=0.6) == [DIDO[0], DIDO[2], DIDO[4]]


def test_threshold_one_turns_it_off(monkeypatch):
    assert pick_diverse(DIDO, 5, text=str, threshold=1.0) == DIDO[:5]
    monkeypatch.setenv("ZEP_MCP_DIVERSE_THRESHOLD", "1")
    assert fetch_size(5) == 5


def test_identical_text_is_a_repeat():
    assert pick_diverse(["A fact.", "A fact.", "Another one."], 5, text=str) == ["A fact.", "Another one."]


def test_fetch_size(monkeypatch):
    monkeypatch.delenv("ZEP_MCP_DIVERSE_THRESHOLD", raising=False)
    monkeypatch.delenv("ZEP_MCP_DIVERSE_FETCH", raising=False)
    assert fetch_size(5) == 20
    assert fetch_size(20) == 50  # Zep's page cap
    monkeypatch.setenv("ZEP_MCP_DIVERSE_FETCH", "2")
    assert fetch_size(5) == 10
