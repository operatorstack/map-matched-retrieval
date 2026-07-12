from mapmatched.eval.corpus import build_section_graph
from mapmatched.eval.types import Passage


def test_same_group_is_adjacent_and_cross_group_clamps() -> None:
    passages = [
        Passage("a1", "text a1", group_key="ArticleA"),
        Passage("a2", "text a2", group_key="ArticleA"),
        Passage("b1", "text b1", group_key="ArticleB"),
        Passage("c1", "text c1"),  # no group -> isolated
    ]
    graph = build_section_graph(passages)
    # same article -> connected (distance 1 via the anchor edge)
    assert graph.distance("a1", "a2") == 1.0
    # different articles -> unreachable -> clamped to maximum_distance
    assert graph.distance("a1", "b1") == graph.maximum_distance
    # ungrouped passage is isolated
    assert graph.distance("a1", "c1") == graph.maximum_distance
    # identity is zero
    assert graph.distance("a1", "a1") == 0.0


def test_empty_groups_yield_all_isolated() -> None:
    passages = [Passage("x", "tx"), Passage("y", "ty")]
    graph = build_section_graph(passages)
    assert graph.distance("x", "y") == graph.maximum_distance
