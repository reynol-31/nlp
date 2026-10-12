import copy

import pytest

from src.fst_analyzer import EPS, FST, FSTAnalyzer
from src.normalization import normalize_word
from src.rule_based_analyzer import RuleBasedAnalyzer
from tests.conftest import shorts


def test_hand_built_fst_transduces():
    """A 4-state toy transducer: 'ab' -> 'X+Y' (the last arc is an ε-input arc)."""
    t = FST()
    s0, s1, s2, s3 = (t.add_state(n) for n in ("start", "mid", "pre-final", "final"))
    t.start = s0
    t.finals.add(s3)
    t.add_arc(s0, "a", "X", s1, "stem")
    t.add_arc(s1, "b", EPS, s2, "suffix")
    t.add_arc(s2, EPS, "+Y", s3, "tags")
    paths = t.transduce("ab")
    assert [p.output for p in paths] == ["X+Y"]
    assert paths[0].states == [s0, s1, s2, s3]
    assert t.transduce("a") == [] and t.transduce("abc") == []
    assert t.generate("X+Y") == ["ab"]


def test_input_symbols_are_single_code_points():
    t = FST()
    s = t.add_state()
    with pytest.raises(ValueError):
        t.add_arc(s, "ab", "", s, "stem")


def test_compiled_fst_structure(fst):
    stats = fst.stats()
    assert stats["states"] > 100 and stats["arcs"] > stats["states"] - 1
    assert stats["accepting_states"] == 1
    assert fst.fst.is_acyclic()


def test_analysis_follows_real_transitions(fst):
    """Every arc on the accepting path exists in the transition table and the tapes line up."""
    word = normalize_word("लड़कों")
    res = fst.analyze(word)
    assert shorts(res) == {"लड़का NOUN MASC PL OBL"}
    arcs = res.trace[0]["arcs"]
    assert arcs[0].src == fst.fst.start
    assert arcs[-1].dst in fst.fst.finals
    for prev, nxt in zip(arcs, arcs[1:]):
        assert prev.dst == nxt.src  # contiguous path
    for a in arcs:
        assert a in fst.fst.arcs_from(a.src, a.inp)  # arc really is in the graph
    assert "".join(a.inp for a in arcs) == word  # input tape = surface word
    assert "".join(a.out for a in arcs) == "लड़का+NOUN+MASC+PL+OBL"  # output tape = lexical form
    kinds = [a.kind for a in arcs]
    assert {"stem", "link", "suffix", "restore", "tags"} <= set(kinds)


def test_removing_a_transition_removes_the_analysis(resources):
    """Proof that output is produced by traversal: delete one arc and the analysis disappears."""
    analyzer = FSTAnalyzer(resources)
    assert analyzer.analyze("लड़कों").known
    broken = copy.deepcopy(analyzer)
    table = broken.fst._arcs
    removed = 0
    for src, by_input in table.items():
        for sym, arcs in by_input.items():
            keep = [a for a in arcs if not (a.kind == "tags" and a.info == "N_M_AA:PL_OBL")]
            removed += len(arcs) - len(keep)
            by_input[sym] = keep
    assert removed >= 1
    assert not broken.analyze("लड़कों").known
    assert broken.analyze("लड़के").known  # other paths untouched


def test_rejected_path_reports_where_it_failed(fst):
    res = fst.analyze("लड़कोंा")
    assert not res.known
    assert "Rejected" in res.message
    assert res.trace and res.trace[0].get("rejected")
    consumed, _ = fst.fst.longest_prefix(normalize_word("लड़कोंा"))
    assert consumed == len(normalize_word("लड़कों"))


def test_unknown_lemma_rejected(fst):
    assert fst.analyze("मकानों").status == "UNKNOWN"
    assert fst.analyze("xyz").status == "UNKNOWN"


def test_irregular_paths(fst):
    assert shorts(fst.analyze("किया")) == {"करना VERB PART PERF MASC SG"}
    assert fst.analyze("किया").primary.source == "fst-irregular"
    assert not fst.analyze("करा").known  # blocked slot was never compiled


def test_ambiguity_gives_multiple_paths(fst):
    assert shorts(fst.analyze("खाने")) == {
        "खाना NOUN MASC SG OBL", "खाना NOUN MASC PL DIR", "खाना VERB INF OBL",
    }


@pytest.mark.parametrize(
    "lexical, surface",
    [
        ("लड़का+NOUN+MASC+PL+OBL", "लड़कों"),
        ("पीना+VERB+FIN+FUT+IND+3+MASC+SG", "पिएगा"),
        ("देना+VERB+PART+PERF+MASC+SG", "दिया"),
        ("करना+VERB+PART+PERF+MASC+SG", "किया"),
    ],
)
def test_generation_inverse_direction(fst, lexical, surface):
    assert normalize_word(surface) in fst.generate(lexical)


def test_generation_rejects_invalid_lexical_string(fst):
    assert fst.generate("लड़का+NOUN+FEM+PL+OBL") == []
    assert fst.generate("not a lexical string") == []


def test_fst_and_rules_are_extensionally_equivalent(resources, fst):
    """With identical resources and no guesser, both analyzers return the same analysis sets
    for every form the paradigms can produce (fair-comparison check)."""
    rb = RuleBasedAnalyzer(resources, enable_guesser=False)
    forms = {resources.stem_of(e) + r.surface for e in resources.entries
             for r in resources.paradigms[e.paradigm].rules if resources.stem_of(e) + r.surface}
    forms |= {f.form for f in resources.irregular}
    for w in sorted(forms):
        a = {x.key() for x in rb.analyze(w).analyses}
        b = {x.key() for x in fst.analyze(w).analyses}
        assert a == b, w
