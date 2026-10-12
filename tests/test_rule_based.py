import pytest

from src.schema import FEATURE_VALUES, FEATURES, NA, AnalyzerResult
from tests.conftest import shorts


@pytest.mark.parametrize(
    "word, expected",
    [
        ("लड़कों", {"लड़का NOUN MASC PL OBL"}),
        ("किताबों", {"किताब NOUN FEM PL OBL"}),
        ("घरों", {"घर NOUN MASC PL OBL"}),
        ("किताबें", {"किताब NOUN FEM PL DIR"}),
        ("भाषाएँ", {"भाषा NOUN FEM PL DIR"}),
    ],
)
def test_regular_nouns(rb, word, expected):
    assert shorts(rb.analyze(word)) == expected


def test_multiple_candidates_for_ambiguous_form(rb):
    res = rb.analyze("लड़के")
    assert shorts(res) == {"लड़का NOUN MASC SG OBL", "लड़का NOUN MASC PL DIR"}
    assert len(res.analyses) == 2


def test_lexical_ambiguity_two_lemmas(rb):
    assert shorts(rb.analyze("बना")) == {"बनना VERB PART PERF MASC SG", "बनाना VERB FIN IMP 2 SG"}


@pytest.mark.parametrize(
    "word, expected",
    [
        ("आदमियों", {"आदमी NOUN MASC PL OBL"}),  # -ई -> -इयों
        ("लड़कियाँ", {"लड़की NOUN FEM PL DIR"}),  # -ई -> -इयाँ
        ("आलुओं", {"आलू NOUN MASC PL OBL"}),  # -ऊ -> -उओं
        ("बहुएँ", {"बहू NOUN FEM PL DIR"}),
        ("चिड़ियाँ", {"चिड़िया NOUN FEM PL DIR"}),
        ("राजाओं", {"राजा NOUN MASC PL OBL"}),  # invariant -आ class
    ],
)
def test_stem_changing_and_exception_classes(rb, word, expected):
    assert shorts(rb.analyze(word)) == expected


def test_invariant_class_does_not_take_regular_suffix(rb):
    assert not rb.analyze("राजे").known  # राजा is not *राजे
    assert not rb.analyze("राजों").known


@pytest.mark.parametrize(
    "word, expected",
    [
        ("खाएगा", {"खाना VERB FIN FUT IND 2 MASC SG", "खाना VERB FIN FUT IND 3 MASC SG"}),
        ("पढ़ेंगे", {f"पढ़ना VERB FIN FUT IND {p} MASC PL" for p in "123"}),
        ("पढ़ती", {"पढ़ना VERB PART HAB FEM SG", "पढ़ना VERB PART HAB FEM PL"}),
        ("पिया", {"पीना VERB PART PERF MASC SG"}),
        ("दीजिए", {"देना VERB FIN IMP 2 PL"}),
        ("खाया", {"खाना VERB PART PERF MASC SG"}),
    ],
)
def test_verb_forms(rb, word, expected):
    assert shorts(rb.analyze(word)) == expected


@pytest.mark.parametrize(
    "word, expected",
    [
        ("किया", {"करना VERB PART PERF MASC SG"}),
        ("गई", {"जाना VERB PART PERF FEM SG"}),
        ("मुझे", {"मैं PRON 1 SG ACC_DAT"}),
        ("था", {"होना AUX FIN PAST IND MASC SG"}),
    ],
)
def test_irregular_forms(rb, word, expected):
    res = rb.analyze(word)
    assert shorts(res) == expected
    assert res.primary.source == "exception"


def test_blocked_slot_is_not_generated(rb, rb_guess):
    assert not rb.analyze("करा").known  # करना has the suppletive perfective किया
    guessed = rb_guess.analyze("करा")
    assert all(a.lemma != "करना" for a in guessed.analyses)  # lexicon stays authoritative


def test_mass_noun_restriction(rb):
    s = shorts(rb.analyze("सोने"))
    assert "सोना NOUN MASC SG OBL" in s
    assert "सोना NOUN MASC PL DIR" not in s
    assert "सोना VERB INF OBL" in s


def test_unknown_word_without_guesser(rb):
    res = rb.analyze("मकानों")
    assert res.status == "UNKNOWN" and res.analyses == []
    assert "Unknown" in res.message


def test_guesser_marks_guesses(rb_guess):
    res = rb_guess.analyze("मकानों")
    assert res.status == "GUESSED"
    assert res.primary.source == "guess"
    assert res.primary.short() == "मकान NOUN MASC PL OBL"


def test_known_word_never_guessed(rb_guess):
    assert rb_guess.analyze("लड़कों").status == "KNOWN"


@pytest.mark.parametrize("bad", ["", "   ", "ladke", "123"])
def test_invalid_input_returns_unknown(rb_guess, bad):
    res = rb_guess.analyze(bad)
    assert isinstance(res, AnalyzerResult) and not res.known


def test_trace_shows_decomposition(rb):
    trace = rb.analyze("लड़कों").trace
    accepted = [t for t in trace if t["outcome"].startswith("accepted")]
    assert accepted and accepted[0]["stem"] == "लड़क" and accepted[0]["suffix"] == "ों"


def test_canonical_output(rb_guess):
    for w in ["लड़के", "पढ़ेगा", "मुझे", "सुंदर", "मकानों"]:
        for a in rb_guess.analyze(w).analyses:
            d = a.to_dict()
            assert set(FEATURES) <= set(d)
            for f in FEATURES:
                assert d[f] == NA or d[f] in FEATURE_VALUES[f]
