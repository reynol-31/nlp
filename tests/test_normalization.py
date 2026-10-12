import pytest

from src.dataset_loader import load_gold
from src.normalization import is_devanagari_word, normalize_word, split_words

ZWJ, ZWNJ, NBSP = chr(0x200D), chr(0x200C), chr(0x00A0)
PRECOMPOSED_ZA = chr(0x095B)  # ज़ as a single code point
NUKTA = chr(0x093C)


def test_nfc_precomposed_nukta_equals_decomposed():
    a = "दरवा" + PRECOMPOSED_ZA + "ा"
    b = "दरवा" + "ज" + NUKTA + "ा"
    assert normalize_word(a, fold_nukta=False) == normalize_word(b, fold_nukta=False)
    assert PRECOMPOSED_ZA not in normalize_word(a, fold_nukta=False)


def test_invisible_characters_removed():
    assert normalize_word("ल" + ZWJ + "ड़का") == normalize_word("लड़का")
    assert normalize_word("किता" + ZWNJ + "ब") == "किताब"


def test_whitespace_and_punctuation_stripped():
    assert normalize_word("  लड़कों।  ") == normalize_word("लड़कों")
    assert normalize_word(NBSP + "किताब?") == "किताब"
    assert normalize_word("") == ""
    assert normalize_word(None) == ""


def test_nasal_folding():
    assert normalize_word("लड़कियाँ") == normalize_word("लड़कियां")
    assert "ँ" in normalize_word("लड़कियाँ", fold_nasal=False)


def test_perso_arabic_nukta_folded_but_not_retroflex():
    assert normalize_word("ख़राब") == normalize_word("खराब")
    assert NUKTA in normalize_word("लड़का")  # ड़ is a distinct letter and is kept
    assert normalize_word("लड़का") != normalize_word("लडका")


@pytest.mark.parametrize(
    "variant, canonical",
    [("नये", "नए"), ("नयी", "नई"), ("गये", "गए"), ("खायी", "खाई"), ("पिये", "पिए"), ("खायेंगे", "खाएंगे")],
)
def test_y_glide_folding(variant, canonical):
    assert normalize_word(variant) == normalize_word(canonical)


@pytest.mark.parametrize("word", ["क्यों", "ये", "गाय", "यह"])
def test_y_glide_not_applied_where_invalid(word):
    assert normalize_word(word) == word


def test_is_devanagari_word():
    assert is_devanagari_word("लड़का")
    assert not is_devanagari_word("ladka")
    assert not is_devanagari_word("लड़का1")
    assert not is_devanagari_word("")


def test_split_words():
    assert split_words(" लड़के  किताबें पढ़ते हैं। ") == ["लड़के", "किताबें", "पढ़ते", "हैं"]


def test_normalization_is_idempotent_on_dataset():
    for item in load_gold():
        once = normalize_word(item.word)
        assert normalize_word(once) == once
