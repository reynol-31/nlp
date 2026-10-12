import json
import shutil

import pytest

from src.dataset_loader import (
    ANNOTATION_FILE,
    GOLD_FILE,
    DatasetError,
    build_gold_records,
    dataset_summary,
    filter_split,
    item_from_record,
    load_gold,
    load_gold_records,
    parse_annotation_file,
    parse_annotation_line,
    split_for_group,
)
from src.normalization import normalize_word
from src.resources import DATA_DIR, ResourceError, load_resources_from

VALID = {
    "id": "X1", "word": "लड़कों", "split": "test", "category": "noun_regular",
    "analyses": [{"lemma": "लड़का", "pos": "NOUN", "gender": "MASC", "number": "PL", "case": "OBL"}],
}


@pytest.fixture(scope="module")
def gold():
    return load_gold()


def test_gold_loads_and_is_well_formed(gold):
    assert 100 <= len(gold) <= 400
    assert len({i.id for i in gold}) == len(gold)
    assert {i.split for i in gold} == {"dev", "test"}
    assert all(i.analyses for i in gold)
    assert all(i.source == "manual" for i in gold)


def test_no_lemma_group_in_both_splits(gold):
    summary = dataset_summary(gold)
    assert summary["group_overlap_between_splits"] == []
    dev_lemmas = {l for i in filter_split(gold, "dev") for l in i.gold_lemmas}
    test_lemmas = {l for i in filter_split(gold, "test") for l in i.gold_lemmas}
    assert not dev_lemmas & test_lemmas


def test_gold_json_in_sync_with_annotation_source():
    rebuilt = build_gold_records(parse_annotation_file(ANNOTATION_FILE))
    stored = json.loads(GOLD_FILE.read_text(encoding="utf-8"))["items"]
    assert rebuilt == stored, "run: python scripts/build_gold.py"


def test_dataset_lexicon_consistency(gold, resources):
    """In-scope categories use only lexicon lemmas; OOV items use only lemmas absent from it."""
    known = resources.known_lemmas()
    for it in gold:
        lemmas = it.gold_lemmas
        if it.category == "oov":
            assert not lemmas & known, it.word
        elif it.category not in ("out_of_scope", "spelling_variant"):
            assert lemmas <= known, it.word


def test_split_is_deterministic():
    assert split_for_group("लड़का") == split_for_group("लड़का")
    assert split_for_group("x", seed="a") in ("dev", "test")


def test_filter_split_rejects_unknown():
    with pytest.raises(DatasetError):
        filter_split([], "train")


# -- malformed input ----------------------------------------------------------
@pytest.mark.parametrize("field", ["id", "word", "split", "category", "analyses"])
def test_missing_field_raises(field):
    rec = dict(VALID)
    del rec[field]
    with pytest.raises(DatasetError, match="missing"):
        item_from_record(rec)


@pytest.mark.parametrize(
    "patch",
    [
        {"split": "train"},
        {"category": "nonsense"},
        {"word": "ladkon"},
        {"analyses": [{"lemma": "लड़का", "pos": "NOUN", "case": "ABLATIVE"}]},
        {"analyses": [{"lemma": "लड़का"}]},
        {"analyses": "लड़का NOUN"},
    ],
)
def test_invalid_values_raise(patch):
    with pytest.raises(DatasetError):
        item_from_record({**VALID, **patch})


def test_duplicate_ids_raise():
    with pytest.raises(DatasetError, match="duplicate"):
        load_gold_records([VALID, VALID])


def test_missing_and_malformed_files(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        load_gold(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(DatasetError, match="not valid JSON"):
        load_gold(bad)
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"items": {"a": 1}}), encoding="utf-8")
    with pytest.raises(DatasetError):
        load_gold(wrong)


def test_annotation_line_parsing():
    rec = parse_annotation_line("लड़के :: noun_regular :: लड़का NOUN MASC SG OBL | लड़का NOUN MASC PL DIR :: note")
    assert rec["word"] == "लड़के" and len(rec["analyses"]) == 2 and rec["note"] == "note"
    assert parse_annotation_line("# comment") is None
    assert parse_annotation_line("   ") is None
    for bad in [
        "लड़के :: noun_regular",
        "लड़के :: verbish :: लड़का NOUN",
        "लड़के :: noun_regular :: लड़का BOGUS",
        "लड़के :: noun_regular :: लड़का NOUN SG | लड़का NOUN SG",
        "लड़के :: noun_regular :: लड़का",
    ]:
        with pytest.raises(DatasetError):
            parse_annotation_line(bad)


def test_resource_validation_catches_bad_lexicon(tmp_path):
    shutil.copy(DATA_DIR / "morphology_rules.json", tmp_path / "morphology_rules.json")
    lex = json.loads((DATA_DIR / "lexicon.json").read_text(encoding="utf-8"))
    lex["entries"].append({"lemma": "किताब", "paradigm": "N_M_AA"})  # does not end in -आ
    (tmp_path / "lexicon.json").write_text(json.dumps(lex, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ResourceError):
        load_resources_from(tmp_path)
    lex["entries"][-1] = {"lemma": "किताब", "paradigm": "NO_SUCH_PARADIGM"}
    (tmp_path / "lexicon.json").write_text(json.dumps(lex, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ResourceError, match="unknown paradigm"):
        load_resources_from(tmp_path)
    (tmp_path / "lexicon.json").unlink()
    with pytest.raises(ResourceError, match="not found"):
        load_resources_from(tmp_path)


def test_lemma_display_preserved(gold):
    assert any(a.lemma == "दरवाज़ा" for it in gold for a in it.analyses)
    assert normalize_word("दरवाज़ा") == normalize_word("दरवाजा")
