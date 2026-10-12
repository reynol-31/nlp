import json

import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from src.dataset_loader import GoldItem, load_gold
from src.error_analysis import AGREEMENT_CATEGORIES, agreement_counts, filter_comparison, insights
from src.evaluation import (
    FST_NAME,
    RULE,
    best_gold_match,
    compute_metrics,
    export_reports,
    feature_pairs,
    prf,
    run_evaluation,
    score_item,
    set_metrics,
    top1_metrics,
)
from src.schema import Analysis, AnalyzerResult


def A(s: str, **meta) -> Analysis:
    lemma, *tags = s.split()
    return Analysis.from_tags(lemma, tags, **meta)


def item(word, *gold, cat="noun_regular"):
    return GoldItem(id="T", word=word, split="test", category=cat, analyses=tuple(A(g) for g in gold))


def result(word, *preds, guessed=False):
    return AnalyzerResult("x", word, word, [A(p) for p in preds], guessed=guessed)


# -- primitives -------------------------------------------------------------
def test_top1_metrics_hand_computed():
    m = top1_metrics([True, False, False, True], [True, True, False, True])
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))
    assert (m["correct"], m["answered"], m["unknown"], m["errors"], m["total"]) == (2, 3, 1, 2, 4)


def test_top1_metrics_empty_and_undefined():
    m = top1_metrics([], [])
    assert m["accuracy"] == m["precision"] == m["recall"] == m["f1"] == 0.0
    assert not m["accuracy_defined"] and not m["precision_defined"]
    m = top1_metrics([False, False], [False, False])  # analyzer abstains everywhere
    assert m["precision"] == 0.0 and not m["precision_defined"]


def test_top1_metrics_rejects_inconsistent_input():
    with pytest.raises(ValueError):
        top1_metrics([True], [False])
    with pytest.raises(ValueError):
        top1_metrics([True], [True, True])


def test_prf_zero_division():
    m = prf(0, 0, 0)
    assert m["precision"] == m["recall"] == m["f1"] == 0.0
    assert not m["precision_defined"] and not m["recall_defined"]
    m = prf(3, 1, 2)
    assert m["precision"] == pytest.approx(0.75) and m["recall"] == pytest.approx(0.6)


def test_set_metrics():
    m = set_metrics([{1, 2}, set(), {5}], [{1}, {3}, {5, 6}])
    assert (m["tp"], m["n_pred"], m["n_gold"]) == (2, 3, 4)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(0.5)


# -- matching ---------------------------------------------------------------
def test_alternative_valid_analysis_counts_as_correct():
    it = item("लड़के", "लड़का NOUN MASC SG OBL", "लड़का NOUN MASC PL DIR")
    r = score_item(it, result("लड़के", "लड़का NOUN MASC PL DIR"))
    assert r.top1_correct and r.error_type == "CORRECT"
    assert r.best_gold.short() == "लड़का NOUN MASC PL DIR"


def test_lemma_match_without_exact_match():
    it = item("लड़कों", "लड़का NOUN MASC PL OBL")
    r = score_item(it, result("लड़कों", "लड़का NOUN MASC SG OBL"))
    assert not r.top1_correct and r.lemma_correct
    assert r.error_type == "WRONG_FEATURES" and r.mismatched == ["number"]


def test_wrong_lemma_and_wrong_pos_classification():
    it = item("पत्तों", "पत्ता NOUN MASC PL OBL", cat="oov")
    assert score_item(it, result("पत्तों", "पत्त NOUN MASC PL OBL")).error_type == "WRONG_LEMMA"
    it2 = item("झंडे", "झंडा NOUN MASC PL DIR", cat="oov")
    assert score_item(it2, result("झंडे", "झंडना VERB PART PERF MASC PL")).error_type == "WRONG_POS"


def test_unknown_prediction_is_counted_not_dropped():
    items = [item("घरों", "घर NOUN MASC PL OBL"), item("मकानों", "मकान NOUN MASC PL OBL", cat="oov")]
    results = [score_item(items[0], result("घरों", "घर NOUN MASC PL OBL")), score_item(items[1], result("मकानों"))]
    assert results[1].error_type == "UNKNOWN"
    m = compute_metrics(results)
    assert m["exact_top1"]["total"] == 2 and m["exact_top1"]["unknown"] == 1
    assert m["exact_top1"]["accuracy"] == pytest.approx(0.5)
    assert m["exact_top1"]["precision"] == pytest.approx(1.0)
    assert m["feature_micro"]["fn"] >= 5  # every gold pair of the unknown item is a false negative


def test_feature_pairs_and_na_handling():
    assert feature_pairs(A("सुंदर ADJ")) == {("lemma", "सुंदर"), ("pos", "ADJ")}
    it = item("अच्छे", "अच्छा ADJ MASC SG OBL")
    r = score_item(it, result("अच्छे", "अच्छा ADJ"))  # predicted NA where gold has values -> FN
    assert r.feat_counts["gender"] == [0, 0, 1] and r.feat_counts["case"] == [0, 0, 1]
    r2 = score_item(item("लाल", "लाल ADJ"), result("लाल", "लाल ADJ MASC SG DIR"))  # value where gold NA -> FP
    assert r2.feat_counts["gender"] == [0, 1, 0]
    assert r2.feat_counts["tense"] == [0, 0, 0]  # NA/NA not counted


def test_best_gold_match_prefers_exact_then_overlap():
    gold = (A("घर NOUN MASC SG DIR"), A("घर NOUN MASC PL DIR"))
    assert best_gold_match(A("घर NOUN MASC PL DIR"), gold).number == "PL"
    assert best_gold_match(A("घर NOUN MASC PL OBL"), gold).number == "PL"
    assert best_gold_match(None, gold) is gold[0]


# -- full pipeline ----------------------------------------------------------
@pytest.fixture(scope="module")
def report():
    return run_evaluation("test", items=load_gold())


def test_exact_accuracy_matches_sklearn(report):
    for name in (RULE, FST_NAME):
        res = report.results[name]
        y_true = [r.best_gold.key() for r in res]
        y_pred = [r.top1.key() if r.top1 else ("UNKNOWN",) for r in res]
        sk = accuracy_score([str(x) for x in y_true], [str(x) for x in y_pred])
        assert report.metrics[name]["exact_top1"]["accuracy"] == pytest.approx(sk)


def test_metric_counts_are_consistent(report):
    for m in report.metrics.values():
        e = m["exact_top1"]
        assert e["total"] == len(report.items)
        assert e["correct"] + e["errors"] == e["total"]
        assert sum(m["error_types"].values()) == e["total"]
        assert m["error_types"]["CORRECT"] == e["correct"]
        assert m["error_types"]["UNKNOWN"] == e["unknown"]
        assert sum(sum(row) for row in m["pos_confusion"]["matrix"]) == e["total"]


def test_evaluation_is_reproducible():
    items = load_gold()
    a = run_evaluation("dev", items=items).to_json_dict()
    b = run_evaluation("dev", items=items).to_json_dict()
    assert json.dumps(a, sort_keys=True, ensure_ascii=False) == json.dumps(b, sort_keys=True, ensure_ascii=False)


def test_empty_split_is_safe():
    rep = run_evaluation("all", items=[])
    for m in rep.metrics.values():
        assert m["exact_top1"]["total"] == 0 and m["exact_top1"]["accuracy"] == 0.0
    assert rep.summary_frame()["Items"].tolist() == [0] * len(rep.metrics)


def test_exported_results_match_report(report, tmp_path):
    export_reports({"test": report}, tmp_path)
    summary = json.loads((tmp_path / "evaluation_summary.json").read_text(encoding="utf-8"))
    df = pd.read_csv(tmp_path / "predictions_test.csv", encoding="utf-8-sig")
    assert len(df) == len(report.items)
    for name, col in ((RULE, "rule"), (FST_NAME, "fst")):
        m = report.metrics[name]["exact_top1"]
        assert int(df[f"{col}_correct"].sum()) == m["correct"]
        assert summary["test"]["metrics"][name]["exact_top1"]["accuracy"] == pytest.approx(m["accuracy"], abs=1e-6)
    js = json.loads((tmp_path / "predictions_test.json").read_text(encoding="utf-8"))
    assert [r["id"] for r in js] == df["id"].tolist()
    # second export is byte-identical (timing is kept in a separate file)
    before = (tmp_path / "evaluation_summary.json").read_bytes()
    export_reports({"test": run_evaluation("test", items=load_gold())}, tmp_path)
    assert (tmp_path / "evaluation_summary.json").read_bytes() == before


def test_error_analysis_helpers(report):
    df = report.comparison_frame()
    counts = agreement_counts(df)
    assert counts["Items"].sum() == len(df)
    assert list(counts["Outcome"]) == list(AGREEMENT_CATEGORIES)
    both_wrong = filter_comparison(df, agreement="Both incorrect")
    assert (~both_wrong["rule_correct"] & ~both_wrong["fst_correct"]).all()
    unk = filter_comparison(df, error_type="UNKNOWN", analyzer=FST_NAME)
    assert (unk["fst_status"] == "UNKNOWN").all()
    assert insights(report)
