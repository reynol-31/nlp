"""Error analysis helpers built on :class:`EvaluationReport` (no Streamlit dependency)."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from .evaluation import ERROR_TYPES, FST_NAME, RULE, EvaluationReport, _col

AGREEMENT_CATEGORIES = (
    "Both correct",
    "Only rule-based correct",
    "Only FST correct",
    "Both incorrect",
)

ERROR_DESCRIPTIONS = {
    "CORRECT": "Primary prediction equals one of the gold analyses.",
    "UNKNOWN": "No analysis returned (word rejected / not covered).",
    "WRONG_POS": "An analysis was returned but with the wrong part of speech.",
    "WRONG_LEMMA": "POS is right but the reconstructed lemma is wrong (typical for guessed unknown words).",
    "WRONG_FEATURES": "Lemma and POS are right but at least one grammatical feature differs.",
}


def filter_comparison(
    df: pd.DataFrame,
    agreement: str | None = None,
    pos: list[str] | None = None,
    categories: list[str] | None = None,
    error_type: str | None = None,
    analyzer: str = RULE,
    feature: str | None = None,
) -> pd.DataFrame:
    """Filter the per-item comparison table. ``feature`` keeps rows where *analyzer* got that feature wrong."""
    out = df
    if agreement:
        out = out[out["agreement"] == agreement]
    if pos:
        out = out[out["gold_pos"].isin(pos)]
    if categories:
        out = out[out["category"].isin(categories)]
    tag = _col(analyzer)
    if error_type:
        out = out[out[f"{tag}_error"] == error_type]
    if feature:
        out = out[out[f"{tag}_mismatched"].fillna("").str.split(",").apply(lambda xs: feature in xs)]
    return out


def agreement_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["agreement"].value_counts()
    return pd.DataFrame(
        {"Outcome": list(AGREEMENT_CATEGORIES), "Items": [int(counts.get(c, 0)) for c in AGREEMENT_CATEGORIES]}
    )


def error_type_table(report: EvaluationReport) -> pd.DataFrame:
    rows = []
    for name, m in report.metrics.items():
        for et in ERROR_TYPES:
            rows.append({"Analyzer": name, "Error type": et, "Items": m["error_types"][et]})
    return pd.DataFrame(rows)


def category_accuracy_table(report: EvaluationReport) -> pd.DataFrame:
    rows = []
    for name, m in report.metrics.items():
        for cat, c in m["by_category"].items():
            rows.append({
                "Analyzer": name, "Category": cat, "Items": c["total"], "Correct": c["correct"],
                "Answered": c["answered"], "Accuracy": c["correct"] / c["total"] if c["total"] else 0.0,
            })
    return pd.DataFrame(rows)


def mismatched_feature_counts(report: EvaluationReport, analyzer: str) -> Counter:
    counter: Counter = Counter()
    for r in report.results[analyzer]:
        if r.error_type not in ("CORRECT", "UNKNOWN"):
            counter.update(r.mismatched)
    return counter


def insights(report: EvaluationReport) -> list[str]:
    """Plain-language observations derived from the current results (no fixed text about numbers)."""
    if RULE not in report.results or FST_NAME not in report.results:
        return []
    df = report.comparison_frame()
    n = len(df)
    if n == 0:
        return ["No items in this split."]
    out = []
    ac = df["agreement"].value_counts()
    out.append(
        f"Both analyzers are correct on {ac.get('Both correct', 0)} of {n} items; "
        f"only the rule-based analyzer on {ac.get('Only rule-based correct', 0)}; "
        f"only the FST on {ac.get('Only FST correct', 0)}; neither on {ac.get('Both incorrect', 0)}."
    )
    in_lex = df[~df["category"].isin(["oov", "out_of_scope"])]
    if len(in_lex):
        out.append(
            f"On the {len(in_lex)} in-lexicon items the rule-based analyzer is correct on "
            f"{int(in_lex['rule_correct'].sum())} and the FST on {int(in_lex['fst_correct'].sum())}: "
            "with identical lexicon and paradigms the two formalisms accept the same forms."
        )
    oov = df[df["category"] == "oov"]
    if len(oov):
        guessed_ok = int(oov["rule_correct"].sum())
        out.append(
            f"On {len(oov)} OOV items (lemma not in lexicon) the FST returns UNKNOWN for "
            f"{int((oov['fst_status'] == 'UNKNOWN').sum())}; the rule-based guesser gets {guessed_ok} fully right "
            f"and {int(oov['rule_lemma_correct'].sum())} with the correct lemma."
        )
    m_r, m_f = report.metrics[RULE]["exact_top1"], report.metrics[FST_NAME]["exact_top1"]
    out.append(
        f"Precision vs recall trade-off: the FST answers {m_f['answered']} items with precision "
        f"{100 * m_f['precision']:.1f}%, the rule-based analyzer answers {m_r['answered']} with precision "
        f"{100 * m_r['precision']:.1f}% (guesses can be wrong) but higher recall "
        f"({100 * m_r['recall']:.1f}% vs {100 * m_f['recall']:.1f}%)."
    )
    wrong_guesses = [r for r in report.results[RULE] if r.result.guessed and not r.top1_correct]
    if wrong_guesses:
        types = Counter(r.error_type for r in wrong_guesses)
        out.append(
            f"{len(wrong_guesses)} guessed analyses are wrong (" +
            ", ".join(f"{k}: {v}" for k, v in types.most_common()) +
            "); suffix guessing cannot tell noun/adjective/verb apart when they share a suffix, "
            "nor recover a dropped -आ (पत्तों -> पत्त vs पत्ता)."
        )
    oos = df[df["category"] == "out_of_scope"]
    if len(oos):
        out.append(
            f"{len(oos)} out-of-scope items (derivation, vocative, -कर) cannot be analysed correctly by either "
            f"system by design; the FST rejects them, the rule-based analyzer sometimes guesses "
            f"({int((oos['rule_status'] == 'GUESSED').sum())} guessed)."
        )
    return out
