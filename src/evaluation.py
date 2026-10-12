"""Evaluation protocol and metrics.

Notation: for item *i*, ``G_i`` is the set of gold analyses (>= 1) and
``P_i`` the ranked list of predicted analyses (empty = UNKNOWN).  The
*primary prediction* is ``P_i[0]``.

1. Exact-analysis, top-1 (PRIMARY METRIC)
   correct_i  = P_i non-empty and P_i[0] ∈ G_i   (match on lemma + all 9 features)
   Accuracy   = Σ correct / N
   Precision  = Σ correct / #answered            (answered = P_i non-empty)
   Recall     = Σ correct / N
   F1         = harmonic mean of P and R
   UNKNOWN outputs are never excluded: they count as errors for accuracy and
   recall, and as abstentions for precision.  Under this protocol accuracy
   equals recall by definition; precision rewards knowing when to abstain.

2. Exact-analysis, full candidate set (ambiguity handling)
   micro P = Σ|P_i ∩ G_i| / Σ|P_i|,  micro R = Σ|P_i ∩ G_i| / Σ|G_i|

3. Lemma, top-1 (secondary): as (1) but correct iff lemma(P_i[0]) ∈ lemmas(G_i).

4. Feature level (micro over feature-value pairs, top-1)
   Each analysis is turned into a set of (feature, value) pairs for lemma and
   every feature whose value is not NA.  The primary prediction is compared
   with the *best-matching* gold analysis (an exact match if there is one,
   otherwise the gold analysis with the largest pair overlap) so that a
   legitimate alternative reading is never penalised.  TP = shared pairs,
   FP = predicted pairs not in gold, FN = gold pairs not predicted.
   NA handling: a predicted NA where gold has a value is a FN; a predicted
   value where gold has NA is a FP; NA/NA is a true negative and not counted.
   UNKNOWN predictions contribute only FNs (all pairs of the first gold analysis).

Zero-division policy: any ratio with a zero denominator is reported as 0.0
and flagged with ``*_defined = False`` so the UI can say "undefined".
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .dataset_loader import GoldItem, dataset_summary, filter_split, load_gold
from .fst_analyzer import FSTAnalyzer
from .normalization import normalize_word
from .resources import Resources, load_resources
from .rule_based_analyzer import RuleBasedAnalyzer
from .schema import FEATURES, NA, Analysis, AnalyzerResult

FEATURE_KEYS: tuple[str, ...] = ("lemma",) + FEATURES
POS_LABELS: tuple[str, ...] = ("NOUN", "PRON", "ADJ", "VERB", "AUX")
UNKNOWN = "UNKNOWN"
RULE, FST_NAME, RULE_ABLATION = "Rule-based", "FST", "Rule-based (no guesser)"
ERROR_TYPES = ("CORRECT", "UNKNOWN", "WRONG_POS", "WRONG_LEMMA", "WRONG_FEATURES")


class Analyzer(Protocol):
    label: str

    def analyze(self, word: str) -> AnalyzerResult: ...


# ---------------------------------------------------------------------------
# metric primitives
def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    f = safe_div(2 * p * r, p + r)
    return {
        "precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn,
        "precision_defined": (tp + fp) > 0, "recall_defined": (tp + fn) > 0,
    }


def top1_metrics(correct: list[bool], answered: list[bool]) -> dict[str, Any]:
    """Accuracy / precision / recall / F1 for single-prediction-per-item evaluation."""
    if len(correct) != len(answered):
        raise ValueError("correct and answered must have the same length")
    if any(c and not a for c, a in zip(correct, answered)):
        raise ValueError("an unanswered item cannot be correct")
    n = len(correct)
    n_correct = sum(correct)
    n_answered = sum(answered)
    p = safe_div(n_correct, n_answered)
    r = safe_div(n_correct, n)
    return {
        "accuracy": safe_div(n_correct, n),
        "precision": p,
        "recall": r,
        "f1": safe_div(2 * p * r, p + r),
        "correct": n_correct,
        "answered": n_answered,
        "unknown": n - n_answered,
        "errors": n - n_correct,
        "total": n,
        "accuracy_defined": n > 0,
        "precision_defined": n_answered > 0,
    }


def set_metrics(pred_sets: list[set], gold_sets: list[set]) -> dict[str, Any]:
    tp = sum(len(p & g) for p, g in zip(pred_sets, gold_sets))
    n_pred = sum(len(p) for p in pred_sets)
    n_gold = sum(len(g) for g in gold_sets)
    out = prf(tp, n_pred - tp, n_gold - tp)
    out.update(n_pred=n_pred, n_gold=n_gold)
    return out


def feature_pairs(a: Analysis | None) -> set[tuple[str, str]]:
    if a is None:
        return set()
    pairs = {("lemma", normalize_word(a.lemma))}
    pairs |= {(f, getattr(a, f)) for f in FEATURES if getattr(a, f) != NA}
    return pairs


def best_gold_match(pred: Analysis | None, gold: list[Analysis] | tuple[Analysis, ...]) -> Analysis:
    """Exact match if present; otherwise max pair overlap, then fewest missed pairs, then gold order."""
    if pred is None:
        return gold[0]
    pk = pred.key()
    for g in gold:
        if g.key() == pk:
            return g
    pp = feature_pairs(pred)
    scored = [(-len(pp & feature_pairs(g)), len(feature_pairs(g) - pp), i) for i, g in enumerate(gold)]
    return gold[min(scored)[2]]


def classify_error(pred: Analysis | None, gold: Analysis) -> tuple[str, list[str]]:
    if pred is None:
        return "UNKNOWN", []
    mismatched = [f for f in FEATURE_KEYS
                  if (normalize_word(getattr(pred, f)) if f == "lemma" else getattr(pred, f))
                  != (normalize_word(getattr(gold, f)) if f == "lemma" else getattr(gold, f))]
    if not mismatched:
        return "CORRECT", []
    if pred.pos != gold.pos:
        return "WRONG_POS", mismatched
    if "lemma" in mismatched:
        return "WRONG_LEMMA", mismatched
    return "WRONG_FEATURES", mismatched


# ---------------------------------------------------------------------------
@dataclass
class ItemResult:
    item: GoldItem
    result: AnalyzerResult
    elapsed_us: float
    top1_correct: bool = False
    lemma_correct: bool = False
    n_overlap: int = 0
    best_gold: Analysis | None = None
    error_type: str = ""
    mismatched: list[str] = field(default_factory=list)
    feat_counts: dict[str, list[int]] = field(default_factory=dict)  # feature -> [tp, fp, fn]

    @property
    def pred(self) -> list[Analysis]:
        return self.result.analyses

    @property
    def top1(self) -> Analysis | None:
        return self.result.primary


def score_item(item: GoldItem, result: AnalyzerResult, elapsed_us: float = 0.0) -> ItemResult:
    top1 = result.primary
    gold_keys = item.gold_keys
    r = ItemResult(item=item, result=result, elapsed_us=elapsed_us)
    r.top1_correct = top1 is not None and top1.key() in gold_keys
    r.lemma_correct = top1 is not None and normalize_word(top1.lemma) in item.gold_lemmas
    r.n_overlap = len({a.key() for a in result.analyses} & gold_keys)
    r.best_gold = best_gold_match(top1, item.analyses)
    r.error_type, r.mismatched = classify_error(top1, r.best_gold)
    pp, gp = feature_pairs(top1), feature_pairs(r.best_gold)
    counts = {f: [0, 0, 0] for f in FEATURE_KEYS}
    for f, _ in pp & gp:
        counts[f][0] += 1
    for f, _ in pp - gp:
        counts[f][1] += 1
    for f, _ in gp - pp:
        counts[f][2] += 1
    r.feat_counts = counts
    return r


def evaluate_analyzer(analyzer: Analyzer, items: list[GoldItem]) -> list[ItemResult]:
    out = []
    if items:  # warm-up call so one-off initialisation does not distort timing
        analyzer.analyze(items[0].word)
    for item in items:
        t0 = time.perf_counter()
        res = analyzer.analyze(item.word)
        dt = (time.perf_counter() - t0) * 1e6
        out.append(score_item(item, res, dt))
    return out


def compute_metrics(results: list[ItemResult]) -> dict[str, Any]:
    correct = [r.top1_correct for r in results]
    answered = [r.result.known for r in results]
    lemma_ok = [r.lemma_correct for r in results]
    metrics: dict[str, Any] = {
        "n_items": len(results),
        "exact_top1": top1_metrics(correct, answered),
        "lemma_top1": top1_metrics(lemma_ok, answered),
        "analysis_set": set_metrics(
            [{a.key() for a in r.pred} for r in results], [r.item.gold_keys for r in results]
        ),
    }
    totals = {f: [0, 0, 0] for f in FEATURE_KEYS}
    for r in results:
        for f, (tp, fp, fn) in r.feat_counts.items():
            totals[f][0] += tp
            totals[f][1] += fp
            totals[f][2] += fn
    metrics["feature_micro"] = prf(*(sum(v[i] for v in totals.values()) for i in range(3)))
    metrics["feature_per_feature"] = {
        f: {**prf(*v), "support": v[0] + v[2]} for f, v in totals.items()
    }

    y_true = [r.best_gold.pos for r in results]
    y_pred = [r.top1.pos if r.top1 else UNKNOWN for r in results]
    labels = list(POS_LABELS) + [UNKNOWN]
    metrics["pos_confusion"] = {
        "labels": labels,
        "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist() if results else [],
    }
    if results:
        p, r_, f, s = precision_recall_fscore_support(
            y_true, y_pred, labels=list(POS_LABELS), zero_division=0
        )
        metrics["pos_per_class"] = {
            lab: {"precision": float(p[i]), "recall": float(r_[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, lab in enumerate(POS_LABELS)
        }
    else:
        metrics["pos_per_class"] = {}
    metrics["error_types"] = {t: sum(1 for r in results if r.error_type == t) for t in ERROR_TYPES}
    metrics["guessed"] = sum(1 for r in results if r.result.guessed)
    metrics["ambiguous_predictions"] = sum(1 for r in results if len(r.pred) > 1)
    by_cat: dict[str, dict[str, int]] = {}
    for r in results:
        c = by_cat.setdefault(r.item.category, {"total": 0, "correct": 0, "answered": 0})
        c["total"] += 1
        c["correct"] += int(r.top1_correct)
        c["answered"] += int(r.result.known)
    metrics["by_category"] = by_cat
    return metrics


# ---------------------------------------------------------------------------
@dataclass
class EvaluationReport:
    split: str
    items: list[GoldItem]
    results: dict[str, list[ItemResult]]
    metrics: dict[str, dict[str, Any]]
    config: dict[str, Any]

    @property
    def analyzers(self) -> list[str]:
        return list(self.results)

    def timing(self) -> dict[str, float]:
        return {
            name: safe_div(sum(r.elapsed_us for r in res), len(res)) for name, res in self.results.items()
        }

    # -- tables -----------------------------------------------------------
    def summary_frame(self) -> pd.DataFrame:
        rows = []
        for name, m in self.metrics.items():
            e, l_, s, f = m["exact_top1"], m["lemma_top1"], m["analysis_set"], m["feature_micro"]
            rows.append({
                "Analyzer": name,
                "Items": e["total"],
                "Exact matches": e["correct"],
                "Errors": e["errors"],
                "Unknown": e["unknown"],
                "Accuracy": e["accuracy"],
                "Precision": e["precision"],
                "Recall": e["recall"],
                "F1": e["f1"],
                "Lemma accuracy": l_["accuracy"],
                "Lemma F1": l_["f1"],
                "Analysis-set P": s["precision"],
                "Analysis-set R": s["recall"],
                "Analysis-set F1": s["f1"],
                "Feature P": f["precision"],
                "Feature R": f["recall"],
                "Feature F1": f["f1"],
            })
        return pd.DataFrame(rows)

    def feature_frame(self) -> pd.DataFrame:
        rows = []
        for name, m in self.metrics.items():
            for feat, v in m["feature_per_feature"].items():
                rows.append({"Analyzer": name, "Feature": feat, "Precision": v["precision"],
                             "Recall": v["recall"], "F1": v["f1"], "Support": v["support"],
                             "TP": v["tp"], "FP": v["fp"], "FN": v["fn"]})
        return pd.DataFrame(rows)

    def comparison_frame(self) -> pd.DataFrame:
        """One row per gold item with every analyzer's primary prediction."""
        names = self.analyzers
        rows = []
        for idx, item in enumerate(self.items):
            row: dict[str, Any] = {
                "id": item.id, "word": item.word, "normalized": item.normalized,
                "split": item.split, "category": item.category, "gold_pos": item.primary_pos,
                "gold": " | ".join(a.short() for a in item.analyses),
                "n_gold": len(item.analyses),
            }
            for name in names:
                r = self.results[name][idx]
                tag = _col(name)
                row[f"{tag}_prediction"] = r.top1.short() if r.top1 else UNKNOWN
                row[f"{tag}_all"] = " | ".join(a.short() for a in r.pred)
                row[f"{tag}_status"] = r.result.status
                row[f"{tag}_correct"] = r.top1_correct
                row[f"{tag}_lemma_correct"] = r.lemma_correct
                row[f"{tag}_error"] = r.error_type
                row[f"{tag}_mismatched"] = ",".join(r.mismatched)
                row[f"{tag}_rule"] = r.top1.rule if r.top1 else ""
            if RULE in self.results and FST_NAME in self.results:
                row["agreement"] = agreement_label(row[f"{_col(RULE)}_correct"], row[f"{_col(FST_NAME)}_correct"])
            rows.append(row)
        return pd.DataFrame(rows)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "config": self.config,
            "dataset": dataset_summary(self.items) if self.items else {},
            "metrics": self.metrics,
        }


def _col(name: str) -> str:
    return {RULE: "rule", FST_NAME: "fst", RULE_ABLATION: "rule_noguess"}.get(name, name.lower())


def agreement_label(rule_ok: bool, fst_ok: bool) -> str:
    if rule_ok and fst_ok:
        return "Both correct"
    if rule_ok:
        return "Only rule-based correct"
    if fst_ok:
        return "Only FST correct"
    return "Both incorrect"


def build_analyzers(
    resources: Resources | None = None, include_ablation: bool = False, enable_guesser: bool = True
) -> dict[str, Analyzer]:
    res = resources or load_resources()
    analyzers: dict[str, Analyzer] = {
        RULE: RuleBasedAnalyzer(res, enable_guesser=enable_guesser),
        FST_NAME: FSTAnalyzer(res),
    }
    if include_ablation:
        analyzers[RULE_ABLATION] = RuleBasedAnalyzer(res, enable_guesser=False)
    return analyzers


def run_evaluation(
    split: str = "test",
    items: list[GoldItem] | None = None,
    analyzers: dict[str, Analyzer] | None = None,
    include_ablation: bool = True,
) -> EvaluationReport:
    all_items = items if items is not None else load_gold()
    selected = filter_split(all_items, split)
    analyzers = analyzers or build_analyzers(include_ablation=include_ablation)
    results = {name: evaluate_analyzer(a, selected) for name, a in analyzers.items()}
    metrics = {name: compute_metrics(res) for name, res in results.items()}
    config = {
        "split": split,
        "analyzers": list(analyzers),
        "primary_metric": "exact-analysis top-1 accuracy",
        "zero_division": "0.0 (flagged as undefined)",
    }
    return EvaluationReport(split, selected, results, metrics, config)


# ---------------------------------------------------------------------------
def _round(obj: Any, nd: int = 6) -> Any:
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, nd) for v in obj]
    return obj


def export_reports(reports: dict[str, EvaluationReport], out_dir: Path | str) -> list[Path]:
    """Write deterministic CSV/JSON results (timing goes to a separate file)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    summary = {split: _round(rep.to_json_dict()) for split, rep in reports.items()}
    p = out / "evaluation_summary.json"
    p.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    written.append(p)
    for split, rep in reports.items():
        cmp_df = rep.comparison_frame()
        p = out / f"predictions_{split}.csv"
        cmp_df.to_csv(p, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
        written.append(p)
        p = out / f"predictions_{split}.json"
        p.write_text(cmp_df.to_json(orient="records", force_ascii=False, indent=1), encoding="utf-8")
        written.append(p)
        p = out / f"metrics_{split}.csv"
        rep.summary_frame().round(6).to_csv(p, index=False, encoding="utf-8-sig")
        written.append(p)
        p = out / f"feature_metrics_{split}.csv"
        rep.feature_frame().round(6).to_csv(p, index=False, encoding="utf-8-sig")
        written.append(p)
    p = out / "timing.json"
    p.write_text(
        json.dumps({s: _round(r.timing(), 2) for s, r in reports.items()}, indent=2) + "\n", encoding="utf-8"
    )
    written.append(p)
    return written
