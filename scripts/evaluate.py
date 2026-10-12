"""Command-line evaluation of both analyzers against the gold standard.

    python scripts/evaluate.py                 # dev, test and all splits -> results/
    python scripts/evaluate.py --split dev     # one split only
    python scripts/evaluate.py --errors 15     # also list up to 15 errors per analyzer

Every number is computed from the analyzers' actual predictions; nothing is
hard-coded.  Results (except timing) are deterministic and byte-identical
across runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation import FST_NAME, RULE, export_reports, run_evaluation  # noqa: E402
from src.dataset_loader import load_gold  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def pct(x: float) -> str:
    return f"{100 * x:6.2f}%"


def print_report(rep, n_errors: int) -> None:
    print(f"\n=== split: {rep.split}  ({len(rep.items)} items) ===")
    header = f"{'Analyzer':26s} {'Acc':>8s} {'P':>8s} {'R':>8s} {'F1':>8s} | {'LemmaAcc':>8s} | {'SetF1':>8s} | {'FeatF1':>8s} | correct/answered/total"
    print(header)
    print("-" * len(header))
    for name, m in rep.metrics.items():
        e = m["exact_top1"]
        print(
            f"{name:26s} {pct(e['accuracy'])} {pct(e['precision'])} {pct(e['recall'])} {pct(e['f1'])} | "
            f"{pct(m['lemma_top1']['accuracy'])} | {pct(m['analysis_set']['f1'])} | "
            f"{pct(m['feature_micro']['f1'])} | {e['correct']}/{e['answered']}/{e['total']}"
        )
    timing = rep.timing()
    print("mean latency per word: " + ", ".join(f"{k}: {v:.1f} µs" for k, v in timing.items()))
    if n_errors:
        for name in (RULE, FST_NAME):
            errs = [r for r in rep.results[name] if not r.top1_correct]
            print(f"\n  {name} errors ({len(errs)}):")
            for r in errs[:n_errors]:
                pred = r.top1.short() if r.top1 else "UNKNOWN"
                print(f"   [{r.item.category}] {r.item.word}: {r.error_type}  pred={pred}  gold={r.best_gold.short()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test", "all"], help="evaluate one split only")
    ap.add_argument("--out", default=str(ROOT / "results"), help="output directory")
    ap.add_argument("--errors", type=int, default=0, help="print up to N errors per analyzer")
    ap.add_argument("--no-export", action="store_true", help="do not write result files")
    args = ap.parse_args()

    items = load_gold()
    splits = [args.split] if args.split else ["dev", "test", "all"]
    reports = {s: run_evaluation(s, items=items, include_ablation=True) for s in splits}
    for rep in reports.values():
        print_report(rep, args.errors)
    if not args.no_export:
        paths = export_reports(reports, args.out)
        print("\nwrote:")
        for p in paths:
            print("  ", Path(p).resolve().relative_to(ROOT))


if __name__ == "__main__":
    main()
