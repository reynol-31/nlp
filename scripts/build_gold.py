"""Build data/gold_standard.json from the human-written annotation file.

    python scripts/build_gold.py

Steps: parse + validate every annotation line, group items by shared lemma
(union-find), assign each lemma group to dev/test with a fixed content hash,
and write the JSON gold standard plus dataset_metadata.json.
The split depends only on the lemma strings and the fixed seed, so it is
fully reproducible and was fixed before any rule tuning.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset_loader import (  # noqa: E402
    ANNOTATION_FILE,
    DEV_FRACTION,
    GOLD_FILE,
    METADATA_FILE,
    SPLIT_SEED,
    build_gold_records,
    dataset_summary,
    load_gold_records,
    parse_annotation_file,
)


def main() -> None:
    records = build_gold_records(parse_annotation_file(ANNOTATION_FILE))
    items = load_gold_records(records)  # round-trip validation
    summary = dataset_summary(items)
    if summary["group_overlap_between_splits"]:
        raise SystemExit(f"lemma leakage between splits: {summary['group_overlap_between_splits']}")

    GOLD_FILE.write_text(
        json.dumps(
            {
                "name": "Hindi inflectional morphology gold standard (project-curated)",
                "version": "1.0",
                "schema": "see data/dataset_metadata.json",
                "items": records,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )

    meta = json.loads(METADATA_FILE.read_text(encoding="utf-8")) if METADATA_FILE.exists() else {}
    meta.setdefault("provenance", {})
    meta["statistics"] = summary
    meta["split"] = {
        "method": "lemma-grouped deterministic hash split",
        "details": (
            "Items that share any lemma are merged into one group (union-find). Each group is assigned "
            f"to 'dev' if sha1('{SPLIT_SEED}:' + group) / 2^32 < {DEV_FRACTION}, else 'test'. "
            "No lemma group occurs in both splits."
        ),
        "seed": SPLIT_SEED,
        "dev_fraction_target": DEV_FRACTION,
        "built_on": date.today().isoformat(),
    }
    METADATA_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {GOLD_FILE.relative_to(ROOT)}: {summary['items']} items "
          f"({summary['by_split']}), {summary['analyses']} gold analyses")
    for cat, counts in summary["by_category"].items():
        print(f"  {cat:18s} dev={counts['dev']:3d} test={counts['test']:3d}")


if __name__ == "__main__":
    main()
