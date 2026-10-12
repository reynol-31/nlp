"""Gold-standard dataset: parsing the annotation source, splitting, loading, validating.

``data/annotations/gold_annotations.txt``  (human-written source)
        │  scripts/build_gold.py  (parse -> validate -> lemma-grouped split)
        ▼
``data/gold_standard.json``                (reproducible evaluation reference)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .normalization import is_devanagari_word, normalize_word
from .resources import DATA_DIR
from .schema import Analysis, SchemaError

GOLD_FILE = DATA_DIR / "gold_standard.json"
ANNOTATION_FILE = DATA_DIR / "annotations" / "gold_annotations.txt"
METADATA_FILE = DATA_DIR / "dataset_metadata.json"

CATEGORIES = (
    "noun_regular",
    "noun_irregular",
    "pronoun",
    "adjective",
    "verb_regular",
    "verb_irregular",
    "lexical_ambiguity",
    "spelling_variant",
    "oov",
    "out_of_scope",
)
SPLITS = ("dev", "test")
SPLIT_SEED = "hindi-morph-split-v1"
DEV_FRACTION = 0.40
REQUIRED_FIELDS = ("id", "word", "split", "category", "analyses")


class DatasetError(ValueError):
    """Raised for missing files, malformed records or invalid annotations."""


@dataclass(frozen=True)
class GoldItem:
    id: str
    word: str
    split: str
    category: str
    analyses: tuple[Analysis, ...]
    group: str = ""
    source: str = "manual"
    note: str = ""

    @property
    def normalized(self) -> str:
        return normalize_word(self.word)

    @property
    def gold_keys(self) -> set[tuple[str, ...]]:
        return {a.key() for a in self.analyses}

    @property
    def gold_lemmas(self) -> set[str]:
        return {normalize_word(a.lemma) for a in self.analyses}

    @property
    def primary_pos(self) -> str:
        return self.analyses[0].pos


# ---------------------------------------------------------------------------
# annotation source
def parse_annotation_line(line: str, lineno: int = 0) -> dict[str, Any] | None:
    """Parse ``word :: category :: lemma TAG.. | lemma TAG.. [:: note]``; None for comments/blank."""
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    parts = [p.strip() for p in text.split("::")]
    if len(parts) < 3:
        raise DatasetError(f"line {lineno}: expected 'word :: category :: analyses', got {text!r}")
    word, category, analyses_txt = parts[0], parts[1], parts[2]
    note = parts[3] if len(parts) > 3 else ""
    if category not in CATEGORIES:
        raise DatasetError(f"line {lineno}: unknown category {category!r}")
    if not is_devanagari_word(normalize_word(word)):
        raise DatasetError(f"line {lineno}: word {word!r} is not a single Devanagari word")
    analyses = []
    for chunk in analyses_txt.split("|"):
        tokens = chunk.split()
        if len(tokens) < 2:
            raise DatasetError(f"line {lineno}: analysis needs a lemma and a POS tag: {chunk!r}")
        try:
            analyses.append(Analysis.from_tags(tokens[0], tokens[1:]))
        except SchemaError as exc:
            raise DatasetError(f"line {lineno} ({word}): {exc}") from exc
    keys = [a.key() for a in analyses]
    if len(set(keys)) != len(keys):
        raise DatasetError(f"line {lineno} ({word}): duplicate analyses")
    return {"word": word, "category": category, "analyses": analyses, "note": note, "line": lineno}


def parse_annotation_file(path: Path | str = ANNOTATION_FILE) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"annotation file not found: {path}")
    records = []
    seen_words: dict[str, int] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        rec = parse_annotation_line(line, lineno)
        if rec is None:
            continue
        if rec["word"] in seen_words:
            raise DatasetError(f"line {lineno}: word {rec['word']} already annotated on line {seen_words[rec['word']]}")
        seen_words[rec["word"]] = lineno
        records.append(rec)
    if not records:
        raise DatasetError(f"no annotations found in {path}")
    return records


def lemma_groups(records: list[dict[str, Any]]) -> list[str]:
    """Union-find over lemmas: items sharing any lemma end up in the same group."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for rec in records:
        lemmas = sorted({normalize_word(a.lemma) for a in rec["analyses"]})
        for other in lemmas[1:]:
            union(lemmas[0], other)
        find(lemmas[0])
    groups = []
    for rec in records:
        first = normalize_word(rec["analyses"][0].lemma)
        groups.append(find(first))
    return groups


def split_for_group(group: str, seed: str = SPLIT_SEED, dev_fraction: float = DEV_FRACTION) -> str:
    """Deterministic, content-based split assignment (no RNG state involved)."""
    h = hashlib.sha1(f"{seed}:{group}".encode("utf-8")).hexdigest()
    return "dev" if int(h[:8], 16) / 0xFFFFFFFF < dev_fraction else "test"


def build_gold_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = lemma_groups(records)
    out = []
    for i, (rec, group) in enumerate(zip(records, groups), 1):
        out.append(
            {
                "id": f"G{i:03d}",
                "word": rec["word"],
                "split": split_for_group(group),
                "category": rec["category"],
                "group": group,
                "source": "manual",
                "note": rec["note"],
                "analyses": [a.to_dict(with_meta=False) for a in rec["analyses"]],
            }
        )
    return out


# ---------------------------------------------------------------------------
# JSON gold standard
def item_from_record(rec: dict[str, Any]) -> GoldItem:
    missing = [f for f in REQUIRED_FIELDS if f not in rec or rec[f] in (None, "", [])]
    if missing:
        raise DatasetError(f"gold record {rec.get('id', '?')} is missing field(s): {', '.join(missing)}")
    if rec["split"] not in SPLITS:
        raise DatasetError(f"gold record {rec['id']}: invalid split {rec['split']!r}")
    if rec["category"] not in CATEGORIES:
        raise DatasetError(f"gold record {rec['id']}: invalid category {rec['category']!r}")
    if not isinstance(rec["analyses"], list):
        raise DatasetError(f"gold record {rec['id']}: 'analyses' must be a list")
    try:
        analyses = tuple(Analysis.from_dict(a) for a in rec["analyses"])
    except (SchemaError, AttributeError, TypeError) as exc:
        raise DatasetError(f"gold record {rec['id']}: {exc}") from exc
    word = str(rec["word"])
    if not is_devanagari_word(normalize_word(word)):
        raise DatasetError(f"gold record {rec['id']}: word {word!r} is not Devanagari")
    return GoldItem(
        id=str(rec["id"]),
        word=word,
        split=rec["split"],
        category=rec["category"],
        analyses=analyses,
        group=str(rec.get("group", "")),
        source=str(rec.get("source", "manual")),
        note=str(rec.get("note", "")),
    )


def load_gold_records(records: Iterable[dict[str, Any]]) -> list[GoldItem]:
    items = [item_from_record(r) for r in records]
    ids = [i.id for i in items]
    if len(set(ids)) != len(ids):
        raise DatasetError("duplicate item ids in gold standard")
    return items


def load_gold(path: Path | str = GOLD_FILE) -> list[GoldItem]:
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"gold standard not found: {path}. Run: python scripts/build_gold.py")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{path.name} is not valid JSON: {exc}") from exc
    records = data.get("items") if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise DatasetError(f"{path.name}: expected a list of items under 'items'")
    return load_gold_records(records)


def filter_split(items: list[GoldItem], split: str) -> list[GoldItem]:
    if split == "all":
        return list(items)
    if split not in SPLITS:
        raise DatasetError(f"unknown split {split!r} (use dev, test or all)")
    return [i for i in items if i.split == split]


def dataset_summary(items: list[GoldItem]) -> dict[str, Any]:
    by_split: dict[str, int] = {}
    by_cat: dict[str, dict[str, int]] = {}
    groups: dict[str, set[str]] = {}
    for it in items:
        by_split[it.split] = by_split.get(it.split, 0) + 1
        by_cat.setdefault(it.category, {s: 0 for s in SPLITS})[it.split] += 1
        groups.setdefault(it.split, set()).add(it.group)
    overlap = set.intersection(*groups.values()) if len(groups) == len(SPLITS) else set()
    normalized = [i.normalized for i in items]
    return {
        "items": len(items),
        "analyses": sum(len(i.analyses) for i in items),
        "ambiguous_items": sum(1 for i in items if len(i.analyses) > 1),
        "by_split": by_split,
        "by_category": by_cat,
        "lemma_groups": {s: len(g) for s, g in groups.items()},
        "group_overlap_between_splits": sorted(overlap),
        "items_sharing_normalized_form": len(normalized) - len(set(normalized)),
    }


def load_metadata(path: Path | str = METADATA_FILE) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
