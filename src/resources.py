"""Load and validate the shared linguistic resources (lexicon + morphology rules).

Both analyzers are built from the same :class:`Resources` object, which
guarantees they share identical lexical and paradigm knowledge.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .normalization import normalize_word
from .schema import NA, SchemaError, features_from_tags

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LEXICON_FILE = "lexicon.json"
RULES_FILE = "morphology_rules.json"


class ResourceError(ValueError):
    """Raised when a lexicon or rules file is missing or inconsistent."""


@dataclass(frozen=True)
class SuffixRule:
    """One inflectional rule: surface suffix <-> feature bundle within a paradigm."""

    paradigm: str
    slot: str
    surface: str  # normalised surface suffix ("" = zero suffix)
    truncate: str  # normalised lemma ending replaced by the suffix
    pos: str
    tags: tuple[str, ...]  # feature tags *including* POS
    label: str
    order: int  # position within the paradigm (for deterministic ranking)

    @property
    def rule_id(self) -> str:
        return f"{self.paradigm}:{self.slot}"

    def features(self) -> dict[str, str]:
        return features_from_tags(self.tags)


@dataclass(frozen=True)
class Paradigm:
    id: str
    pos: str
    description: str
    truncate: str
    lemma_regex: re.Pattern[str]
    guessable: bool
    guess_prior: float
    rules: tuple[SuffixRule, ...]
    order: int


@dataclass(frozen=True)
class LexEntry:
    lemma: str  # normalised
    display: str  # as written in the lexicon
    paradigm: str
    pos: str
    gloss: str
    blocked_slots: frozenset[str] = frozenset()
    restrict: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def allows(self, rule: SuffixRule) -> bool:
        """Lexical constraints: blocked slots and feature restrictions."""
        if rule.slot in self.blocked_slots:
            return False
        feats = rule.features()
        for feat, allowed in self.restrict:
            if feats.get(feat, NA) not in allowed:
                return False
        return True


@dataclass(frozen=True)
class IrregularForm:
    form: str  # normalised
    display_form: str
    lemma: str  # normalised
    display_lemma: str
    tags: tuple[str, ...]
    note: str
    index: int


@dataclass
class Resources:
    paradigms: dict[str, Paradigm]
    entries: list[LexEntry]
    irregular: list[IrregularForm]
    guesser_config: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.by_lemma: dict[str, list[LexEntry]] = {}
        for e in self.entries:
            self.by_lemma.setdefault(e.lemma, []).append(e)
        self.irregular_by_form: dict[str, list[IrregularForm]] = {}
        for f in self.irregular:
            self.irregular_by_form.setdefault(f.form, []).append(f)
        self.display_lemma: dict[str, str] = {e.lemma: e.display for e in self.entries}
        for f in self.irregular:
            self.display_lemma.setdefault(f.lemma, f.display_lemma)
        self.all_rules: list[SuffixRule] = [r for p in self.paradigms.values() for r in p.rules]

    def stem_of(self, entry: LexEntry) -> str:
        trunc = self.paradigms[entry.paradigm].truncate
        return entry.lemma[: len(entry.lemma) - len(trunc)] if trunc else entry.lemma

    def known_lemmas(self) -> set[str]:
        return set(self.by_lemma) | {f.lemma for f in self.irregular}

    def summary(self) -> dict[str, int]:
        return {
            "lemmas": len({e.lemma for e in self.entries} | {f.lemma for f in self.irregular}),
            "lexicon_entries": len(self.entries),
            "paradigms": len(self.paradigms),
            "suffix_rules": len(self.all_rules),
            "irregular_forms": len(self.irregular_by_form),
            "irregular_analyses": len(self.irregular),
        }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ResourceError(f"required data file not found: {path}")
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise ResourceError(f"{path.name} is not valid JSON: {exc}") from exc


def _build_paradigms(rules_json: dict[str, Any]) -> dict[str, Paradigm]:
    slot_table = rules_json.get("verb_slots", {})
    columns = slot_table.get("columns", [])
    paradigms: dict[str, Paradigm] = {}
    for order, (pid, p) in enumerate(rules_json.get("paradigms", {}).items()):
        try:
            pos = p["pos"]
            truncate = normalize_word(p.get("truncate", ""))
            regex = re.compile(p["lemma_regex"])
        except (KeyError, re.error) as exc:
            raise ResourceError(f"paradigm {pid}: {exc}") from exc
        raw_suffixes: list[dict[str, Any]]
        if "slot_column" in p:
            col = p["slot_column"]
            if col not in columns:
                raise ResourceError(f"paradigm {pid}: unknown slot column {col!r}")
            ci = columns.index(col)
            raw_suffixes = [
                {"slot": r["slot"], "surface": r["surface"][ci], "tags": r["tags"], "label": r["label"]}
                for r in slot_table["rows"]
            ]
        else:
            raw_suffixes = p.get("suffixes", [])
        if not raw_suffixes:
            raise ResourceError(f"paradigm {pid} has no suffixes")
        rules = []
        seen_slots = set()
        for i, s in enumerate(raw_suffixes):
            if s["slot"] in seen_slots:
                raise ResourceError(f"paradigm {pid}: duplicate slot {s['slot']}")
            seen_slots.add(s["slot"])
            tags = (pos, *s["tags"])
            try:
                features_from_tags(tags)
            except SchemaError as exc:
                raise ResourceError(f"paradigm {pid} slot {s['slot']}: {exc}") from exc
            rules.append(
                SuffixRule(
                    paradigm=pid,
                    slot=s["slot"],
                    surface=normalize_word(s["surface"]) if s["surface"] else "",
                    truncate=truncate,
                    pos=pos,
                    tags=tags,
                    label=s.get("label", ""),
                    order=i,
                )
            )
        paradigms[pid] = Paradigm(
            id=pid,
            pos=pos,
            description=p.get("description", ""),
            truncate=truncate,
            lemma_regex=regex,
            guessable=bool(p.get("guessable", False)),
            guess_prior=float(p.get("guess_prior", 0.0)),
            rules=tuple(rules),
            order=order,
        )
    if not paradigms:
        raise ResourceError("no paradigms defined in rules file")
    return paradigms


def _build_lexicon(lex_json: dict[str, Any], paradigms: dict[str, Paradigm]):
    entries: list[LexEntry] = []
    seen: set[tuple[str, str]] = set()
    for i, e in enumerate(lex_json.get("entries", [])):
        if "lemma" not in e or "paradigm" not in e:
            raise ResourceError(f"lexicon entry #{i} needs 'lemma' and 'paradigm': {e}")
        pid = e["paradigm"]
        if pid not in paradigms:
            raise ResourceError(f"lexicon entry {e['lemma']}: unknown paradigm {pid!r}")
        par = paradigms[pid]
        lemma = normalize_word(e["lemma"])
        if not par.lemma_regex.match(lemma):
            raise ResourceError(f"lemma {e['lemma']} does not fit paradigm {pid} ({par.lemma_regex.pattern})")
        if par.truncate and not lemma.endswith(par.truncate):
            raise ResourceError(f"lemma {e['lemma']} does not end in {par.truncate!r} required by {pid}")
        if (lemma, pid) in seen:
            raise ResourceError(f"duplicate lexicon entry {e['lemma']} / {pid}")
        seen.add((lemma, pid))
        blocked = frozenset(e.get("blocked_slots", []))
        unknown = blocked - {r.slot for r in par.rules}
        if unknown:
            raise ResourceError(f"lemma {e['lemma']}: blocked slots not in {pid}: {sorted(unknown)}")
        restrict = tuple(sorted((k, tuple(v)) for k, v in e.get("restrict", {}).items()))
        entries.append(
            LexEntry(
                lemma=lemma,
                display=e["lemma"],
                paradigm=pid,
                pos=par.pos,
                gloss=e.get("gloss", ""),
                blocked_slots=blocked,
                restrict=restrict,
            )
        )

    irregular: list[IrregularForm] = []
    for i, f in enumerate(lex_json.get("irregular_forms", [])):
        if not f.get("form") or not f.get("lemma") or not f.get("analyses"):
            raise ResourceError(f"irregular form #{i} needs form, lemma and analyses: {f}")
        for tags in f["analyses"]:
            try:
                features_from_tags(tags)
            except SchemaError as exc:
                raise ResourceError(f"irregular form {f['form']}: {exc}") from exc
            irregular.append(
                IrregularForm(
                    form=normalize_word(f["form"]),
                    display_form=f["form"],
                    lemma=normalize_word(f["lemma"]),
                    display_lemma=f["lemma"],
                    tags=tuple(tags),
                    note=f.get("note", ""),
                    index=len(irregular),
                )
            )
    return entries, irregular


def load_resources_from(data_dir: Path | str = DATA_DIR) -> Resources:
    data_dir = Path(data_dir)
    rules_json = _read_json(data_dir / RULES_FILE)
    lex_json = _read_json(data_dir / LEXICON_FILE)
    paradigms = _build_paradigms(rules_json)
    entries, irregular = _build_lexicon(lex_json, paradigms)
    return Resources(
        paradigms=paradigms,
        entries=entries,
        irregular=irregular,
        guesser_config=dict(rules_json.get("guesser", {})),
        meta={"rules": rules_json.get("meta", {}), "lexicon": lex_json.get("meta", {})},
    )


@lru_cache(maxsize=4)
def load_resources(data_dir: str = str(DATA_DIR)) -> Resources:
    """Cached loader (resources are treated as read-only)."""
    return load_resources_from(data_dir)
