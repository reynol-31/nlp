"""Canonical morphological analysis format shared by both analyzers and the evaluator.

An analysis is a lemma plus a fixed set of feature slots.  A slot that does
not apply to a word (e.g. *tense* on a noun) holds the explicit value ``NA``.

Lexical-side string (used by the FST output tape and by the gold annotation
file)::

    लड़का+NOUN+MASC+PL+OBL

The lemma comes first, followed by the non-NA feature values in the fixed
``FEATURES`` order.  Every tag value is unique across features, so the string
can be parsed back unambiguously.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .normalization import normalize_word

NA = "NA"

# Order matters: it defines the lexical-string order and table column order.
FEATURES: tuple[str, ...] = (
    "pos",
    "verbform",
    "tense",
    "aspect",
    "mood",
    "person",
    "gender",
    "number",
    "case",
)
KEY_FIELDS: tuple[str, ...] = ("lemma",) + FEATURES

FEATURE_VALUES: dict[str, tuple[str, ...]] = {
    "pos": ("NOUN", "PRON", "ADJ", "VERB", "AUX"),
    # CONV (conjunctive participle -कर) is annotated in the gold data but not
    # produced by either analyzer: it is outside the implemented scope.
    "verbform": ("INF", "PART", "FIN", "CONV"),
    "tense": ("PRES", "PAST", "FUT"),
    "aspect": ("HAB", "PERF"),
    "mood": ("IND", "SUBJ", "IMP"),
    "person": ("1", "2", "3"),
    "gender": ("MASC", "FEM"),
    "number": ("SG", "PL"),
    # VOC (vocative) likewise exists only in gold annotations of out-of-scope items.
    "case": ("DIR", "OBL", "ACC_DAT", "VOC"),
}

TAG_TO_FEATURE: dict[str, str] = {
    value: feat for feat, values in FEATURE_VALUES.items() for value in values
}

FEATURE_LABELS: dict[str, str] = {
    "lemma": "Lemma",
    "pos": "POS",
    "verbform": "Verb form",
    "tense": "Tense",
    "aspect": "Aspect",
    "mood": "Mood",
    "person": "Person",
    "gender": "Gender",
    "number": "Number",
    "case": "Case",
}

VALUE_LABELS: dict[str, str] = {
    "NOUN": "noun", "PRON": "pronoun", "ADJ": "adjective", "VERB": "verb", "AUX": "auxiliary",
    "INF": "infinitive", "PART": "participle", "FIN": "finite", "CONV": "conjunctive participle",
    "PRES": "present", "PAST": "past", "FUT": "future",
    "HAB": "habitual (imperfective)", "PERF": "perfective",
    "IND": "indicative", "SUBJ": "subjunctive", "IMP": "imperative",
    "1": "1st person", "2": "2nd person", "3": "3rd person",
    "MASC": "masculine", "FEM": "feminine", "SG": "singular", "PL": "plural",
    "DIR": "direct", "OBL": "oblique", "ACC_DAT": "accusative/dative", "VOC": "vocative",
    NA: "not applicable",
}


class SchemaError(ValueError):
    """Raised when tags or feature values violate the canonical schema."""


def features_from_tags(tags: Iterable[str]) -> dict[str, str]:
    """Map a bag of tag values (e.g. ``["NOUN", "MASC", "PL", "OBL"]``) to feature slots."""
    feats = {f: NA for f in FEATURES}
    for tag in tags:
        tag = tag.strip()
        if not tag:
            continue
        feat = TAG_TO_FEATURE.get(tag)
        if feat is None:
            raise SchemaError(f"unknown tag {tag!r}")
        if feats[feat] != NA and feats[feat] != tag:
            raise SchemaError(f"conflicting values for {feat}: {feats[feat]!r} and {tag!r}")
        feats[feat] = tag
    if feats["pos"] == NA:
        raise SchemaError(f"analysis has no POS tag: {list(tags)!r}")
    return feats


@dataclass(frozen=True)
class Analysis:
    """One morphological analysis in canonical form.

    Only ``lemma`` and the ``FEATURES`` slots take part in equality checks
    during evaluation (see :meth:`key`).  ``suffix``, ``rule`` and ``source``
    are explanatory metadata.
    """

    lemma: str
    pos: str
    verbform: str = NA
    tense: str = NA
    aspect: str = NA
    mood: str = NA
    person: str = NA
    gender: str = NA
    number: str = NA
    case: str = NA
    suffix: str = field(default="", compare=False)
    rule: str = field(default="", compare=False)
    source: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        for feat in FEATURES:
            value = getattr(self, feat)
            if value != NA and value not in FEATURE_VALUES[feat]:
                raise SchemaError(f"invalid value {value!r} for feature {feat!r}")
        if self.pos == NA:
            raise SchemaError("pos must not be NA")
        if not self.lemma:
            raise SchemaError("lemma must not be empty")

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_tags(cls, lemma: str, tags: Iterable[str], **meta: str) -> "Analysis":
        return cls(lemma=lemma, **features_from_tags(tags), **meta)

    @classmethod
    def from_lexical(cls, lexical: str, **meta: str) -> "Analysis":
        """Parse ``लड़का+NOUN+MASC+PL+OBL``."""
        parts = lexical.split("+")
        if len(parts) < 2 or not parts[0]:
            raise SchemaError(f"malformed lexical string {lexical!r}")
        return cls.from_tags(parts[0], parts[1:], **meta)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Analysis":
        if "lemma" not in data or "pos" not in data:
            raise SchemaError(f"analysis needs 'lemma' and 'pos': {data!r}")
        kwargs = {f: str(data.get(f, NA) or NA) for f in FEATURES}
        for meta in ("suffix", "rule", "source"):
            if data.get(meta):
                kwargs[meta] = str(data[meta])
        return cls(lemma=str(data["lemma"]), **kwargs)

    # -- views ----------------------------------------------------------------
    def key(self) -> tuple[str, ...]:
        """Comparison key: normalised lemma + all feature values."""
        return (normalize_word(self.lemma),) + tuple(getattr(self, f) for f in FEATURES)

    def features(self) -> dict[str, str]:
        return {f: getattr(self, f) for f in FEATURES}

    def tags(self) -> list[str]:
        return [getattr(self, f) for f in FEATURES if getattr(self, f) != NA]

    def lexical_string(self) -> str:
        return "+".join([self.lemma] + self.tags())

    def to_dict(self, with_meta: bool = True) -> dict[str, str]:
        d = {"lemma": self.lemma, **self.features()}
        if with_meta:
            d.update(suffix=self.suffix, rule=self.rule, source=self.source)
        return d

    def short(self) -> str:
        """Compact display string, e.g. ``लड़का NOUN MASC PL OBL``."""
        return " ".join([self.lemma] + self.tags())


def dedupe(analyses: Iterable[Analysis]) -> list[Analysis]:
    """Remove analyses with identical keys, keeping the first (highest-ranked)."""
    seen: set[tuple[str, ...]] = set()
    out: list[Analysis] = []
    for a in analyses:
        k = a.key()
        if k not in seen:
            seen.add(k)
            out.append(a)
    return out


@dataclass
class AnalyzerResult:
    """What an analyzer returns for one input word."""

    analyzer: str
    word: str
    normalized: str
    analyses: list[Analysis]
    message: str = ""
    trace: list[dict[str, Any]] = field(default_factory=list)
    guessed: bool = False  # True when analyses come from unknown-word guessing

    @property
    def known(self) -> bool:
        return bool(self.analyses)

    @property
    def status(self) -> str:
        if not self.analyses:
            return "UNKNOWN"
        return "GUESSED" if self.guessed else "KNOWN"

    @property
    def primary(self) -> Analysis | None:
        return self.analyses[0] if self.analyses else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "analyzer": self.analyzer,
            "word": self.word,
            "normalized": self.normalized,
            "status": self.status,
            "primary": self.primary.to_dict() if self.primary else None,
            "analyses": [a.to_dict() for a in self.analyses],
            "message": self.message,
        }
