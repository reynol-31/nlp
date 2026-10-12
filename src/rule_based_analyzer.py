"""Analyzer A: rule-based morphological analysis by suffix stripping.

Algorithm for an input word *w* (after normalisation):

1. **Exception lookup** -- irregular / suppletive forms (किया, गया, मुझे, था ...)
   are looked up in the irregular-form table of the lexicon.
2. **Suffix stripping** -- every paradigm rule whose surface suffix *s* ends *w*
   proposes a decomposition ``w = stem + s``.  The lemma is reconstructed by
   *restoring* the paradigm's truncated ending: ``lemma = stem + truncate``
   (e.g. लड़कों = लड़क + ों  ->  लड़क + ा = लड़का).
3. **Lexicon validation** -- a hypothesis survives only if the reconstructed
   lemma is in the lexicon *with that paradigm*, and the lemma does not block
   the slot (करना has no regular *करा) or restrict its features (mass nouns).
4. **Ranking** -- irregular forms first, then longer suffix matches before
   shorter ones (longest-match principle), then paradigm/slot order from the
   rules file.  Duplicate analyses are removed.
5. **Unknown-word guessing (optional)** -- if nothing was validated, rules of
   *guessable* paradigms are applied without the lexicon check, constrained by
   the paradigm's lemma shape (``lemma_regex``) and a minimum stem length.
   Hypotheses are ranked by suffix length, then by a paradigm prior, and the
   best ``max_hypotheses`` are returned, flagged as guesses.
6. If still nothing, an UNKNOWN result is returned -- never a fabricated analysis.
"""

from __future__ import annotations

from collections import defaultdict

from .normalization import is_devanagari_word, normalize_word
from .resources import Resources, SuffixRule, load_resources
from .schema import Analysis, AnalyzerResult, dedupe


class RuleBasedAnalyzer:
    name = "Rule-based"

    def __init__(self, resources: Resources | None = None, enable_guesser: bool = True) -> None:
        self.res = resources or load_resources()
        self.enable_guesser = enable_guesser
        self._rules_by_surface: dict[str, list[SuffixRule]] = defaultdict(list)
        for rule in self.res.all_rules:
            self._rules_by_surface[rule.surface].append(rule)
        par_order = {pid: p.order for pid, p in self.res.paradigms.items()}
        for rules in self._rules_by_surface.values():
            rules.sort(key=lambda r: (par_order[r.paradigm], r.order))
        # longest suffix first; "" (zero suffix) last
        self._surfaces = sorted(self._rules_by_surface, key=lambda s: (-len(s), s))
        self._known_lemmas = self.res.known_lemmas()
        cfg = self.res.guesser_config
        self.max_hypotheses = int(cfg.get("max_hypotheses", 2))
        self.min_stem_length = int(cfg.get("min_stem_length", 2))

    @property
    def label(self) -> str:
        return self.name + ("" if self.enable_guesser else " (no guesser)")

    # ------------------------------------------------------------------
    def analyze(self, word: str) -> AnalyzerResult:
        norm = normalize_word(word)
        if not norm:
            return AnalyzerResult(self.label, word, norm, [], "Empty input.")
        if not is_devanagari_word(norm):
            return AnalyzerResult(self.label, word, norm, [], "Input is not a single Devanagari word.")

        trace: list[dict[str, str]] = []
        candidates: list[Analysis] = []

        # 1. exception table
        for f in self.res.irregular_by_form.get(norm, []):
            a = Analysis.from_tags(f.display_lemma, f.tags, suffix="", rule="IRREGULAR", source="exception")
            candidates.append(a)
        trace.append(
            {
                "step": "1. exception table",
                "suffix": "",
                "stem": "",
                "lemma hypothesis": "",
                "rules": "IRREGULAR",
                "outcome": f"{len(candidates)} irregular analysis(es)" if candidates else "no irregular form",
            }
        )

        # 2+3. suffix stripping with lexicon validation
        for surface in self._surfaces:
            if surface and not norm.endswith(surface):
                continue
            stem = norm[: len(norm) - len(surface)] if surface else norm
            if not stem:
                continue
            by_paradigm: dict[str, list[SuffixRule]] = defaultdict(list)
            for rule in self._rules_by_surface[surface]:
                by_paradigm[rule.paradigm].append(rule)
            for pid, rules in by_paradigm.items():
                lemma = stem + rules[0].truncate
                entries = [e for e in self.res.by_lemma.get(lemma, []) if e.paradigm == pid]
                if not entries:
                    outcome = "rejected: lemma not in lexicon with this paradigm"
                else:
                    accepted, blocked = [], []
                    for entry in entries:
                        for rule in rules:
                            if entry.allows(rule):
                                accepted.append(rule.slot)
                                candidates.append(
                                    Analysis.from_tags(
                                        entry.display, rule.tags, suffix=surface, rule=rule.rule_id, source="lexicon"
                                    )
                                )
                            else:
                                blocked.append(rule.slot)
                    outcome = f"accepted: {', '.join(accepted)}" if accepted else ""
                    if blocked:
                        outcome += ("; " if outcome else "") + f"blocked by lexicon: {', '.join(blocked)}"
                trace.append(
                    {
                        "step": "2. suffix rule",
                        "suffix": surface or "∅",
                        "stem": stem,
                        "lemma hypothesis": lemma,
                        "rules": pid,
                        "outcome": outcome,
                    }
                )

        analyses = dedupe(candidates)
        if analyses:
            return AnalyzerResult(
                self.label, word, norm, analyses,
                f"{len(analyses)} lexicon-validated analysis(es).", trace,
            )

        # 5. unknown-word guessing
        if self.enable_guesser:
            guesses = self._guess(norm, trace)
            if guesses:
                return AnalyzerResult(
                    self.label, word, norm, guesses,
                    "Lemma not in lexicon: analyses are GUESSES from suffix rules.", trace, guessed=True,
                )
        return AnalyzerResult(
            self.label, word, norm, [],
            "Unknown word: no rule produced a lemma that is in the lexicon"
            + ("" if self.enable_guesser else " (guesser disabled)") + ".",
            trace,
        )

    # ------------------------------------------------------------------
    def _guess(self, norm: str, trace: list[dict[str, str]]) -> list[Analysis]:
        hyps: dict[tuple[str, str], list[SuffixRule]] = defaultdict(list)
        for surface in self._surfaces:
            if not surface or not norm.endswith(surface):
                continue
            stem = norm[: len(norm) - len(surface)]
            if len(stem) < self.min_stem_length:
                continue
            for rule in self._rules_by_surface[surface]:
                par = self.res.paradigms[rule.paradigm]
                if not par.guessable:
                    continue
                lemma = stem + rule.truncate
                if lemma in self._known_lemmas:  # lexicon is authoritative for known lemmas
                    continue
                if not par.lemma_regex.match(lemma):
                    continue
                hyps[(lemma, rule.paradigm)].append(rule)

        def score(item):
            (lemma, pid), rules = item
            par = self.res.paradigms[pid]
            return (-len(rules[0].surface), -par.guess_prior, par.order)

        ranked = sorted(hyps.items(), key=score)
        out: list[Analysis] = []
        for rank, ((lemma, pid), rules) in enumerate(ranked):
            kept = rank < self.max_hypotheses
            trace.append(
                {
                    "step": "3. guess",
                    "suffix": rules[0].surface,
                    "stem": lemma[: len(lemma) - len(rules[0].truncate)] if rules[0].truncate else lemma,
                    "lemma hypothesis": lemma,
                    "rules": pid,
                    "outcome": (f"kept (rank {rank + 1}, prior {self.res.paradigms[pid].guess_prior})"
                                if kept else f"discarded (rank {rank + 1})"),
                }
            )
            if kept:
                for rule in rules:
                    out.append(Analysis.from_tags(lemma, rule.tags, suffix=rule.surface, rule=rule.rule_id, source="guess"))
        return dedupe(out)
