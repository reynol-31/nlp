"""Analyzer B: a finite-state transducer (FST) for Hindi inflection.

The transducer is a directed graph of numbered states.  Every arc carries an
*input* symbol (one Unicode code point of the surface word, or ε = "" for an
epsilon move) and an *output* string on the lexical tape.  Analysis = reading
the surface word on the input tape from the start state; every path that
consumes the whole word and ends in an accepting state writes one lexical
string such as ``लड़का+NOUN+MASC+PL+OBL`` on the output tape.

Topology (compiled from the same lexicon + rules files as the rule-based analyzer)::

    START ─ε→ STEM-ROOT ─ल:ल→ ─ड:ड→ ─़:़→ ─क:क→ (stem end of लड़का)
                                                    │ ε:ε  (continuation class N_M_AA)
                                                    ▼
                                         SUFFIX-ROOT ─ो:ε→ ─ं:ε→ ─ε:"ा"→ ─ε:"+NOUN+MASC+PL+OBL"→ FINAL
                                              (surface suffix)  (restore lemma ending)  (emit tags)
    START ─ε→ IRREGULAR-ROOT ─क:ε→ ─ि:ε→ ─य:ε→ ─ा:ε→ ─ε:"करना+VERB+PART+PERF+MASC+SG"→ FINAL

* Stems are stored in a letter trie that copies characters to the output.
* Each (paradigm, blocked-slots, restrictions) combination is compiled once
  into a *continuation network* (as in Xerox lexc continuation classes).
  Suffix arcs consume surface characters and write nothing; an ε-arc then
  restores the lemma ending that the stem had dropped (the truncation), and a
  final ε-arc writes the feature tags.
* Irregular forms are separate full-word paths.

The FST is non-deterministic (ambiguity = several accepting paths) and acyclic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from .normalization import is_devanagari_word, normalize_word
from .resources import Resources, SuffixRule, load_resources
from .schema import Analysis, AnalyzerResult, SchemaError, dedupe

EPS = ""


@dataclass(frozen=True)
class Arc:
    src: int
    inp: str  # one code point, or EPS
    out: str  # output string (may be EPS)
    dst: int
    kind: str  # "stem" | "link" | "suffix" | "restore" | "tags" | "irregular" | "enter"
    info: str = ""  # rule id / paradigm, for explanation


@dataclass
class Path:
    arcs: list[Arc]

    @property
    def output(self) -> str:
        return "".join(a.out for a in self.arcs)

    @property
    def states(self) -> list[int]:
        return [self.arcs[0].src] + [a.dst for a in self.arcs] if self.arcs else []


@dataclass
class FST:
    """A minimal, explicit finite-state transducer."""

    start: int = 0
    finals: set[int] = field(default_factory=set)
    state_labels: dict[int, str] = field(default_factory=dict)
    # src -> input symbol -> arcs (insertion-ordered)
    _arcs: dict[int, dict[str, list[Arc]]] = field(default_factory=dict)
    n_arcs: int = 0

    def add_state(self, label: str = "") -> int:
        sid = len(self.state_labels)
        self.state_labels[sid] = label
        self._arcs[sid] = {}
        return sid

    def add_arc(self, src: int, inp: str, out: str, dst: int, kind: str, info: str = "") -> Arc:
        if len(inp) > 1:
            raise ValueError("input symbols are single code points (or ε)")
        arc = Arc(src, inp, out, dst, kind, info)
        self._arcs[src].setdefault(inp, []).append(arc)
        self.n_arcs += 1
        return arc

    def arcs_from(self, state: int, inp: str | None = None) -> list[Arc]:
        table = self._arcs.get(state, {})
        if inp is None:
            return [a for arcs in table.values() for a in arcs]
        return table.get(inp, [])

    def find_arc(self, src: int, inp: str, out: str, kind: str) -> Arc | None:
        for a in self._arcs[src].get(inp, []):
            if a.out == out and a.kind == kind:
                return a
        return None

    @property
    def n_states(self) -> int:
        return len(self.state_labels)

    @property
    def input_alphabet(self) -> set[str]:
        return {sym for t in self._arcs.values() for sym in t if sym != EPS}

    # -- traversal --------------------------------------------------------
    def transduce(self, word: str, max_eps: int = 16) -> list[Path]:
        """All accepting paths that consume *word* completely (depth-first, arc order)."""
        results: list[Path] = []
        n = len(word)

        def dfs(state: int, pos: int, arcs: list[Arc], eps_run: int) -> None:
            if pos == n and state in self.finals:
                results.append(Path(list(arcs)))
            if eps_run < max_eps:  # guard; the compiled graph has no ε-cycles anyway
                for arc in self.arcs_from(state, EPS):
                    arcs.append(arc)
                    dfs(arc.dst, pos, arcs, eps_run + 1)
                    arcs.pop()
            if pos < n:
                for arc in self.arcs_from(state, word[pos]):
                    arcs.append(arc)
                    dfs(arc.dst, pos + 1, arcs, 0)
                    arcs.pop()

        dfs(self.start, 0, [], 0)
        return results

    def longest_prefix(self, word: str) -> tuple[int, list[Arc]]:
        """For rejected words: how many input symbols could be consumed at most, and along which path."""
        best: tuple[int, list[Arc]] = (0, [])
        n = len(word)

        def dfs(state: int, pos: int, arcs: list[Arc], eps_run: int) -> None:
            nonlocal best
            if pos > best[0]:
                best = (pos, list(arcs))
            if eps_run < 16:
                for arc in self.arcs_from(state, EPS):
                    arcs.append(arc)
                    dfs(arc.dst, pos, arcs, eps_run + 1)
                    arcs.pop()
            if pos < n:
                for arc in self.arcs_from(state, word[pos]):
                    arcs.append(arc)
                    dfs(arc.dst, pos + 1, arcs, 0)
                    arcs.pop()

        dfs(self.start, 0, [], 0)
        return best

    def generate(self, lexical: str) -> list[str]:
        """Inverse direction: lexical string -> surface forms (output tape read as input)."""
        found: list[str] = []

        def dfs(state: int, pos: int, surface: list[str], depth: int) -> None:
            if depth > 200:
                return
            if pos == len(lexical) and state in self.finals:
                found.append("".join(surface))
            for arc in self.arcs_from(state):
                if arc.out and not lexical.startswith(arc.out, pos):
                    continue
                surface.append(arc.inp)
                dfs(arc.dst, pos + len(arc.out), surface, depth + 1)
                surface.pop()

        dfs(self.start, 0, [], 0)
        return list(dict.fromkeys(found))

    def is_acyclic(self) -> bool:
        WHITE, GREY, BLACK = 0, 1, 2
        color = dict.fromkeys(self.state_labels, WHITE)
        for root in self.state_labels:
            if color[root] != WHITE:
                continue
            stack: list[tuple[int, Iterator[Arc]]] = [(root, iter(self.arcs_from(root)))]
            color[root] = GREY
            while stack:
                node, it = stack[-1]
                arc = next(it, None)
                if arc is None:
                    color[node] = BLACK
                    stack.pop()
                elif color[arc.dst] == GREY:
                    return False
                elif color[arc.dst] == WHITE:
                    color[arc.dst] = GREY
                    stack.append((arc.dst, iter(self.arcs_from(arc.dst))))
        return True


# ----------------------------------------------------------------------------
def compile_fst(res: Resources) -> FST:
    """Compile lexicon + paradigms + irregular forms into an FST."""
    fst = FST()
    fst.start = fst.add_state("START")
    final = fst.add_state("FINAL (accept)")
    fst.finals.add(final)
    stem_root = fst.add_state("STEM-LEXICON root")
    irr_root = fst.add_state("IRREGULAR-LEXICON root")
    fst.add_arc(fst.start, EPS, EPS, stem_root, "enter", "regular stems")
    fst.add_arc(fst.start, EPS, EPS, irr_root, "enter", "irregular full forms")

    def trie_step(src: int, ch: str, out: str, kind: str, label: str) -> int:
        arc = fst.find_arc(src, ch, out, kind)
        if arc is not None:
            return arc.dst
        dst = fst.add_state(label)
        fst.add_arc(src, ch, out, dst, kind)
        return dst

    continuation_cache: dict[tuple, int] = {}

    def continuation(pid: str, rules: tuple[SuffixRule, ...]) -> int:
        key = (pid, tuple(r.slot for r in rules))
        if key in continuation_cache:
            return continuation_cache[key]
        par = res.paradigms[pid]
        root = fst.add_state(f"SUFFIX root [{pid}]")
        continuation_cache[key] = root
        for rule in rules:
            node = root
            for i, ch in enumerate(rule.surface):
                node = trie_step(node, ch, EPS, "suffix", f"[{pid}] suffix '{rule.surface[: i + 1]}'")
            if par.truncate:
                restored = fst.add_state(f"[{pid}] restored '{par.truncate}'")
                fst.add_arc(node, EPS, par.truncate, restored, "restore", rule.rule_id)
                node = restored
            fst.add_arc(node, EPS, "+" + "+".join(_ordered_tags(rule)), final, "tags", rule.rule_id)
        return root

    # regular stems
    for entry in res.entries:
        stem = res.stem_of(entry)
        node = stem_root
        for i, ch in enumerate(stem):
            node = trie_step(node, ch, ch, "stem", f"stem '{stem[: i + 1]}'")
        allowed = tuple(r for r in res.paradigms[entry.paradigm].rules if entry.allows(r))
        cont = continuation(entry.paradigm, allowed)
        if not any(a.kind == "link" and a.dst == cont for a in fst.arcs_from(node, EPS)):
            fst.add_arc(node, EPS, EPS, cont, "link", f"{entry.display} -> {entry.paradigm}")

    # irregular full forms: consume surface, emit the whole lexical string at the end
    for form in res.irregular:
        node = irr_root
        for i, ch in enumerate(form.form):
            node = trie_step(node, ch, EPS, "irregular", f"irregular '{form.form[: i + 1]}'")
        out = "+".join([form.lemma, *_ordered_tags_from(form.tags)])
        if fst.find_arc(node, EPS, out, "tags") is None:
            fst.add_arc(node, EPS, out, final, "tags", f"IRREGULAR: {form.display_form}")
    return fst


def _ordered_tags_from(tags) -> list[str]:
    a = Analysis.from_tags("x", tags)
    return a.tags()


def _ordered_tags(rule: SuffixRule) -> list[str]:
    return _ordered_tags_from(rule.tags)


# ----------------------------------------------------------------------------
class FSTAnalyzer:
    name = "FST"
    label = "FST"

    def __init__(self, resources: Resources | None = None) -> None:
        self.res = resources or load_resources()
        self.fst = compile_fst(self.res)

    def stats(self) -> dict[str, int]:
        return {
            "states": self.fst.n_states,
            "arcs": self.fst.n_arcs,
            "accepting_states": len(self.fst.finals),
            "input_alphabet": len(self.fst.input_alphabet),
        }

    def analyze(self, word: str) -> AnalyzerResult:
        norm = normalize_word(word)
        if not norm:
            return AnalyzerResult(self.label, word, norm, [], "Empty input.")
        if not is_devanagari_word(norm):
            return AnalyzerResult(self.label, word, norm, [], "Input is not a single Devanagari word.")
        paths = self.fst.transduce(norm)
        analyses: list[Analysis] = []
        traces: list[dict] = []
        for i, path in enumerate(paths):
            try:
                a = self._to_analysis(path)
            except SchemaError:  # pragma: no cover - would indicate a compiler bug
                continue
            analyses.append(a)
            traces.append({"path": i + 1, "output": path.output, "arcs": path.arcs})
        analyses = dedupe(analyses)
        if analyses:
            return AnalyzerResult(
                self.label, word, norm, analyses,
                f"{len(paths)} accepting path(s).", traces,
            )
        consumed, arcs = self.fst.longest_prefix(norm)
        if consumed == len(norm):
            why = f"all {consumed} symbols were consumed but no path ends in an accepting state"
        else:
            why = (f"stuck after consuming {consumed} of {len(norm)} symbols "
                   f"('{norm[:consumed]}' | '{norm[consumed:]}'): no arc for '{norm[consumed]}'")
        return AnalyzerResult(
            self.label, word, norm, [], f"Rejected by the FST: {why}.",
            [{"path": 0, "output": "", "arcs": arcs, "rejected": True}],
        )

    def _to_analysis(self, path: Path) -> Analysis:
        out = path.output
        lemma, _, rest = out.partition("+")
        suffix = "".join(a.inp for a in path.arcs if a.kind == "suffix")
        rule = next((a.info for a in reversed(path.arcs) if a.kind == "tags"), "")
        source = "fst-irregular" if any(a.kind == "irregular" for a in path.arcs) else "fst"
        display = self.res.display_lemma.get(lemma, lemma)
        return Analysis.from_tags(display, rest.split("+"), suffix=suffix, rule=rule, source=source)

    def generate(self, lexical: str) -> list[str]:
        """Surface forms for a lexical string such as ``लड़का+NOUN+MASC+PL+OBL``."""
        try:
            a = Analysis.from_lexical(lexical.strip())
        except SchemaError:
            return []
        canon = "+".join([normalize_word(a.lemma), *a.tags()])
        return self.fst.generate(canon)

    # -- explanation helpers --------------------------------------------
    def path_table(self, arcs: list[Arc]) -> list[dict[str, str]]:
        rows = []
        for i, a in enumerate(arcs, 1):
            rows.append(
                {
                    "#": str(i),
                    "from": f"q{a.src}",
                    "input": a.inp or "ε",
                    "output": a.out or "ε",
                    "to": f"q{a.dst}" + (" (final)" if a.dst in self.fst.finals else ""),
                    "arc type": a.kind,
                    "to-state description": self.fst.state_labels.get(a.dst, ""),
                }
            )
        return rows

    def path_dot(self, arcs: list[Arc], rejected: bool = False) -> str:
        """Graphviz DOT source for one path (rendered in the UI)."""
        lines = [
            "digraph path {",
            "rankdir=LR; bgcolor=transparent;",
            'node [shape=circle, fontsize=11, fontname="Nirmala UI, Mangal, sans-serif"];',
            'edge [fontsize=11, fontname="Nirmala UI, Mangal, sans-serif"];',
        ]
        states = [arcs[0].src] + [a.dst for a in arcs] if arcs else [self.fst.start]
        for s in dict.fromkeys(states):
            shape = "doublecircle" if s in self.fst.finals else "circle"
            lines.append(f'q{s} [label="q{s}", shape={shape}];')
        for a in arcs:
            inp = a.inp or "ε"
            out = (a.out or "ε").replace('"', '\\"')
            lines.append(f'q{a.src} -> q{a.dst} [label="{inp} : {out}"];')
        if rejected:
            last = states[-1]
            lines.append('X [label="✗ no arc", shape=box, color=red, fontcolor=red];')
            lines.append(f"q{last} -> X [style=dashed, color=red];")
        lines.append("}")
        return "\n".join(lines)
