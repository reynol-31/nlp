# Morphological Analyzer for Hindi
### A Comparative Evaluation of Rule-Based and Finite-State Transducer Approaches

## Abstract

This project implements two morphological analyzers for a documented subset of Hindi inflection:
a **rule-based suffix-stripping analyzer** and a **finite-state transducer (FST)**. Both are
compiled from the same lexicon (≈120 lemmas) and the same paradigm table (18 paradigms, 197 suffix
rules, plus 49 irregular full forms). They are evaluated against a manually curated gold standard
of 366 Hindi words (564 gold analyses), split by lemma into a development set (170) and a held-out test
set (196). On the held-out test set the rule-based analyzer reaches **95.92 % exact-analysis accuracy**
(precision 97.41 %, F1 96.66 %) and the FST **89.29 %** (precision 100 %, F1 94.34 %). On words whose
lemma is in the lexicon the two give identical results. The whole gap comes from unknown-word handling:
the rule-based analyzer guesses the analysis of out-of-lexicon words, and the FST rejects them.

> **Scope disclaimer.** This is an educational implementation of a *subset* of Hindi inflectional
> morphology. It is not a complete or production-grade Hindi morphological analyzer.

---

## 1. Problem statement and objectives

Hindi marks gender, number and case on nouns, adjectives and pronouns, and marks aspect, mood, tense, person,
gender and number on verbs. Search, MT, parsing and language-learning tools need to map a surface form
such as **लड़कों** to its lemma and features (**लड़का + NOUN + MASC + PL + OBL**).

Objectives:
1. Implement a genuine rule-based analyzer (decomposition and lexicon validation, not just lookup).
2. Implement a genuine FST that performs explicit state transitions.
3. Give both the same linguistic resources and the same canonical output format.
4. Evaluate both on the same gold standard with accuracy, precision, recall and F1.
5. Present the results in an interactive Streamlit app that you can explain in a viva.

## 2. Scope (what is covered)

| Class | Paradigms | Examples |
|---|---|---|
| Nouns | masc ‑आ (inflecting), masc ‑आ invariant, masc consonant, masc ‑ई, masc ‑ऊ, fem ‑ई, fem ‑इया, fem consonant, fem ‑आ, fem ‑ऊ, fem ‑उ | लड़का/लड़के/लड़कों, राजा/राजाओं, आदमियों, लड़कियाँ, किताबें, भाषाएँ, बहुएँ |
| Pronouns | suppletive personal pronouns (direct, oblique, fused acc/dat) | मैं, मुझ, मुझे, उन्हें, आप |
| Adjectives | inflecting ‑आ, ‑या (नया), invariant | अच्छा/अच्छे/अच्छी, नए/नई, सुंदर |
| Verbs | consonant stems, ‑आ/‑ओ stems, ‑ई stems (पीना), देना/लेना; infinitive (dir/obl), habitual & perfective participles, subjunctive, imperative (incl. polite ‑इए), future | पढ़ता, खाया, पिया, दीजिए, खाएगा, पढ़ेंगे |
| Irregular | करना, जाना, होना (perfective, contracted subjunctive/future, copula है/था) | किया, गया, हुआ, हो, होगा, हैं, थीं |

**Inflection vs. derivation.** Only inflection is implemented. Derivational forms (अच्छाई, लड़कपन,
मिठास, दोस्ती) are in the gold standard as deliberately *out-of-scope* items. Their gold lemma is the
derived word itself, and we do not decompose them.

Not implemented: vocative, conjunctive participle ‑कर, oblique participles, possessive pronouns,
postpositions, numerals, causative formation as a process, honorific/politeness features.

## 3. Architecture

```
hindi-morphological-analyzer/
├── app.py                         Streamlit UI (5 pages)
├── requirements.txt               pinned dependencies
├── data/
│   ├── lexicon.json               lemmas + paradigm class + irregular full forms   (shared)
│   ├── morphology_rules.json      paradigms, suffix rules, verb slot table, guesser priors (shared)
│   ├── annotations/gold_annotations.txt   human-written gold annotations (source)
│   ├── gold_standard.json         generated evaluation reference (with dev/test split)
│   └── dataset_metadata.json      provenance, schema, split, statistics, correction log
├── src/
│   ├── normalization.py           Unicode / spelling normalisation
│   ├── schema.py                  canonical Analysis + AnalyzerResult
│   ├── resources.py               loads & validates lexicon and rules
│   ├── rule_based_analyzer.py     Analyzer A
│   ├── fst_analyzer.py            Analyzer B (FST class, compiler, analyzer)
│   ├── dataset_loader.py          annotation parser, lemma-grouped split, gold loader
│   ├── evaluation.py              metrics, evaluation runner, exports
│   └── error_analysis.py          agreement / error-type analysis
├── scripts/
│   ├── build_gold.py              annotations -> gold_standard.json
│   └── evaluate.py                command-line evaluation -> results/
├── tests/                         pytest suite (107 tests)
└── results/                       generated CSV / JSON results
```

The analyzers do not depend on Streamlit, so you can use them from plain Python:

```python
from src.rule_based_analyzer import RuleBasedAnalyzer
from src.fst_analyzer import FSTAnalyzer
print(RuleBasedAnalyzer().analyze("लड़कों").primary.short())   # लड़का NOUN MASC PL OBL
print(FSTAnalyzer().generate("पीना+VERB+FIN+FUT+IND+3+MASC+SG"))  # ['पिएगा']
```

## 4. Canonical representation and linguistic assumptions

Every analysis is a lemma plus nine feature slots. `NA` means *not applicable*:

| Field | Values |
|---|---|
| pos | NOUN, PRON, ADJ, VERB, AUX |
| verbform | INF, PART, FIN (CONV only in gold for out-of-scope ‑कर) |
| tense | PRES, PAST, FUT |
| aspect | HAB, PERF |
| mood | IND, SUBJ, IMP |
| person | 1, 2, 3 |
| gender | MASC, FEM |
| number | SG, PL |
| case | DIR, OBL, ACC_DAT (VOC only in gold for out-of-scope vocatives) |

Lexical string: `लड़का+NOUN+MASC+PL+OBL` (lemma, then the non-NA values in a fixed order).

Assumptions and decisions:
* **Context-free annotation.** A word gets *every* analysis that is possible in isolation. For example, लड़के has
  2 analyses (sg obl / pl dir), अच्छी has 4, and पढ़ें has 3.
* Infinitives are treated as verbal nouns: case DIR/OBL, no gender/number agreement.
* Participles carry gender/number but no person; tense is carried by auxiliaries (है/था), which are analysed separately as AUX.
* तुम and आप are grammatically plural (person 2, PL); politeness is not encoded (पढ़ो and पढ़िए are both IMP 2 PL, with different rule labels).
* Invariant adjectives (सुंदर) have no agreement features (all NA).
* Mass noun सोना 'gold' is restricted to the singular (a lexical constraint used by both analyzers).

### Normalisation (`src/normalization.py`)
NFC (this also canonicalises precomposed nukta letters), removal of ZWJ/ZWNJ/ZWSP/BOM, stripping of
whitespace and punctuation (incl. ।), folding of Perso-Arabic nukta (ख़राब = खराब; ड़/ढ़ are kept),
chandrabindu → anusvara (लड़कियाँ = लड़कियां), and y-glide folding after vowels (नये = नए, खायी = खाई).
The same function is applied to the input, the lexicon, the rules and the gold data.

## 5. Analyzer A: rule-based (`src/rule_based_analyzer.py`)

1. **Exception table**: irregular and suppletive forms (किया, गया, था, मुझे …).
2. **Suffix stripping**: every rule whose surface suffix ends the word proposes `word = stem + suffix`.
3. **Lemma restoration**: `lemma = stem + truncate` (e.g. लड़क + ों → लड़क + ा = **लड़का**; पि + या → प + ीना = **पीना**).
4. **Lexicon validation**: the lemma must exist *with that paradigm*. Lexical constraints block slots
   (करना has no regular *करा) and restrict features (mass nouns).
5. **Ranking**: irregular forms first, then the longest suffix first, then the paradigm/slot order from the rules file. Duplicates are removed.
6. **Unknown-word guesser** (optional; on by default): if nothing is validated, guessable paradigms are
   applied without the lexicon, constrained by a lemma-shape regex and a minimum stem length, and ranked
   by suffix length and then by a paradigm prior. The top 2 hypotheses are returned and marked **GUESSED**.
7. Otherwise the analyzer returns **UNKNOWN**, with no fabricated analysis.

Every step is recorded in a trace, which the UI displays.

**Limitations:** guesses cannot separate noun/adjective/verb readings that share a suffix (‑े, ‑ी, ‑ा), and they
cannot tell whether a dropped ‑आ must be restored (पत्तों → पत्त or पत्ता?).

## 6. Analyzer B: finite-state transducer (`src/fst_analyzer.py`)

The `FST` class is an explicit graph: numbered states, one start state, one accepting state, and arcs
`(src, input, output, dst)`. Input symbols are single Unicode code points or ε. The output is a string,
which may be a multi-character tag such as `+NOUN+MASC+PL+OBL`.

```
START ─ε→ STEM-ROOT ─ल:ल→ ─ड:ड→ ─़:़→ ─क:क→ (end of stem लड़क)
                                              │ ε:ε   link to continuation class N_M_AA
                                              ▼
                                   SUFFIX-ROOT ─ो:ε→ ─ं:ε→ ─ε:ा→ ─ε:+NOUN+MASC+PL+OBL→ FINAL
                                       (consume suffix)     (restore -ा)  (emit tags)
START ─ε→ IRREGULAR-ROOT ─क:ε→ ─ि:ε→ ─य:ε→ ─ा:ε→ ─ε:करना+VERB+PART+PERF+MASC+SG→ FINAL
```

* **Compilation:** stems go into a letter trie whose arcs copy each character. Each (paradigm, allowed slots)
  combination is compiled once into a continuation network, like *lexc* continuation classes. Suffix arcs
  consume characters, a *restore* arc writes the lemma ending back, and a *tag* arc writes the features.
  Irregular forms are separate full-word paths.
* **Traversal:** depth-first search over (state, input position), following ε-arcs and arcs whose input
  symbol matches the next character. Every path that consumes the whole word and ends in the accepting
  state produces one analysis. Ambiguity means several accepting paths.
* **Rejection:** when no path accepts, the analyzer reports the longest prefix it could consume and the
  character that had no arc (e.g. `stuck after 'लड़कों' | 'ा'`).
* **Generation:** the same graph runs in reverse, reading the output tape and producing the input tape
  (`लड़का+NOUN+MASC+PL+OBL → लड़कों`).
* Size: **876 states, 1 313 arcs, 50 input symbols**. The graph is acyclic (checked by a test).

`tests/test_fst.py::test_removing_a_transition_removes_the_analysis` proves that the output really comes
from traversal: deleting one tag arc makes लड़कों unanalysable, while लड़के still works.

## 7. Fair comparison

* Both analyzers are built from the **same** `Resources` object (lexicon + rules + irregular forms).
* Both return the same `Analysis` objects.
* `test_fst_and_rules_are_extensionally_equivalent` generates *every* form that the paradigms allow
  (all lemmas × all slots + irregular forms). It checks that the rule-based analyzer *without guesser* and
  the FST return identical analysis sets for each form.
* **Disclosed difference:** only the rule-based analyzer has an unknown-word guesser. It is a common
  feature of rule-based stemmers, while a lexc-style FST is closed-lexicon by default. The evaluation also
  reports the **ablation** "Rule-based (no guesser)" so that the effect is isolated.

## 8. Dataset and gold standard

* **Provenance:** manually curated for this project (`source = manual` on every item). No external corpus
  was used, and no item is claimed to come from one. UD_Hindi-HDTB was considered but **not accessed or used** (see
  `data/dataset_metadata.json`). The annotations were drafted with AI assistance and should be
  reviewed by a fluent Hindi speaker before any use beyond coursework.
* **Source → reference:** `data/annotations/gold_annotations.txt` (one line per word, e.g.
  `लड़के :: noun_regular :: लड़का NOUN MASC SG OBL | लड़का NOUN MASC PL DIR`) is validated and converted
  by `scripts/build_gold.py` into `data/gold_standard.json`.
* **Size:** 366 words, 564 gold analyses, 128 ambiguous words.

| Category | dev | test |
|---|---|---|
| noun_regular | 30 | 46 |
| noun_irregular | 17 | 11 |
| pronoun | 9 | 12 |
| adjective | 11 | 12 |
| verb_regular | 53 | 56 |
| verb_irregular | 25 | 26 |
| lexical_ambiguity | 4 | 5 |
| spelling_variant | 2 | 8 |
| oov (lemma not in lexicon) | 17 | 12 |
| out_of_scope | 2 | 8 |
| **total** | **170** | **196** |

* **Split (leakage control):** items that share any lemma are merged with union-find (e.g. बना links
  बनना and बनाना). Each group goes to dev if `sha1(seed:group)/2³² < 0.4`. **No lemma appears in both splits**
  (checked by a test). The split was fixed before tuning.
* **Tuning protocol:** the only tuned parameters are the guesser's paradigm priors. They were changed once,
  after looking at **dev** errors only (logged in `morphology_rules.json → meta.tuning_log`). The test split was then
  evaluated once.
* **Annotation correction C1 (logged):** the test run showed that गायें was annotated only as 'cows'.
  Under the project's own ये/ए spelling convention it is also the subjunctive of गाना. The missing readings
  were added. This changed both analyzers' test score by +1 item, so it does not affect the comparison. No
  code or rule was changed after the test run. Details are in `dataset_metadata.json → annotation_corrections`.
* **Limitations of the split:** the lexicon is a dictionary and contains the lemmas of in-scope *test*
  items. The split therefore prevents rule *tuning* on test forms, not lexical knowledge of test lemmas.
  The OOV category measures behaviour on unseen lemmas. 8 spelling-variant items normalise to forms that
  also appear in canonical spelling.

## 9. Evaluation protocol and metric definitions

For item *i*: gold set *Gᵢ* (≥ 1 analyses) and ranked prediction list *Pᵢ* (empty = UNKNOWN). The primary prediction is *Pᵢ[0]*.

**Exact analysis, top-1 (primary):** correctᵢ = *Pᵢ[0] ∈ Gᵢ* (lemma and all 9 features match).

```
Accuracy  = #correct / N
Precision = #correct / #answered        (answered = Pᵢ non-empty)
Recall    = #correct / N
F1        = 2·P·R / (P + R)
```

UNKNOWN outputs are never dropped: they are errors for accuracy and recall, and abstentions for precision.
As a result, **accuracy = recall by construction**, and precision measures how reliable an answer is when
the analyzer gives one. A legitimate alternative reading of an ambiguous word counts as correct, because
any member of *Gᵢ* is accepted.

**Lemma (secondary):** same formulas, but correct iff lemma(*Pᵢ[0]*) ∈ lemmas(*Gᵢ*).

**Analysis set (ambiguity handling):** micro `P = Σ|Pᵢ∩Gᵢ| / Σ|Pᵢ|`, `R = Σ|Pᵢ∩Gᵢ| / Σ|Gᵢ|`.

**Feature level:** micro P/R/F1 over (feature, value) pairs (lemma + every non-NA feature). The pairs of
*Pᵢ[0]* are compared with the best-matching gold analysis: the exact match if there is one, otherwise the one
with the largest overlap. A predicted NA where gold has a value counts as FN. A value where gold is NA counts as FP.
NA/NA is not counted. For an UNKNOWN word, all pairs of its first gold analysis count as FN.

**Zero division:** a ratio with denominator 0 is reported as 0.0 and flagged as *undefined*.
POS confusion matrices and per-POS scores use scikit-learn. The exact accuracy is cross-checked against
`sklearn.metrics.accuracy_score` in the tests. Every score is computed from live predictions; none is hard-coded.

## 10. Results (actual output of `python scripts/evaluate.py`)

### Held-out test split (196 words), the reported result

| Analyzer | Accuracy | Precision | Recall | F1 | correct / answered / total | Lemma acc. | Analysis-set F1 | Feature F1 |
|---|---|---|---|---|---|---|---|---|
| **Rule-based** | **95.92 %** | 97.41 % | **95.92 %** | **96.66 %** | 188 / 193 / 196 | 95.92 % | 93.03 % | 97.74 % |
| **FST** | 89.29 % | **100.00 %** | 89.29 % | 94.34 % | 175 / 175 / 196 | 89.29 % | **94.28 %** | 94.65 % |
| Rule-based, no guesser (ablation) | 89.29 % | 100.00 % | 89.29 % | 94.34 % | 175 / 175 / 196 | 89.29 % | 94.28 % | 94.65 % |

Analysis-set P/R on test: rule-based 91.99 % / 94.10 %, FST 100.00 % / 89.18 %.
Feature-level P/R on test: rule-based 98.42 % / 97.07 %, FST 100.00 % / 89.84 %.

### Development split (170 words)

| Analyzer | Accuracy | Precision | Recall | F1 | correct / answered / total |
|---|---|---|---|---|---|
| Rule-based | 96.47 % | 97.04 % | 96.47 % | 96.76 % | 164 / 169 / 170 |
| FST | 88.82 % | 100.00 % | 88.82 % | 94.08 % | 151 / 151 / 170 |

All 366 words: rule-based 96.17 % accuracy (P 97.24 %, F1 96.70 %); FST 89.07 % accuracy (P 100 %, F1 94.22 %).

Mean analysis time is about 30–40 µs per word for both analyzers on the development machine. It varies
from run to run and is reported only for information (`results/timing.json`).

## 11. Error analysis (test split)

* **Agreement:** both analyzers are correct on 175 words, only the rule-based analyzer on 13, only the FST on 0, and neither on 8.
* **In-lexicon words (176):** both analyzers are correct on the same 175. The single shared error is पीएगा, a
  spelling with an unshortened stem vowel that normalisation does not cover.
* **OOV words (12):** the FST rejects all 12. The rule-based guesser gets 11 right. It misreads
  मीठी (adjective 'sweet') as a feminine noun.
* **Wrong guesses (5)** are all POS confusions caused by shared suffixes: मीठी → noun, पीएगा → noun
  (the lexicon blocks a guess for the known lemma पीना), अच्छाई → *अच्छाना PERF FEM (a derivational noun read
  as a verb), बच्चो → verb, करा → noun.
* **Out of scope (8):** derivation, vocative and ‑कर cannot be decomposed by either analyzer. The FST rejects all 8.
  The rule-based guesser happens to be right on 2 of them: दोस्ती (guessed as a fem ‑ई noun, which is its
  inflectional analysis) and पढ़ाई (guessed as the perfective of पढ़ाना, one of its valid gold readings).
* **Known merge:** y-glide folding maps गायें ('cows') to गाएं, so both analyzers lose the noun reading.
  This is a visible analysis-set recall error (see correction C1).

**Interpretation.** With the same linguistic knowledge, the two formalisms accept exactly the same forms.
The FST gives a precise and transparent closed system that also runs in reverse for generation. The
rule-based design makes it easy to add a heuristic guesser, which trades precision (100 % → 97.4 %) for
recall (89.3 % → 95.9 %) on this dataset. The test set is small: 12 OOV and 8 out-of-scope items drive the
whole difference, so no claim of statistical significance is made.

## 12. Limitations

* Small, hand-curated lexicon (≈120 lemmas) and dataset (366 words). The results do not generalise to running text.
* Context-free: no disambiguation in context, and ambiguous readings are ranked by a fixed rule order.
* The gold standard was curated with AI assistance by a single annotator. There is no inter-annotator agreement.
* Lossy normalisation (ँ→ं, ये→ए) merges a few distinct forms.
* The lexicon includes test lemmas, so the split controls rule tuning, not vocabulary.
* The unimplemented phenomena are listed in §2.

## 13. Future improvements

Add an FST guesser (wildcard stem arcs), so that both analyzers handle unknown words; add derivation (‑ई, ‑पन, ‑आस)
and the vocative and ‑कर forms; disambiguate in context with a POS tagger; extend the lexicon from a licensed
lexical resource; convert a licensed treebank (after verifying its licence) for an in-context evaluation;
add a second annotator and report agreement; port the FST to HFST/foma for comparison.

## 14. Installation and running (Windows PowerShell, VS Code)

1. Install **Python 3.11** from python.org (tick "Add python.exe to PATH" and keep the `py` launcher).
2. Open the folder `hindi-morphological-analyzer` in VS Code (*File → Open Folder*), then open a terminal (*Ctrl+`*).
3. Create and activate a virtual environment, install the dependencies, test, evaluate and launch:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest                      # 107 tests
python scripts/evaluate.py            # prints metrics, writes results/
python -m streamlit run app.py        # opens http://localhost:8501
```

4. Open **http://localhost:8501** in a browser if it does not open automatically.

Optional: `python scripts/evaluate.py --split test --errors 20` lists errors. `python scripts/build_gold.py`
rebuilds the gold JSON after you edit `data/annotations/gold_annotations.txt`.

### Outputs in `results/`
`evaluation_summary.json` (all metrics, all splits), `metrics_<split>.csv`, `feature_metrics_<split>.csv`,
`predictions_<split>.csv/.json` (per-word gold vs both predictions), and `timing.json`. Apart from
timing, these files are byte-identical across runs. The app's *Dataset & methodology* page also offers CSV/JSON downloads.

## 15. Troubleshooting

| Problem | Fix |
|---|---|
| `Activate.ps1 cannot be loaded because running scripts is disabled` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `.\.venv\Scripts\activate.bat` from cmd. |
| `py -3.11` not found | Install Python 3.11, or use another installed 3.10–3.12 (`py -0` lists them). |
| pip fails to build numpy/pandas | Upgrade pip first (`python -m pip install --upgrade pip`). The pinned versions ship Windows wheels for Python 3.11. |
| Hindi shows as boxes or `?` in the terminal | Use Windows Terminal or VS Code. `scripts/evaluate.py` already writes UTF-8; if needed run `chcp 65001` or set `$env:PYTHONIOENCODING="utf-8"`. The CSV files are UTF-8 with BOM, so Excel opens them correctly. Install a Devanagari font (Nirmala UI is preinstalled on Windows 10/11). |
| `gold standard not found` / `required data file not found` | Make sure the `data/` folder is complete. Run `python scripts/build_gold.py`. |
| `ModuleNotFoundError: src` | Run commands from the project root folder (the one that contains `app.py`). |
| `streamlit` not recognised | Use `python -m streamlit run app.py` with the venv activated. |
| Port 8501 in use | `python -m streamlit run app.py --server.port 8502`. |
| Typed word not analysed | Type a single Devanagari word. Unsupported words are shown as UNKNOWN by design. |

## 16. Live-demo walkthrough (≈ 5 minutes)

1. **Overview page:** state the problem, show the two approaches table and the live headline numbers.
2. **Word analyzer → लड़कों:** both analyzers agree. Open the *FST state-transition path* and walk through the
   stem arcs, the suffix arcs, the restore arc (‑ा) and the tag arc. Then open the rule trace (suffix ‑ों → stem लड़क → लड़का ✓).
3. **लड़के / खाने / बना:** ambiguity. Show the multiple candidates and the selectable FST paths.
4. **किया, पिया:** irregular and stem-change forms. Use the *generate* box with `पीना+VERB+FIN+FUT+IND+3+MASC+SG` → पिएगा.
5. **मकानों / दीवारें:** the FST rejects them (show where it got stuck). The rule-based analyzer GUESSES correctly.
   **पूछा** gives a wrong guess. **लड़कोंा** is invalid and rejected by both.
6. **Evaluation dashboard (test split):** metric tiles, P/R/F1 chart, feature-level chart, confusion
   matrices. Tick the *ablation* box to show that without the guesser the rule-based analyzer equals the FST.
7. **Error analysis:** filter *Only rule-based correct*, then *Both incorrect*, and explain the error types.
8. **Dataset & methodology:** provenance, lemma-grouped split, the correction log, and the downloads.
