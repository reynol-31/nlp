"""Streamlit demo: Hindi morphological analysis -- rule-based vs finite-state transducer.

Run:  python -m streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dataset_loader import GOLD_FILE, dataset_summary, load_gold, load_metadata
from src.error_analysis import (
    AGREEMENT_CATEGORIES,
    ERROR_DESCRIPTIONS,
    agreement_counts,
    category_accuracy_table,
    error_type_table,
    filter_comparison,
    insights,
    mismatched_feature_counts,
)
from src.evaluation import ERROR_TYPES, FST_NAME, RULE, RULE_ABLATION, run_evaluation
from src.fst_analyzer import FSTAnalyzer
from src.normalization import codepoints, normalize_word, split_words
from src.resources import DATA_DIR, load_resources
from src.rule_based_analyzer import RuleBasedAnalyzer
from src.schema import FEATURE_LABELS, FEATURES, NA, VALUE_LABELS

st.set_page_config(page_title="Hindi Morphological Analyzer", page_icon="🔤", layout="wide")

# Validated categorical slots (dataviz reference palette): blue, orange, aqua.
COLORS = {RULE: "#2a78d6", FST_NAME: "#eb6834", RULE_ABLATION: "#1baf7a"}
PAGES = ["Overview", "Word analyzer", "Evaluation dashboard", "Error analysis", "Dataset & methodology"]
SAMPLES = {
    "Nouns": ["लड़कों", "लड़के", "किताबें", "लड़कियाँ", "आदमियों", "राजाओं", "चिड़ियाँ", "बहुएँ"],
    "Verbs": ["खाएगा", "पढ़ेंगे", "पढ़ती", "खाया", "पिया", "दीजिए", "बैठिए", "सोएँगे"],
    "Irregular / pronouns": ["किया", "गया", "हुआ", "था", "हैं", "मुझे", "उन्हें", "तुम्हें"],
    "Adjectives": ["अच्छी", "बड़े", "नई", "सुंदर"],
    "Ambiguous": ["खाने", "बना", "जीता", "सोने", "हो"],
    "Unknown / unsupported": ["मकानों", "दीवारें", "पूछा", "खाकर", "अच्छाई", "लड़कोंा"],
}


# ---------------------------------------------------------------------------
def data_fingerprint() -> tuple:
    files = [DATA_DIR / "lexicon.json", DATA_DIR / "morphology_rules.json", GOLD_FILE]
    return tuple((f.name, f.stat().st_mtime if f.exists() else 0) for f in files)


@st.cache_resource(show_spinner=False)
def get_analyzers(_fp: tuple):
    load_resources.cache_clear()
    res = load_resources()
    return {
        RULE: RuleBasedAnalyzer(res, enable_guesser=True),
        FST_NAME: FSTAnalyzer(res),
        RULE_ABLATION: RuleBasedAnalyzer(res, enable_guesser=False),
    }, res


@st.cache_resource(show_spinner="Running evaluation on the gold standard ...")
def get_report(split: str, _fp: tuple):
    analyzers, _ = get_analyzers(_fp)
    return run_evaluation(split, items=load_gold(), analyzers=analyzers)


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def label_value(v: str) -> str:
    return f"{v} ({VALUE_LABELS[v]})" if v in VALUE_LABELS and v != NA else ("—" if v == NA else v)


def bar(df, x, y, title, ylab="Score (%)", color="Analyzer", rng=(0, 105), **kw):
    fig = px.bar(df, x=x, y=y, color=color, barmode="group", color_discrete_map=COLORS,
                 text_auto=".1f", title=title, labels={y: ylab}, **kw)
    fig.update_traces(marker_line_width=0, textposition="outside", cliponaxis=False)
    fig.update_layout(yaxis_range=list(rng) if rng else None, legend_title_text="",
                      margin=dict(t=60, b=10, l=10, r=10), height=380, bargap=0.25, bargroupgap=0.06)
    return fig


# ---------------------------------------------------------------------------
fp = data_fingerprint()
try:
    ANALYZERS, RES = get_analyzers(fp)
    GOLD = load_gold()
except Exception as exc:  # show a helpful message instead of a stack trace
    st.error(f"Could not load the linguistic resources or gold standard: {exc}")
    st.info("Check that the data/ folder is complete. Rebuild the gold standard with "
            "`python scripts/build_gold.py`.")
    st.stop()

with st.sidebar:
    st.title("🔤 Hindi Morphology")
    st.caption("Rule-based vs Finite-State Transducer")
    page = st.radio("Page", PAGES, label_visibility="collapsed")
    st.divider()
    split = st.selectbox(
        "Evaluation split", ["test", "dev", "all"],
        format_func={"test": "Held-out test", "dev": "Development", "all": "All items"}.get,
        help="Rules were tuned on the development split only. Report the held-out test split.",
    )
    show_ablation = st.checkbox("Show ablation: rule-based without guesser", value=False,
                                help="Same rule-based analyzer with the unknown-word guesser switched off.")
    st.divider()
    st.caption(f"Lexicon: {RES.summary()['lemmas']} lemmas · {RES.summary()['suffix_rules']} suffix rules · "
               f"FST: {ANALYZERS[FST_NAME].stats()['states']} states / {ANALYZERS[FST_NAME].stats()['arcs']} arcs · "
               f"Gold: {len(GOLD)} words")

REPORT = get_report(split, fp)
SPLIT_NAME = {"test": "held-out TEST split", "dev": "DEVELOPMENT split", "all": "ALL items (dev + test)"}[split]
SHOWN = [RULE, FST_NAME] + ([RULE_ABLATION] if show_ablation else [])


# ===========================================================================
def page_overview():
    st.title("Morphological Analyzer for Hindi")
    st.subheader("A comparative evaluation of rule-based and finite-state transducer approaches")
    st.markdown(
        """
**Problem.** Hindi is morphologically rich: one lemma surfaces in many inflected forms
(लड़का, लड़के, लड़कों; खाना, खाता, खाया, खाएगा, खाइए …). NLP systems for search, machine translation,
parsing and language learning need to map each surface form to its **lemma** and **grammatical
features**. This project builds two analyzers for a documented subset of Hindi inflection and
measures how well each recovers gold-standard analyses.
"""
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### What is morphological analysis?")
        st.markdown("Splitting a word into its **stem** and **inflectional suffix** and reading off the features:")
        st.code("लड़कों  =  लड़क + ों\n        →  lemma लड़का, NOUN, masculine, plural, oblique\n"
                "लेक्सिकल रूप (lexical form):  लड़का+NOUN+MASC+PL+OBL", language=None)
        st.markdown(
            "**Why Hindi?** Gender, number and case are marked on nouns, adjectives and verbs; verbs inflect "
            "for aspect, mood, tense, person, gender and number; there are stem alternations (पीना → पिया), "
            "suppletive forms (करना → किया, जाना → गया) and widespread spelling variation (ँ/ं, ये/ए, nukta)."
        )
    with c2:
        st.markdown("#### The two approaches")
        st.table(pd.DataFrame(
            {
                "Rule-based analyzer": [
                    "Procedural: strip suffix, restore lemma ending, check lexicon",
                    "Exception table → suffix rules → lexicon validation → ranking",
                    "Optional guesser for unknown lemmas (higher recall, lower precision)",
                    "Trace of rule hypotheses",
                ],
                "Finite-state transducer": [
                    "Declarative: a graph of states with input:output arcs",
                    "Stem trie → continuation class → suffix arcs → tag arcs",
                    "Closed lexicon: unknown lemmas are rejected",
                    "State/transition path; also runs in reverse (generation)",
                ],
            },
            index=["Idea", "Pipeline", "Unknown words", "Explanation"],
        ))
    st.markdown("#### Methodology")
    st.markdown(
        """
1. **Shared resources** – one lexicon (`data/lexicon.json`) and one paradigm table (`data/morphology_rules.json`) feed *both* analyzers, so they encode identical linguistic knowledge.
2. **Shared output format** – every analysis is a lemma plus 9 feature slots (POS, verb form, tense, aspect, mood, person, gender, number, case), with `NA` where a feature does not apply.
3. **Gold standard** – manually annotated words with *all* valid analyses, split into development / held-out test sets **by lemma group** (no lemma in both splits).
4. **Tuning on dev only** – the only tuned parameters (guesser priors) were adjusted using development errors; the test split was evaluated once.
5. **Metrics** – exact-analysis accuracy / precision / recall / F1 (primary), lemma accuracy, analysis-set and feature-level P/R/F1, all computed live from the analyzers' actual output.
"""
    )
    m = REPORT.metrics
    st.markdown(f"#### Headline result ({SPLIT_NAME}, {len(REPORT.items)} words, computed live)")
    cols = st.columns(4)
    cols[0].metric("Rule-based exact accuracy", pct(m[RULE]["exact_top1"]["accuracy"]))
    cols[1].metric("FST exact accuracy", pct(m[FST_NAME]["exact_top1"]["accuracy"]))
    cols[2].metric("Rule-based precision", pct(m[RULE]["exact_top1"]["precision"]))
    cols[3].metric("FST precision", pct(m[FST_NAME]["exact_top1"]["precision"]))
    st.caption("See *Evaluation dashboard* for definitions, sample counts and charts.")


# ===========================================================================
def analysis_table(a) -> pd.DataFrame:
    rows = [("Lemma", a.lemma)] + [(FEATURE_LABELS[f], label_value(getattr(a, f))) for f in FEATURES
                                   if getattr(a, f) != NA]
    rows += [("Suffix", a.suffix if a.suffix else "∅ (zero / full form)"), ("Rule / path", a.rule),
             ("Source", a.source)]
    return pd.DataFrame(rows, columns=["Field", "Value"])


def candidates_frame(analyses) -> pd.DataFrame:
    return pd.DataFrame([{"#": i + 1, "lemma": a.lemma, **{FEATURE_LABELS[f]: getattr(a, f) for f in FEATURES},
                          "suffix": a.suffix, "rule": a.rule} for i, a in enumerate(analyses)])


def show_result(name: str, res):
    st.markdown(f"### {name}")
    badge = {"KNOWN": "🟢 KNOWN", "GUESSED": "🟡 GUESSED (lemma not in lexicon)", "UNKNOWN": "🔴 UNKNOWN"}[res.status]
    st.markdown(f"**Status:** {badge}  \n**Input:** `{res.word}` → normalised `{res.normalized}`")
    st.caption(res.message)
    if res.known:
        st.markdown("**Primary analysis**")
        st.dataframe(analysis_table(res.primary), hide_index=True, width="stretch")
        st.markdown(f"**All candidate analyses ({len(res.analyses)})**")
        st.dataframe(candidates_frame(res.analyses), hide_index=True, width="stretch")
    else:
        st.info("No analysis is produced for this input — nothing is fabricated.")


def page_analyzer():
    st.title("Interactive word analyzer")
    c1, c2 = st.columns([1, 2])
    with c1:
        group = st.selectbox("Sample group", list(SAMPLES))
        sample = st.selectbox("Sample word", SAMPLES[group])
    with c2:
        text = st.text_input("…or type a Hindi word in Devanagari", value=sample,
                             help="Spelling variants such as लड़कियां / लड़कियाँ, नये / नए are normalised.")
        mode = st.radio("Run", ["Both analyzers", "Rule-based only", "FST only"], horizontal=True)
    words = split_words(text)
    if not words:
        st.warning("Please enter a word.")
        return
    if len(words) > 1:
        st.info(f"Several words entered; analysing the first one: **{words[0]}**")
    word = words[0]

    gold = [it for it in GOLD if it.normalized == normalize_word(word)]
    if gold:
        st.success("This word is in the gold standard (" + ", ".join(f"{g.id} · {g.split} · {g.category}" for g in gold)
                   + "): " + " | ".join(a.short() for a in gold[0].analyses))

    names = {"Both analyzers": [RULE, FST_NAME], "Rule-based only": [RULE], "FST only": [FST_NAME]}[mode]
    cols = st.columns(len(names))
    results = {}
    for col, name in zip(cols, names):
        with col:
            results[name] = ANALYZERS[name].analyze(word)
            show_result(name, results[name])

    if RULE in results:
        with st.expander("Rule-based: rule application trace", expanded=False):
            st.caption("Every suffix rule whose surface suffix matches the end of the word proposes stem + suffix; "
                       "the restored lemma must exist in the lexicon with that paradigm.")
            st.dataframe(pd.DataFrame(results[RULE].trace), hide_index=True, width="stretch")
    if FST_NAME in results:
        fst = ANALYZERS[FST_NAME]
        res = results[FST_NAME]
        with st.expander("FST: state-transition path", expanded=True):
            if not res.trace:
                st.write("No path (input rejected before traversal).")
            else:
                options = [f"path {t['path']}: {t['output']}" if t.get("output") else "longest partial path (rejected)"
                           for t in res.trace]
                idx = st.selectbox("Accepting path", range(len(options)), format_func=options.__getitem__) \
                    if len(options) > 1 else 0
                t = res.trace[idx]
                st.graphviz_chart(fst.path_dot(t["arcs"], rejected=t.get("rejected", False)))
                st.dataframe(pd.DataFrame(fst.path_table(t["arcs"])), hide_index=True, width="stretch")
                st.caption("Each row is one transition `input : output`. ε = no symbol. Stem arcs copy letters, "
                           "suffix arcs consume the ending, the restore arc writes back the lemma ending, the "
                           "tag arc writes the features; q1 is the single accepting state.")
        with st.expander("FST in reverse: generate surface forms from a lexical string"):
            default = res.primary.lexical_string() if res.known else "लड़का+NOUN+MASC+PL+OBL"
            lex = st.text_input("Lexical string", value=default)
            forms = fst.generate(lex)
            st.write("Surface form(s): " + (", ".join(f"**{f}**" for f in forms) if forms else "_none (not generated)_"))
    with st.expander("Unicode code points of the normalised input"):
        st.code("\n".join(codepoints(normalize_word(word))), language=None)


# ===========================================================================
def metric_tiles(name: str):
    e = REPORT.metrics[name]["exact_top1"]
    st.markdown(f"**{name}**")
    c = st.columns(4)
    c[0].metric("Accuracy", pct(e["accuracy"]), help="exact top-1 matches / all items")
    c[1].metric("Precision", pct(e["precision"]) if e["precision_defined"] else "undefined",
                help="exact top-1 matches / items the analyzer answered (non-UNKNOWN)")
    c[2].metric("Recall", pct(e["recall"]), help="exact top-1 matches / all items (equals accuracy by definition)")
    c[3].metric("F1-score", pct(e["f1"]), help="harmonic mean of precision and recall")
    st.caption(f"{e['correct']} exact matches · {e['errors']} errors · {e['unknown']} unknown · "
               f"{e['answered']} answered · n = {e['total']}")


def page_dashboard():
    st.title("Comparative evaluation dashboard")
    st.markdown(f"All figures below are computed now from the analyzers' predictions on the **{SPLIT_NAME}** "
                f"(**{len(REPORT.items)} words**, {sum(len(i.analyses) for i in REPORT.items)} gold analyses).")
    with st.expander("Metric definitions", expanded=False):
        st.markdown(
            r"""
* **Exact-analysis (primary)** – the analyzer's *first-ranked* analysis is correct iff it equals one of the gold analyses on lemma **and** all 9 features.
  $\text{Accuracy} = \frac{\#\text{correct}}{N}$, $\text{Precision} = \frac{\#\text{correct}}{\#\text{answered}}$, $\text{Recall} = \frac{\#\text{correct}}{N}$, $F_1 = \frac{2PR}{P+R}$.
  UNKNOWN outputs are *not* dropped: they are errors for accuracy/recall and abstentions for precision. Hence accuracy = recall by definition, while precision shows how reliable an answer is when one is given.
* **Lemma (secondary)** – same, but only the lemma must match.
* **Analysis set** – micro P/R/F1 over *all* returned candidates vs *all* gold analyses ($TP=\sum|P_i\cap G_i|$): rewards finding every reading of ambiguous words without over-generating.
* **Feature level** – micro P/R/F1 over (feature, value) pairs of the first-ranked analysis vs the best-matching gold analysis; `NA` never counts as a value (predicted NA where gold has a value = FN; a value where gold is NA = FP).
* **Zero division** – a ratio with denominator 0 is shown as 0 / “undefined”.
"""
        )
    metric_tiles(RULE)
    metric_tiles(FST_NAME)
    if show_ablation:
        metric_tiles(RULE_ABLATION)

    st.subheader("Side-by-side comparison")
    sf = REPORT.summary_frame()
    sf = sf[sf["Analyzer"].isin(SHOWN)].copy()
    pct_cols = [c for c in sf.columns if c not in ("Analyzer", "Items", "Exact matches", "Errors", "Unknown")]
    sf[pct_cols] = (sf[pct_cols] * 100).round(2)
    st.dataframe(sf, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(c, format="%.2f%%") for c in pct_cols})

    long = []
    for name in SHOWN:
        m = REPORT.metrics[name]
        long += [
            {"Analyzer": name, "Metric": "Exact accuracy", "Value": 100 * m["exact_top1"]["accuracy"]},
            {"Analyzer": name, "Metric": "Lemma accuracy", "Value": 100 * m["lemma_top1"]["accuracy"]},
        ]
    c1, c2 = st.columns(2)
    c1.plotly_chart(bar(pd.DataFrame(long), "Metric", "Value", "Accuracy (top-1)"), width="stretch")
    prf_rows = []
    for name in SHOWN:
        e = REPORT.metrics[name]["exact_top1"]
        prf_rows += [{"Analyzer": name, "Metric": k.capitalize() if k != "f1" else "F1", "Value": 100 * e[k]}
                     for k in ("precision", "recall", "f1")]
    c2.plotly_chart(bar(pd.DataFrame(prf_rows), "Metric", "Value", "Exact-analysis precision / recall / F1"),
                    width="stretch")

    st.subheader("Feature-level performance")
    ff = REPORT.feature_frame()
    ff = ff[ff["Analyzer"].isin(SHOWN)].copy()
    ff["F1 (%)"] = ff["F1"] * 100
    ff["Feature"] = ff["Feature"].map(FEATURE_LABELS)
    fig = bar(ff, "Feature", "F1 (%)", "Feature-level F1 by feature (micro over feature-value pairs)",
              ylab="F1 (%)", hover_data={"Support": True, "TP": True, "FP": True, "FN": True})
    st.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        rows = []
        for name in SHOWN:
            fm, s = REPORT.metrics[name]["feature_micro"], REPORT.metrics[name]["analysis_set"]
            rows.append({"Analyzer": name, "Feature P": fm["precision"], "Feature R": fm["recall"],
                         "Feature F1": fm["f1"], "Set P": s["precision"], "Set R": s["recall"], "Set F1": s["f1"],
                         "Predicted analyses": s["n_pred"], "Gold analyses": s["n_gold"]})
        df = pd.DataFrame(rows)
        for c in ("Feature P", "Feature R", "Feature F1", "Set P", "Set R", "Set F1"):
            df[c] = (df[c] * 100).round(2)
        st.markdown("**Feature-level and analysis-set micro scores (%)**")
        st.dataframe(df, hide_index=True, width="stretch")
    with c2:
        ct = category_accuracy_table(REPORT)
        ct = ct[ct["Analyzer"].isin(SHOWN)]
        st.markdown("**Exact accuracy by dataset category (%)**")
        pv = ct.pivot(index="Category", columns="Analyzer", values="Accuracy").mul(100).round(1)
        pv.insert(0, "Items", ct.drop_duplicates("Category").set_index("Category")["Items"])
        st.dataframe(pv, width="stretch")

    st.subheader("Error distribution")
    et = error_type_table(REPORT)
    et = et[et["Analyzer"].isin(SHOWN)]
    c1, c2 = st.columns(2)
    c1.plotly_chart(bar(et, "Error type", "Items", "Outcome of the first-ranked analysis", ylab="Words",
                        rng=None, category_orders={"Error type": list(ERROR_TYPES)}), width="stretch")
    ag = agreement_counts(REPORT.comparison_frame())
    fig = px.bar(ag, x="Outcome", y="Items", text_auto=True, title="Agreement between the two analyzers",
                 labels={"Items": "Words"}, color_discrete_sequence=["#2a78d6"])
    fig.update_traces(marker_line_width=0, textposition="outside", cliponaxis=False)
    fig.update_layout(height=380, margin=dict(t=60, b=10, l=10, r=10))
    c2.plotly_chart(fig, width="stretch")

    st.subheader("POS confusion matrices")
    st.caption("Rows: gold POS (of the best-matching gold analysis); columns: POS of the first-ranked prediction. "
               "UNKNOWN = no prediction. Computed with scikit-learn.")
    cols = st.columns(len(SHOWN))
    for col, name in zip(cols, SHOWN):
        cm = REPORT.metrics[name]["pos_confusion"]
        if not cm["matrix"]:
            col.info("No items.")
            continue
        fig = px.imshow(cm["matrix"], x=cm["labels"], y=cm["labels"], text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="Predicted POS", y="Gold POS", color="Words"), title=name, aspect="auto")
        fig.update_layout(height=380, margin=dict(t=50, b=10, l=10, r=10), coloraxis_showscale=False)
        col.plotly_chart(fig, width="stretch")
    with st.expander("Per-POS precision / recall / F1 (scikit-learn) – small-sample caveat"):
        rows = [{"Analyzer": n, "POS": p, **{k: round(100 * v, 2) if k != "support" else v for k, v in d.items()}}
                for n in SHOWN for p, d in REPORT.metrics[n]["pos_per_class"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Some POS classes have very few items (e.g. AUX), so per-class scores are descriptive only; "
                   "no claim of statistical significance is made.")
    t = REPORT.timing()
    st.caption("Mean analysis time per word on this machine (informative only, varies between runs): "
               + ", ".join(f"{k}: {v:.1f} µs" for k, v in t.items() if k in SHOWN))


# ===========================================================================
def page_errors():
    st.title("Error analysis")
    df = REPORT.comparison_frame()
    st.markdown(f"**{SPLIT_NAME}** · {len(df)} words")
    for line in insights(REPORT):
        st.markdown(f"- {line}")
    st.divider()
    c1, c2, c3 = st.columns(3)
    agreement = c1.selectbox("Outcome", ["(all)"] + list(AGREEMENT_CATEGORIES), index=4)
    pos = c2.multiselect("Gold POS", sorted(df["gold_pos"].unique()))
    cats = c3.multiselect("Dataset category", sorted(df["category"].unique()))
    c4, c5, c6 = st.columns(3)
    who = c4.selectbox("Error filters apply to", [RULE, FST_NAME])
    etype = c5.selectbox("Error type", ["(any)"] + [e for e in ERROR_TYPES if e != "CORRECT"],
                         help="\n".join(f"{k}: {v}" for k, v in ERROR_DESCRIPTIONS.items()))
    feature = c6.selectbox("Mismatched feature", ["(any)", "lemma"] + list(FEATURES),
                           help="Only rows where the selected analyzer got this feature wrong.")
    out = filter_comparison(
        df, agreement=None if agreement == "(all)" else agreement, pos=pos or None, categories=cats or None,
        error_type=None if etype == "(any)" else etype, analyzer=who,
        feature=None if feature == "(any)" else feature,
    )
    st.markdown(f"**{len(out)} matching words**")
    view = out[["id", "word", "category", "gold", "rule_prediction", "rule_status", "rule_error",
                "fst_prediction", "fst_status", "fst_error", "agreement"]].rename(columns={
        "gold": "Gold analyses", "rule_prediction": "Rule-based prediction", "rule_status": "RB status",
        "rule_error": "RB error", "fst_prediction": "FST prediction", "fst_status": "FST status",
        "fst_error": "FST error"})
    st.dataframe(view, hide_index=True, width="stretch", height=420)
    c1, c2 = st.columns(2)
    for col, name in ((c1, RULE), (c2, FST_NAME)):
        cnt = mismatched_feature_counts(REPORT, name)
        col.markdown(f"**{name}: features wrong in wrong-but-answered predictions**")
        col.write(", ".join(f"{FEATURE_LABELS[k]}: {v}" for k, v in cnt.most_common()) or "none")
    with st.expander("Error type definitions"):
        st.table(pd.DataFrame(ERROR_DESCRIPTIONS.items(), columns=["Error type", "Meaning"]))
    st.markdown(
        """
**Strengths and weaknesses (interpretation of the data above)**
* With the *same* lexicon and paradigms the two analyzers accept exactly the same in-lexicon forms (a unit test checks this for every generated form), so differences come from **unknown-word handling**, not from the formalism.
* The **FST** is a closed system: perfect or near-perfect precision, but every out-of-lexicon word is UNKNOWN. It is bidirectional (generation) and its paths are easy to inspect.
* The **rule-based** analyzer's guesser recovers many OOV words from suffixes, raising recall, but suffix sharing between nouns, adjectives and verbs (‑े, ‑ी, ‑ा) and dropped lemma endings (पत्तों → पत्त/पत्ता) cause wrong guesses, lowering precision.
* **Out-of-scope** morphology (derivation, vocative, ‑कर) defeats both by design; spelling outside the normalisation rules (पीएगा) as well.
"""
    )


# ===========================================================================
def page_dataset():
    st.title("Dataset & methodology")
    meta = load_metadata()
    prov = meta.get("provenance", {})
    st.subheader("Provenance")
    st.markdown(f"**{prov.get('name', 'Gold standard')}** — source type: `{prov.get('source_type', '?')}`, "
                f"created {prov.get('created', '?')}")
    st.markdown(prov.get("source_description", ""))
    st.markdown(f"*Annotation process:* {prov.get('annotation_process', '')}")
    st.markdown(f"*Licence:* {prov.get('license', '')}")
    for ext in prov.get("external_datasets_considered", []):
        st.markdown(f"*Considered, not used:* [{ext['name']}]({ext['url']}) — {ext['reason']}")

    summ = dataset_summary(GOLD)
    st.subheader("Size and split")
    c = st.columns(5)
    c[0].metric("Words", summ["items"])
    c[1].metric("Gold analyses", summ["analyses"])
    c[2].metric("Ambiguous words", summ["ambiguous_items"])
    c[3].metric("Dev / test", f"{summ['by_split'].get('dev', 0)} / {summ['by_split'].get('test', 0)}")
    c[4].metric("Lemma groups in both splits", len(summ["group_overlap_between_splits"]))
    st.markdown(meta.get("split", {}).get("details", ""))
    cat = pd.DataFrame(summ["by_category"]).T
    cat["total"] = cat.sum(axis=1)
    cat.insert(0, "description", [meta.get("annotation_schema", {}).get("categories", {}).get(i, "") for i in cat.index])
    st.dataframe(cat, width="stretch")

    st.subheader("Annotation schema")
    sch = meta.get("annotation_schema", {}).get("analysis_fields", {})
    st.table(pd.DataFrame(sch.items(), columns=["Field", "Values"]))
    st.caption(meta.get("annotation_schema", {}).get("not_applicable", ""))

    st.subheader("Linguistic coverage (shared by both analyzers)")
    rows = []
    for pid, p in RES.paradigms.items():
        lemmas = [e.display for e in RES.entries if e.paradigm == pid]
        rows.append({"Paradigm": pid, "POS": p.pos, "Lemmas": len(lemmas), "Suffix rules": len(p.rules),
                     "Examples": ", ".join(lemmas[:4]), "Description": p.description})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Plus {RES.summary()['irregular_forms']} irregular / suppletive full forms "
               f"({RES.summary()['irregular_analyses']} analyses): pronouns, करना, जाना, होना.")

    corr = meta.get("annotation_corrections", [])
    if corr:
        st.subheader("Annotation corrections log")
        for c_ in corr:
            st.markdown(f"**{c_['id']} – {c_['word']}** (found {c_['found']}): {c_['reason']} *Effect:* {c_['effect']}")

    st.subheader("Known limitations")
    st.markdown(
        """
* Covers a **subset** of Hindi inflection (≈120 lemmas, 18 paradigms); not a production analyzer.
* Context-free: analyses of an isolated word; no disambiguation in context. Ranking of ambiguous readings is a fixed rule order.
* Not implemented: derivation, compounds, vocative, conjunctive ‑कर, oblique participles, possessive pronouns, postpositions, numerals, causative morphology as a process, honorific/politeness features.
* Normalisation folds ँ→ं and ये→ए; this merges a few distinct forms (गायें ‘cows’ ≡ गाएं ‘may sing’).
* The gold standard was curated by the project author with AI assistance and is small (366 words); it should be reviewed by a fluent speaker. Results do not generalise to running text.
* The lexicon contains the lemmas of in-scope test items (it is a dictionary); OOV items test unknown-word behaviour.
"""
    )

    st.subheader("Download evaluation results")
    cmp_df = REPORT.comparison_frame()
    c1, c2, c3 = st.columns(3)
    c1.download_button("Predictions (CSV)", cmp_df.to_csv(index=False).encode("utf-8-sig"),
                       f"predictions_{split}.csv", "text/csv")
    c2.download_button("Metrics summary (JSON)",
                       json.dumps(REPORT.to_json_dict(), ensure_ascii=False, indent=2).encode("utf-8"),
                       f"evaluation_summary_{split}.json", "application/json")
    c3.download_button("Predictions (JSON)", cmp_df.to_json(orient="records", force_ascii=False, indent=1).encode("utf-8"),
                       f"predictions_{split}.json", "application/json")
    st.caption(f"Downloads reflect the currently selected split ({SPLIT_NAME}). "
               "`python scripts/evaluate.py` writes the same files for all splits to results/.")


{"Overview": page_overview, "Word analyzer": page_analyzer, "Evaluation dashboard": page_dashboard,
 "Error analysis": page_errors, "Dataset & methodology": page_dataset}[page]()
