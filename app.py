"""AI vs. Human Text Detector - Streamlit web application.

Loads the six models trained in the notebook and lets a user analyze a document
or pasted text: prediction with calibrated confidence, a word-level explanation,
text statistics, a side-by-side comparison of all models, and a downloadable
report. All model logic lives in src/; this file is the interface only.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # quiet TensorFlow startup logs

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.features import FEATURE_NAMES, linguistic_features
from src.inference import Detector
from src.extract import extract_text, ExtractionError
from src.explain import explain_prediction, lime_explanation
from src.report import pdf_report, text_report
from src.examples import EXAMPLE_HUMAN, EXAMPLE_AI

PROJECT_ROOT = Path(__file__).resolve().parent
HUMAN_COLOR = "#1f7a63"
AI_COLOR = "#c2620a"

st.set_page_config(page_title="AI vs. Human Text Detector",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { max-width: 1180px; padding-top: 1.6rem; }

/* Header */
.hero { padding: 1.4rem 1.6rem; border-radius: 16px; margin-bottom: 1.4rem;
        background: linear-gradient(120deg, #1f2937 0%, #374151 60%, #4b5563 100%);
        color: #fff; box-shadow: 0 6px 24px rgba(31,41,55,.18); }
.hero h1 { font-size: 1.85rem; font-weight: 800; margin: 0; letter-spacing: -.02em; }
.hero p  { margin: .35rem 0 0; color: #d1d5db; font-size: .98rem; }

/* Verdict card */
.verdict { padding: 1.3rem 1.5rem; border-radius: 14px; color: white;
           font-size: 1.6rem; font-weight: 800; letter-spacing: -.01em;
           box-shadow: 0 6px 20px rgba(0,0,0,.12); }
.verdict small { display:block; font-size: .92rem; font-weight: 500; opacity: .92;
                 margin-top: .25rem; }
.note { background:#fff7ed; border-left:4px solid #c2620a; padding:.8rem 1.05rem;
        border-radius:8px; color:#7c2d12; font-size:.92rem; margin-top:.8rem; }

/* Metric cards */
div[data-testid="stMetric"] { background:#f9fafb; border:1px solid #eceef1;
        border-radius:12px; padding:.85rem 1rem; }
div[data-testid="stMetricLabel"] { color:#6b7280; }

/* Tabs */
button[data-baseweb="tab"] { font-weight:600; font-size:.97rem; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; }

/* Sidebar headings */
section[data-testid="stSidebar"] h3 { font-size:.8rem; text-transform:uppercase;
        letter-spacing:.06em; color:#9ca3af; margin-bottom:.3rem; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------- cached loaders
@st.cache_resource(show_spinner="Loading models...")
def get_detector() -> Detector:
    return Detector()


@st.cache_data(show_spinner=False)
def model_accuracies() -> dict[str, float]:
    path = PROJECT_ROOT / "reports" / "test_metrics.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, index_col=0)
    return df["accuracy"].to_dict()


@st.cache_data(show_spinner=False)
def all_probabilities(text: str) -> dict[str, float]:
    return get_detector().predict_all(text)


@st.cache_data(show_spinner=False)
def text_statistics(text: str) -> dict[str, float]:
    return linguistic_features(text)


@st.cache_data(show_spinner=False)
def explanation(text: str, model_name: str) -> dict:
    return explain_prediction(get_detector(), model_name, text)


@st.cache_data(show_spinner=False)
def sentence_scores(text: str, model_name: str):
    return get_detector().sentence_scores(model_name, text)


# ------------------------------------------------------------------- UI helpers
def verdict_banner(label: str, confidence: float):
    color = AI_COLOR if label == "AI" else HUMAN_COLOR
    phrase = "Likely AI-generated" if label == "AI" else "Likely human-written"
    st.markdown(
        f'<div class="verdict" style="background:{color}">{phrase}'
        f'<small>Confidence {confidence:.1%}</small></div>',
        unsafe_allow_html=True)


def confidence_gauge(p_ai: float):
    color = AI_COLOR if p_ai >= 0.5 else HUMAN_COLOR
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p_ai * 100,
        number={"suffix": "%", "font": {"size": 30}},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": color},
               "steps": [{"range": [0, 50], "color": "#eef3f1"},
                         {"range": [50, 100], "color": "#fbf1e7"}],
               "threshold": {"line": {"color": "#6b7280", "width": 2}, "value": 50}},
        title={"text": "Probability AI-generated", "font": {"size": 14}}))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=50, b=10))
    st.plotly_chart(fig, width="stretch")


def diverging_words(toward_ai, toward_human):
    items = sorted(toward_human + toward_ai, key=lambda kv: kv[1])
    if not items:
        st.info("No strongly weighted words found in this passage.")
        return
    fig = go.Figure(go.Bar(
        x=[v for _, v in items], y=[w for w, _ in items], orientation="h",
        marker_color=[AI_COLOR if v > 0 else HUMAN_COLOR for _, v in items]))
    fig.update_layout(height=28 * len(items) + 80,
                      margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="contribution toward Human (left) / AI (right)")
    st.plotly_chart(fig, width="stretch")


# ----------------------------------------------------------------- view: tabs
def tab_prediction(text, probs, selected):
    p_ai = probs[selected]
    label = "AI" if p_ai >= 0.5 else "Human"
    confidence = max(p_ai, 1 - p_ai)
    ai_votes = sum(1 for p in probs.values() if p >= 0.5)
    total = len(probs)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Verdict", label)
    c2.metric("Confidence", f"{confidence:.0%}")
    c3.metric("Models agreeing", f"{max(ai_votes, total - ai_votes)} of {total}")
    c4.metric("Words analyzed", f"{len(text.split()):,}")

    st.write("")
    left, right = st.columns([1.1, 1])
    with left:
        verdict_banner(label, confidence)
        st.caption(f"Selected model: {selected}")
        if 0.4 <= p_ai <= 0.6:
            st.markdown('<div class="note">This is a borderline case. The model is '
                        'close to undecided, so treat the result with caution.</div>',
                        unsafe_allow_html=True)
    with right:
        confidence_gauge(p_ai)


def tab_explanation(text, selected):
    detector = get_detector()
    kind = detector.index[selected]["kind"]
    exp = explanation(text, selected)

    if exp["kind"] == "signed":
        st.write("Words in this passage pushing the prediction toward each class. "
                 "Longer bars had more influence.")
        diverging_words(exp["toward_ai"], exp["toward_human"])
    elif exp["kind"] == "importance":
        st.write("Words from this passage that the model relies on most. These "
                 "reflect what the model learned overall, not this passage alone.")
        if exp["words"]:
            fig = go.Figure(go.Bar(
                x=[v for _, v in exp["words"]][::-1],
                y=[w for w, _ in exp["words"]][::-1],
                orientation="h", marker_color="#4C72B0"))
            fig.update_layout(height=28 * len(exp["words"]) + 80,
                              margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_title="feature importance")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("None of this passage's words were among the model's top features.")
    else:
        st.markdown('<div class="note">The neural-network models do not expose '
                    'readable weights. Use the on-demand explanation below, or see '
                    'the Statistics tab for interpretable style features.</div>',
                    unsafe_allow_html=True)

    st.divider()
    st.caption("LIME builds a local, model-agnostic explanation by probing the "
               "model with variations of your text. It works for every model but "
               "takes a few seconds.")
    if st.button(f"Explain '{selected}' with LIME"):
        with st.spinner("Probing the model..."):
            weights = lime_explanation(detector, selected, text)
        toward_ai = [(w, v) for w, v in weights if v > 0]
        toward_human = [(w, v) for w, v in weights if v < 0]
        diverging_words(toward_ai, toward_human)


def tab_highlights(text, selected):
    scored = sentence_scores(text, selected)
    if not scored:
        st.info("Not enough sentence-length text to highlight. Add a few full "
                "sentences.")
        return
    st.write("Each sentence is shaded by how AI-like the selected model finds it. "
             "Hover a sentence to see its probability.")
    spans = []
    for sent, p in scored:
        if p >= 0.5:
            bg = f"rgba(194,98,10,{0.15 + (p - 0.5) * 1.4:.2f})"
        else:
            bg = f"rgba(31,122,99,{0.15 + (0.5 - p) * 1.4:.2f})"
        safe = sent.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        spans.append(f'<span title="P(AI) {p:.0%}" style="background:{bg}; '
                     f'padding:2px 5px; border-radius:5px;">{safe}</span>')
    st.markdown(f'<div style="line-height:2.4; font-size:1.02rem;">'
                f'{" ".join(spans)}</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<span style="background:rgba(31,122,99,.6);padding:2px 8px;'
                'border-radius:5px;">human-leaning</span>&nbsp;&nbsp;'
                '<span style="background:rgba(194,98,10,.6);padding:2px 8px;'
                'border-radius:5px;">AI-leaning</span>', unsafe_allow_html=True)


def tab_statistics(text):
    stats = text_statistics(text)
    headline = [
        ("Words", f"{stats['word_count']:.0f}", None),
        ("Avg sentence length", f"{stats['avg_sentence_length']:.1f}", "words"),
        ("Sentence variation", f"{stats['sentence_length_std']:.1f}",
         "higher = more human-like burstiness"),
        ("Vocabulary richness", f"{stats['type_token_ratio']:.2f}",
         "unique / total words"),
        ("Readability (Flesch)", f"{stats['flesch_reading_ease']:.0f}",
         "higher = easier to read"),
    ]
    cols = st.columns(len(headline))
    for col, (name, value, help_) in zip(cols, headline):
        col.metric(name, value, help=help_)
    st.divider()
    table = pd.DataFrame(
        {"feature": FEATURE_NAMES, "value": [stats[f] for f in FEATURE_NAMES]}
    ).set_index("feature").round(3)
    st.dataframe(table, width="stretch", height=580)


def tab_compare(probs):
    order = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    fig = go.Figure(go.Bar(
        x=[p * 100 for _, p in order], y=[n for n, _ in order], orientation="h",
        marker_color=[AI_COLOR if p >= 0.5 else HUMAN_COLOR for _, p in order],
        text=[f"{p:.0%}" for _, p in order], textposition="auto"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="probability AI-generated (%)",
                      xaxis_range=[0, 100])
    fig.add_vline(x=50, line_dash="dash", line_color="#6b7280")
    st.plotly_chart(fig, width="stretch")

    ai_votes = sum(1 for p in probs.values() if p >= 0.5)
    total = len(probs)
    if ai_votes in (0, total):
        st.success(f"All {total} models agree: "
                   f"{'AI-generated' if ai_votes == total else 'human-written'}.")
    else:
        st.warning(f"The models disagree ({ai_votes} of {total} say AI). "
                   "Disagreement is itself a sign of low certainty.")

    table = pd.DataFrame([
        {"model": n, "P(AI)": f"{p:.1%}", "verdict": "AI" if p >= 0.5 else "Human"}
        for n, p in order]).set_index("model")
    st.dataframe(table, width="stretch")


def tab_about():
    st.subheader("How it works")
    st.write(
        "Your text is cleaned the same way the training data was, converted to "
        "features, and scored by the selected model. The three traditional models "
        "(SVM, Decision Tree, AdaBoost) read TF-IDF word features; the three neural "
        "networks (Feedforward, LSTM, CNN) were trained alongside them for "
        "comparison.")
    acc = model_accuracies()
    if acc:
        st.subheader("Test-set accuracy")
        df = pd.DataFrame({"accuracy": acc}).sort_values("accuracy", ascending=False)
        st.dataframe(df.style.format({"accuracy": "{:.1%}"}), width="stretch")
    st.subheader("Limitations")
    st.markdown(
        "- Automated detection is **probabilistic and can be wrong**. Do not treat "
        "a result as proof.\n"
        "- Detectors are known to be **biased against non-native English writers** "
        "and are **easily evaded by paraphrasing**.\n"
        "- These models were trained on one dataset from a narrow set of sources; "
        "accuracy on very different text may be lower.\n"
        "- When the models disagree or confidence is near 50%, the result is "
        "genuinely uncertain.")


# --------------------------------------------------------------------- sidebar
def read_input() -> str:
    st.sidebar.markdown("### Input")
    mode = st.sidebar.radio("Source", ["Paste text", "Upload document"],
                            label_visibility="collapsed")
    if mode == "Paste text":
        return st.sidebar.text_area("Paste text", height=220,
                                    placeholder="Paste a paragraph or more here...",
                                    label_visibility="collapsed")
    upload = st.sidebar.file_uploader("Upload a PDF, Word, or text file",
                                      type=["pdf", "docx", "txt"])
    if upload is None:
        return ""
    try:
        text = extract_text(upload.name, upload.getvalue())
        st.sidebar.success(f"Read {len(text.split()):,} words from {upload.name}")
        return text
    except ExtractionError as exc:
        st.sidebar.error(str(exc))
        return ""


def main():
    st.markdown(
        '<div class="hero"><h1>AI vs. Human Text Detector</h1>'
        '<p>Detect whether a passage was written by a person or generated by AI, '
        'compare six models, and see exactly why.</p></div>',
        unsafe_allow_html=True)

    detector = get_detector()
    text = read_input()

    st.sidebar.markdown("### Model")
    acc = model_accuracies()
    selected = st.sidebar.selectbox(
        "Model", detector.model_names, label_visibility="collapsed",
        format_func=lambda n: f"{n}  ({acc.get(n, 0):.0%})" if acc else n)

    analyze = st.sidebar.button("Analyze", type="primary", width="stretch")
    if analyze and text.strip():
        st.session_state["text"] = text
    if analyze and not text.strip():
        st.sidebar.error("Enter or upload some text first.")

    stored = st.session_state.get("text", "")
    if not stored:
        st.markdown("#### Try it instantly")
        e1, e2 = st.columns(2)
        if e1.button("Load a human-written example", width="stretch"):
            st.session_state["text"] = EXAMPLE_HUMAN
            st.rerun()
        if e2.button("Load an AI-generated example", width="stretch"):
            st.session_state["text"] = EXAMPLE_AI
            st.rerun()

        st.write("")
        h1, h2, h3 = st.columns(3)
        h1.markdown("**1. Add text**  \nPaste text or upload a PDF or Word file "
                    "in the sidebar.")
        h2.markdown("**2. Pick a model**  \nChoose any of the six trained "
                    "classifiers.")
        h3.markdown("**3. See why**  \nGet a verdict, confidence, and a word- and "
                    "sentence-level explanation.")
        st.divider()
        tab_about()
        return

    if len(stored.split()) < 20:
        st.warning("Short passages are hard to classify reliably. For a trustworthy "
                   "result, use at least a few sentences.")

    probs = all_probabilities(stored)
    tabs = st.tabs(["Prediction", "Highlights", "Explanation", "Statistics",
                    "Compare models", "About"])
    with tabs[0]:
        tab_prediction(stored, probs, selected)
    with tabs[1]:
        tab_highlights(stored, selected)
    with tabs[2]:
        tab_explanation(stored, selected)
    with tabs[3]:
        tab_statistics(stored)
    with tabs[4]:
        tab_compare(probs)
    with tabs[5]:
        tab_about()

    # Downloadable report
    p_ai = probs[selected]
    prediction = {"label": "AI" if p_ai >= 0.5 else "Human",
                  "confidence": max(p_ai, 1 - p_ai)}
    stats = text_statistics(stored)
    st.sidebar.markdown("### Report")
    st.sidebar.download_button(
        "Download PDF", pdf_report(stored, selected, prediction, probs, stats),
        file_name="detection_report.pdf", mime="application/pdf",
        width="stretch")
    st.sidebar.download_button(
        "Download text", text_report(stored, selected, prediction, probs, stats),
        file_name="detection_report.txt", mime="text/plain",
        width="stretch")


if __name__ == "__main__":
    main()
