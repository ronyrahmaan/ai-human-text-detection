"""Handcrafted linguistic features.

These capture writing style rather than topic: how long sentences are and how
much that varies, vocabulary richness, punctuation habits, and readability.
They are computed on the original text (before whitespace normalization) so
casing, digits, and punctuation are preserved.

Used by both the training notebook and the Streamlit app.
"""
from __future__ import annotations

import re

import numpy as np
import textstat
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords


def _ensure_nltk_data():
    """Download the corpora we rely on if they are not already present.

    Keeps the app working on a fresh server (e.g. a cloud deploy) where NLTK
    data has not been installed. No-op once the data is on disk.
    """
    for resource in ("punkt", "punkt_tab", "stopwords", "wordnet"):
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)


_ensure_nltk_data()
_STOPWORDS = set(stopwords.words("english"))
_WORD = re.compile(r"[A-Za-z']+")

# Order matters: this is the column order of the feature matrix.
FEATURE_NAMES = [
    "word_count",
    "avg_word_length",
    "avg_sentence_length",
    "sentence_length_std",      # burstiness: humans vary sentence length more
    "type_token_ratio",         # vocabulary richness
    "hapax_ratio",              # share of words used exactly once
    "stopword_ratio",
    "uppercase_ratio",
    "digit_ratio",
    "comma_rate",               # the *_rate features are per 100 words
    "semicolon_rate",
    "quote_rate",
    "exclamation_rate",
    "question_rate",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
]


def linguistic_features(text: str) -> dict[str, float]:
    """Return the named linguistic features for one passage."""
    text = str(text)
    words = _WORD.findall(text.lower())
    n_words = len(words)
    letters = [c for c in text if c.isalpha()]

    if n_words == 0:
        return {name: 0.0 for name in FEATURE_NAMES}

    sentences = sent_tokenize(text) or [text]
    sent_word_counts = [len(_WORD.findall(s)) for s in sentences]

    unique = set(words)
    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    hapax = sum(1 for c in counts.values() if c == 1)

    per_100 = 100.0 / n_words
    feats = {
        "word_count": float(n_words),
        "avg_word_length": float(np.mean([len(w) for w in words])),
        "avg_sentence_length": float(np.mean(sent_word_counts)),
        "sentence_length_std": float(np.std(sent_word_counts)),
        "type_token_ratio": len(unique) / n_words,
        "hapax_ratio": hapax / n_words,
        "stopword_ratio": sum(1 for w in words if w in _STOPWORDS) / n_words,
        "uppercase_ratio": (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0.0,
        "digit_ratio": sum(1 for c in text if c.isdigit()) / len(text) if text else 0.0,
        "comma_rate": text.count(",") * per_100,
        "semicolon_rate": text.count(";") * per_100,
        "quote_rate": (text.count('"') + text.count("'")) * per_100,
        "exclamation_rate": text.count("!") * per_100,
        "question_rate": text.count("?") * per_100,
        "flesch_reading_ease": float(textstat.flesch_reading_ease(text)),
        "flesch_kincaid_grade": float(textstat.flesch_kincaid_grade(text)),
    }
    return feats


def feature_matrix(texts) -> np.ndarray:
    """Stack linguistic features for many passages into a 2-D array."""
    rows = [linguistic_features(t) for t in texts]
    return np.array([[r[name] for name in FEATURE_NAMES] for r in rows], dtype=float)
