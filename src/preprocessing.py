"""Dataset loading and text cleaning.

The cleaning here is deliberately light. Detecting AI vs. human writing is a
stylometric problem (telling two writing styles apart), not a topic problem.
Function words, punctuation, and casing carry much of the signal, so unlike a
typical topic-classification pipeline we do not strip stopwords or punctuation.
We only normalize whitespace, which document parsers tend to mangle.
"""
from __future__ import annotations

import re
from pathlib import Path

import ftfy
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DATA = PROJECT_ROOT / "data" / "training_data" / "train_data_with_labels.xlsx"

LABEL_NAMES = {0: "Human", 1: "AI"}

_WHITESPACE = re.compile(r"\s+")


def load_dataset(path: Path | str = TRAIN_DATA) -> pd.DataFrame:
    """Load the labeled passages and remove exact duplicates and empty rows.

    The text is run through ftfy to repair mojibake. A large share of passages
    contain double-encoded smart quotes (for example "don‚Äôt" for "don't");
    left unrepaired these turn into garbage tokens and an encoding artifact the
    classifier could exploit. Repairing first also lets us deduplicate passages
    that differ only by encoding.
    """
    df = pd.read_excel(path)
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].map(lambda t: ftfy.fix_text(str(t)))
    df = df[df["text"].str.strip().astype(bool)]
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to single spaces and trim the ends.

    Punctuation, casing, and stopwords are left untouched on purpose.
    """
    return _WHITESPACE.sub(" ", str(text)).strip()


def clean_text(text: str) -> str:
    """Repair encoding then normalize whitespace.

    This is the single entry point the Streamlit app uses on user-supplied text
    so it gets exactly the same treatment as the training data.
    """
    return normalize_whitespace(ftfy.fix_text(str(text)))
