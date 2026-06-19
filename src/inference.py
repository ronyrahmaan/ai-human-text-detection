"""Load the trained models and run predictions.

This is the single bridge between the artifacts saved by the notebook and the
Streamlit app. Every model is fed exactly the way it was trained: the sklearn
models and the feedforward net read TF-IDF, the LSTM and CNN read integer token
sequences. Raw text is always passed through the same `clean_text` used in
training, so the app and the notebook agree.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from .preprocessing import PROJECT_ROOT, clean_text

MODELS_DIR = PROJECT_ROOT / "models"
SEQ_LEN = 300  # must match training (Section 3)


class Detector:
    """Holds every trained model and produces probabilities from raw text."""

    def __init__(self, models_dir: Path | str = MODELS_DIR):
        import keras  # heavy import kept local so module import stays cheap

        self.models_dir = Path(models_dir)
        with open(self.models_dir / "model_index.json") as f:
            self.index = json.load(f)

        self.tfidf = joblib.load(self.models_dir / "tfidf_vectorizer.pkl")
        self.feature_names = self.tfidf.get_feature_names_out()

        self.models = {}
        for name, meta in self.index.items():
            path = self.models_dir / meta["file"]
            self.models[name] = (joblib.load(path) if meta["kind"] == "sklearn"
                                 else keras.models.load_model(path))

        # Rebuild the sequence vectorizer from the saved vocabulary.
        with open(self.models_dir / "keras_text_vocab.json") as f:
            vocab = json.load(f)
        real_tokens = [t for t in vocab if t not in ("", "[UNK]")]
        self.text_vectorizer = keras.layers.TextVectorization(
            max_tokens=len(vocab), output_mode="int",
            output_sequence_length=SEQ_LEN)
        self.text_vectorizer.set_vocabulary(real_tokens)

    @property
    def model_names(self) -> list[str]:
        return list(self.index)

    def _proba(self, model_name: str, cleaned_texts: list[str]) -> np.ndarray:
        """P(AI) for a list of already-cleaned passages."""
        kind = self.index[model_name]["kind"]
        model = self.models[model_name]
        if kind == "sklearn":
            X = self.tfidf.transform(cleaned_texts)
            return model.predict_proba(X)[:, 1]
        if kind == "keras_tfidf":
            X = self.tfidf.transform(cleaned_texts).toarray().astype("float32")
            return model.predict(X, verbose=0).ravel()
        seq = self.text_vectorizer(np.array(cleaned_texts)).numpy()  # keras_seq
        return model.predict(seq, verbose=0).ravel()

    def predict(self, text: str, model_name: str) -> dict:
        """Single prediction with a label and confidence."""
        p = float(self._proba(model_name, [clean_text(text)])[0])
        return {
            "model": model_name,
            "p_ai": p,
            "label": "AI" if p >= 0.5 else "Human",
            "confidence": max(p, 1.0 - p),
        }

    def predict_all(self, text: str) -> dict[str, float]:
        """P(AI) from every model, for the side-by-side comparison."""
        cleaned = [clean_text(text)]
        return {name: float(self._proba(name, cleaned)[0]) for name in self.index}

    def proba_from_raw(self, model_name: str, texts: list[str]) -> np.ndarray:
        """Probabilities for raw (uncleaned) texts. Used by LIME."""
        return self._proba(model_name, [clean_text(t) for t in texts])

    def sentence_scores(self, model_name: str, text: str,
                        min_words: int = 4) -> list[tuple[str, float]]:
        """Score each sentence so the app can highlight which parts read as AI."""
        from nltk.tokenize import sent_tokenize

        sentences = [s.strip() for s in sent_tokenize(clean_text(text)) if s.strip()]
        scored = [s for s in sentences if len(s.split()) >= min_words]
        if not scored:
            return []
        probs = self._proba(model_name, scored)
        return list(zip(scored, (float(p) for p in probs)))
