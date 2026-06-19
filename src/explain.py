"""Explain why a model made its prediction.

Three cases, in order of how directly interpretable the model is:

- Linear SVM: the prediction is a weighted sum, so each word's signed
  contribution to *this* passage is exact (tf-idf value times its weight).
- Decision Tree / AdaBoost: only a global `feature_importances_` is available,
  so we show the influential words that appear in the passage and say plainly
  that this reflects the model overall, not this passage alone.
- Neural networks: no readable weights. We say so and rely on the linguistic
  statistics panel instead.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from .preprocessing import clean_text


def _linear_coef(model) -> np.ndarray | None:
    """Class-1 coefficient vector for a linear model, averaged if calibrated."""
    if isinstance(model, CalibratedClassifierCV):
        coefs = []
        for cc in model.calibrated_classifiers_:
            est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
            if est is not None and hasattr(est, "coef_"):
                coefs.append(est.coef_.ravel())
        return np.mean(coefs, axis=0) if coefs else None
    if hasattr(model, "coef_"):
        return model.coef_.ravel()
    return None


def explain_prediction(detector, model_name: str, text: str, top_k: int = 10) -> dict:
    """Return the words that drove the prediction, shaped for display."""
    model = detector.models[model_name]
    names = detector.feature_names
    x = detector.tfidf.transform([clean_text(text)])
    present = x.toarray().ravel()

    coef = _linear_coef(model)
    if coef is not None:
        contrib = present * coef
        nz = np.nonzero(contrib)[0]
        order = nz[np.argsort(contrib[nz])]
        toward_human = [(names[i], float(contrib[i])) for i in order[:top_k]]
        toward_ai = [(names[i], float(contrib[i])) for i in order[::-1][:top_k]]
        return {"kind": "signed",
                "toward_ai": [t for t in toward_ai if t[1] > 0],
                "toward_human": [t for t in toward_human if t[1] < 0]}

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_ * (present > 0)
        idx = np.argsort(importance)[::-1]
        top = [(names[i], float(importance[i])) for i in idx[:top_k] if importance[i] > 0]
        return {"kind": "importance", "words": top}

    return {"kind": "none"}


def lime_explanation(detector, model_name: str, text: str,
                     num_features: int = 10, num_samples: int = 400):
    """Model-agnostic explanation via LIME. Works for any model, including the
    neural networks, but is slower, so the app runs it only on request."""
    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer(class_names=["Human", "AI"])

    def predict_proba(texts):
        p_ai = detector.proba_from_raw(model_name, list(texts))
        return np.column_stack([1.0 - p_ai, p_ai])

    exp = explainer.explain_instance(
        clean_text(text), predict_proba,
        num_features=num_features, num_samples=num_samples)
    return exp.as_list()  # [(word, weight toward AI), ...]
