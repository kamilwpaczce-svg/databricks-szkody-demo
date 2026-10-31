"""Pipeline scikit-learn "szkoda do weryfikacji" i metryki (Etap 15).

Skopiowane z underwriter_agent_demo@a1e0a3e, Etap 15.

`hgb` = ColumnTransformer + HistGradientBoostingClassifier (model docelowy),
`logreg` = ColumnTransformer + LogisticRegression (baseline). Dane syntetyczne -
porownanie ma pokazac cykl MLOps, a nie jakosc modelu.
"""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from szkody.cechy import CECHY_KATEGORIALNE, CECHY_LICZBOWE

RODZAJE_MODELU = ("hgb", "logreg")


def _kodowanie_kategorii() -> OneHotEncoder:
    # handle_unknown="ignore": nieznany rodzaj mienia w serwowaniu daje wiersz zer, nie wyjatek.
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def zbuduj_pipeline(rodzaj: str) -> Pipeline:
    """Etap 15: niedopasowany Pipeline dla `rodzaj` in {"hgb", "logreg"}."""
    if rodzaj == "hgb":
        przetwarzanie = ColumnTransformer(
            [("liczby", "passthrough", CECHY_LICZBOWE), ("kategorie", _kodowanie_kategorii(), CECHY_KATEGORIALNE)]
        )
        # Silna regularyzacja: przy 3500 wierszach i ~15% pozytywnych domyslne HGB przeucza sie
        # i przegrywa z logreg na PR AUC (zmierzone 26.10, Etap 15).
        klasyfikator: Any = HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.05,
            max_leaf_nodes=6,
            min_samples_leaf=50,
            l2_regularization=2.0,
            early_stopping=False,
            random_state=0,
        )
    elif rodzaj == "logreg":
        przetwarzanie = ColumnTransformer(
            [("liczby", StandardScaler(), CECHY_LICZBOWE), ("kategorie", _kodowanie_kategorii(), CECHY_KATEGORIALNE)]
        )
        klasyfikator = LogisticRegression(max_iter=1000, random_state=0)
    else:
        raise ValueError(f"Nieznany rodzaj modelu {rodzaj!r}; dozwolone: {RODZAJE_MODELU}")
    return Pipeline([("przetwarzanie", przetwarzanie), ("klasyfikator", klasyfikator)])


def ocen(model: Pipeline, X, y) -> dict[str, float]:
    """Etap 15: `roc_auc`, `pr_auc` (average precision - glowna metryka bramki), `brier`.

    Zbior z jedna klasa w `y` -> ValueError z scikit-learn (glosno, bez metryk-zludzen).
    """
    prawdopodobienstwa = model.predict_proba(X)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y, prawdopodobienstwa)),
        "pr_auc": float(average_precision_score(y, prawdopodobienstwa)),
        "brier": float(brier_score_loss(y, prawdopodobienstwa)),
    }
