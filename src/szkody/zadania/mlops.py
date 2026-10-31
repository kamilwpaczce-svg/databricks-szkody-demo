"""Cienka warstwa MLflow dla `train.py` (Etap 22): trening, logowanie, rejestracja, bramka.

Wszystkie DECYZJE sa w `train_logika` (czyste funkcje). Tu tylko wywolania MLflow, dzieki
czemu da sie to przetestowac lokalnie na MLflow ze sqlite (bez Databricks); URI rejestru
(`databricks-uc`) ustawia skrypt `train.py`, nie ta warstwa.
"""

from __future__ import annotations

from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature

from szkody.cechy import CECHY
from szkody.generator import ETYKIETA
from szkody.model import RODZAJE_MODELU, ocen, zbuduj_pipeline
from szkody.zadania.train_logika import (
    ALIAS_CHAMPIONA,
    metryki_z_tagow,
    wybierz_lepszy,
    zaplanuj_bramke,
)

_KODY_BRAKU = {"RESOURCE_DOES_NOT_EXIST", "NOT_FOUND"}
_PROBKA_SYGNATURY = 100


def brak_zasobu(blad: Exception) -> bool:
    """True tylko dla 'nie ma takiego modelu/aliasu' (kod albo HTTP 404); inne bledy (np. 401) lecimy dalej."""
    if not isinstance(blad, MlflowException):
        return False
    return blad.error_code in _KODY_BRAKU or blad.get_http_status_code() == 404


def _argument_nazwy_modelu() -> dict[str, str]:
    # MLflow 3 woli `name=`, starszy (klastry z MLflow 2.x) zna tylko `artifact_path=`.
    wersja_glowna = int(mlflow.__version__.split(".")[0])
    return {"name": "model"} if wersja_glowna >= 3 else {"artifact_path": "model"}


def pobierz_championa(klient: MlflowClient, nazwa: str) -> tuple[str, dict[str, float]] | None:
    """(wersja, metryki z tagow) championa wskazanego aliasem albo None, gdy aliasu/modelu nie ma."""
    try:
        wersja = klient.get_model_version_by_alias(nazwa, ALIAS_CHAMPIONA)
    except MlflowException as e:
        if brak_zasobu(e):
            return None
        raise
    return str(wersja.version), metryki_z_tagow(dict(wersja.tags or {}))


def trenuj_modele(train: pd.DataFrame, test: pd.DataFrame, seed: int, tagi_runu: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Trenuje `hgb` i `logreg`, kazdy w osobnym runie MLflow (parametry, metryki, model z sygnatura).

    Zwraca `{rodzaj: {"metryki", "run_id", "model_uri"}}`. Modeli NIE rejestruje - to robi bramka.
    """
    wyniki: dict[str, dict[str, Any]] = {}
    probka = train[CECHY].head(_PROBKA_SYGNATURY)
    for rodzaj in RODZAJE_MODELU:
        with mlflow.start_run(run_name=rodzaj) as run:
            model = zbuduj_pipeline(rodzaj).fit(train[CECHY], train[ETYKIETA])
            metryki = ocen(model, test[CECHY], test[ETYKIETA])
            mlflow.set_tags(tagi_runu)
            mlflow.log_params({"rodzaj": rodzaj, "seed": seed, "wiersze_train": len(train), "wiersze_test": len(test)})
            mlflow.log_metrics(metryki)
            info = mlflow.sklearn.log_model(
                model,
                signature=infer_signature(probka, model.predict_proba(probka)),
                input_example=probka.head(5),
                pyfunc_predict_fn="predict_proba",
                # MLflow 3.17 domyslnie uzywa skops, ktory odrzuca HistGradientBoosting (TreePredictor
                # "untrusted") - jawny pickle dziala tak samo na starszym i nowszym MLflow.
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
                **_argument_nazwy_modelu(),
            )
            wyniki[rodzaj] = {"metryki": metryki, "run_id": run.info.run_id, "model_uri": info.model_uri}
    return wyniki


def zarejestruj_z_bramka(
    klient: MlflowClient, nazwa: str, model_uri: str, metryki: dict[str, float], rodzaj: str, partia: int
) -> dict[str, Any]:
    """Rejestruje wersje, puszcza przez bramke, ustawia tagi i (przy promocji) alias `champion`.

    Odrzucenie NIE jest wyjatkiem: wersja zostaje bez aliasu z tagiem `bramka=odrzucony`.
    """
    champion = pobierz_championa(klient, nazwa)
    wersja_championa, metryki_championa = champion if champion else (None, None)
    wersja = str(mlflow.register_model(model_uri, nazwa).version)
    plan = zaplanuj_bramke(metryki_championa, wersja_championa, metryki, rodzaj, partia, wersja)
    for klucz, wartosc in plan.tagi_wersji.items():
        klient.set_model_version_tag(nazwa, wersja, klucz, wartosc)
    if plan.promuj:
        for klucz, wartosc in plan.tagi_poprzedniego.items():
            klient.set_model_version_tag(nazwa, plan.poprzedni_champion, klucz, wartosc)
        # Ostatni krok: dopiero tu alias przechodzi na nowa wersje.
        klient.set_registered_model_alias(nazwa, plan.alias, wersja)
    return {
        "wersja": wersja,
        "promowany": plan.promuj,
        "powod": plan.powod,
        "champion_przed": wersja_championa,
        "champion_po": wersja if plan.promuj else wersja_championa,
    }


def trenuj_i_zarejestruj(
    train: pd.DataFrame, test: pd.DataFrame, seed: int, nazwa_modelu: str, tagi_runu: dict[str, str]
) -> dict[str, Any]:
    """Pelny krok treningu: oba modele -> wybor lepszego -> rejestracja + bramka. Zwraca podsumowanie."""
    wyniki = trenuj_modele(train, test, seed, tagi_runu)
    lepszy = wybierz_lepszy({r: w["metryki"] for r, w in wyniki.items()})
    wybrany = wyniki[lepszy]
    wynik = zarejestruj_z_bramka(MlflowClient(), nazwa_modelu, wybrany["model_uri"], wybrany["metryki"], lepszy, seed)
    wynik.update({"rodzaj": lepszy, "metryki": wybrany["metryki"], "wszystkie": {r: w["metryki"] for r, w in wyniki.items()}})
    return wynik
