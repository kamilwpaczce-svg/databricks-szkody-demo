"""Warstwa MLflow na PRAWDZIWYM MLflow (sqlite w katalogu tymczasowym), bez Databricks.

Sprawdza kolejnosc krokow, tagi i alias. NIE sprawdza Unity Catalog (osobne zachowania REST) -
to wychodzi dopiero na workspace (zadanie 3).
"""

import mlflow
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, RESOURCE_DOES_NOT_EXIST

from szkody.generator import generuj_zbior
from szkody.zadania.mlops import brak_zasobu, pobierz_championa, trenuj_i_zarejestruj
from szkody.zadania.train_logika import przygotuj_dane

NAZWA = "szkoda_test"


@pytest.fixture()
def mlflow_lokalny(tmp_path):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    id_eksperymentu = mlflow.create_experiment("szkody_test", artifact_location=(tmp_path / "artefakty").as_uri())
    mlflow.set_experiment(experiment_id=id_eksperymentu)
    yield MlflowClient()
    mlflow.set_tracking_uri(None)
    mlflow.set_registry_uri(None)


@pytest.fixture(scope="module")
def dane():
    return przygotuj_dane(generuj_zbior(1500, 1), 1)


def uruchom(dane, seed=1):
    train, test = dane
    return trenuj_i_zarejestruj(train, test, seed, NAZWA, {"partia": str(seed), "przesuniecie": "{}"})


def test_brak_zasobu_rozpoznaje_tylko_nieistniejacy_zasob():
    assert brak_zasobu(MlflowException("x", error_code=RESOURCE_DOES_NOT_EXIST)) is True
    assert brak_zasobu(MlflowException("x", error_code=INTERNAL_ERROR)) is False
    assert brak_zasobu(ValueError("x")) is False


def test_brak_championa_gdy_modelu_nie_ma(mlflow_lokalny):
    assert pobierz_championa(mlflow_lokalny, NAZWA) is None


def test_pierwszy_bieg_promuje_i_ustawia_alias_tagi_i_sygnature(mlflow_lokalny, dane):
    wynik = uruchom(dane)
    assert wynik["promowany"] is True and wynik["champion_przed"] is None and wynik["champion_po"] == "1"
    assert wynik["rodzaj"] in ("hgb", "logreg") and set(wynik["wszystkie"]) == {"hgb", "logreg"}

    wersja = mlflow_lokalny.get_model_version_by_alias(NAZWA, "champion")
    assert str(wersja.version) == "1"  # lokalny sqlite zwraca int, UC str - kod uzywa str()
    assert wersja.tags["bramka"] == "promowany"
    assert float(wersja.tags["pr_auc"]) == pytest.approx(wynik["metryki"]["pr_auc"], abs=1e-6)
    assert {"roc_auc", "brier", "rodzaj", "partia", "bramka_powod"} <= set(wersja.tags)
    info = mlflow.models.get_model_info(f"models:/{NAZWA}/1")
    assert info.signature is not None and info.saved_input_example_info is not None

    # Oba modele zalogowane jako osobne runy z metrykami (jeden bieg, zeby nie trenowac drugi raz).
    runy = mlflow.search_runs(output_format="list")
    assert sorted(r.data.params["rodzaj"] for r in runy) == ["hgb", "logreg"]
    assert all({"pr_auc", "roc_auc", "brier"} <= set(r.data.metrics) for r in runy)
    assert all(r.data.tags["partia"] == "1" for r in runy)


def test_odrzucenie_zostawia_alias_i_oznacza_wersje_bez_wyjatku(mlflow_lokalny, dane):
    uruchom(dane)
    mlflow_lokalny.set_model_version_tag(NAZWA, "1", "pr_auc", "0.999000")  # champion nie do pobicia

    wynik = uruchom(dane, seed=1)  # nie rzuca - odrzucenie to wynik, nie blad
    assert wynik["promowany"] is False and wynik["champion_po"] == "1" and wynik["wersja"] == "2"

    assert str(mlflow_lokalny.get_model_version_by_alias(NAZWA, "champion").version) == "1"
    odrzucona = mlflow_lokalny.get_model_version(NAZWA, "2")
    assert odrzucona.tags["bramka"] == "odrzucony" and "ponizej progu" in odrzucona.tags["bramka_powod"]
    assert not odrzucona.aliases


def test_promocja_przenosi_alias_i_wycofuje_poprzedniego(mlflow_lokalny, dane):
    uruchom(dane)
    mlflow_lokalny.set_model_version_tag(NAZWA, "1", "pr_auc", "0.000000")  # slaby champion

    wynik = uruchom(dane, seed=1)
    assert wynik["promowany"] is True and wynik["champion_przed"] == "1" and wynik["champion_po"] == "2"
    assert str(mlflow_lokalny.get_model_version_by_alias(NAZWA, "champion").version) == "2"
    stary = mlflow_lokalny.get_model_version(NAZWA, "1")
    assert stary.tags["bramka"] == "wycofany" and "wersje 2" in stary.tags["bramka_powod"]


def test_champion_bez_tagu_metryki_to_glosny_blad(mlflow_lokalny, dane):
    uruchom(dane)
    mlflow_lokalny.delete_model_version_tag(NAZWA, "1", "pr_auc")
    with pytest.raises(ValueError, match="nie ma tagu"):
        uruchom(dane)
