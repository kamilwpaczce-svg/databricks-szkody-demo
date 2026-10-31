"""Model: oba pipeline'y trenuja sie, `ocen` zwraca 3 metryki."""

import pytest
from sklearn.model_selection import train_test_split

from szkody.cechy import CECHY
from szkody.generator import ETYKIETA, generuj_zbior
from szkody.model import RODZAJE_MODELU, ocen, zbuduj_pipeline


@pytest.fixture(scope="module")
def podzial():
    df = generuj_zbior(3500, 1)
    train, test = train_test_split(df, test_size=0.3, stratify=df[ETYKIETA], random_state=1)
    return train, test


@pytest.mark.parametrize("rodzaj", RODZAJE_MODELU)
def test_pipeline_trenuje_sie_i_ocena_zwraca_trzy_metryki(rodzaj, podzial):
    train, test = podzial
    model = zbuduj_pipeline(rodzaj).fit(train[CECHY], train[ETYKIETA])
    metryki = ocen(model, test[CECHY], test[ETYKIETA])
    assert set(metryki) == {"roc_auc", "pr_auc", "brier"}
    assert 0.5 < metryki["roc_auc"] <= 1.0
    assert 0.0 < metryki["pr_auc"] <= 1.0
    assert 0.0 <= metryki["brier"] < 0.5


def test_nieznany_rodzaj_modelu_to_blad():
    with pytest.raises(ValueError, match="Nieznany rodzaj"):
        zbuduj_pipeline("xgb")


def test_ocena_na_jednej_klasie_nie_rzuca_wyjatku_w_tej_wersji_sklearn(podzial):
    # Docstring `ocen` (kopia z Etapu 15) obiecuje ValueError, ale sklearn 1.9.1 daje tylko
    # ostrzezenie i roc_auc=NaN, pr_auc=0.0. Dlatego train.py ma wlasny straznik klas
    # (`zadania.train_logika.sprawdz_klasy`), a ten test przypina faktyczne zachowanie.
    train, test = podzial
    model = zbuduj_pipeline("logreg").fit(train[CECHY], train[ETYKIETA])
    tylko_zera = test[test[ETYKIETA] == 0]
    with pytest.warns(UserWarning):
        metryki = ocen(model, tylko_zera[CECHY], tylko_zera[ETYKIETA])
    assert metryki["roc_auc"] != metryki["roc_auc"]  # NaN
