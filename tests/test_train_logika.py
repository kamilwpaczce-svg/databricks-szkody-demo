"""Czysta logika treningu i bramki: split, straznik klas, wybor modelu, plan tagow/aliasu."""

import numpy as np
import pandas as pd
import pytest

from szkody.cechy import CECHY
from szkody.generator import ETYKIETA, generuj_zbior
from szkody.zadania.train_logika import (
    UDZIAL_TESTU,
    metryki_z_tagow,
    przygotuj_dane,
    sprawdz_klasy,
    wybierz_lepszy,
    zaplanuj_bramke,
    znormalizuj_typy,
)

METRYKI = {"pr_auc": 0.60, "roc_auc": 0.80, "brier": 0.10}


def test_split_jak_w_etapie_15_30_proc_stratyfikowany_deterministyczny():
    df = generuj_zbior(3500, 1)
    train, test = przygotuj_dane(df, 1)
    assert len(train) + len(test) == 3500
    assert len(test) == pytest.approx(3500 * UDZIAL_TESTU, abs=1)
    assert train[ETYKIETA].mean() == pytest.approx(test[ETYKIETA].mean(), abs=0.01)  # stratyfikacja
    train2, test2 = przygotuj_dane(df, 1)
    pd.testing.assert_frame_equal(train, train2)
    pd.testing.assert_frame_equal(test, test2)


def test_split_identyczny_z_bezposrednim_wywolaniem_etapu_15():
    from sklearn.model_selection import train_test_split

    df = generuj_zbior(3500, 3)
    oczekiwany, _ = train_test_split(df, test_size=0.3, stratify=df[ETYKIETA], random_state=3)
    train, _ = przygotuj_dane(df, 3)
    assert list(train["suma_ubezpieczenia"]) == list(oczekiwany["suma_ubezpieczenia"])


def test_typy_po_normalizacji_jak_po_sparku_int32():
    df = generuj_zbior(50, 1)
    for kolumna in ("dni_do_zgloszenia", "liczba_zalacznikow", "szkody_3_lata", "dni_od_poczatku_polisy", ETYKIETA):
        df[kolumna] = df[kolumna].astype("int32")
    wynik = znormalizuj_typy(df)
    assert wynik["dni_do_zgloszenia"].dtype == np.int64
    assert wynik[ETYKIETA].dtype == np.int64
    assert wynik["udzial_kwoty"].dtype == np.float64


def test_pusty_zbior_to_blad():
    with pytest.raises(ValueError, match="Pusty zbior"):
        przygotuj_dane(generuj_zbior(10, 1).iloc[0:0], 1)


def test_zbior_z_jedna_klasa_to_glosny_blad():
    df = generuj_zbior(500, 1)
    df[ETYKIETA] = 0
    with pytest.raises(ValueError, match="jedna klase"):
        przygotuj_dane(df, 1)


def test_sprawdz_klasy_dwie_klasy_ok():
    sprawdz_klasy(pd.Series([0, 1, 0]), "x")
    with pytest.raises(ValueError):
        sprawdz_klasy(pd.Series([1, 1]), "x")


def test_wybierz_lepszy_po_pr_auc():
    assert wybierz_lepszy({"hgb": {"pr_auc": 0.7}, "logreg": {"pr_auc": 0.6}}) == "hgb"
    assert wybierz_lepszy({"hgb": {"pr_auc": 0.5}, "logreg": {"pr_auc": 0.6}}) == "logreg"


def test_wybierz_lepszy_remis_daje_pierwszy_z_rodzajow():
    assert wybierz_lepszy({"hgb": {"pr_auc": 0.6}, "logreg": {"pr_auc": 0.6}}) == "hgb"


def test_wybierz_lepszy_nan_to_blad():
    with pytest.raises(ValueError, match="NaN"):
        wybierz_lepszy({"hgb": {"pr_auc": float("nan")}, "logreg": {"pr_auc": 0.6}})


def test_metryki_z_tagow_i_bledy():
    assert metryki_z_tagow({"pr_auc": "0.612300", "inne": "x"}) == {"pr_auc": 0.6123}
    with pytest.raises(ValueError, match="nie ma tagu"):
        metryki_z_tagow({"roc_auc": "0.8"})
    with pytest.raises(ValueError):
        metryki_z_tagow({"pr_auc": "nan"})


def test_plan_pierwsza_wersja_promocja_z_aliasem_champion():
    plan = zaplanuj_bramke(None, None, METRYKI, "hgb", 1, "1")
    assert plan.promuj is True and plan.alias == "champion"
    assert plan.tagi_wersji["bramka"] == "promowany"
    assert plan.tagi_wersji["pr_auc"] == "0.600000"
    assert plan.tagi_wersji["rodzaj"] == "hgb" and plan.tagi_wersji["partia"] == "1"
    assert plan.poprzedni_champion is None and plan.tagi_poprzedniego == {}
    assert "brak championa" in plan.tagi_wersji["bramka_powod"]


def test_plan_promocja_wycofuje_poprzedniego():
    plan = zaplanuj_bramke({"pr_auc": 0.58}, "3", METRYKI, "logreg", 2, "4")
    assert plan.promuj is True and plan.alias == "champion"
    assert plan.poprzedni_champion == "3"
    assert plan.tagi_poprzedniego == {"bramka": "wycofany", "bramka_powod": "zastapiony przez wersje 4"}


def test_plan_granica_tolerancji_promuje():
    plan = zaplanuj_bramke({"pr_auc": 0.61}, "1", METRYKI, "hgb", 2, "2")  # 0.60 == 0.61 - 0.01
    assert plan.promuj is True


def test_plan_odrzucenie_bez_aliasu_z_tagiem_i_powodem():
    plan = zaplanuj_bramke({"pr_auc": 0.80}, "1", METRYKI, "hgb", 2, "2")
    assert plan.promuj is False and plan.alias is None
    assert plan.tagi_wersji["bramka"] == "odrzucony"
    assert "odrzucony" in plan.tagi_wersji["bramka_powod"] and "ponizej progu" in plan.tagi_wersji["bramka_powod"]
    assert plan.poprzedni_champion is None and plan.tagi_poprzedniego == {}


def test_kolumny_cech_sa_w_planie_danych():
    train, _ = przygotuj_dane(generuj_zbior(500, 1), 1)
    assert list(train.columns) == [*CECHY, ETYKIETA]
