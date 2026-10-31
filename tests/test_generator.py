"""Generator: deterministycznosc i zgodnosc kolumn (kopia z Etapu 15 musi dzialac tak samo)."""

import pandas as pd
import pytest

from szkody.cechy import CECHY
from szkody.generator import ETYKIETA, generuj_zbior


def test_kolumny_to_cechy_i_etykieta():
    df = generuj_zbior(200, 1)
    assert list(df.columns) == [*CECHY, ETYKIETA]
    assert len(df) == 200


def test_deterministyczny_dla_seeda():
    pd.testing.assert_frame_equal(generuj_zbior(500, 7), generuj_zbior(500, 7))


def test_rozne_seedy_daja_rozne_dane():
    assert not generuj_zbior(500, 1).equals(generuj_zbior(500, 2))


def test_etykieta_tylko_0_i_1_i_obie_klasy_wystepuja():
    df = generuj_zbior(3500, 1)
    assert set(df[ETYKIETA].unique()) == {0, 1}


def test_nieznana_cecha_w_przesunieciu_to_blad():
    with pytest.raises(ValueError, match="Nieznane cechy"):
        generuj_zbior(10, 1, {"nie_ma_takiej": 1.0})


def test_przesuniecie_zmienia_dane_ale_nie_ksztalt():
    baza = generuj_zbior(500, 1)
    dryf = generuj_zbior(500, 1, {"udzial_kwoty": 0.3})
    assert list(dryf.columns) == list(baza.columns)
    assert dryf["udzial_kwoty"].mean() > baza["udzial_kwoty"].mean()
