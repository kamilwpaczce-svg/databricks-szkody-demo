"""Reguly jakosci: te same napisy SQL, ktore wykona Spark, sprawdzone lokalnie w DuckDB.

DuckDB wybrany zamiast rownoleglej implementacji w Pythonie: test wykonuje DOKLADNIE
wyrazenia z `jakosc.py`, wiec nie powstaje druga kopia logiki, ktora moglaby sie rozjechac.
"""

import duckdb
import pandas as pd
import pytest

from szkody.cechy import CECHY, RODZAJE_MIENIA
from szkody.generator import _ZAKRESY, ETYKIETA, generuj_zbior
from szkody.jakosc import REGULY, wyrazenie_poprawny, wyrazenie_powod


def ocen_wiersze(df: pd.DataFrame) -> pd.DataFrame:
    """Dopisuje kolumny `poprawny` i `powod` z wyrazen SQL (DuckDB)."""
    polaczenie = duckdb.connect()
    polaczenie.register("t", df)
    zapytanie = f"SELECT *, {wyrazenie_poprawny()} AS poprawny, {wyrazenie_powod()} AS powod FROM t"
    return polaczenie.execute(zapytanie).df()


def jeden_wiersz(**zmiany) -> pd.DataFrame:
    """Poprawny wiersz bazowy z nadpisaniami; kolumny object, zeby None byl prawdziwym NULL."""
    wiersz = {
        "suma_ubezpieczenia": 1_000_000.0,
        "udzial_kwoty": 0.4,
        "dni_do_zgloszenia": 3,
        "dni_od_poczatku_polisy": 100,
        "rodzaj_mienia": "budynek",
        "liczba_zalacznikow": 2,
        "szkody_3_lata": 0,
        ETYKIETA: 0,
    }
    wiersz.update(zmiany)
    return pd.DataFrame({k: pd.Series([v], dtype="object") for k, v in wiersz.items()})


def test_reguly_maja_unikalne_nazwy_i_opisy():
    nazwy = [r.nazwa for r in REGULY]
    assert len(nazwy) == len(set(nazwy))
    assert all(r.opis and r.wyrazenie for r in REGULY)


def test_reguly_pokrywaja_cechy_zakresy_slownik_i_etykiete():
    nazwy = {r.nazwa for r in REGULY}
    assert {f"brak_null_{c}" for c in [*CECHY, ETYKIETA]} <= nazwy
    assert {f"zakres_{c}" for c in _ZAKRESY} <= nazwy
    assert {"rodzaj_dozwolony", "etykieta_0_1"} <= nazwy


def test_wszystkie_wiersze_generatora_przechodza_reguly():
    wynik = ocen_wiersze(generuj_zbior(3500, 1))
    assert len(wynik) == 3500
    assert wynik["poprawny"].all()
    assert wynik["powod"].isna().all()


def test_wiersze_z_dryfem_nadal_przechodza_reguly():
    # Dryf kowariatow to dane POPRAWNE (generator obcina do zakresu) - nie moze konczyc w odrzuconych.
    dryf = {"suma_ubezpieczenia": 1.0, "udzial_kwoty": 0.3, "dni_do_zgloszenia": 5.0, "rodzaj_mienia": 0.3}
    wynik = ocen_wiersze(generuj_zbior(3500, 2, dryf))
    assert wynik["poprawny"].all()


def test_wiersz_bazowy_jest_poprawny():
    wynik = ocen_wiersze(jeden_wiersz())
    assert bool(wynik.loc[0, "poprawny"]) is True
    assert pd.isna(wynik.loc[0, "powod"])


@pytest.mark.parametrize(
    "zmiana, oczekiwany_powod",
    [
        ({"suma_ubezpieczenia": 50_000.0}, "zakres_suma_ubezpieczenia"),
        ({"suma_ubezpieczenia": 25_000_000.0}, "zakres_suma_ubezpieczenia"),
        ({"udzial_kwoty": -0.1}, "zakres_udzial_kwoty"),
        ({"udzial_kwoty": 1.6}, "zakres_udzial_kwoty"),
        ({"dni_do_zgloszenia": 61}, "zakres_dni_do_zgloszenia"),
        ({"dni_od_poczatku_polisy": -1}, "zakres_dni_od_poczatku_polisy"),
        ({"liczba_zalacznikow": 7}, "zakres_liczba_zalacznikow"),
        ({"szkody_3_lata": 5}, "zakres_szkody_3_lata"),
        ({"rodzaj_mienia": "statek"}, "rodzaj_dozwolony"),
        ({ETYKIETA: 2}, "etykieta_0_1"),
        ({ETYKIETA: -1}, "etykieta_0_1"),
        ({"suma_ubezpieczenia": None}, "brak_null_suma_ubezpieczenia"),
        ({"rodzaj_mienia": None}, "brak_null_rodzaj_mienia"),
        ({"szkody_3_lata": None}, "brak_null_szkody_3_lata"),
        ({ETYKIETA: None}, "brak_null_do_weryfikacji"),
    ],
)
def test_zepsuty_wiersz_odrzucony_z_wlasciwym_powodem(zmiana, oczekiwany_powod):
    wynik = ocen_wiersze(jeden_wiersz(**zmiana))
    assert bool(wynik.loc[0, "poprawny"]) is False
    assert wynik.loc[0, "powod"] == oczekiwany_powod


def test_wartosci_na_granicach_zakresu_sa_poprawne():
    granice = jeden_wiersz(
        suma_ubezpieczenia=100_000.0, udzial_kwoty=1.5, dni_do_zgloszenia=60, dni_od_poczatku_polisy=365
    )
    assert bool(ocen_wiersze(granice).loc[0, "poprawny"]) is True


def test_powod_to_pierwsza_zlamana_regula():
    # Zlamane dwie: brak NULL (regula wczesniejsza) i zakres - raportujemy pierwsza w kolejnosci REGULY.
    wynik = ocen_wiersze(jeden_wiersz(szkody_3_lata=99, rodzaj_mienia=None))
    assert wynik.loc[0, "powod"] == "brak_null_rodzaj_mienia"


def test_nan_liczbowy_tez_jest_odrzucony():
    df = jeden_wiersz()
    df["udzial_kwoty"] = pd.Series([float("nan")])
    assert bool(ocen_wiersze(df).loc[0, "poprawny"]) is False


def test_mieszany_zbior_odrzuca_tylko_zepsute():
    df = pd.concat([jeden_wiersz(), jeden_wiersz(rodzaj_mienia="statek"), jeden_wiersz()], ignore_index=True)
    wynik = ocen_wiersze(df)
    assert list(wynik["poprawny"]) == [True, False, True]
    assert RODZAJE_MIENIA  # slownik niepusty: bez tego test powyzej nic by nie znaczyl
