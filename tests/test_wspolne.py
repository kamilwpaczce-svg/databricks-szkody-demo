"""Czyste funkcje wspolne zadan joba: nazwy, argumenty, DDL, MERGE, progi, partia, rzutowanie."""

import duckdb
import pytest

from szkody.cechy import CECHY
from szkody.generator import ETYKIETA
from szkody.jakosc import wyrazenie_poprawny, wyrazenie_powod
from szkody.zadania.wspolne import (
    KLUCZ_SILVER,
    KOLUMNA_ID,
    KOLUMNA_PARTIA,
    TYPY_BRONZE,
    TYPY_SILVER,
    BladJakosci,
    ddl_inicjalizacji,
    ddl_schematu,
    ddl_tabeli,
    parsuj_przesuniecie,
    przygotuj_partie,
    sprawdz_identyfikator,
    sprawdz_prog_odrzuconych,
    sql_merge,
    tabele,
    wyrazenia_rzutowania,
    zbuduj_parser,
)


def test_nazwy_tabel_wg_spec():
    t = tabele("szkody_demo")
    assert t.bronze == "szkody_demo.bronze.zgloszenia_raw"
    assert t.silver == "szkody_demo.silver.zgloszenia"
    assert t.odrzucone == "szkody_demo.silver.zgloszenia_odrzucone"
    assert t.gold == "szkody_demo.gold.cechy_szkod"
    assert t.model == "szkody_demo.ml.szkoda_do_weryfikacji"


def test_katalog_z_innej_nazwy():
    assert tabele("test_kat").gold == "test_kat.gold.cechy_szkod"


@pytest.mark.parametrize("zla", ["a b", "a;DROP", "a.b", "1abc", "", "a`b"])
def test_niedozwolona_nazwa_katalogu(zla):
    with pytest.raises(ValueError, match="Niedozwolona"):
        sprawdz_identyfikator(zla)


def test_ddl_inicjalizacji_katalog_i_cztery_schematy_idempotentnie():
    ddl = ddl_inicjalizacji("szkody_demo")
    assert ddl[0] == "CREATE CATALOG IF NOT EXISTS szkody_demo"
    assert ddl[1:] == [f"CREATE SCHEMA IF NOT EXISTS szkody_demo.{s}" for s in ("bronze", "silver", "gold", "ml")]
    assert all("IF NOT EXISTS" in s for s in ddl)


def test_ddl_tabeli_zawiera_wszystkie_kolumny_i_delta():
    ddl = ddl_tabeli("k.bronze.t", TYPY_BRONZE, {"_zaladowano": "TIMESTAMP"})
    assert ddl.startswith("CREATE TABLE IF NOT EXISTS k.bronze.t (")
    assert ddl.endswith(") USING DELTA")
    for kolumna in [*CECHY, ETYKIETA, "id_wiersza", "_partia", "_zaladowano"]:
        assert f"{kolumna} " in ddl


def test_ddl_schematu_bez_create():
    napis = ddl_schematu({"a": "INT", "b": "STRING"})
    assert napis == "a INT, b STRING"


def test_typy_silver_wezsze_niz_bronze():
    assert TYPY_BRONZE["liczba_zalacznikow"] == "BIGINT"
    assert TYPY_SILVER["liczba_zalacznikow"] == "INT"
    assert TYPY_SILVER[KOLUMNA_ID] == "BIGINT" and TYPY_SILVER[KOLUMNA_PARTIA] == "BIGINT"


def test_sql_merge_po_kluczu_partia_id():
    sql = sql_merge("k.silver.z", "nowe", KLUCZ_SILVER, ["_partia", "id_wiersza", "x"])
    assert "MERGE INTO k.silver.z AS t USING nowe AS s" in sql
    assert "ON t._partia = s._partia AND t.id_wiersza = s.id_wiersza" in sql
    assert "WHEN MATCHED THEN UPDATE SET _partia = s._partia, id_wiersza = s.id_wiersza, x = s.x" in sql
    assert "WHEN NOT MATCHED THEN INSERT (_partia, id_wiersza, x) VALUES (s._partia, s.id_wiersza, s.x)" in sql


def test_prog_odrzuconych_na_granicy_przechodzi_ponad_nie():
    assert sprawdz_prog_odrzuconych(35, 3500) == pytest.approx(0.01)  # dokladnie 1% - jeszcze OK
    with pytest.raises(BladJakosci, match="powyzej progu"):
        sprawdz_prog_odrzuconych(36, 3500)


def test_prog_zero_odrzuconych_ok():
    assert sprawdz_prog_odrzuconych(0, 3500) == 0.0


def test_pusta_partia_to_blad_a_nie_zero_odrzuconych():
    with pytest.raises(BladJakosci, match="Pusta partia"):
        sprawdz_prog_odrzuconych(0, 0)


def test_parsuj_przesuniecie_puste_i_poprawne():
    assert parsuj_przesuniecie("{}") == {}
    assert parsuj_przesuniecie("") == {}
    assert parsuj_przesuniecie(None) == {}
    assert parsuj_przesuniecie('{"udzial_kwoty": 0.3, "szkody_3_lata": 1}') == {"udzial_kwoty": 0.3, "szkody_3_lata": 1.0}


@pytest.mark.parametrize(
    "tekst, fragment",
    [
        ("{nie json", "poprawnym JSON"),
        ("[1, 2]", "obiektem JSON"),
        ('{"nie_ma": 1}', "Nieznane cechy"),
        ('{"udzial_kwoty": "duzo"}', "musi byc liczba"),
        ('{"udzial_kwoty": true}', "musi byc liczba"),
    ],
)
def test_parsuj_przesuniecie_bledy(tekst, fragment):
    with pytest.raises(ValueError, match=fragment):
        parsuj_przesuniecie(tekst)


def test_parser_wartosci_domyslne_i_jawne():
    parser = zbuduj_parser("opis")
    domyslne = parser.parse_args([])
    assert (domyslne.katalog, domyslne.seed, domyslne.przesuniecie) == ("szkody_demo", 1, "{}")
    jawne = parser.parse_args(["--katalog", "k", "--seed", "7", "--przesuniecie", '{"udzial_kwoty": 0.2}'])
    assert (jawne.katalog, jawne.seed, jawne.przesuniecie) == ("k", 7, '{"udzial_kwoty": 0.2}')


def test_parser_seed_nie_liczba_to_blad():
    with pytest.raises(SystemExit):
        zbuduj_parser("opis").parse_args(["--seed", "abc"])


def test_przygotuj_partie_kolumny_klucze_i_determinizm():
    p1 = przygotuj_partie(300, 5)
    assert list(p1.columns) == list(TYPY_BRONZE)
    assert list(p1[KOLUMNA_ID]) == list(range(300))
    assert set(p1[KOLUMNA_PARTIA]) == {5}
    assert p1.equals(przygotuj_partie(300, 5))
    assert not p1.equals(przygotuj_partie(300, 6))


def test_przygotuj_partie_z_dryfem_zmienia_dane():
    baza = przygotuj_partie(500, 1)
    dryf = przygotuj_partie(500, 1, {"udzial_kwoty": 0.3})
    assert dryf["udzial_kwoty"].mean() > baza["udzial_kwoty"].mean()


def test_przygotuj_partie_zero_wierszy_to_blad():
    with pytest.raises(ValueError):
        przygotuj_partie(0, 1)


def _silver_w_duckdb(raw):
    """To samo, co silver.py robi w Sparku: TRY_CAST -> poprawny/powod, ale na DuckDB."""
    polaczenie = duckdb.connect()
    polaczenie.register("raw", raw)
    rzutowane = ", ".join(wyrazenia_rzutowania(TYPY_SILVER))
    zapytanie = (
        f"SELECT *, {wyrazenie_poprawny()} AS poprawny, {wyrazenie_powod()} AS powod "
        f"FROM (SELECT {rzutowane} FROM raw)"
    )
    return polaczenie.execute(zapytanie).df()


def test_silver_czysta_partia_nic_nie_odrzuca_i_typy_sie_zgadzaja():
    wynik = _silver_w_duckdb(przygotuj_partie(3500, 1))
    assert len(wynik) == 3500 and wynik["poprawny"].all()
    assert str(wynik["liczba_zalacznikow"].dtype) == "int32"  # INT po rzutowaniu z BIGINT


def test_silver_odrzuca_zepsute_wiersze_z_powodem():
    raw = przygotuj_partie(100, 1).astype({"rodzaj_mienia": object, "udzial_kwoty": object, "dni_do_zgloszenia": object})
    raw.loc[3, "rodzaj_mienia"] = "statek"
    raw.loc[4, "udzial_kwoty"] = None
    raw.loc[5, "dni_do_zgloszenia"] = 999
    wynik = _silver_w_duckdb(raw)
    odrzucone = wynik[~wynik["poprawny"]].set_index("id_wiersza")["powod"].to_dict()
    assert odrzucone == {
        3: "rodzaj_dozwolony",
        4: "brak_null_udzial_kwoty",
        5: "zakres_dni_do_zgloszenia",
    }


def test_try_cast_nieparsowalny_tekst_daje_null_a_nie_wyjatek():
    polaczenie = duckdb.connect()
    wyraz = wyrazenia_rzutowania({"dni_do_zgloszenia": "INT"})[0]
    wynik = polaczenie.execute(f"SELECT {wyraz} FROM (SELECT 'abc' AS dni_do_zgloszenia)").fetchone()
    assert wynik == (None,)
