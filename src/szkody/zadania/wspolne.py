"""Czyste funkcje wspolne dla zadan joba (Etap 22): nazwy tabel, argumenty, DDL, MERGE, progi.

Nic tu nie importuje pyspark ani mlflow - wszystko da sie przetestowac lokalnie. Skrypty
`bronze.py`, `silver.py`, `gold.py`, `train.py` skladaja te funkcje z cienkim kodem Sparka.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from szkody.cechy import CECHY
from szkody.generator import _PRZESUWALNE, ETYKIETA, generuj_zbior

DOMYSLNY_KATALOG = "szkody_demo"
SCHEMATY = ("bronze", "silver", "gold", "ml")
KOLUMNA_PARTIA = "_partia"
KOLUMNA_ID = "id_wiersza"
KOLUMNA_ZALADOWANO = "_zaladowano"
KLUCZ_SILVER = (KOLUMNA_PARTIA, KOLUMNA_ID)
PROG_ODRZUCONYCH = 0.01  # job pada, gdy odrzuconych > 1% partii
LICZBA_WIERSZY_PARTII = 3500
NAZWA_MODELU = "szkoda_do_weryfikacji"

# Typy BRONZE = surowe, tak jak wychodzi z pandas (ints -> BIGINT). Silver rzutuje na wlasciwe.
_TYPY_CECH_BRONZE = {
    "suma_ubezpieczenia": "DOUBLE",
    "udzial_kwoty": "DOUBLE",
    "dni_do_zgloszenia": "BIGINT",
    "dni_od_poczatku_polisy": "BIGINT",
    "rodzaj_mienia": "STRING",
    "liczba_zalacznikow": "BIGINT",
    "szkody_3_lata": "BIGINT",
}
TYPY_BRONZE: dict[str, str] = {
    KOLUMNA_ID: "BIGINT",
    **{c: _TYPY_CECH_BRONZE[c] for c in CECHY},
    ETYKIETA: "BIGINT",
    KOLUMNA_PARTIA: "BIGINT",
}
# Silver: waskie typy dla licznikow i etykiety.
TYPY_SILVER: dict[str, str] = {
    **{k: ("INT" if v == "BIGINT" else v) for k, v in TYPY_BRONZE.items()},
    KOLUMNA_ID: "BIGINT",
    KOLUMNA_PARTIA: "BIGINT",
}
KOLUMNY_GOLD: list[str] = [*CECHY, ETYKIETA, KOLUMNA_PARTIA]
# Tabela odrzuconych: klucz + powod (nazwa pierwszej zlamanej reguly) + surowy wiersz jako JSON.
TYPY_ODRZUCONE: dict[str, str] = {
    KOLUMNA_PARTIA: "BIGINT",
    KOLUMNA_ID: "BIGINT",
    "powod": "STRING",
    "surowy": "STRING",
    "odrzucono": "TIMESTAMP",
}

_IDENTYFIKATOR = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class Tabele:
    """Pelne nazwy tabel (katalog.schemat.tabela) dla jednego katalogu."""

    bronze: str
    silver: str
    odrzucone: str
    gold: str
    model: str


def sprawdz_identyfikator(nazwa: str) -> str:
    """Katalog idzie z linii polecen do SQL - dopuszczamy tylko zwykle identyfikatory."""
    if not _IDENTYFIKATOR.match(nazwa):
        raise ValueError(f"Niedozwolona nazwa katalogu {nazwa!r}: litery, cyfry, podkreslnik")
    return nazwa


def tabele(katalog: str = DOMYSLNY_KATALOG) -> Tabele:
    katalog = sprawdz_identyfikator(katalog)
    return Tabele(
        bronze=f"{katalog}.bronze.zgloszenia_raw",
        silver=f"{katalog}.silver.zgloszenia",
        odrzucone=f"{katalog}.silver.zgloszenia_odrzucone",
        gold=f"{katalog}.gold.cechy_szkod",
        model=f"{katalog}.ml.{NAZWA_MODELU}",
    )


def ddl_inicjalizacji(katalog: str = DOMYSLNY_KATALOG) -> list[str]:
    """Instrukcje SQL tworzace schematy (idempotentne) w ISTNIEJACYM katalogu.

    Bez CREATE CATALOG: metastore bez storage_root odrzuca je nawet z IF NOT EXISTS
    (INVALID_STATE, 31.10), wiec katalog musi juz istniec (katalog workspace'u).
    """
    katalog = sprawdz_identyfikator(katalog)
    return [f"CREATE SCHEMA IF NOT EXISTS {katalog}.{schemat}" for schemat in SCHEMATY]


def ddl_tabeli(nazwa: str, typy: dict[str, str], dodatkowe: dict[str, str] | None = None) -> str:
    """CREATE TABLE IF NOT EXISTS ... USING DELTA dla slownika kolumna -> typ SQL."""
    kolumny = {**typy, **(dodatkowe or {})}
    lista = ", ".join(f"{k} {v}" for k, v in kolumny.items())
    return f"CREATE TABLE IF NOT EXISTS {nazwa} ({lista}) USING DELTA"


def ddl_schematu(typy: dict[str, str]) -> str:
    """Napis schematu do `spark.createDataFrame(..., schema=...)` (DDL, bez CREATE TABLE)."""
    return ", ".join(f"{k} {v}" for k, v in typy.items())


def wyrazenia_rzutowania(typy_docelowe: dict[str, str]) -> list[str]:
    """`TRY_CAST(k AS typ) AS k`: blad rzutowania daje NULL (a NULL lapie regula braku wartosci),
    zamiast wyjatku calego zadania na jednym zepsutym wierszu."""
    return [f"TRY_CAST({k} AS {t}) AS {k}" for k, t in typy_docelowe.items()]


def sql_merge(cel: str, widok: str, klucze: tuple[str, ...], kolumny: list[str]) -> str:
    """MERGE po kluczu: istniejace wiersze nadpisane, nowe dopisane (idempotentny rerun partii)."""
    warunek = " AND ".join(f"t.{k} = s.{k}" for k in klucze)
    ustaw = ", ".join(f"{k} = s.{k}" for k in kolumny)  # lewa strona bez aliasu: najbezpieczniejsza w Delta MERGE
    wstaw_k = ", ".join(kolumny)
    wstaw_w = ", ".join(f"s.{k}" for k in kolumny)
    return (
        f"MERGE INTO {cel} AS t USING {widok} AS s ON {warunek} "
        f"WHEN MATCHED THEN UPDATE SET {ustaw} "
        f"WHEN NOT MATCHED THEN INSERT ({wstaw_k}) VALUES ({wstaw_w})"
    )


class BladJakosci(RuntimeError):
    """Za duzo odrzuconych wierszy albo pusta partia - job ma sie zakonczyc bledem."""


def sprawdz_prog_odrzuconych(odrzucone: int, wszystkie: int, prog: float = PROG_ODRZUCONYCH) -> float:
    """Zwraca udzial odrzuconych; wyjatek, gdy partia jest pusta albo udzial > `prog`."""
    if wszystkie <= 0:
        raise BladJakosci("Pusta partia: brak wierszy w bronze dla tego seeda (zly --seed albo bronze nie zaladowal)")
    udzial = odrzucone / wszystkie
    if udzial > prog:
        raise BladJakosci(
            f"Odrzucono {odrzucone} z {wszystkie} wierszy ({udzial:.2%}) - powyzej progu {prog:.0%}; "
            "szczegoly w tabeli silver.zgloszenia_odrzucone"
        )
    return udzial


def parsuj_przesuniecie(tekst: str | None) -> dict[str, float]:
    """JSON `{"cecha": liczba}` -> slownik. Puste/`{}` -> {}. Nieznana cecha lub nie-liczba -> ValueError."""
    if tekst is None or not tekst.strip():
        return {}
    try:
        dane = json.loads(tekst)
    except json.JSONDecodeError as e:
        raise ValueError(f"--przesuniecie nie jest poprawnym JSON-em: {e}") from e
    if not isinstance(dane, dict):
        raise ValueError("--przesuniecie musi byc obiektem JSON, np. {\"udzial_kwoty\": 0.3}")
    nieznane = set(dane) - _PRZESUWALNE
    if nieznane:
        raise ValueError(f"Nieznane cechy w --przesuniecie: {sorted(nieznane)}; dozwolone: {sorted(_PRZESUWALNE)}")
    for klucz, wartosc in dane.items():
        if isinstance(wartosc, bool) or not isinstance(wartosc, (int, float)):
            raise ValueError(f"--przesuniecie[{klucz!r}] musi byc liczba, jest {wartosc!r}")
    return {k: float(v) for k, v in dane.items()}


def zbuduj_parser(opis: str) -> argparse.ArgumentParser:
    """Wspolne argumenty zadan: --katalog, --seed, --przesuniecie (uzywa go tylko bronze)."""
    parser = argparse.ArgumentParser(description=opis)
    parser.add_argument("--katalog", default=DOMYSLNY_KATALOG, help="katalog Unity Catalog")
    parser.add_argument("--seed", type=int, default=1, help="seed partii (= wartosc kolumny _partia)")
    parser.add_argument("--przesuniecie", default="{}", help="JSON z przesunieciem cech (dryf), np. {\"udzial_kwoty\": 0.3}")
    return parser


def przygotuj_partie(n: int, seed: int, przesuniecie: dict[str, float] | None = None) -> pd.DataFrame:
    """Surowa partia dla bronze: `id_wiersza`, 7 cech, etykieta, `_partia` (= seed). Bez `_zaladowano` (dodaje Spark)."""
    if n <= 0:
        raise ValueError("Liczba wierszy partii musi byc dodatnia")
    df = generuj_zbior(n, seed, przesuniecie)
    df.insert(0, KOLUMNA_ID, np.arange(len(df), dtype="int64"))
    df[KOLUMNA_PARTIA] = np.int64(seed)
    return df[list(TYPY_BRONZE)]
