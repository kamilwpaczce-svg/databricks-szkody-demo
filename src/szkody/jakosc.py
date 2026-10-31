"""Reguly jakosci danych jako DANE (Etap 22): nazwa + wyrazenie SQL + opis.

Wyrazenie reguly jest PRAWDZIWE dla wiersza poprawnego i napisane w skladni zgodnej ze
Spark SQL (i z DuckDB, na ktorym testy sprawdzaja to samo wyrazenie lokalnie). Spark
w `silver.py` wykonuje DOKLADNIE te same napisy (`F.expr`), wiec nie ma drugiej kopii
logiki do rozjechania.

Zrodla regul (nic nie jest wpisane drugi raz):
- zakresy liczbowe = `_ZAKRESY` generatora (generator je obcina, wiec poza nimi dane
  nie powstaja - wiersz poza zakresem oznacza uszkodzony zaladunek, nie dryf),
- `rodzaj_mienia` in `RODZAJE_MIENIA`,
- etykieta in {0, 1},
- brak NULL w kolumnach `CECHY` i w etykiecie.

Wartosc NULL w regule daje NULL w SQL; `COALESCE(..., FALSE)` zamienia to na "zla", wiec
brak wartosci nigdy nie przechodzi po cichu.
"""

from __future__ import annotations

from dataclasses import dataclass

from szkody.cechy import CECHY, RODZAJE_MIENIA
from szkody.generator import _ZAKRESY, ETYKIETA

# Nazwa reguly dla wiersza, ktory jest poprawny - nigdy nie trafia do tabeli odrzuconych.
POWOD_BRAK = None


@dataclass(frozen=True)
class Regula:
    nazwa: str
    wyrazenie: str  # SQL, prawdziwe dla wiersza poprawnego
    opis: str


def _liczba_sql(wartosc: int | float) -> str:
    return repr(float(wartosc)) if isinstance(wartosc, float) else str(int(wartosc))


def _napis_sql(tekst: str) -> str:
    return "'" + tekst.replace("'", "''") + "'"


def zbuduj_reguly() -> list[Regula]:
    """Lista regul w stalej kolejnosci: najpierw braki (NULL), potem zakresy, potem slowniki."""
    reguly = [
        Regula(f"brak_null_{kolumna}", f"{kolumna} IS NOT NULL", f"kolumna {kolumna} nie moze byc pusta")
        for kolumna in [*CECHY, ETYKIETA]
    ]
    for kolumna, (dol, gora) in _ZAKRESY.items():
        reguly.append(
            Regula(
                f"zakres_{kolumna}",
                f"{kolumna} BETWEEN {_liczba_sql(dol)} AND {_liczba_sql(gora)}",
                f"{kolumna} w zakresie [{dol}, {gora}]",
            )
        )
    dozwolone = ", ".join(_napis_sql(r) for r in RODZAJE_MIENIA)
    reguly.append(Regula("rodzaj_dozwolony", f"rodzaj_mienia IN ({dozwolone})", f"rodzaj_mienia jedna z: {RODZAJE_MIENIA}"))
    reguly.append(Regula("etykieta_0_1", f"{ETYKIETA} IN (0, 1)", f"{ETYKIETA} to 0 albo 1"))
    return reguly


REGULY: list[Regula] = zbuduj_reguly()


def wyrazenie_poprawny(reguly: list[Regula] | None = None) -> str:
    """Jedno wyrazenie SQL: TRUE, gdy wiersz spelnia WSZYSTKIE reguly (NULL liczy sie jako FALSE)."""
    reguly = REGULY if reguly is None else reguly
    if not reguly:
        return "TRUE"
    return "COALESCE(" + " AND ".join(f"({r.wyrazenie})" for r in reguly) + ", FALSE)"


def wyrazenie_powod(reguly: list[Regula] | None = None) -> str:
    """Wyrazenie SQL: nazwa PIERWSZEJ zlamanej reguly, a dla wiersza poprawnego NULL."""
    reguly = REGULY if reguly is None else reguly
    galezie = " ".join(f"WHEN NOT COALESCE(({r.wyrazenie}), FALSE) THEN {_napis_sql(r.nazwa)}" for r in reguly)
    return f"CASE {galezie} ELSE NULL END"
