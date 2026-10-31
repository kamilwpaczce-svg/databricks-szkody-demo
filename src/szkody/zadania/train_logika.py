"""Czysta logika treningu i bramki (Etap 22) - bez Sparka i bez MLflow.

Cienki kod (`train.py`, `mlops.py`) tylko czyta/zapisuje; DECYZJE (jaki split, ktory model
lepszy, promowac czy odrzucic, jakie tagi i alias ustawic) zapadaja tutaj i sa testowane lokalnie.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd
from sklearn.model_selection import train_test_split

from szkody.bramka import METRYKA, decyzja
from szkody.cechy import CECHY
from szkody.generator import ETYKIETA
from szkody.model import RODZAJE_MODELU

UDZIAL_TESTU = 0.3  # jak w Etapie 15 (pipeline_ml/komponenty/przygotuj.py)
ALIAS_CHAMPIONA = "champion"
NAZWY_METRYK = ("pr_auc", "roc_auc", "brier")
_KOLUMNY_FLOAT = ("suma_ubezpieczenia", "udzial_kwoty")


def znormalizuj_typy(df: pd.DataFrame) -> pd.DataFrame:
    """Wspolne typy niezaleznie od Sparka (INT -> int32 w pandas): float64 dla kwot, int64 dla reszty.

    Wazne dla sygnatury modelu w rejestrze: ta sama sygnatura po kazdym biegu.
    """
    wynik = df.copy()
    for kolumna in [*CECHY, ETYKIETA]:
        if kolumna == "rodzaj_mienia":
            wynik[kolumna] = wynik[kolumna].astype(str)
        elif kolumna in _KOLUMNY_FLOAT:
            wynik[kolumna] = wynik[kolumna].astype("float64")
        else:
            wynik[kolumna] = wynik[kolumna].astype("int64")
    return wynik


def sprawdz_klasy(y: pd.Series, nazwa: str) -> None:
    """Glosny blad, gdy zbior ma mniej niz dwie klasy (sklearn tylko ostrzega i zwraca NaN/0.0)."""
    if y.nunique() < 2:
        raise ValueError(f"Zbior {nazwa} ma jedna klase ({sorted(y.unique())}) - metryki bylyby bez znaczenia")


def przygotuj_dane(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test): podzial stratyfikowany po etykiecie, `random_state=seed`, jak w Etapie 15."""
    if len(df) == 0:
        raise ValueError("Pusty zbior treningowy: gold nie ma wierszy dla tej partii")
    df = znormalizuj_typy(df[[*CECHY, ETYKIETA]])
    sprawdz_klasy(df[ETYKIETA], "calosc")
    train, test = train_test_split(df, test_size=UDZIAL_TESTU, stratify=df[ETYKIETA], random_state=seed)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    sprawdz_klasy(train[ETYKIETA], "treningowy")
    sprawdz_klasy(test[ETYKIETA], "testowy")
    return train, test


def wybierz_lepszy(metryki_wg_rodzaju: dict[str, dict[str, float]]) -> str:
    """Rodzaj modelu z najwyzszym PR AUC. Remis -> wczesniejszy w `RODZAJE_MODELU`. NaN -> ValueError."""
    najlepszy, wynik_najlepszego = None, -math.inf
    for rodzaj in RODZAJE_MODELU:
        wynik = metryki_wg_rodzaju[rodzaj][METRYKA]
        if math.isnan(wynik):
            raise ValueError(f"{METRYKA} dla {rodzaj!r} to NaN - nie da sie wybrac lepszego modelu")
        if wynik > wynik_najlepszego:
            najlepszy, wynik_najlepszego = rodzaj, wynik
    return najlepszy


def metryki_z_tagow(tagi: dict[str, str]) -> dict[str, float]:
    """Metryki championa z tagow wersji. Brak tagu / NaN -> ValueError (nie 'brak championa' po cichu)."""
    if METRYKA not in tagi:
        raise ValueError(f"Champion nie ma tagu {METRYKA!r} - nie da sie porownac; tagi: {sorted(tagi)}")
    wartosc = float(tagi[METRYKA])
    if not math.isfinite(wartosc):
        raise ValueError(f"Tag {METRYKA!r} championa ma wartosc {tagi[METRYKA]!r}")
    return {METRYKA: wartosc}


@dataclass(frozen=True)
class PlanBramki:
    """Co ma zrobic cienki kod MlflowClient po decyzji bramki."""

    promuj: bool
    powod: str
    tagi_wersji: dict[str, str]  # tagi nowej wersji
    alias: str | None  # alias do ustawienia na nowej wersji (None = odrzucenie)
    poprzedni_champion: str | None = None
    tagi_poprzedniego: dict[str, str] = field(default_factory=dict)


def zaplanuj_bramke(
    champion: dict[str, float] | None,
    wersja_championa: str | None,
    challenger: dict[str, float],
    rodzaj: str,
    partia: int,
    nowa_wersja: str,
) -> PlanBramki:
    """Decyzja bramki -> plan tagow i aliasu. Odrzucenie to wynik, nie blad: wersja zostaje bez aliasu."""
    promuj, powod = decyzja(champion, challenger)
    tagi = {nazwa: f"{challenger[nazwa]:.6f}" for nazwa in NAZWY_METRYK}
    tagi.update({"rodzaj": rodzaj, "partia": str(partia), "bramka_powod": powod})
    if not promuj:
        return PlanBramki(False, powod, {**tagi, "bramka": "odrzucony"}, alias=None)
    plan_poprzedniego: dict[str, str] = {}
    if wersja_championa is not None:
        plan_poprzedniego = {"bramka": "wycofany", "bramka_powod": f"zastapiony przez wersje {nowa_wersja}"}
    return PlanBramki(
        True,
        powod,
        {**tagi, "bramka": "promowany"},
        alias=ALIAS_CHAMPIONA,
        poprzedni_champion=wersja_championa,
        tagi_poprzedniego=plan_poprzedniego,
    )
