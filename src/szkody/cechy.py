"""Cechy modelu ryzyka szkody - JEDNO zrodlo prawdy dla treningu i regul jakosci (Etap 15).

Skopiowane z underwriter_agent_demo@a1e0a3e, Etap 15. Wziete TYLKO stale (CECHY,
CECHY_KATEGORIALNE, CECHY_LICZBOWE, RODZAJE_MIENIA); `cechy_ze_stanu` (stan grafu agenta)
zostalo w underwriterze - tu go nie potrzeba.

Generator (`szkody.generator`) produkuje wiersze z kolumnami `CECHY`, dzieki czemu nie ma
rozjazdu trening/serwowanie (training-serving skew).
"""

from __future__ import annotations

CECHY: list[str] = [
    "suma_ubezpieczenia",
    "udzial_kwoty",
    "dni_do_zgloszenia",
    "dni_od_poczatku_polisy",
    "rodzaj_mienia",
    "liczba_zalacznikow",
    "szkody_3_lata",
]
CECHY_KATEGORIALNE: list[str] = ["rodzaj_mienia"]
CECHY_LICZBOWE: list[str] = [c for c in CECHY if c not in CECHY_KATEGORIALNE]
RODZAJE_MIENIA: list[str] = ["budynek", "maszyny", "zapasy"]
