"""Generator syntetycznego zbioru "szkoda do weryfikacji" (Etap 15).

Skopiowane z underwriter_agent_demo@a1e0a3e, Etap 15.

Dane sa FIKCYJNE: etykieta pochodzi z ukrytej reguly (logit) plus szum - celem etapu
jest cykl MLOps, nie jakosc modelu. Kolumny = `CECHY` + `do_weryfikacji` (0/1).

Parametr `przesuniecie` wstrzykuje dryf kowariatow (bez zmiany pozostalych cech):
- `suma_ubezpieczenia`: przesuniecie w skali logarytmicznej (mnoznik exp(x)),
- cechy liczbowe: dodawane wprost (w jednostkach cechy), potem obciecie do zakresu,
  cechy calkowite sa zaokraglane,
- `rodzaj_mienia`: o ile wzrasta prawdopodobienstwo "zapasy" (reszta skalowana proporcjonalnie).
Etykieta jest liczona z regulu PO przesunieciu (dryf kowariatow, ta sama regula).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from szkody.cechy import CECHY, RODZAJE_MIENIA

ETYKIETA = "do_weryfikacji"
_ZAKRESY = {
    "suma_ubezpieczenia": (100_000.0, 20_000_000.0),
    "udzial_kwoty": (0.0, 1.5),
    "dni_do_zgloszenia": (0, 60),
    "dni_od_poczatku_polisy": (0, 365),
    "liczba_zalacznikow": (0, 6),
    "szkody_3_lata": (0, 4),
}
_PROPORCJE_RODZAJU = np.array([0.5, 0.3, 0.2])  # budynek, maszyny, zapasy
_PRZESUWALNE = set(_ZAKRESY) | {"rodzaj_mienia"}
_SZUM_LOGITU = 0.6
_WYRAZ_WOLNY = -3.6


def _proporcje_rodzaju(przesuniecie: float) -> np.ndarray:
    zapasy = float(np.clip(_PROPORCJE_RODZAJU[2] + przesuniecie, 0.0, 0.95))
    reszta = _PROPORCJE_RODZAJU[:2] / _PROPORCJE_RODZAJU[:2].sum() * (1.0 - zapasy)
    return np.array([reszta[0], reszta[1], zapasy])


def _logit_reguly(df: pd.DataFrame) -> np.ndarray:
    """Ukryta regula etykiety (SPEC_ETAP15, decyzja 2)."""
    return (
        _WYRAZ_WOLNY
        + 2.4 * df["udzial_kwoty"].to_numpy()
        + 1.0 * (df["udzial_kwoty"].to_numpy() > 1.0)
        + 1.6 * (df["dni_od_poczatku_polisy"].to_numpy() < 30)
        + 0.55 * df["szkody_3_lata"].to_numpy()
        + 1.3 * (df["dni_do_zgloszenia"].to_numpy() > 14)
        - 0.25 * df["liczba_zalacznikow"].to_numpy()
        + 0.4 * (df["rodzaj_mienia"].to_numpy() == "zapasy")
    )


def generuj_zbior(n: int, seed: int, przesuniecie: dict[str, float] | None = None) -> pd.DataFrame:
    """Etap 15: zbior `n` wierszy (cechy + `do_weryfikacji`), deterministyczny dla `seed`."""
    przesuniecie = przesuniecie or {}
    nieznane = set(przesuniecie) - _PRZESUWALNE
    if nieznane:
        raise ValueError(f"Nieznane cechy w przesunieciu: {sorted(nieznane)}")

    rng = np.random.default_rng(seed)
    # Wszystkie losowania w stalej kolejnosci, niezaleznie od przesuniecia.
    su = rng.lognormal(mean=np.log(1_200_000), sigma=0.9, size=n)
    udzial = rng.beta(1.6, 4.0, size=n) * 1.5
    dni_zgl = np.floor(rng.exponential(scale=6.0, size=n))
    dni_pocz = rng.integers(0, 366, size=n)
    zalaczniki = np.minimum(rng.poisson(2.5, size=n), 6)
    szkody = rng.choice(5, size=n, p=[0.5, 0.25, 0.12, 0.08, 0.05])
    u_rodzaj = rng.random(n)
    szum = rng.normal(0.0, _SZUM_LOGITU, size=n)
    u_etykieta = rng.random(n)

    proporcje = _proporcje_rodzaju(przesuniecie.get("rodzaj_mienia", 0.0))
    indeks_rodzaju = np.searchsorted(np.cumsum(proporcje), u_rodzaj, side="right").clip(0, 2)

    kolumny = {
        "suma_ubezpieczenia": su * np.exp(przesuniecie.get("suma_ubezpieczenia", 0.0)),
        "udzial_kwoty": udzial + przesuniecie.get("udzial_kwoty", 0.0),
        "dni_do_zgloszenia": dni_zgl + przesuniecie.get("dni_do_zgloszenia", 0.0),
        "dni_od_poczatku_polisy": dni_pocz + przesuniecie.get("dni_od_poczatku_polisy", 0.0),
        "liczba_zalacznikow": zalaczniki + przesuniecie.get("liczba_zalacznikow", 0.0),
        "szkody_3_lata": szkody + przesuniecie.get("szkody_3_lata", 0.0),
    }
    dane = {}
    for nazwa, wartosci in kolumny.items():
        dol, gora = _ZAKRESY[nazwa]
        wartosci = np.clip(wartosci, dol, gora)
        if nazwa not in ("suma_ubezpieczenia", "udzial_kwoty"):
            wartosci = np.rint(wartosci).astype(int)
        dane[nazwa] = wartosci
    dane["rodzaj_mienia"] = np.array(RODZAJE_MIENIA, dtype=object)[indeks_rodzaju]

    df = pd.DataFrame(dane)[CECHY]
    prawdopodobienstwo = 1.0 / (1.0 + np.exp(-(_logit_reguly(df) + szum)))
    df[ETYKIETA] = (u_etykieta < prawdopodobienstwo).astype(int)
    return df
