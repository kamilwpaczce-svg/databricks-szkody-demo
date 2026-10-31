"""Bramka champion/challenger na PR AUC (Etap 15, SPEC decyzja 4).

Skopiowane z underwriter_agent_demo@a1e0a3e, Etap 15.

Wzorzec z repo `ergonode_kategorie_herbert` (`bramka.py:decyzja`) - skopiowany, nie
importowany. Czysta funkcja bez MLflow: rejestr i aliasy obsluguje pipeline (zadanie 3).
"""

from __future__ import annotations

METRYKA = "pr_auc"
TOLERANCJA = 0.01
_EPS = 1e-12  # szum zmiennoprzecinkowy: 0.80 - 0.79 nie moze odrzucic wyniku "dokladnie na granicy"


def decyzja(champion: dict | None, challenger: dict, tol: float = TOLERANCJA) -> tuple[bool, str]:
    """Etap 15: (czy_promowac, powod). Promocja, gdy PR AUC challengera >= champion - tol.

    Brak championa (pierwsza wersja) -> promuj.
    """
    if champion is None:
        return True, "brak championa - pierwsza wersja zostaje championem"
    wynik_champion = champion[METRYKA]
    wynik_challenger = challenger[METRYKA]
    prog = wynik_champion - tol
    porownanie = f"{METRYKA} challengera {wynik_challenger:.4f} vs champion {wynik_champion:.4f} (tol {tol})"
    if wynik_challenger >= prog - _EPS:
        return True, f"promocja: {porownanie}"
    return False, f"odrzucony: {porownanie}, ponizej progu {prog:.4f}"
