"""Bramka champion/challenger - przypadki z underwritera (Etap 15)."""

from szkody.bramka import METRYKA, TOLERANCJA, decyzja


def test_metryka_i_tolerancja_jak_w_etapie_15():
    assert METRYKA == "pr_auc"
    assert TOLERANCJA == 0.01


def test_brak_championa_promuje():
    promuj, powod = decyzja(None, {"pr_auc": 0.5})
    assert promuj is True
    assert "brak championa" in powod


def test_lepszy_challenger_promowany():
    promuj, powod = decyzja({"pr_auc": 0.60}, {"pr_auc": 0.65})
    assert promuj is True
    assert powod.startswith("promocja")


def test_dokladnie_na_granicy_tolerancji_promuje():
    # 0.80 - 0.79 = 0.010000000000000009 w float: bez _EPS to odrzucilo by wynik "dokladnie na granicy".
    promuj, _ = decyzja({"pr_auc": 0.80}, {"pr_auc": 0.79})
    assert promuj is True


def test_tuz_ponizej_granicy_odrzuca():
    promuj, powod = decyzja({"pr_auc": 0.80}, {"pr_auc": 0.789})
    assert promuj is False
    assert powod.startswith("odrzucony")
    assert "ponizej progu 0.7900" in powod


def test_wyraznie_gorszy_odrzucony():
    promuj, _ = decyzja({"pr_auc": 0.70}, {"pr_auc": 0.50})
    assert promuj is False


def test_wlasna_tolerancja():
    assert decyzja({"pr_auc": 0.80}, {"pr_auc": 0.75}, tol=0.0)[0] is False
    assert decyzja({"pr_auc": 0.80}, {"pr_auc": 0.75}, tol=0.05)[0] is True
