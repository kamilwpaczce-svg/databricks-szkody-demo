"""Zadanie 4 joba: TRENING - hgb i logreg na gold, rejestracja lepszego w Unity Catalog, bramka.

Dane (3500 wierszy) czyta Spark, ale trening idzie na pandas/scikit-learn: przy tej wielkosci
Spark ML niczego nie dodaje (to jest swiadoma decyzja, nie przeoczenie). Split jak w Etapie 15
(30% test, stratyfikacja, random_state=seed). Oba modele logowane w MLflow; lepszy (PR AUC)
rejestrowany jako `<katalog>.ml.szkoda_do_weryfikacji` z sygnatura i przykladem wejscia.
Bramka: alias `champion` wg `bramka.decyzja`; odrzucenie to wynik (tag), nie blad zadania.
"""

from __future__ import annotations

import json

from szkody.cechy import CECHY
from szkody.generator import ETYKIETA
from szkody.zadania.mlops import trenuj_i_zarejestruj
from szkody.zadania.train_logika import przygotuj_dane
from szkody.zadania.wspolne import KOLUMNA_PARTIA, parsuj_przesuniecie, tabele, zbuduj_parser

DOMYSLNY_EKSPERYMENT = "/Shared/szkody_demo"


def main(argv: list[str] | None = None) -> None:
    parser = zbuduj_parser(__doc__)
    parser.add_argument("--eksperyment", default=DOMYSLNY_EKSPERYMENT, help="sciezka eksperymentu MLflow w workspace")
    args = parser.parse_args(argv)
    przesuniecie = parsuj_przesuniecie(args.przesuniecie)
    t = tabele(args.katalog)

    import mlflow
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(args.eksperyment)

    spark = SparkSession.builder.getOrCreate()
    gold = spark.table(t.gold).filter(F.col(KOLUMNA_PARTIA) == args.seed).select(*CECHY, ETYKIETA)
    train, test = przygotuj_dane(gold.toPandas(), args.seed)

    tagi_runu = {"partia": str(args.seed), "przesuniecie": json.dumps(przesuniecie, sort_keys=True)}
    wynik = trenuj_i_zarejestruj(train, test, args.seed, t.model, tagi_runu)
    print(f"train: wiersze train={len(train)} test={len(test)}; metryki: {json.dumps(wynik['wszystkie'])}")
    print(f"bramka: {wynik['powod']}")
    print(
        f"rejestr: {t.model} wersja {wynik['wersja']} ({wynik['rodzaj']}), promowany={wynik['promowany']}, "
        f"champion {wynik['champion_przed']} -> {wynik['champion_po']}"
    )


if __name__ == "__main__":
    main()
