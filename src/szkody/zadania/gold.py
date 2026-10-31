"""Zadanie 3 joba: GOLD - 7 cech + etykieta + `_partia`, gotowe do treningu.

Czyta z silver partie `_partia = --seed` i zapisuje ja do `gold.cechy_szkod` przez
`replaceWhere` (nadpisuje tylko te partie - rerun jest idempotentny, inne partie zostaja).
Pusta partia = wyjatek (nie trenujemy na niczym).
"""

from __future__ import annotations

from szkody.zadania.wspolne import (
    KOLUMNA_PARTIA,
    KOLUMNY_GOLD,
    TYPY_SILVER,
    BladJakosci,
    ddl_tabeli,
    tabele,
    zbuduj_parser,
)


def uruchom(spark, katalog: str, seed: int) -> int:
    """Cienki kod Sparka. Zwraca liczbe wierszy zapisanych dla partii."""
    from pyspark.sql import functions as F

    t = tabele(katalog)
    spark.sql(ddl_tabeli(t.gold, {k: TYPY_SILVER[k] for k in KOLUMNY_GOLD}))
    gold = spark.table(t.silver).filter(F.col(KOLUMNA_PARTIA) == seed).select(*KOLUMNY_GOLD)
    liczba = gold.count()
    if liczba == 0:
        raise BladJakosci(f"Silver nie ma wierszy dla _partia={seed} - gold nie ma z czego powstac")
    (
        gold.write.format("delta")
        .mode("overwrite")
        .option("replaceWhere", f"{KOLUMNA_PARTIA} = {int(seed)}")
        .saveAsTable(t.gold)
    )
    return liczba


def main(argv: list[str] | None = None) -> None:
    args = zbuduj_parser(__doc__).parse_args(argv)
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    liczba = uruchom(spark, args.katalog, args.seed)
    print(f"gold: _partia={args.seed} zapisano {liczba} wierszy")


if __name__ == "__main__":
    main()
