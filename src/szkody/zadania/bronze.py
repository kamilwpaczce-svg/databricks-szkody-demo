"""Zadanie 1 joba `szkody_pipeline`: BRONZE - surowa partia syntetycznych zgloszen (append).

Tworzy katalog i schematy (`CREATE ... IF NOT EXISTS`), generuje `generuj_zbior(3500, seed,
przesuniecie)` i DOPISUJE do `bronze.zgloszenia_raw` z kolumnami `id_wiersza`, `_partia`
(= seed) i `_zaladowano`. Powtorny bieg tego samego seeda dopisze druga kopie - deduplikuje
ja silver (po `_partia`, `id_wiersza`, z najnowszym `_zaladowano`).
"""

from __future__ import annotations

from szkody.zadania.wspolne import (
    KOLUMNA_ZALADOWANO,
    LICZBA_WIERSZY_PARTII,
    TYPY_BRONZE,
    ddl_inicjalizacji,
    ddl_schematu,
    ddl_tabeli,
    parsuj_przesuniecie,
    przygotuj_partie,
    tabele,
    zbuduj_parser,
)


def uruchom(spark, katalog: str, seed: int, przesuniecie: dict[str, float]) -> int:
    """Cienki kod Sparka: DDL, ramka z pandas, append. Zwraca liczbe dopisanych wierszy."""
    from pyspark.sql import functions as F

    t = tabele(katalog)
    for instrukcja in ddl_inicjalizacji(katalog):
        spark.sql(instrukcja)
    spark.sql(ddl_tabeli(t.bronze, TYPY_BRONZE, {KOLUMNA_ZALADOWANO: "TIMESTAMP"}))
    partia = przygotuj_partie(LICZBA_WIERSZY_PARTII, seed, przesuniecie)
    ramka = spark.createDataFrame(partia, schema=ddl_schematu(TYPY_BRONZE))
    ramka = ramka.withColumn(KOLUMNA_ZALADOWANO, F.current_timestamp())
    ramka.write.format("delta").mode("append").saveAsTable(t.bronze)
    return len(partia)


def main(argv: list[str] | None = None) -> None:
    args = zbuduj_parser(__doc__).parse_args(argv)
    przesuniecie = parsuj_przesuniecie(args.przesuniecie)  # zly JSON/cecha pada PRZED startem Sparka
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    liczba = uruchom(spark, args.katalog, args.seed, przesuniecie)
    print(f"bronze: dopisano {liczba} wierszy, _partia={args.seed}, przesuniecie={przesuniecie}")


if __name__ == "__main__":
    main()
