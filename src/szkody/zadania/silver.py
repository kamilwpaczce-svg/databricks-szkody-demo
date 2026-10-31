"""Zadanie 2 joba: SILVER - deduplikacja, typy, reguly jakosci (`szkody.jakosc`), MERGE.

Czyta z bronze TYLKO partie `_partia = --seed`, deduplikuje po (`_partia`, `id_wiersza`)
(zostaje najnowszy `_zaladowano`), rzutuje typy (`TRY_CAST`), dzieli wiersze na poprawne
(MERGE do `silver.zgloszenia`) i odrzucone (MERGE do `silver.zgloszenia_odrzucone` z powodem
i surowym wierszem). Spark wykonuje DOKLADNIE te wyrazenia SQL, ktore testy sprawdzaja w
DuckDB. Na koncu wyjatek, gdy odrzuconych > 1% partii albo partia jest pusta.
"""

from __future__ import annotations

from szkody.jakosc import wyrazenie_poprawny, wyrazenie_powod
from szkody.zadania.wspolne import (
    KLUCZ_SILVER,
    KOLUMNA_ID,
    KOLUMNA_PARTIA,
    KOLUMNA_ZALADOWANO,
    TYPY_ODRZUCONE,
    TYPY_SILVER,
    ddl_tabeli,
    sprawdz_prog_odrzuconych,
    sql_merge,
    tabele,
    wyrazenia_rzutowania,
    zbuduj_parser,
)

KOLUMNY_SILVER = [*TYPY_SILVER, KOLUMNA_ZALADOWANO]


def uruchom(spark, katalog: str, seed: int) -> dict[str, float]:
    """Cienki kod Sparka. Zwraca `{"wszystkie", "odrzucone", "udzial"}`; rzuca BladJakosci ponad progiem."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    t = tabele(katalog)
    spark.sql(ddl_tabeli(t.silver, TYPY_SILVER, {KOLUMNA_ZALADOWANO: "TIMESTAMP"}))
    spark.sql(ddl_tabeli(t.odrzucone, TYPY_ODRZUCONE))

    surowe = spark.table(t.bronze).filter(F.col(KOLUMNA_PARTIA) == seed)
    okno = Window.partitionBy(*KLUCZ_SILVER).orderBy(F.col(KOLUMNA_ZALADOWANO).desc())
    bez_duplikatow = surowe.withColumn("_rn", F.row_number().over(okno)).filter("_rn = 1").drop("_rn")
    typowane = bez_duplikatow.selectExpr(
        *wyrazenia_rzutowania(TYPY_SILVER), KOLUMNA_ZALADOWANO, "to_json(struct(*)) AS _surowy"
    )
    ocenione = (
        typowane.withColumn("_poprawny", F.expr(wyrazenie_poprawny()))
        .withColumn("_powod", F.expr(wyrazenie_powod()))
        .cache()
    )
    poprawne = ocenione.filter("_poprawny").select(*KOLUMNY_SILVER)
    odrzucone = ocenione.filter("NOT _poprawny").select(
        KOLUMNA_PARTIA,
        KOLUMNA_ID,
        F.col("_powod").alias("powod"),
        F.col("_surowy").alias("surowy"),
        F.current_timestamp().alias("odrzucono"),
    )
    wszystkie, liczba_odrzuconych = ocenione.count(), odrzucone.count()

    poprawne.createOrReplaceTempView("_silver_nowe")
    spark.sql(sql_merge(t.silver, "_silver_nowe", KLUCZ_SILVER, KOLUMNY_SILVER))
    odrzucone.createOrReplaceTempView("_odrzucone_nowe")
    spark.sql(sql_merge(t.odrzucone, "_odrzucone_nowe", KLUCZ_SILVER, list(TYPY_ODRZUCONE)))

    # Tabele zapisane PRZED sprawdzeniem progu: odrzucone da sie obejrzec po padnietym biegu.
    udzial = sprawdz_prog_odrzuconych(liczba_odrzuconych, wszystkie)
    return {"wszystkie": wszystkie, "odrzucone": liczba_odrzuconych, "udzial": udzial}


def main(argv: list[str] | None = None) -> None:
    args = zbuduj_parser(__doc__).parse_args(argv)
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    wynik = uruchom(spark, args.katalog, args.seed)
    print(f"silver: _partia={args.seed} wierszy={wynik['wszystkie']} odrzuconych={wynik['odrzucone']} ({wynik['udzial']:.2%})")


if __name__ == "__main__":
    main()
