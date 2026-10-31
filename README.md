# databricks_szkody_demo

Mini-demo Azure Databricks: ten sam problem co Etap 15 w `underwriter_agent_demo` (syntetyczna
"szkoda do weryfikacji", 7 cech, HistGradientBoosting vs regresja logistyczna, bramka
champion/challenger na PR AUC), przeniesiony na Databricks: medallion w Delta + MLflow w Unity Catalog,
wdrazany jako Databricks Asset Bundle. Specyfikacja: `SPEC_ETAP22.md`.

**Stan: kod gotowy i przetestowany lokalnie, wdrozenie czeka na workspace** (zadanie 2 SPEC).
Nic z tego nie zostalo jeszcze uruchomione na Databricks - szczegoly w sekcji "Niesprawdzone".

## Struktura

```
databricks.yml                 bundle `szkody_demo`: job szkody_pipeline (4 zadania), targety dev i demo
src/szkody/
  generator.py cechy.py model.py bramka.py   kopie z underwriter_agent_demo@a1e0a3e (Etap 15)
  jakosc.py                    reguly jakosci jako dane: nazwa + wyrazenie SQL + opis
  zadania/
    wspolne.py                 czyste funkcje: nazwy tabel, argumenty, DDL, MERGE, prog odrzuconych
    bronze.py silver.py gold.py train.py   skrypty zadan (spark_python_task), cienki kod Sparka
    train_logika.py            split, wybor modelu, plan bramki (tagi i alias) - czyste funkcje
    mlops.py                   cienka warstwa MLflow (trening, rejestracja, bramka)
tests/                         pytest, wszystko lokalnie (bez Sparka i Databricks)
infra/                         Terraform workspace'u (osobny tfstate; nie dotyka stanu underwritera)
```

## Przeplyw danych

| Zadanie | Wejscie | Wyjscie | Uwagi |
|---|---|---|---|
| `bronze` | `generuj_zbior(3500, seed, przesuniecie)` | `bronze.zgloszenia_raw` (append) | kolumny `id_wiersza`, `_partia` (= seed), `_zaladowano`; tworzy katalog i schematy `IF NOT EXISTS` |
| `silver` | bronze, partia `_partia = seed` | `silver.zgloszenia`, `silver.zgloszenia_odrzucone` | dedup, `TRY_CAST`, reguly z `jakosc.py`, MERGE po (`_partia`, `id_wiersza`); wyjatek, gdy odrzuconych > 1% albo partia pusta |
| `gold` | silver, ta partia | `gold.cechy_szkod` | 7 cech + etykieta + `_partia`, `replaceWhere` (rerun idempotentny) |
| `trening` | gold, ta partia -> pandas | model `szkody_demo.ml.szkoda_do_weryfikacji` | split jak w Etapie 15; oba modele w MLflow; lepszy zarejestrowany; alias `champion` wg bramki |

Trening idzie na pandas/scikit-learn, nie na Spark ML: przy 3500 wierszach rozproszony trening
nic nie daje - to swiadoma decyzja. Spark robi tu to, w czym jest dobry (tabele Delta, MERGE, reguly SQL).

Bramka: `bramka.decyzja` (PR AUC challengera >= champion - 0.01). Metryki championa pochodza z tagow
wersji (`pr_auc`, `roc_auc`, `brier`). Promocja przenosi alias `champion` i oznacza poprzednia wersje
tagiem `bramka=wycofany`. Odrzucenie zostawia wersje bez aliasu z tagiem `bramka=odrzucony` i powodem -
job NIE pada (to wynik, nie blad). Decyzja trafia do logu zadania (`bramka: ...`).

Parametry joba: `seed` (domyslnie `1`; to jednoczesnie numer partii) i `przesuniecie`
(JSON, domyslnie `{}`, np. `{"udzial_kwoty": 0.3, "dni_do_zgloszenia": 5}`). Kazda partia ma wlasny seed;
ponowny bieg tego samego seeda jest bezpieczny (bronze dopisze kopie, silver ja deduplikuje).

## Jak uruchomic testy

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
.venv/Scripts/python.exe -m pytest -q
```

Testy `test_mlops.py` trenuja modele na prawdziwym MLflow (sqlite w katalogu tymczasowym) - okolo
minuty. Testy jakosci wykonuja *te same* wyrazenia SQL, ktore Spark dostaje w `silver.py`, ale w DuckDB.
Nie ma lokalnie Javy ani pyspark, wiec samo wywolanie Sparka (`uruchom()` w skryptach) nie jest testowane.

## Biblioteki w zadaniach (wheel vs requirements)

Wybrany wheel pakietu `szkody`: `artifacts.szkody_whl` w `databricks.yml` buduje go komenda
`uv build --wheel` (wymaga `uv` na maszynie robiacej deploy), a kazde zadanie ma
`libraries: - whl: ./dist/*.whl`. Skrypty zadan (`spark_python_task`) importuja `szkody.*` z zainstalowanego
wheela, wiec kod zrodlowy w `src/` jest jedynym zrodlem prawdy (zadnej kopii w notebookach).
Zaleznosci wheela (numpy, pandas, scikit-learn, mlflow) bez pinow - na runtime ML sa juz zainstalowane.
Przy zmianie kodu podbij `version` w `pyproject.toml` albo wyczysc `dist/`, zeby w `dist/` nie lezaly dwa wheele.

## Niesprawdzone do czasu biegu na Databricks

- `databricks bundle validate` (CLI jeszcze nie ma): zgodnosc `databricks.yml` ze schematem, w tym
  `workspace.host: ${var.host}`, `libraries.whl: ./dist/*.whl`, `{{job.parameters.*}}` w `parameters`
  zadania oraz przekazanie JSON-a `{}` jako argumentu.
- `spark_version` w `databricks.yml` to PLACEHOLDER (`17.3.x-cpu-ml-scala2.13`) - potwierdzic w zadaniu 2.
- Unity Catalog: czy auto-metastore jest aktywny na trialu w swedencentral i czy `CREATE CATALOG IF NOT EXISTS`
  przejdzie bez `MANAGED LOCATION`. Plan B (`hive_metastore` + rejestr workspace'u) NIE jest zakodowany.
- `data_security_mode: SINGLE_USER` na jednowezlowym klastrze z UC; `/Users/<user>/szkody_demo` jako eksperyment.
- Zachowanie Delta: `replaceWhere` przez `saveAsTable` w gold, MERGE z widoku tymczasowego,
  `createDataFrame(pandas, schema=DDL)` w bronze.
- MLflow na Databricks: `mlflow.register_model` do UC z URI modelu z `log_model`, kod bledu "brak aliasu/modelu"
  w UC (kod `RESOURCE_DOES_NOT_EXIST`/`NOT_FOUND` albo HTTP 404 uznawany za brak championa), typ pola `version`.
  Lokalnie sprawdzone tylko na MLflow 3.17.2 ze sqlite.
- Zaleznosc `mlflow` w wheelu na runtime, gdzie jest tylko `mlflow-skinny`: pip dociagnie pelny `mlflow`.

## Infra i koszty

Workspace tworzy Terraform w `infra/` (osobny lokalny tfstate; plan/apply uruchamia Kamil). Job cluster
jednowezlowy `Standard_DS3_v2` (4 vCPU = limit regionu), wiec job i klaster interaktywny nie moga chodzic
naraz. Destroy razem z reszta 08.11.2026.
