# databricks_szkody_demo

Mini-demo Azure Databricks: ten sam problem co Etap 15 w `underwriter_agent_demo` (syntetyczna
"szkoda do weryfikacji", 7 cech, HistGradientBoosting vs regresja logistyczna, bramka
champion/challenger na PR AUC), przeniesiony na Databricks: medallion w Delta + MLflow w Unity Catalog,
wdrazany jako Databricks Asset Bundle. Specyfikacja: `SPEC_ETAP22.md`.

> **Dane sa syntetyczne** (generator z ukryta regula + szum). Celem jest cykl data engineering + MLOps
> na Databricks, nie jakosc modelu.

**Stan (31.10.2026): wdrozone i uruchomione na Azure Databricks** - 3 biegi joba zielone, bramka
raz promowala, raz odrzucila model (sekcja "Przebiegi"). Workspace jest tymczasowy (usuwany 08.11.2026).

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
| `bronze` | `generuj_zbior(3500, seed, przesuniecie)` | `bronze.zgloszenia_raw` (append) | kolumny `id_wiersza`, `_partia` (= seed), `_zaladowano`; tworzy schematy `IF NOT EXISTS` w istniejacym katalogu |
| `silver` | bronze, partia `_partia = seed` | `silver.zgloszenia`, `silver.zgloszenia_odrzucone` | dedup, `TRY_CAST`, reguly z `jakosc.py`, MERGE po (`_partia`, `id_wiersza`); wyjatek, gdy odrzuconych > 1% albo partia pusta |
| `gold` | silver, ta partia | `gold.cechy_szkod` | 7 cech + etykieta + `_partia`, `replaceWhere` (rerun idempotentny) |
| `trening` | gold, ta partia -> pandas | model `<katalog>.ml.szkoda_do_weryfikacji` | split jak w Etapie 15; oba modele w MLflow; lepszy zarejestrowany; alias `champion` wg bramki |

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

## Przebiegi (Azure Databricks, 31.10.2026)

Job `szkody_pipeline`, runtime 17.3 LTS ML, klaster jobowy jednowezlowy `Standard_D4ds_v5`, Unity Catalog.
Kazdy bieg = nowa partia 3500 wierszy; trening na partii (test = 1050 wierszy). Bieg trwa ok. 6-7 min,
z czego wiekszosc to start klastra.

| # | partia | przesuniecie (dryf) | odrzucone w silver | najlepszy model | PR AUC | ROC AUC | decyzja bramki |
|---|---|---|---|---|---|---|---|
| 1 | seed 1 | brak | 0 / 3500 | v1 `hgb` | 0,3951 | 0,7525 | brak championa -> v1 championem |
| 2 | seed 2 | `{"suma_ubezpieczenia": 0.5, "rodzaj_mienia": 0.25}` | 0 / 3500 | v2 `hgb` | 0,4181 | 0,7823 | `0,4181 >= 0,3951 - 0,01` -> **promocja**, champion 1 -> 2 |
| 3 | seed 3 | brak | 0 / 3500 | v3 `logreg` | 0,3790 | 0,7454 | `0,3790 < 0,4181 - 0,01` -> **odrzucony**, champion zostaje 2 |

Log zadania `trening` z biegu 3 (doslownie):

```
bramka: odrzucony: pr_auc challengera 0.3790 vs champion 0.4181 (tol 0.01), ponizej progu 0.4081
rejestr: <katalog>.ml.szkoda_do_weryfikacji wersja 3 (logreg), promowany=False, champion 2 -> 2
```

Przesuniecie w biegu 2 zostalo dobrane tak, zeby nie lamac regul jakosci (sprawdzone lokalnie w DuckDB:
0 naruszen) - inaczej job padlby w `silver` na progu 1% odrzuconych, co tez jest poprawnym zachowaniem.

## Ten sam problem: Azure ML (underwriter, Etap 15) vs Databricks (to repo)

| | Azure ML (`underwriter_agent_demo`, `docs/MLOPS.md`) | Azure Databricks (to repo) |
|---|---|---|
| Orkiestracja | pipeline Azure ML: 4 komponenty (argparse, katalogi wej./wyj.) | job z bundla: 4 zadania `spark_python_task`, kod z wheela |
| Dane | pliki w katalogach komponentow, jedna partia na bieg | tabele Delta bronze/silver/gold w Unity Catalog, partie dopisywane, MERGE, `replaceWhere` |
| Jakosc danych | walidacja w kodzie Pythona | reguly jako dane -> wyrazenia SQL; te same napisy testowane lokalnie w DuckDB i wykonywane przez Spark |
| Compute | serverless `Standard_DS2_v2` | klaster jobowy single-node `D4ds_v5`, ginie po jobie |
| Rejestr modeli | rejestr Azure ML przez MLflow; **aliasy nie dzialaja (404)**, `register_model` nie dziala -> champion = tag modelu | Unity Catalog (`databricks-uc`): `register_model` i **natywny alias `champion`** dzialaja |
| Bramka | `ryzyko/bramka.py` (PR AUC, tolerancja 0,01) | ta sama funkcja (kopia), wynik = alias + tagi wersji |
| Wdrozenie | `pipeline_ml/uruchom.py` (SDK) | `databricks bundle deploy` (deklaratywnie, targety dev/demo) |
| Infra | Terraform, wspolny stan kilku etapow | Terraform, osobny lokalny stan |

Wniosek: przy 3500-5000 wierszach Spark nie przyspiesza treningu (tu trening i tak idzie na pandas),
ale Databricks daje dwie rzeczy, ktorych w Azure ML trzeba bylo obchodzic: tabele Delta z historia
partii i rejestr w Unity Catalog z dzialajacymi aliasami.

## Czego nauczyl bieg na zywo (31.10)

- Azure Free Trial nie przyjmuje sku `trial` workspace'u (`TrialSkuNotSupportedForSubscriptionType`) -> `premium`.
- `Standard_DS3_v2` niedostepny w swedencentral (`CLOUD_PROVIDER_RESOURCE_STOCKOUT`) -> `Standard_D4ds_v5`.
- Auto-metastore UC nowego workspace'u nie ma `storage_root`: `CREATE CATALOG IF NOT EXISTS` pada nawet,
  gdy nic nie trzeba tworzyc (`INVALID_STATE`) -> pipeline uzywa katalogu workspace'u (zmienna `katalog`)
  i tworzy tylko schematy.
- `workspace.host` w `databricks.yml` nie przyjmuje `${var.*}` -> host poza repo, z `DATABRICKS_HOST`.
- Databricks CLI z `auth_type=azure-cli` wymaga `DATABRICKS_AZURE_TENANT_ID`, gdy `az` ma kilka tenantow.
- `bundle run --params` dzieli wartosci po przecinku, wiec JSON z kilkoma kluczami lepiej podac przez
  `databricks jobs run-now --json`.

## Uruchomienie na workspace

```bash
export DATABRICKS_HOST=https://adb-<id>.azuredatabricks.net   # output `url_workspace` z infra/
export DATABRICKS_AUTH_TYPE=azure-cli DATABRICKS_AZURE_TENANT_ID=<tenant>
databricks bundle deploy
databricks bundle run szkody_pipeline --params seed=1
```

## Infra i koszty

Workspace tworzy Terraform w `infra/` (osobny lokalny tfstate). Job cluster jednowezlowy
`Standard_D4ds_v5` (4 vCPU = caly limit regionu na subskrypcji trial), wiec job i klaster interaktywny
nie moga chodzic naraz. Koszt: VM tylko na czas biegu + DBU Premium Jobs. Destroy 08.11.2026.
