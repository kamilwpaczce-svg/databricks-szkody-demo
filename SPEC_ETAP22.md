# SPEC Etap 22 — mini-demo Azure Databricks (szkody do weryfikacji)

Data: 31.10.2026. Decyzja Kamila 31.10 12:52: „rób databricks” — Azure Databricks TERAZ (przed 9.11, kredyt trialu),
start 31.10–01.11. Cel biznesowy: zamknąć lukę „Databricks” z ogłoszenia Empik (AI Engineer, ważne do 22.11)
i wysłać CV ok. 02–04.11 zamiast 12–15.11. Źródło planu: `underwriter_agent_demo/PLAN_ETAPY_20-22.md`, sekcja Etap 22.

## Zakres

Ten sam problem co Etap 15 underwritera (syntetyczne „szkoda do weryfikacji”, 7 cech, HGB vs logreg, bramka
champion/challenger na PR AUC) — przeniesiony na Databricks, żeby README mogło uczciwie porównać
„ten sam problem: Azure ML/MLflow w underwriterze vs Databricks”.

Poza zakresem: Delta Live Tables / Lakeflow, Feature Store online, serving endpoint, VNet injection, dane realne.

## Decyzje

1. **Osobne repo** `projekty/databricks_szkody_demo` (git lokalnie; GitHub i push — dopiero za zgodą Kamila).
   Z underwritera KOPIUJEMY (nie importujemy): `ryzyko/generator.py`, `ryzyko/cechy.py` (tylko stałe CECHY,
   bez `cechy_ze_stanu`), `ryzyko/model.py`, `ryzyko/bramka.py`. Nagłówek każdego pliku: „skopiowane z
   underwriter_agent_demo@a1e0a3e, Etap 15”.
2. **Infra = osobny Terraform w tym repo, osobny lokalny tfstate.** NIGDY nie dotykamy `etap13.tfstate` underwritera.
   RG `rg-databricks-demo` (swedencentral), `azurerm_databricks_workspace` sku `premium` (`trial` odrzucony 31.10:
   TrialSkuNotSupportedForSubscriptionType na Free Trial; DBU z kredytu, szac. 2–5 $ na całe demo — NIESPRAWDZONE). Bez VNet injection.
   `plan`/`apply` uruchamia Kamil. Rejestracja providera `Microsoft.Databricks` — Kamil (`az provider register`).
3. **Compute:** job cluster jednowęzłowy `Standard_D4ds_v5` (DS3_v2 odpadł 31.10: SkuNotAvailable w swedencentral; 4 vCPU = cały limit regionu 4/4, zmierzone 31.10),
   `spark_version` LTS ML (bieżące LTS z `databricks clusters spark-versions`), `num_workers: 0`,
   `spark.databricks.cluster.profile: singleNode`. Klaster jobowy ginie po jobie. Klaster interaktywny
   — tylko ręcznie, `autotermination_minutes: 10`. Dwa klastry naraz NIE wejdą w limit — job i klaster
   interaktywny nie mogą chodzić jednocześnie.
4. **Unity Catalog:** zakładamy auto-metastore nowego workspace'u (NIESPRAWDZONE dla trialu w swedencentral —
   sprawdzić w zadaniu 2). Katalog `szkody_demo`, schematy `bronze`, `silver`, `gold`, `ml`.
   Plan B, gdy UC nieaktywny: `hive_metastore` + rejestr modeli workspace'u; README mówi to wprost.
5. **Medallion (Delta):**
   - `bronze.zgloszenia_raw` — wynik `generuj_zbior(n=3500, seed)` + kolumny `_partia` (seed), `_zaladowano` (timestamp);
     dopisywanie (append), każda partia osobno. Partia z dryfem: parametr `przesuniecie` (JSON) w jobie.
   - `silver.zgloszenia` — PySpark: deduplikacja, typy, odrzucenie wierszy łamiących reguły z `jakosc.py`
     (zakresy z generatora, rodzaj_mienia ∈ RODZAJE_MIENIA, etykieta ∈ {0,1}); odrzucone do `silver.zgloszenia_odrzucone`
     z powodem. MERGE po kluczu (`_partia`, `id_wiersza`).
   - `gold.cechy_szkod` — 7 cech + etykieta + `_partia`, gotowe do treningu.
   Reguły jakości jako czyste dane w `src/szkody/jakosc.py` (lista reguł → wyrażenie SQL), testowane lokalnie
   na pandas; Spark wykonuje to samo wyrażenie. Job kończy się wyjątkiem, gdy odrzuconych > 1% partii.
6. **Trening:** `train.py` — czyta `gold.cechy_szkod` do pandas (3500 wierszy, Spark nie jest tu potrzebny — README
   mówi to wprost), split jak w Etapie 15, trenuje `hgb` i `logreg`, loguje do MLflow (eksperyment w workspace),
   rejestruje lepszy w UC jako `szkody_demo.ml.szkoda_do_weryfikacji` z sygnaturą i przykładem wejścia.
7. **Bramka:** `bramka.decyzja` (skopiowana). Champion = alias `champion` w UC; metryki championa z tagów wersji.
   Promocja → `set_registered_model_alias`. Odrzucenie → wersja zostaje bez aliasu, tag `bramka=odrzucony` z powodem.
   Job NIE pada przy odrzuceniu (to wynik, nie błąd).
8. **Databricks Asset Bundle** (`databricks.yml`): job `szkody_pipeline` z 4 zadaniami: `bronze` → `silver` → `gold`
   → `trening`; parametry `seed`, `przesuniecie`. Targety `dev` (mode development) i `demo`.
9. **CI (GitHub Actions)**: `pytest` + `databricks bundle validate` na PR; `bundle deploy -t demo` na main.
   Uwierzytelnienie: Azure OIDC (`azure/login`) + Databricks CLI `auth_type: azure-cli`. Service principal
   underwritera (ten sam app registration, nowy federated credential dla nowego repo) — dodanie go do
   workspace'u robi Kamil. Uruchamianie joba w CI: tylko `workflow_dispatch` (koszt VM), nie przy każdym pushu.
10. **Koszt:** VM D4ds_v5 + dyski + IP podczas biegu joba (~kilka–kilkanaście min/bieg); DBU Premium jobs z kredytu.
    Po każdej sesji: Cost Management. Destroy razem z resztą 08.11 (`terraform destroy` W TYM repo + kontrola,
    że managed RG `databricks-rg-*` zniknęła).

## Zadania

| # | Co | Gdzie | Kto | Kryterium |
|---|---|---|---|---|
| 0 | Szkielet repo: `pyproject.toml`, `src/szkody/` (generator, cechy, model, bramka — kopie), `jakosc.py`, testy pytest | lokalnie | worker | `pytest` zielony; kopie identyczne z underwriterem poza nagłówkiem (diff) |
| 1 | Kod zadań joba (`bronze.py`, `silver.py`, `gold.py`, `train.py`) + `databricks.yml` | lokalnie | worker | `pytest` zielony; `databricks bundle validate` (po zadaniu 2) |
| 2 | Terraform workspace + provider + instalacja Databricks CLI; sprawdzenie UC i spark_version | Azure | Kamil (apply), sesja (pliki tf, polecenia) | workspace działa, `databricks current-user me` OK, katalog widać |
| 3 | `bundle deploy -t dev` + bieg joba: partia czysta (seed 1) → partia z dryfem → druga czysta | Azure | sesja + Kamil | 3 biegi zielone; tabele Delta; wersje w UC; alias `champion`; jedna decyzja bramki w logu |
| 4 | GitHub Actions (OIDC) | GitHub | sesja; repo/push za zgodą | validate + deploy z Actions zielone |
| 5 | README: porównanie Azure ML vs Databricks, koszty, zrzuty (`docs/`), `BAZA_DOWODOW.md`, CV Empik | lokalnie | sesja | CV 2 strony, Databricks z poziomem ZBUDOWANE |

Postęp: commit po każdym zadaniu (repo lokalne), stan w tabeli niżej.

## Ryzyka

- Sku `trial` lub auto-UC niedostępne na subskrypcji Free Trial → plan B z decyzji 2/4, jawnie w README.
- Limit 4 vCPU: równoległy klaster interaktywny zablokuje job (błąd quota) — przed biegiem sprawdzić, że nic nie chodzi.
- Zapomniany klaster → koszt: tylko job clusters; interaktywny z autoterminacją 10 min.
- `spending limit On`: po wyczerpaniu kredytu subskrypcja się wyłącza — kontrola budżetu po każdym biegu.

## Stan

| # | Status | Commit / dowód |
|---|---|---|
| 0 | ZROBIONE 31.10 (lokalnie) | df66b47; pytest 41 zielonych; kopie = underwriter@a1e0a3e poza importem i dopiskiem (diff --strip-trailing-cr) |
| 1 | KOD GOTOWY 31.10, niezwalidowany/niewdrozony | b415b56; pytest 110 zielonych lokalnie; `bundle validate` i bieg na Databricks NIESPRAWDZONE (lista w README) |
| 2 | ZROBIONE 31.10 | 3201a36; workspace premium (host poza repo, `DATABRICKS_HOST`), UC auto-metastore BEZ storage_root -> katalog workspace'u `dbw_szkody_demo_kk1` (CREATE CATALOG odrzucane, b0fd3c2); CLI wymaga `DATABRICKS_AZURE_TENANT_ID` |
| 3 | ZROBIONE 31.10 | job 377714994460053, 17.3 LTS ML, D4ds_v5. Bieg 1 seed=1 czysty: v1 hgb PR AUC 0,395 -> champion (brak poprzedniego). Bieg 2 seed=2 dryf `{suma_ubezpieczenia:0.5, rodzaj_mienia:0.25}` (run 793309151496645, 15:01-15:08): v2 hgb 0,418 -> PROMOCJA. Bieg 3 seed=3 czysty (run 956735281161038): v3 logreg 0,379 < 0,408 -> ODRZUCONY, champion zostaje v2. Odrzuconych w silver 0/3500 w kazdej partii |
| 4 | — | |
| 5 | — | |
