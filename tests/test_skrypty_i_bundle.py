"""Skrypty zadan (bez Sparka) i struktura databricks.yml (bez Databricks CLI).

`databricks bundle validate` dopiero w zadaniu 2 - tu tylko to, co da sie sprawdzic lokalnie:
skladnia YAML i zgodnosc z SPEC (4 zadania w lancuchu, klaster jednowezlowy, parametry joba,
targety, pliki istnieja, argumenty zgodne z parserem skryptow).
"""

import importlib
import sys
from pathlib import Path

import pytest
import yaml

from szkody.zadania.wspolne import zbuduj_parser

KORZEN = Path(__file__).resolve().parents[1]
MODULY = ["bronze", "silver", "gold", "train"]


@pytest.fixture(scope="module")
def bundle():
    return yaml.safe_load((KORZEN / "databricks.yml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("modul", MODULY)
def test_skrypt_importuje_sie_bez_pyspark(modul):
    assert "pyspark" not in sys.modules
    mod = importlib.import_module(f"szkody.zadania.{modul}")
    assert callable(mod.main)
    assert "pyspark" not in sys.modules  # pyspark tylko wewnatrz main()/uruchom()


@pytest.mark.parametrize("modul", MODULY)
def test_zly_argument_konczy_sie_przed_startem_sparka(modul):
    mod = importlib.import_module(f"szkody.zadania.{modul}")
    with pytest.raises(SystemExit):
        mod.main(["--seed", "nie_liczba"])


@pytest.mark.parametrize("modul", ["bronze", "train"])
def test_zly_json_przesuniecia_pada_zanim_ruszy_spark(modul):
    mod = importlib.import_module(f"szkody.zadania.{modul}")
    with pytest.raises(ValueError, match="JSON"):
        mod.main(["--przesuniecie", "{zly"])


def test_bundle_nazwa_i_targety(bundle):
    assert bundle["bundle"]["name"] == "szkody_demo"
    assert bundle["targets"]["dev"]["mode"] == "development" and bundle["targets"]["dev"]["default"] is True
    assert "demo" in bundle["targets"]


def test_bundle_host_i_spark_version_to_zmienne_z_placeholderem(bundle):
    assert bundle["workspace"]["host"].startswith("https://adb-")
    assert "spark_version" in bundle["variables"] and "default" in bundle["variables"]["spark_version"]


def test_job_cztery_zadania_w_lancuchu(bundle):
    zadania = bundle["resources"]["jobs"]["szkody_pipeline"]["tasks"]
    assert [z["task_key"] for z in zadania] == ["bronze", "silver", "gold", "trening"]
    zaleznosci = {z["task_key"]: [d["task_key"] for d in z.get("depends_on", [])] for z in zadania}
    assert zaleznosci == {"bronze": [], "silver": ["bronze"], "gold": ["silver"], "trening": ["gold"]}


def test_job_parametry_seed_i_przesuniecie_z_domyslnymi(bundle):
    parametry = {p["name"]: p["default"] for p in bundle["resources"]["jobs"]["szkody_pipeline"]["parameters"]}
    assert parametry == {"seed": "1", "przesuniecie": "{}"}


def test_job_klaster_jednowezlowy_ds3_v2(bundle):
    job = bundle["resources"]["jobs"]["szkody_pipeline"]
    klaster = job["job_clusters"][0]["new_cluster"]
    assert klaster["num_workers"] == 0
    assert klaster["node_type_id"] == "${var.node_type_id}"
    assert bundle["variables"]["node_type_id"]["default"] == "Standard_D4ds_v5"
    assert klaster["spark_conf"]["spark.databricks.cluster.profile"] == "singleNode"
    assert klaster["custom_tags"]["ResourceClass"] == "SingleNode"
    assert klaster["spark_version"] == "${var.spark_version}"
    assert job["max_concurrent_runs"] == 1  # limit 4 vCPU w regionie


def test_zadania_wskazuja_istniejace_skrypty_i_ten_sam_klaster(bundle):
    job = bundle["resources"]["jobs"]["szkody_pipeline"]
    klucz_klastra = job["job_clusters"][0]["job_cluster_key"]
    for zadanie in job["tasks"]:
        assert zadanie["job_cluster_key"] == klucz_klastra
        sciezka = KORZEN / zadanie["spark_python_task"]["python_file"]
        assert sciezka.is_file(), sciezka
        assert zadanie["libraries"] == [{"whl": "./dist/*.whl"}]


def test_parametry_zadan_parsuja_sie_parserem_skryptow(bundle):
    """Argumenty z yml (po podstawieniu wartosci domyslnych) musza przejsc przez argparse skryptow."""
    job = bundle["resources"]["jobs"]["szkody_pipeline"]
    podstawienia = {
        "${var.katalog}": "szkody_demo",
        "{{job.parameters.seed}}": "1",
        "{{job.parameters.przesuniecie}}": "{}",
        "${var.eksperyment}": "/Shared/x",
    }
    for zadanie in job["tasks"]:
        argv = [podstawienia.get(a, a) for a in zadanie["spark_python_task"]["parameters"]]
        parser = zbuduj_parser("x")
        parser.add_argument("--eksperyment", default="")
        args = parser.parse_args(argv)
        assert (args.katalog, args.seed, args.przesuniecie) == ("szkody_demo", 1, "{}")


def test_artefakt_wheel_budowany_przez_bundle(bundle):
    artefakt = bundle["artifacts"]["szkody_whl"]
    assert artefakt["type"] == "whl" and artefakt["path"] == "."
    assert "build" in artefakt["build"]
