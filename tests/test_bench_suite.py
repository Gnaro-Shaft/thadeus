"""Bout en bout de l'étage 0 : config -> exécution -> artefact.

Sert de test de non-régression du *patron* que tous les étages suivants
reprendront, pas seulement du benchmark lui-même.
"""

from __future__ import annotations

import json

import pytest

from thadeus.bench.kernels import BENCHMARKS
from thadeus.bench.suite import run_suite, summarize
from thadeus.core.artifacts import ARTIFACT_ROOT
from thadeus.core.config import load_config

MINIMAL = {
    "label": "test",
    "device": "cpu",
    "seed": 7,
    "defaults": {"warmup": 0, "iters": 1},
    "benchmarks": [
        {"name": "matmul", "sizes": [64], "dtypes": ["float32"]},
        {"name": "bandwidth", "megabytes": 1, "dtypes": ["float32"]},
        {"name": "attention", "batch": 1, "heads": 2, "head_dim": 16, "seq_lens": [32]},
    ],
}


@pytest.fixture
def artifacts_in(tmp_path, monkeypatch):
    """Redirige les artefacts vers un répertoire jetable."""
    monkeypatch.setattr("thadeus.core.artifacts.ARTIFACT_ROOT", tmp_path)
    return tmp_path


def test_les_benchmarks_sont_enregistres():
    assert set(BENCHMARKS.names()) == {"matmul", "bandwidth", "attention"}


def test_execution_produit_un_artefact_complet(artifacts_in):
    artifact = run_suite(MINIMAL)
    assert artifact.exists(), "meta.json doit marquer l'achèvement"
    assert artifact.path.is_relative_to(artifacts_in)

    rows = json.loads((artifact.path / "results.json").read_text(encoding="utf-8"))
    assert {r["benchmark"] for r in rows} == {"matmul", "bandwidth", "attention"}
    assert all(r["backend"] == "cpu" for r in rows), "chaque mesure porte sa machine"

    meta = artifact.read_meta()
    assert meta["config"] == MINIMAL, "la config exacte est rejouable depuis l'artefact"
    assert meta["device"]["backend"] == "cpu"


def test_ne_rejoue_pas_un_artefact_existant(artifacts_in):
    first = run_suite(MINIMAL)
    mtime = first.meta_path.stat().st_mtime_ns
    second = run_suite(MINIMAL)
    assert second.path == first.path
    assert second.meta_path.stat().st_mtime_ns == mtime, "aucune réécriture"


def test_force_rejoue(artifacts_in):
    run_suite(MINIMAL)
    artifact = run_suite(MINIMAL, force=True)
    assert artifact.exists()


def test_cle_inconnue_dans_la_config_rejetee(artifacts_in):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        run_suite({**MINIMAL, "sed": 7})  # faute de frappe sur "seed"


def test_summarize_extrait_les_chiffres_de_decision():
    rows = [
        {"benchmark": "matmul", "dtype": "bfloat16", "tflops": 30.0},
        {"benchmark": "matmul", "dtype": "float32", "tflops": 7.5},
        {"benchmark": "bandwidth", "gb_per_s": 173.0},
        {"benchmark": "attention", "seq_len": 1024, "tflops": 12.0},
    ]
    out = summarize(rows)
    assert out["peak_tflops"]["bfloat16"] == 30.0
    assert out["bf16_over_fp32"] == 4.0  # le facteur qui interdit le fp32 sur MPS
    assert out["peak_gb_per_s"] == 173.0


def test_les_configs_du_depot_sont_valides():
    # Garde-fou : une config cassée doit échouer ici, pas au milieu d'un run.
    for name in ("bench/base.toml", "bench/quick.toml"):
        cfg = load_config(name)
        assert cfg["benchmarks"], f"{name} ne déclare aucun benchmark"


def test_quick_herite_de_base():
    base = load_config("bench/base.toml")
    quick = load_config("bench/quick.toml")
    assert quick["seed"] == base["seed"], "hérité"
    assert quick["label"] != base["label"], "surchargé"
    assert quick["defaults"]["iters"] < base["defaults"]["iters"]


def test_artifact_root_par_defaut_dans_le_depot():
    assert ARTIFACT_ROOT.name == "artifacts"
