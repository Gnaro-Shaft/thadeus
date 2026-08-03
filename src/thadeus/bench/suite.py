"""Orchestration du banc de mesure : config -> exécution -> artefact.

Illustre de bout en bout le patron que tous les étages suivants reprendront :

    charger la config -> valider par un schéma -> ouvrir l'artefact
    -> exécuter -> écrire les résultats -> écrire meta.json en dernier

Le ``meta.json`` écrit en dernier fait office de marqueur d'achèvement : un run
interrompu laisse un répertoire sans métadonnées, donc considéré comme absent,
donc rejoué. Aucune reprise silencieuse sur des données partielles.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from thadeus.bench.kernels import run_benchmark
from thadeus.core.artifacts import Artifact, open_artifact
from thadeus.core.config import Schema
from thadeus.core.device import describe, resolve_device
from thadeus.core.logs import get_logger
from thadeus.core.seeding import seed_everything

__all__ = ["BenchConfig", "run_suite", "summarize"]

log = get_logger(__name__)


class BenchConfig(Schema):
    """Schéma d'une campagne de mesures.

    Hérite de :class:`~thadeus.core.config.Schema`, donc ``extra="forbid"`` :
    une clé mal orthographiée dans le TOML lève une erreur au lieu d'être
    ignorée.
    """

    label: str = "kernels"
    device: str = "auto"
    seed: int = 1337
    defaults: dict[str, Any] = Field(default_factory=dict)
    benchmarks: list[dict[str, Any]] = Field(default_factory=list)


def run_suite(raw_config: dict[str, Any], *, force: bool = False) -> Artifact:
    """Exécute la campagne décrite par ``raw_config``.

    Args:
        raw_config: config déjà chargée et fusionnée (voir
            :func:`~thadeus.core.config.load_config`).
        force: rejoue même si l'artefact existe déjà. Par défaut on ne rejoue
            pas : le hash de config garantit qu'un artefact existant a été
            produit par exactement cette config.

    Returns:
        L'artefact, achevé (``meta.json`` écrit).
    """
    cfg = BenchConfig(**raw_config)
    artifact = open_artifact("bench", cfg.label, raw_config)

    if artifact.exists() and not force:
        log.info("Artefact déjà présent, rien à refaire : %s", artifact)
        return artifact

    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    info = describe(device)
    log.info("Machine : %s", info)

    rows: list[dict[str, Any]] = []
    for spec in cfg.benchmarks:
        rows.extend(run_benchmark(spec, device, defaults=cfg.defaults))

    artifact.write_json("results.json", rows)
    artifact.write_meta(raw_config, device=info.to_dict(), n_results=len(rows))
    log.info("%d mesures écrites dans %s", len(rows), artifact)
    return artifact


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Extrait les quelques chiffres qui pilotent les décisions du projet.

    Volontairement minimal : la crête bf16 dimensionne le budget de calcul, et
    le rapport bf16/fp32 justifie l'interdiction du fp32 sur le chemin chaud.
    Le reste vit dans ``results.json``.
    """
    matmul = [r for r in rows if r.get("benchmark") == "matmul"]
    peak = {
        dtype: max((r["tflops"] for r in matmul if r["dtype"] == dtype), default=None)
        for dtype in {r["dtype"] for r in matmul}
    }
    bf16, fp32 = peak.get("bfloat16"), peak.get("float32")
    return {
        "peak_tflops": peak,
        "bf16_over_fp32": (bf16 / fp32) if bf16 and fp32 else None,
        "peak_gb_per_s": max(
            (r["gb_per_s"] for r in rows if r.get("benchmark") == "bandwidth"), default=None
        ),
        "attention_tflops": {
            r["seq_len"]: r["tflops"] for r in rows if r.get("benchmark") == "attention"
        },
    }
