"""Orchestration de l'étage données : sources -> nettoyage -> dédup -> mélange -> shards.

Deux phases séparées par un passage sur disque, et ce n'est pas une commodité :

1. **Collecte** — chaque source est lue, nettoyée, dédupliquée, puis écrite dans
   ses propres shards. On mesure alors ce qu'elle fournit *réellement*.
2. **Mélange** — on ne peut planifier la composition qu'une fois ces volumes
   connus. Planifier avant reviendrait à faire un budget sans connaître ses
   revenus.

La séparation permet aussi de rejouer le mélange (secondes) sans refaire la
collecte (heures) : changer les proportions du corpus ne doit pas coûter une
nouvelle nuit de téléchargement.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import Field

from thadeus.core.artifacts import Artifact, open_artifact
from thadeus.core.config import Schema
from thadeus.core.logs import get_logger
from thadeus.core.seeding import derive_seed, seed_everything
from thadeus.data.clean import build_pipeline
from thadeus.data.dedup import Deduplicator, MinHashDeduplicator
from thadeus.data.mix import Mixture, plan_mixture
from thadeus.data.novelty import compare_to_existing
from thadeus.data.schema import Document, estimate_tokens, format_tokens
from thadeus.data.shard import ShardWriter, iter_documents, shard_paths
from thadeus.data.sources import SOURCES

__all__ = ["CorpusConfig", "DedupSpec", "SourceSpec", "build_corpus"]

log = get_logger(__name__)


class SourceSpec(Schema):
    """Une source du corpus, son poids et sa chaîne de nettoyage."""

    name: str
    label: str
    weight: float = 1.0
    max_repeats: float = 1.0
    filters: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class DedupSpec(Schema):
    enabled: bool = True
    exact: bool = True
    bands: int = 16
    rows: int = 8
    shingle_size: int = 5
    max_words: int = 5000


class ShardSpec(Schema):
    max_documents: int = 100_000
    max_bytes: int = 256 * 1024 * 1024


class CorpusConfig(Schema):
    """Description complète d'un corpus."""

    label: str = "corpus"
    total_tokens: int = 2_000_000_000
    seed: int = 1337
    sources: list[SourceSpec] = Field(default_factory=list)
    filters: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    dedup: DedupSpec = Field(default_factory=DedupSpec)
    shard: ShardSpec = Field(default_factory=ShardSpec)


def _make_deduplicator(spec: DedupSpec, seed: int) -> Deduplicator:
    if not spec.enabled:
        return Deduplicator(minhash=None, exact=False)
    return Deduplicator(
        exact=spec.exact,
        minhash=MinHashDeduplicator(
            bands=spec.bands,
            rows=spec.rows,
            shingle_size=spec.shingle_size,
            max_words=spec.max_words,
            seed=seed,
        ),
    )


def _collect(cfg: CorpusConfig, artifact: Artifact) -> dict[str, Any]:
    """Phase 1 — lit, nettoie et déduplique chaque source dans ses propres shards.

    La déduplication est **globale**, partagée par toutes les sources : un
    article de Wikipédia repris par une page web doit être détecté même s'il
    arrive par deux sources différentes. C'est aussi pourquoi cette phase ne se
    reprend pas à mi-parcours — l'état du dédupliqueur vit en mémoire, et
    reprendre sans lui produirait un corpus avec des doublons invisibles.
    """
    dedup = _make_deduplicator(cfg.dedup, derive_seed(cfg.seed, "dedup"))
    report: dict[str, Any] = {"sources": {}, "filters": {}, "failed_sources": {}}

    for spec in cfg.sources:
        out_dir = artifact.path / "sources" / spec.label
        stream: Iterator[Document] = SOURCES.build(
            {
                "name": spec.name,
                # **La graine du flux dérive de celle du corpus.** Sans cela,
                # une source garde la graine écrite en dur dans son bloc TOML :
                # deux collectes de même config rendent alors exactement les
                # mêmes documents, quoi qu'on demande par ailleurs.
                #
                # Ce n'était pas théorique : cinq collectes quotidiennes lancées
                # avec `--set seed=<jour>` se sont révélées identiques à 98-99 %,
                # parce que la surcharge n'atteignait que la déduplication et
                # l'entrelacement. Cinq jours de téléchargement pour zéro
                # document nouveau, sans le moindre signal d'erreur.
                #
                # La dérivation dépend aussi du **label** : deux sources tirant
                # du même dataset prélèvent des tranches différentes, ce que les
                # configs obtenaient jusqu'ici en codant des graines à la main.
                "seed": derive_seed(cfg.seed, "source", spec.label),
                # Placé après, un réglage explicite l'emporte toujours — pour
                # rejouer une tranche précise à l'identique.
                **spec.options,
            },
            label=spec.label,
        )

        if spec.filters:
            if spec.filters not in cfg.filters:
                raise KeyError(
                    f"source {spec.label!r} référence la chaîne de filtres "
                    f"{spec.filters!r}, absente de [filters]. "
                    f"Définies : {', '.join(cfg.filters) or '(aucune)'}"
                )
            cleaner = build_pipeline(cfg.filters[spec.filters], name=spec.filters)
            stream = cleaner.apply(stream)
        else:
            cleaner = None

        # Une source qui échoue ne doit pas emporter les heures déjà investies
        # dans les précédentes. On consigne l'échec, on continue, et le plan de
        # mélange signalera la source manquante — plutôt qu'un corpus amputé
        # dont on croirait connaître la composition.
        with ShardWriter(
            out_dir, max_documents=cfg.shard.max_documents, max_bytes=cfg.shard.max_bytes
        ) as writer:
            try:
                writer.write_all(dedup.apply(stream))
            except Exception as exc:  # noqa: BLE001 — réseau, format, licence, script…
                log.error(
                    "Source %s ABANDONNÉE après %d documents : %s: %s",
                    spec.label,
                    writer.n_documents,
                    type(exc).__name__,
                    str(exc)[:200],
                )
                report["failed_sources"][spec.label] = f"{type(exc).__name__}: {str(exc)[:300]}"

        if cleaner is not None:
            cleaner.log_summary()
            report["filters"][spec.label] = cleaner.stats()

        report["sources"][spec.label] = {
            "documents": writer.n_documents,
            "words": writer.n_words,
            "tokens_estimated": estimate_tokens(writer.n_words),
            "shards": writer.n_shards,
        }
        log.info(
            "Source %s : %d documents, ~%s tokens",
            spec.label,
            writer.n_documents,
            format_tokens(estimate_tokens(writer.n_words)),
        )

    dedup.log_summary()
    report["dedup"] = dedup.stats()
    return report


def _interleave(
    artifact: Artifact,
    mixture: Mixture,
    seed: int,
) -> Iterator[Document]:
    """Entrelace les sources selon le plan, en flux.

    Le tirage pondéré évite les longues plages homogènes : un modèle qui voit
    200 000 articles d'encyclopédie d'affilée puis 200 000 pages web subit un
    changement de distribution en cours d'entraînement, ce qui déstabilise
    l'optimisation. On veut chaque lot représentatif du corpus entier.

    Les sources dont le plan demande plus de tokens qu'elles n'en ont sont
    relues depuis le début — c'est la répétition explicitement autorisée par
    ``max_repeats``, comptabilisée dans le plan.
    """
    rng = random.Random(seed)
    budgets = {p.source: p.take_tokens for p in mixture.plans if p.take_tokens > 0}
    if not budgets:
        return

    readers: dict[str, Iterator[Document]] = {}
    directories = {p.source: artifact.path / "sources" / p.source for p in mixture.plans}

    def reader(source: str) -> Iterator[Document]:
        while True:
            produced = False
            for doc in iter_documents(directories[source]):
                produced = True
                yield doc
            if not produced:  # source vide : ne jamais boucler à l'infini
                return

    for source in budgets:
        readers[source] = reader(source)

    while budgets:
        sources = list(budgets)
        source = rng.choices(sources, weights=[budgets[s] for s in sources], k=1)[0]
        try:
            doc = next(readers[source])
        except StopIteration:
            del budgets[source]
            continue

        yield doc
        budgets[source] -= estimate_tokens(doc.n_words)
        if budgets[source] <= 0:
            del budgets[source]


def build_corpus(raw_config: dict[str, Any], *, force: bool = False) -> Artifact:
    """Construit un corpus complet et retourne son artefact achevé."""
    cfg = CorpusConfig(**raw_config)
    if not cfg.sources:
        raise ValueError("aucune source déclarée dans la config")

    artifact = open_artifact("data", cfg.label, raw_config)
    if artifact.exists() and not force:
        log.info("Corpus déjà construit : %s", artifact)
        return artifact

    seed_everything(cfg.seed)
    artifact.create()

    report = _collect(cfg, artifact)

    available = {label: info["tokens_estimated"] for label, info in report["sources"].items()}
    mixture = plan_mixture(
        available,
        {spec.label: spec.weight for spec in cfg.sources},
        total_tokens=cfg.total_tokens,
        max_repeats={spec.label: spec.max_repeats for spec in cfg.sources},
    )
    mixture.log_summary()

    with ShardWriter(
        artifact.path / "corpus",
        max_documents=cfg.shard.max_documents,
        max_bytes=cfg.shard.max_bytes,
    ) as writer:
        writer.write_all(_interleave(artifact, mixture, derive_seed(cfg.seed, "interleave")))

    final = {
        "documents": writer.n_documents,
        "words": writer.n_words,
        "tokens_estimated": estimate_tokens(writer.n_words),
        "shards": writer.n_shards,
        "documents_per_source": writer.per_source,
        "tokens_per_source": {
            src: estimate_tokens(words) for src, words in writer.words_per_source.items()
        },
    }
    # Ce que la collecte apporte de NOUVEAU, et pas seulement ce qu'elle pèse.
    # Sans cette mesure, cinq collectes quasi identiques ont annoncé « 2,03 Md »
    # chacune pendant cinq nuits sans que rien ne le signale.
    nouveaute = compare_to_existing(
        artifact.path / "corpus", root=artifact.root / "data", exclude=artifact.path.name
    )

    artifact.write_json(
        "report.json",
        {**report, "mixture": mixture.to_dict(), "corpus": final, "nouveaute": nouveaute},
    )
    artifact.write_meta(raw_config, corpus=final, warnings=mixture.warnings)

    log.info(
        "Corpus : %d documents, ~%s tokens, %d shards -> %s",
        final["documents"],
        format_tokens(final["tokens_estimated"]),
        final["shards"],
        artifact.path / "corpus",
    )
    return artifact


def corpus_dir(artifact: Artifact) -> Path:
    """Répertoire des shards finaux — l'entrée de l'étage tokenizer."""
    return artifact.path / "corpus"


def peek(artifact: Artifact, n: int = 3) -> list[Document]:
    """Quelques documents du corpus final, pour vérification humaine.

    Regarder le corpus avec ses yeux attrape des catégories entières de
    problèmes qu'aucune statistique ne montre : balisage résiduel, langue
    inattendue, troncatures.
    """
    paths = shard_paths(corpus_dir(artifact))
    return list(iter_documents(paths[:1], limit=n)) if paths else []
