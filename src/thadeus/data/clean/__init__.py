"""Filtres de nettoyage — unitaires, composables, et qui rendent des comptes.

Un filtre est une fonction ``Document -> Document | None`` : il renvoie le
document (éventuellement modifié) ou ``None`` pour le rejeter. Cette signature
unique fait qu'une normalisation et un rejet sont le même objet, donc
composables dans n'importe quel ordre.

Ils s'enchaînent en :class:`CleaningPipeline`, qui **compte les rejets par
filtre**. Ce comptage n'est pas de la décoration : sans lui, on découvre qu'un
filtre trop strict a supprimé 80 % d'une source seulement en constatant que le
modèle final est mauvais. Avec lui, on le voit en une ligne de journal.

Les chaînes sont déclarées en config et référencées par les sources :

    [filters.fr_text]
    steps = [
        {name = "normalize"},
        {name = "min_words", min_words = 60},
        {name = "language_is", lang = "fr", min_score = 0.06},
    ]
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

from thadeus.core.logs import get_logger
from thadeus.core.registry import Registry
from thadeus.data.schema import Document

log = get_logger(__name__)

Filter = Callable[[Document], Document | None]

FILTERS: Registry[Filter] = Registry("data_filter")


class CleaningPipeline:
    """Enchaîne des filtres et tient le compte des rejets.

    Les statistiques sont ventilées par filtre **et par source** : un seuil qui
    convient à Wikipédia peut décimer les notes personnelles, et l'agrégat
    global masquerait exactement ce cas.
    """

    def __init__(self, steps: Sequence[tuple[str, Filter]], *, name: str = "pipeline") -> None:
        self.name = name
        self.steps = list(steps)
        self.n_seen = 0
        self.n_kept = 0
        self.rejected: dict[str, int] = {step: 0 for step, _ in self.steps}
        self.rejected_by_source: dict[str, dict[str, int]] = {}

    def __call__(self, doc: Document) -> Document | None:
        self.n_seen += 1
        current = doc
        for step_name, step in self.steps:
            result = step(current)
            if result is None:
                self.rejected[step_name] += 1
                per_source = self.rejected_by_source.setdefault(doc.source, {})
                per_source[step_name] = per_source.get(step_name, 0) + 1
                return None
            current = result
        self.n_kept += 1
        return current

    def apply(self, documents: Iterable[Document]) -> Iterator[Document]:
        """Filtre un flux, en ne laissant passer que les documents retenus."""
        for doc in documents:
            kept = self(doc)
            if kept is not None:
                yield kept

    def stats(self) -> dict[str, Any]:
        """Rapport de nettoyage, écrit dans les métadonnées de l'artefact."""
        return {
            "name": self.name,
            "seen": self.n_seen,
            "kept": self.n_kept,
            "keep_rate": self.n_kept / self.n_seen if self.n_seen else 0.0,
            "rejected_by_step": dict(self.rejected),
            "rejected_by_source": self.rejected_by_source,
        }

    def log_summary(self) -> None:
        stats = self.stats()
        log.info(
            "[%s] %d/%d retenus (%.1f %%)",
            self.name,
            stats["kept"],
            stats["seen"],
            100 * stats["keep_rate"],
        )
        for step, count in sorted(self.rejected.items(), key=lambda kv: -kv[1]):
            if count:
                log.info("    %-24s rejette %7d (%.1f %%)", step, count, 100 * count / self.n_seen)


def build_pipeline(
    specs: Sequence[str | Mapping[str, Any]],
    *,
    name: str = "pipeline",
) -> CleaningPipeline:
    """Construit une chaîne depuis des specs de config."""
    steps: list[tuple[str, Filter]] = []
    for spec in specs:
        step_name = spec if isinstance(spec, str) else spec["name"]
        steps.append((step_name, FILTERS.build(spec)))
    return CleaningPipeline(steps, name=name)


from thadeus.data.clean import quality  # noqa: E402,F401

__all__ = ["FILTERS", "CleaningPipeline", "Filter", "build_pipeline", "quality"]
