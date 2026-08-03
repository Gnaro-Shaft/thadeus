"""Mélange des sources — décider *combien* de chaque, et le dire honnêtement.

Une composition cible (« 55 % de français, 25 % de documentation technique,
15 % de code ») est un souhait. Ce que le corpus contient vraiment dépend de ce
que chaque source peut fournir après nettoyage et déduplication. L'écart entre
les deux est la chose la plus importante à mesurer de tout cet étage : croire
qu'on entraîne sur 55 % de français alors qu'on est à 30 % rend tout diagnostic
ultérieur faux.

Ce module refuse donc de combler un manque en inventant des données. Quand une
source ne peut pas tenir son poids, deux issues seulement :

- **répéter**, dans la limite explicite de ``max_repeats`` — au-delà, le modèle
  apprend par cœur au lieu d'apprendre ;
- **redistribuer** le déficit vers les sources qui ont de la marge, et **le
  signaler**.

C'est exactement le cas du Vault Obsidian : ~1 M de tokens disponibles pour une
cible à 5 % d'un corpus de 1,7 Md, soit un facteur 85. Le plan le dira au lieu
de le masquer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from thadeus.core.logs import get_logger
from thadeus.data.schema import format_tokens

__all__ = ["Mixture", "SourcePlan", "plan_mixture"]

log = get_logger(__name__)


@dataclass(frozen=True)
class SourcePlan:
    """Ce qu'on prend à une source, et ce qu'on n'a pas pu prendre."""

    source: str
    weight: float
    available_tokens: int
    target_tokens: int
    take_tokens: int
    max_repeats: float

    @property
    def repeats(self) -> float:
        """Nombre de passages sur la source. Au-delà de ~4, la mémorisation guette."""
        return self.take_tokens / self.available_tokens if self.available_tokens else 0.0

    @property
    def shortfall_tokens(self) -> int:
        return max(0, self.target_tokens - self.take_tokens)

    @property
    def is_short(self) -> bool:
        return self.shortfall_tokens > 0


@dataclass
class Mixture:
    """Plan de mélange complet, avec la composition réellement atteinte."""

    plans: list[SourcePlan]
    requested_tokens: int
    warnings: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(p.take_tokens for p in self.plans)

    def effective_weights(self) -> dict[str, float]:
        """La composition obtenue — à comparer aux poids demandés."""
        total = self.total_tokens
        return {p.source: (p.take_tokens / total if total else 0.0) for p in self.plans}

    def take_for(self, source: str) -> SourcePlan | None:
        return next((p for p in self.plans if p.source == source), None)

    def to_dict(self) -> dict:
        return {
            "requested_tokens": self.requested_tokens,
            "total_tokens": self.total_tokens,
            "warnings": self.warnings,
            "sources": [
                {
                    "source": p.source,
                    "weight_requested": p.weight,
                    "weight_effective": self.effective_weights()[p.source],
                    "available_tokens": p.available_tokens,
                    "target_tokens": p.target_tokens,
                    "take_tokens": p.take_tokens,
                    "repeats": round(p.repeats, 2),
                    "shortfall_tokens": p.shortfall_tokens,
                }
                for p in self.plans
            ],
        }

    def log_summary(self) -> None:
        log.info(
            "Mélange : %s tokens sur %s demandés",
            format_tokens(self.total_tokens),
            format_tokens(self.requested_tokens),
        )
        effective = self.effective_weights()
        for plan in self.plans:
            log.info(
                "    %-18s demandé %5.1f %%  obtenu %5.1f %%  (%s tokens, x%.2f passages)",
                plan.source,
                100 * plan.weight,
                100 * effective[plan.source],
                format_tokens(plan.take_tokens),
                plan.repeats,
            )
        for warning in self.warnings:
            log.warning("    %s", warning)


def plan_mixture(
    available: dict[str, int],
    weights: dict[str, float],
    *,
    total_tokens: int,
    max_repeats: dict[str, float] | float = 1.0,
    redistribute: bool = True,
) -> Mixture:
    """Calcule combien prendre à chaque source.

    Args:
        available: tokens réellement disponibles par source, **après** nettoyage
            et déduplication. Utiliser le volume brut ici fausserait tout le plan.
        weights: composition souhaitée. Normalisée automatiquement.
        total_tokens: taille visée du corpus final.
        max_repeats: nombre de passages autorisés, global ou par source. 1.0 =
            aucune répétition.
        redistribute: reverse le déficit d'une source aux sources qui ont de la
            marge. Sans cela, un corpus visant 2 Md en produirait 1,4 en silence.

    Returns:
        Le plan, avec la composition effective et les avertissements.
    """
    if not weights:
        raise ValueError("aucune source pondérée")

    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("la somme des poids doit être strictement positive")
    normalized = {src: w / weight_sum for src, w in weights.items()}

    def repeats_for(source: str) -> float:
        return max_repeats if isinstance(max_repeats, int | float) else max_repeats.get(source, 1.0)

    capacity = {
        src: int(available.get(src, 0) * repeats_for(src)) for src in normalized
    }

    take: dict[str, int] = {}
    warnings: list[str] = []
    remaining = total_tokens
    open_sources = dict(normalized)

    # Point fixe : on sert les sources saturées, on redistribue leur déficit
    # aux autres, et on recommence — une source peut saturer à son tour.
    for _ in range(len(normalized) + 1):
        if not open_sources or remaining <= 0:
            break
        share = sum(open_sources.values())
        saturated = False
        for src, weight in list(open_sources.items()):
            target = int(remaining * weight / share)
            if target >= capacity[src]:
                take[src] = capacity[src]
                del open_sources[src]
                saturated = True
        if not saturated:
            for src, weight in open_sources.items():
                take[src] = int(remaining * weight / share)
            open_sources.clear()
            break
        remaining = total_tokens - sum(take.values())
        if not redistribute:
            for src, weight in open_sources.items():
                take[src] = min(int(total_tokens * weight), capacity[src])
            break

    plans = []
    for src, weight in normalized.items():
        plan = SourcePlan(
            source=src,
            weight=weight,
            available_tokens=available.get(src, 0),
            target_tokens=int(total_tokens * weight),
            take_tokens=take.get(src, 0),
            max_repeats=repeats_for(src),
        )
        plans.append(plan)

        if plan.available_tokens == 0:
            warnings.append(f"{src} : aucune donnée disponible")
        elif plan.is_short:
            warnings.append(
                f"{src} : {plan.take_tokens / 1e6:.1f} M tokens disponibles pour "
                f"{plan.target_tokens / 1e6:.1f} M visés — poids effectif réduit. "
                f"Élargir la source, ou augmenter max_repeats en acceptant la mémorisation."
            )
        elif plan.repeats > 2.0:
            warnings.append(
                f"{src} : {plan.repeats:.1f} passages sur les mêmes données — "
                f"risque de mémorisation par cœur."
            )

    mixture = Mixture(plans=plans, requested_tokens=total_tokens, warnings=warnings)
    if mixture.total_tokens < 0.95 * total_tokens:
        mixture.warnings.append(
            f"corpus final à {mixture.total_tokens / 1e9:.2f} Md tokens au lieu de "
            f"{total_tokens / 1e9:.2f} Md : les sources ne suffisent pas."
        )
    return mixture
