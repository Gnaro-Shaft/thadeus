"""Planificateurs de taux d'apprentissage.

Le taux d'apprentissage est l'hyperparamètre le plus sensible d'un entraînement,
et sa **trajectoire** compte autant que sa valeur : trop élevé au départ, le
modèle diverge avant d'avoir appris quoi que ce soit ; trop élevé à la fin, il
n'affine jamais.

Deux formes disponibles, et le choix n'est pas cosmétique ici.

**Cosinus** — la référence. Sa faiblesse : la décroissance dépend de la durée
totale, fixée *avant* de démarrer. Prolonger un run ou l'arrêter plus tôt donne
un modèle qui n'a pas fini sa décroissance, donc mauvais.

**WSD** (Warmup-Stable-Decay) — chauffe, puis **palier constant**, puis
décroissance courte sur les derniers pourcents. La propriété qui nous intéresse :
on peut **arrêter le palier quand on veut** et ne payer que la décroissance.
Pour un projet qui entraîne par nuits, avec un budget H100 encore inconnu, c'est
la différence entre pouvoir décider quand s'arrêter et devoir s'engager à
l'avance. Un checkpoint pris pendant le palier reste aussi un bon point de
départ pour continuer — ce qu'un cosinus à moitié parcouru n'est pas.
"""

from __future__ import annotations

import math

from thadeus.core.registry import Registry

__all__ = ["SCHEDULES", "cosine", "wsd"]

SCHEDULES: Registry = Registry("lr_schedule")


def _warmup_factor(step: int, warmup_steps: int) -> float | None:
    """Facteur de chauffe linéaire, ou ``None`` si la chauffe est terminée.

    Démarre à ``1 / warmup_steps`` et non à 0 : un premier pas à taux nul est
    un pas perdu, et certains optimiseurs à état s'initialisent mal dessus.
    """
    if warmup_steps <= 0 or step >= warmup_steps:
        return None
    return (step + 1) / warmup_steps


@SCHEDULES.register("cosine")
def cosine(
    *,
    total_steps: int,
    warmup_steps: int = 0,
    min_ratio: float = 0.1,
):
    """Chauffe linéaire puis décroissance en cosinus jusqu'à ``min_ratio``.

    Args:
        min_ratio: plancher, en fraction du taux de base. Descendre jusqu'à zéro
            gaspille les derniers pas ; 10 % est l'usage courant.
    """

    def factor(step: int) -> float:
        warm = _warmup_factor(step, warmup_steps)
        if warm is not None:
            return warm
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return factor


@SCHEDULES.register("wsd", aliases=("trapezoid",))
def wsd(
    *,
    total_steps: int,
    warmup_steps: int = 0,
    decay_fraction: float = 0.1,
    min_ratio: float = 0.0,
):
    """Chauffe, palier constant, puis décroissance finale.

    Args:
        decay_fraction: part finale de l'entraînement consacrée à la
            décroissance. 10 % suffit — c'est le résultat qui rend WSD
            intéressant : la quasi-totalité du budget se passe à taux maximal.

    La décroissance est **linéaire au carré** (``(1-p)²``) plutôt que linéaire :
    elle passe plus de temps aux taux élevés, ce qui empiriquement donne une
    perte finale légèrement meilleure à budget égal.
    """
    decay_steps = max(1, int(total_steps * decay_fraction))
    stable_end = max(warmup_steps, total_steps - decay_steps)

    def factor(step: int) -> float:
        warm = _warmup_factor(step, warmup_steps)
        if warm is not None:
            return warm
        if step < stable_end:
            return 1.0
        progress = min(1.0, (step - stable_end) / max(1, total_steps - stable_end))
        return min_ratio + (1 - min_ratio) * (1 - progress) ** 2

    return factor


@SCHEDULES.register("constant")
def constant(*, total_steps: int = 0, warmup_steps: int = 0):
    """Taux constant après chauffe — pour les tests et les diagnostics.

    Utile quand on cherche à isoler un effet : une perte qui bouge sous un taux
    constant ne peut pas être attribuée au planificateur.
    """

    def factor(step: int) -> float:
        warm = _warmup_factor(step, warmup_steps)
        return warm if warm is not None else 1.0

    return factor
