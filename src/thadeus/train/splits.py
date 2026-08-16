"""Tenir une part du corpus à l'écart — et garantir qu'elle le reste.

**Le problème que ce module résout.** La validation était d'abord prélevée à la
fin du corpus, en supposant qu'un corpus mélangé rendait sa queue
représentative. La mesure a démenti l'hypothèse : la composition **dérive** le
long du fichier, parce que les sources s'épuisent à des rythmes différents
pendant l'entrelacement. La queue de ``thadeus_v1`` contenait 27 % de code
contre 12 % ailleurs, et le code est bien plus facile à prédire que la prose.

Conséquence : la perte de validation valait 1,65 quand l'entraînement était à
2,34. Ce n'était pas un écart de régime mais **deux distributions
différentes**, et une dégradation du français serait passée inaperçue.

**La solution.** Prélever N blocs régulièrement espacés plutôt qu'une tranche
finale. Même volume tenu à l'écart, mais un échantillon qui suit la composition
réelle du corpus. Le découpage ne dépend que de la taille du corpus et de deux
paramètres : il est donc reproductible sans rien mémoriser.

**L'étanchéité est obtenue par construction, pas par rejet.** Les positions
sont tirées dans l'espace cumulé des places *utilisables* d'un jeu de zones,
puis projetées sur les zones réelles. Une fenêtre d'entraînement ne peut donc
pas chevaucher un bloc de validation, même partiellement — alors qu'un tirage
uniforme suivi d'un rejet laisserait passer les chevauchements de bord, et
qu'une fuite ici rendrait la validation complaisante sans qu'aucune erreur ne
se manifeste.
"""

from __future__ import annotations

import numpy as np

__all__ = ["decouper", "placements", "tirer_positions"]

Zone = tuple[int, int]  # (début, longueur)


def decouper(n_tokens: int, val_tokens: int, val_blocks: int) -> tuple[list[Zone], list[Zone]]:
    """Répartit la validation en blocs et rend ``(zones_val, zones_train)``.

    Chaque bloc est **centré** dans son tronçon : le tout début et la toute fin
    du corpus restent à l'entraînement, ce qui évite qu'un bloc coïncide avec
    une frontière de shard.

    Les deux listes sont complémentaires et couvrent exactement ``[0,
    n_tokens)`` — c'est cette propriété qui rend l'étanchéité vérifiable.
    """
    if val_tokens <= 0:
        return [], [(0, n_tokens)]

    blocs = max(1, val_blocks)
    taille = val_tokens // blocs
    if taille <= 0:
        raise ValueError(
            f"val_tokens ({val_tokens}) réparti sur {blocs} blocs donne des blocs vides. "
            f"Réduire val_blocks, ou augmenter val_tokens."
        )
    troncon = n_tokens // blocs
    if troncon < taille:
        raise ValueError(
            f"val_tokens ({val_tokens}) ne laisse pas de place à l'entraînement : "
            f"chaque tronçon de {troncon} tokens devrait contenir un bloc de {taille}."
        )

    val = [(i * troncon + (troncon - taille) // 2, taille) for i in range(blocs)]

    train: list[Zone] = []
    curseur = 0
    for debut, longueur in val:
        if debut > curseur:
            train.append((curseur, debut - curseur))
        curseur = debut + longueur
    if curseur < n_tokens:
        train.append((curseur, n_tokens - curseur))
    return val, train


def placements(zones: list[Zone], window: int) -> tuple[np.ndarray, np.ndarray]:
    """Débuts de zones et nombre de positions utilisables dans chacune.

    Une zone plus courte que la fenêtre en offre zéro : elle reste dans la
    liste mais ne sera jamais tirée, puisque le tirage se fait dans l'espace
    cumulé des positions utilisables.
    """
    debuts = np.array([d for d, _ in zones], dtype=np.int64)
    utiles = np.array([max(0, longueur - window + 1) for _, longueur in zones], dtype=np.int64)
    return debuts, utiles


def tirer_positions(
    rng: np.random.Generator, n: int, zones: list[Zone], window: int, *, nom: str = "split"
) -> np.ndarray:
    """Tire ``n`` départs de fenêtre, tous entièrement contenus dans ``zones``.

    Le tirage porte sur l'index cumulé des positions utilisables, jamais sur le
    corpus entier : c'est ce qui rend le chevauchement impossible plutôt
    qu'improbable.
    """
    debuts, utiles = placements(zones, window)
    total = int(utiles.sum())
    if total <= 0:
        plus_longue = max((longueur for _, longueur in zones), default=0)
        raise ValueError(
            f"{nom} : aucune zone ne peut contenir une fenêtre de {window} tokens "
            f"({len(zones)} zones, la plus longue fait {plus_longue} tokens)"
        )

    cumul = np.cumsum(utiles)
    index = rng.integers(0, total, size=n, dtype=np.int64)
    zone = np.searchsorted(cumul, index, side="right")
    offset_zone = index - np.concatenate(([0], cumul[:-1]))[zone]
    return debuts[zone] + offset_zone


def parcourir(zones: list[Zone], window: int):
    """Parcourt les zones de bout en bout, fenêtre par fenêtre, sans recouvrement.

    Pour une évaluation déterministe qui couvre l'ensemble d'un split plutôt
    que d'en échantillonner un morceau.
    """
    for debut, longueur in zones:
        position = debut
        while position + window <= debut + longueur:
            yield position
            position += window
