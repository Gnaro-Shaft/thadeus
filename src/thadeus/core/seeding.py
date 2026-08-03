"""Reproductibilité.

Un A/B d'architecture ne vaut rien si la différence observée peut venir du
tirage aléatoire. Ce module rend chaque run rejouable : même config + même
graine = même trajectoire.

Nuance importante, à ne pas confondre : ``seed_everything`` fixe l'initialisation
et l'ordre des données ; le mode ``deterministic`` en plus force les *noyaux* de
calcul à être déterministes, ce qui coûte du débit. On veut le premier toujours,
le second seulement pour déboguer un écart inexplicable.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["SeedState", "derive_seed", "new_generator", "seed_everything", "worker_init_fn"]

log = logging.getLogger(__name__)

_MAX_SEED = 2**31 - 1


@dataclass(frozen=True)
class SeedState:
    """Ce qui a été fixé — consigné dans les métadonnées d'artefact."""

    seed: int
    deterministic: bool


def seed_everything(seed: int, *, deterministic: bool = False) -> SeedState:
    """Fixe toutes les sources d'aléa du processus.

    Args:
        seed: la graine maîtresse.
        deterministic: force des noyaux déterministes. Ralentit, et sur MPS
            l'effet est partiel — à réserver au débogage d'un écart entre deux
            exécutions censées être identiques.
    """
    seed = int(seed) % (_MAX_SEED + 1)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        log.info("Mode déterministe actif — débit réduit, à n'utiliser que pour déboguer.")
    else:
        torch.backends.cudnn.benchmark = True

    return SeedState(seed=seed, deterministic=deterministic)


def derive_seed(master: int, *tags: str | int) -> int:
    """Dérive une sous-graine stable à partir de la graine maîtresse.

    Chaque étage tire sa propre graine (``derive_seed(seed, "dataloader", rank)``)
    au lieu de partager un état global. Deux étages restent ainsi indépendants :
    ajouter un tirage dans l'un ne décale pas la séquence de l'autre — sinon,
    changer le chargeur de données changerait l'initialisation du modèle, et
    l'A/B deviendrait ininterprétable.
    """
    value = master
    for tag in tags:
        value = (value * 1_000_003 + (hash(str(tag)) & 0xFFFF_FFFF)) & 0xFFFF_FFFF
    return value % (_MAX_SEED + 1)


def new_generator(seed: int, device: torch.device | None = None) -> torch.Generator:
    """Crée un générateur explicite plutôt que de dépendre de l'état global.

    MPS n'expose pas de générateur attaché au device : on retombe sur le CPU,
    ce qui reste correct puisqu'un générateur ne sert qu'à produire des tirages
    que l'on transfère ensuite.
    """
    if device is not None and device.type == "cuda":
        gen = torch.Generator(device=device)
    else:
        gen = torch.Generator()
    gen.manual_seed(int(seed) % (_MAX_SEED + 1))
    return gen


def worker_init_fn(worker_id: int) -> None:
    """Donne à chaque worker de DataLoader sa propre graine.

    Sans cela, tous les workers partagent la graine du processus parent et
    produisent le même aléa — un bug classique qui divise silencieusement la
    diversité effective des données.
    """
    base = torch.initial_seed() % (_MAX_SEED + 1)
    seed = derive_seed(base, "worker", worker_id)
    random.seed(seed)
    np.random.seed(seed)
