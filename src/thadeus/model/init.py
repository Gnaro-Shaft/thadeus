"""Initialisation des poids et classement des paramètres.

Deux règles d'initialisation, et la seconde est celle qu'on oublie.

1. **Loi normale d'écart-type modeste** (0,02) sur les projections. Recette GPT-2.

2. **Les projections de sortie des sous-blocs sont divisées par √(2·n_layers).**
   Le chemin résiduel additionne la contribution de chaque sous-bloc ; sans cette
   mise à l'échelle, la variance de l'activation croît linéairement avec la
   profondeur, et un modèle profond démarre avec des activations qui saturent.
   Le facteur 2 vient des deux sous-blocs par couche.

**Le classement des paramètres** (:func:`parameter_groups`) est ce sur quoi
reposent Muon et muP. Trois familles, et la distinction n'est pas cosmétique :

- ``hidden`` — matrices 2D des couches cachées. Lignes et colonnes y jouent des
  rôles symétriques, ce qui est exactement la condition sous laquelle
  l'orthogonalisation de Muon a un sens.
- ``embedding`` — tables d'entrée et de sortie. Ce sont des **vecteurs indexés**,
  pas des transformations linéaires : les orthogonaliser n'aurait aucun sens, et
  muP leur applique des règles de mise à l'échelle distinctes.
- ``vector`` — gains de normalisation et biais. Jamais de décroissance de poids :
  les pousser vers zéro revient à désactiver progressivement les normalisations,
  bug silencieux qui se manifeste comme une convergence médiocre.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from thadeus.model.config import ModelConfig

__all__ = ["classify_parameters", "initialize", "parameter_groups"]

# Projections qui écrivent dans le flux résiduel — celles qu'il faut atténuer.
_RESIDUAL_OUTPUTS = ("o_proj", "down_proj")


def initialize(model: nn.Module, cfg: ModelConfig) -> None:
    """Initialise tous les poids du modèle, en place."""
    std = cfg.init_std

    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    if cfg.scale_residual_init:
        scale = (2 * cfg.n_layers) ** -0.5
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in _RESIDUAL_OUTPUTS:
                with torch.no_grad():
                    module.weight.mul_(scale)


def classify_parameters(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    """Range les paramètres en ``hidden`` / ``embedding`` / ``vector``.

    Le classement se fait par **rôle**, déduit du module propriétaire, et non par
    dimension seule : une table d'embedding et une matrice de couche cachée sont
    toutes deux 2D, mais Muon ne doit voir que la seconde.

    Les poids partagés entre entrée et sortie ne sont comptés qu'une fois — les
    lister deux fois ferait appliquer deux mises à jour par pas, soit un taux
    d'apprentissage effectif doublé sur la table, en silence.
    """
    groups: dict[str, list[nn.Parameter]] = {"hidden": [], "embedding": [], "vector": []}
    seen: set[int] = set()

    for module in model.modules():
        kind = "embedding" if isinstance(module, nn.Embedding) else None
        for param in module.parameters(recurse=False):
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            if param.ndim < 2:
                groups["vector"].append(param)
            elif kind == "embedding":
                groups["embedding"].append(param)
            else:
                groups["hidden"].append(param)

    return groups


def parameter_groups(
    model: nn.Module,
    *,
    weight_decay: float = 0.1,
    lr_scales: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Construit les groupes passés à l'optimiseur.

    Args:
        weight_decay: appliquée aux matrices et embeddings, **jamais** aux
            vecteurs.
        lr_scales: multiplicateurs de taux d'apprentissage par famille. C'est
            par là que muP transfère les hyperparamètres entre largeurs (voir
            :mod:`thadeus.optim.mup`) ; sans muP, tout vaut 1.

    Returns:
        Un groupe par famille non vide, portant sa clé ``kind`` — les
        constructeurs d'optimiseur s'en servent pour router les matrices cachées
        vers Muon et le reste vers AdamW.
    """
    scales = lr_scales or {}
    classified = classify_parameters(model)

    groups: list[dict[str, Any]] = []
    for kind, params in classified.items():
        if not params:
            continue
        groups.append(
            {
                "params": params,
                "kind": kind,
                "weight_decay": 0.0 if kind == "vector" else weight_decay,
                "lr_scale": scales.get(kind, 1.0),
            }
        )
    return groups


def lm_head_of(model: nn.Module) -> nn.Linear | None:
    """Retrouve la projection de sortie, si elle n'est pas partagée avec l'entrée.

    Quand les embeddings sont partagés, la sortie **est** la table d'entrée :
    elle est déjà classée en ``embedding`` et ne doit pas être traitée deux fois.
    """
    head = getattr(model, "lm_head", None)
    embedding = getattr(model, "embedding", None)
    if head is None or embedding is None:
        return head
    return None if head.weight is embedding.weight else head
