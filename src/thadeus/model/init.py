"""Initialisation des poids.

Deux règles, et la seconde est celle qu'on oublie.

1. **Loi normale d'écart-type modeste** (0,02) sur les projections. Rien
   d'original, c'est la recette GPT-2.

2. **Les projections de sortie des sous-blocs sont divisées par √(2·n_layers).**
   Le chemin résiduel additionne la contribution de chaque sous-bloc ; sans cette
   mise à l'échelle, la variance de l'activation croît linéairement avec la
   profondeur, et un modèle profond démarre avec des activations qui saturent
   déjà. Le facteur 2 vient des deux sous-blocs par couche (attention et
   feed-forward).

Ce module prépare aussi la **Phase 5 (muP)**. La paramétrisation à transfert
d'hyperparamètres consiste précisément à faire dépendre l'initialisation et le
taux d'apprentissage de la largeur du modèle, selon des règles différentes par
type de matrice. :func:`parameter_groups` expose déjà cette classification :
muP consistera à attacher des multiplicateurs à ces groupes, sans rien
restructurer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from thadeus.model.config import ModelConfig

__all__ = ["initialize", "parameter_groups"]

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


def parameter_groups(model: nn.Module, *, weight_decay: float = 0.1) -> list[dict]:
    """Classe les paramètres pour l'optimiseur.

    Deux groupes, sur un critère de dimension plutôt que de nom : **on ne
    régularise que les matrices**. Appliquer une décroissance de poids aux gains
    de normalisation et aux biais les pousse vers zéro, ce qui revient à
    désactiver progressivement les normalisations — un bug silencieux qui se
    manifeste comme une convergence médiocre, jamais comme une erreur.

    La classification par dimension prépare aussi muP (Phase 5), qui distingue
    matrices et vecteurs pour leurs règles de mise à l'échelle.
    """
    matrices, vectors = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (matrices if param.dim() >= 2 else vectors).append(param)

    return [
        {"params": matrices, "weight_decay": weight_decay, "kind": "matrix"},
        {"params": vectors, "weight_decay": 0.0, "kind": "vector"},
    ]
