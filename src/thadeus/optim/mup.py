"""muP — régler les hyperparamètres sur un modèle jouet, les transférer au grand.

**Pourquoi c'est le cœur du dispositif ici.** Le budget cloud est de 15,24
crédits, soit un seul grand run. Trouver le bon taux d'apprentissage par
tâtonnement sur un modèle de 188 M coûterait plusieurs runs, c'est-à-dire tout
le budget. muP permet de le chercher sur un modèle de 5 M — gratuit, quelques
minutes sur le Mac — et de le **transférer tel quel**.

**L'idée.** Dans la paramétrisation standard, le taux d'apprentissage optimal
dérive quand le modèle s'élargit : ce qui marche à `d_model = 256` diverge à
2048. muP corrige cette dérive en faisant dépendre de la largeur trois choses —
l'échelle d'initialisation, le taux d'apprentissage par famille de paramètres,
et le facteur appliqué aux logits — de sorte que l'optimum devienne
**indépendant de la largeur**.

Les règles, avec ``m = d_model / d_base`` :

| Famille | Initialisation | Taux d'apprentissage | Multiplicateur |
|---|---|---|---|
| ``embedding`` (entrée) | inchangée | inchangé | ×1 |
| ``hidden`` | ÷ √m | ÷ m | ×1 |
| sortie (logits) | ÷ m | ÷ m | ×(1/m) |
| ``vector`` | inchangée | inchangé | ×1 |

**Honnêteté sur ce qui est implémenté.** On applique les règles ci-dessus, qui
sont le cœur de muP. On n'applique **pas** le passage de l'attention en
``1/d_head`` au lieu de ``1/√d_head`` : notre modèle utilise QK-norm, qui borne
déjà l'amplitude des logits d'attention et rend cette correction largement
redondante. Surtout, l'interaction de muP avec **Muon** est moins établie que
son interaction avec Adam — l'orthogonalisation impose déjà une norme de pas.

Conséquence de méthode : **on ne fait pas confiance à la théorie, on mesure le
transfert.** :mod:`scripts/lr_sweep.py` entraîne plusieurs largeurs à plusieurs
taux et vérifie que l'optimum ne bouge pas. Si la courbe ne s'aligne pas, muP
n'est pas appliqué correctement — et un budget de crédits n'est pas l'endroit
où le découvrir.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from thadeus.core.config import Schema
from thadeus.core.logs import get_logger

if TYPE_CHECKING:
    from thadeus.model.config import ModelConfig

__all__ = ["MupConfig", "apply_mup", "lr_scales", "width_multiplier"]

log = get_logger(__name__)


class MupConfig(Schema):
    """Réglage de la paramétrisation à transfert.

    Args:
        enabled: à ``False``, tout vaut 1 et le modèle est en paramétrisation
            standard. C'est le témoin indispensable : sans lui, on ne peut pas
            montrer que muP change quelque chose.
        base_d_model: largeur du modèle **de référence**, celui sur lequel on
            règle les hyperparamètres. Les multiplicateurs sont tous relatifs à
            elle, donc un modèle à cette largeur n'est pas modifié.
        base_init_std: écart-type d'initialisation à la largeur de référence.
    """

    enabled: bool = False
    base_d_model: int = 256
    base_init_std: float = 0.02


def width_multiplier(cfg: ModelConfig, mup: MupConfig) -> float:
    """``m = d_model / base_d_model`` — le seul nombre dont tout dépend."""
    return cfg.d_model / mup.base_d_model


def lr_scales(cfg: ModelConfig, mup: MupConfig) -> dict[str, float]:
    """Multiplicateurs de taux d'apprentissage par famille de paramètres.

    Seules les matrices cachées voient leur taux divisé par ``m``. Les
    embeddings et les vecteurs gardent le leur : leur nombre de paramètres par
    « unité de sens » ne croît pas avec la largeur de la même façon.
    """
    if not mup.enabled:
        return {"hidden": 1.0, "embedding": 1.0, "vector": 1.0}
    m = width_multiplier(cfg, mup)
    return {"hidden": 1.0 / m, "embedding": 1.0, "vector": 1.0}


def logit_scale(cfg: ModelConfig, mup: MupConfig) -> float:
    """Facteur appliqué aux logits.

    Sans lui, les logits croissent avec la largeur et la perte initiale s'écarte
    de ``ln(V)`` — le modèle démarre déjà confiant sur des prédictions
    arbitraires, ce qui déforme les premiers pas d'entraînement.
    """
    return 1.0 / width_multiplier(cfg, mup) if mup.enabled else 1.0


@torch.no_grad()
def apply_mup(model: nn.Module, cfg: ModelConfig, mup: MupConfig) -> dict[str, float]:
    """Réajuste l'initialisation selon les règles de muP, en place.

    À appeler **après** :func:`~thadeus.model.init.initialize` : on part de
    l'initialisation standard et on la corrige, plutôt que de dupliquer la
    logique d'initialisation.

    Returns:
        Les facteurs appliqués, à consigner dans les métadonnées du run — un
        run muP dont on ignore les multiplicateurs n'est pas reproductible.
    """
    from thadeus.model.init import classify_parameters, lm_head_of

    if not mup.enabled:
        return {"width_multiplier": 1.0, "hidden_init": 1.0, "output_init": 1.0}

    m = width_multiplier(cfg, mup)
    groups = classify_parameters(model)

    # Matrices cachées : variance ∝ 1/fan_in, soit un écart-type divisé par √m.
    hidden_factor = m**-0.5
    for param in groups["hidden"]:
        param.mul_(hidden_factor)

    # Projection de sortie : divisée par m, pas par √m. Elle lit le flux
    # résiduel dont la dimension croît avec la largeur, et c'est ce qui garde
    # les logits d'échelle constante.
    head = lm_head_of(model)
    output_factor = 1.0 / m
    if head is not None:
        head.weight.mul_(output_factor * m**0.5)  # annule le √m déjà appliqué
    else:
        # Embeddings partagés : la table sert d'entrée ET de sortie. On ne peut
        # pas la réajuster sans casser le côté entrée — c'est le facteur de
        # logits (voir `logit_scale`) qui joue ce rôle à sa place.
        output_factor = 1.0

    factors = {
        "width_multiplier": m,
        "hidden_init": hidden_factor,
        "output_init": output_factor,
        "logit_scale": logit_scale(cfg, mup),
        **{f"lr_scale_{k}": v for k, v in lr_scales(cfg, mup).items()},
    }
    log.info(
        "muP actif : m = %.2f (d_model %d / base %d) · init cachée ×%.3f · lr cachée ×%.3f",
        m,
        cfg.d_model,
        mup.base_d_model,
        hidden_factor,
        factors["lr_scale_hidden"],
    )
    return factors
