"""Réseaux à propagation avant.

C'est là que vit la majorité des paramètres d'un Transformer — environ deux
tiers dans nos configurations. Le choix de la dimension cachée est donc le
principal levier sur la taille du modèle.

**SwiGLU** remplace le MLP classique par un produit de deux projections, dont
l'une passe par une activation. Le coût : trois matrices au lieu de deux. Le
compromis retenu partout est de réduire la dimension cachée à ``8/3 × d_model``
au lieu de ``4 × d_model``, ce qui rend le nombre de paramètres **identique**
au MLP classique — SwiGLU est donc un gain de qualité à budget constant, pas un
grossissement déguisé.

Le MLP classique reste disponible comme témoin : « SwiGLU vaut mieux » doit être
une mesure sur notre corpus, pas une croyance recopiée.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from thadeus.model.blocks import FFNS

__all__ = ["MLP", "SwiGLU", "swiglu_hidden_dim"]


def swiglu_hidden_dim(d_model: int, *, multiple_of: int = 64, expansion: float = 8 / 3) -> int:
    """Dimension cachée SwiGLU à budget de paramètres équivalent à un MLP ×4.

    L'arrondi à un multiple de 64 n'est pas cosmétique : les noyaux de produit
    matriciel travaillent par tuiles, et une dimension mal alignée laisse des
    unités de calcul inutilisées. Sur une machine où le calcul est la ressource
    limitante, c'est du débit perdu pour rien.
    """
    hidden = int(expansion * d_model)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


@FFNS.register("swiglu")
class SwiGLU(nn.Module):
    """Feed-forward à porte, activation SiLU.

    ``down(silu(gate(x)) * up(x))`` — la branche ``gate`` décide, multiplicativement,
    de ce qui passe de la branche ``up``. C'est ce mécanisme de porte, et non
    l'activation elle-même, qui explique le gain.
    """

    def __init__(
        self,
        *,
        d_model: int,
        hidden_dim: int | None = None,
        bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = hidden_dim if hidden_dim is not None else swiglu_hidden_dim(d_model)
        self.gate_proj = nn.Linear(d_model, hidden, bias=bias)
        self.up_proj = nn.Linear(d_model, hidden, bias=bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return self.dropout(out) if self.dropout is not None else out


@FFNS.register("mlp", aliases=("gelu",))
class MLP(nn.Module):
    """MLP classique à deux couches — témoin de comparaison."""

    def __init__(
        self,
        *,
        d_model: int,
        hidden_dim: int | None = None,
        bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = hidden_dim if hidden_dim is not None else 4 * d_model
        self.up_proj = nn.Linear(d_model, hidden, bias=bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.down_proj(F.gelu(self.up_proj(x), approximate="tanh"))
        return self.dropout(out) if self.dropout is not None else out
