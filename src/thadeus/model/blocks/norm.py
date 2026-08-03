"""Normalisations.

Une seule subtilité, mais elle est structurante : **la normalisation calcule en
fp32 même quand le modèle tourne en bf16**.

Cela paraît contredire la règle « bf16 sur tout le chemin chaud » établie en
Phase 0. Ce n'en est pas une exception arbitraire : la règle vise les **produits
matriciels**, où MPS perd un facteur 4 en fp32. Une normalisation ne fait aucun
produit matriciel — elle est bornée par la bande passante mémoire, où l'écart
fp32/bf16 n'est que d'un facteur 2 sur un volume déjà faible.

En échange, on évite le vrai danger : la somme des carrés en bf16 perd de la
précision dès que la dimension grandit (bf16 n'a que 8 bits de mantisse), ce qui
fabrique des gradients bruités. C'est le compromis que font tous les modèles
récents, et il vaut la peine d'être compris plutôt que recopié.
"""

from __future__ import annotations

import torch
from torch import nn

from thadeus.model.blocks import NORMS

__all__ = ["LayerNorm", "RMSNorm"]


@NORMS.register("rmsnorm", aliases=("rms",))
class RMSNorm(nn.Module):
    """Root Mean Square normalization.

    Par rapport à :class:`LayerNorm`, on retire le recentrage (soustraction de
    la moyenne) et le biais. On garde donc uniquement la remise à l'échelle.
    Moins d'opérations, moins de paramètres, et aucune perte de qualité observée
    — c'est pourquoi tous les modèles récents l'ont adoptée.
    """

    def __init__(self, dim: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * self.weight.float()).to(dtype)

    def extra_repr(self) -> str:
        return f"dim={self.weight.numel()}, eps={self.eps}"


@NORMS.register("layernorm", aliases=("ln",))
class LayerNorm(nn.Module):
    """LayerNorm classique — gardée comme témoin de comparaison.

    Présente pour que « RMSNorm vaut mieux » soit une mesure sur notre corpus et
    non une croyance recopiée d'un article.
    """

    def __init__(self, dim: int, *, eps: float = 1e-5, bias: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        out = nn.functional.layer_norm(
            x.float(),
            (self.weight.numel(),),
            self.weight.float(),
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return out.to(dtype)
