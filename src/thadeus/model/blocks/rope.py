"""Encodage positionnel rotatif (RoPE).

L'idée : plutôt que d'**ajouter** un vecteur de position à l'embedding, on fait
**tourner** chaque paire de dimensions de q et k d'un angle proportionnel à la
position. Le produit scalaire entre deux vecteurs ainsi tournés ne dépend plus
que de leur **écart** de position, jamais de leur position absolue.

Conséquence pratique qui compte pour nous : le modèle apprend des relations
relatives, ce qui l'aide à généraliser à des longueurs de contexte qu'il n'a pas
vues à l'entraînement. Avec un budget de calcul qui nous force à entraîner sur
des séquences courtes, c'est une propriété qu'on ne peut pas se permettre de
perdre.

Le cache de cosinus/sinus est calculé une fois pour toutes et partagé par toutes
les couches — le recalculer à chaque couche serait du gaspillage de bande
passante, la ressource la plus rare après le calcul.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["RotaryEmbedding", "apply_rope"]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Fait tourner les deux moitiés d'un vecteur de 90°.

    Convention « moitiés » (Llama, HF) plutôt qu'« entrelacée » (article
    original) : mathématiquement équivalentes, mais elles ne sont **pas
    interchangeables**. Mélanger les deux entre l'entraînement et l'inférence
    produit un modèle qui semble intact et génère du charabia.
    """
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applique la rotation à un tenseur ``(batch, seq, heads, head_dim)``.

    **Attention à la disposition** : on travaille en ``(B, S, H, D)``, c'est-à-dire
    *avant* la transposition qui prépare l'attention — et non en ``(B, H, S, D)``
    comme la plupart des implémentations.

    Ce n'est pas un détail de style. Mesuré sur M5 Pro : une opération élément par
    élément sur le tenseur transposé (donc non contigu) est **2× plus lente**.
    Faire tourner RoPE avant la transposition la rend gratuite en comparaison, et
    évite d'insérer un ``.contiguous()`` qui coûterait une copie complète.
    """
    return x * cos + _rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    """Cache des cosinus/sinus, étendu à la demande.

    Args:
        head_dim: dimension d'une tête. Doit être paire — RoPE agit sur des
            paires de dimensions.
        base: base des fréquences. Plus elle est grande, plus les fréquences
            basses sont lentes, et plus le modèle peut extrapoler loin. 10 000
            est la valeur historique ; les modèles à contexte long montent à
            500 000.
    """

    def __init__(self, head_dim: int, *, base: float = 10_000.0, max_seq_len: int = 4096) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim doit être pair pour RoPE, reçu {head_dim}")
        self.head_dim = head_dim
        self.base = base
        # `persistent=False` : ces tampons se recalculent à partir des
        # hyperparamètres, les stocker dans les checkpoints serait du poids mort
        # — et surtout figerait la longueur de contexte du modèle sauvegardé.
        self.register_buffer("cos", torch.empty(0), persistent=False)
        self.register_buffer("sin", torch.empty(0), persistent=False)
        self._build(max_seq_len, torch.device("cpu"), torch.float32)

    def _build(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        exponents = torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (self.base ** (exponents / self.head_dim))
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        # Chaque angle sert aux deux moitiés du vecteur : on duplique.
        emb = torch.cat((angles, angles), dim=-1)
        # Forme (1, seq, 1, head_dim) : diffusion sur la disposition (B, S, H, D),
        # celle où les tenseurs sont encore contigus (voir `apply_rope`).
        self.cos = emb.cos().to(dtype)[None, :, None, :]
        self.sin = emb.sin().to(dtype)[None, :, None, :]

    def forward(self, seq_len: int, *, device: torch.device, dtype: torch.dtype):
        """Retourne ``(cos, sin)`` tronqués à ``seq_len``.

        Le cache s'étend automatiquement si l'on dépasse la longueur prévue :
        une évaluation à contexte plus long que l'entraînement ne doit pas
        planter.
        """
        if self.cos.shape[1] < seq_len or self.cos.device != device or self.cos.dtype != dtype:
            self._build(max(seq_len, self.cos.shape[1]), device, dtype)
        return self.cos[:, :seq_len], self.sin[:, :seq_len]

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, base={self.base}, cached={self.cos.shape[1]}"
