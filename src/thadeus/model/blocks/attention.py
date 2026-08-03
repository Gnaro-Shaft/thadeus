"""Attention causale à requêtes groupées (GQA).

C'est ici que se joue le pari « architecture » de la Phase 6, et le banc de
Phase 0 a déjà dit où regarder : sur MPS, l'attention plafonne à **6,9 TFLOPS
contre 30 pour le produit matriciel dense**. C'est le goulot du modèle, pas les
matmuls.

**GQA** partage un même couple clé/valeur entre plusieurs têtes de requête.
Le gain principal n'est pas le calcul — c'est la taille du cache clé/valeur, qui
domine la mémoire en génération. Avec 12 têtes de requête pour 4 têtes clé/valeur,
le cache est divisé par 3.

Ici, GQA joue surtout le rôle de **témoin** : c'est la référence contre laquelle
MLA (attention à cache latent) sera mesurée en Phase 6. Une variante ne sera
adoptée que si elle bat ce témoin à budget de FLOPs constant.

**QK-norm** normalise requêtes et clés avant le produit scalaire. Cela borne
l'amplitude des logits d'attention, ce qui permet d'entraîner à taux
d'apprentissage plus élevé sans divergence — exactement ce qu'on veut quand
chaque heure de calcul compte et qu'un run divergé est une nuit perdue.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from thadeus.model.blocks import ATTENTIONS
from thadeus.model.blocks.norm import RMSNorm
from thadeus.model.blocks.rope import apply_rope

__all__ = ["GroupedQueryAttention"]

# `enable_gqa` évite de matérialiser les têtes clé/valeur répétées. Le `or ""`
# n'est pas de la superstition : `__doc__` vaut None quand Python tourne avec
# -OO, et l'import du modèle échouerait alors sur un AttributeError obscur.
_SDPA_HAS_GQA = "enable_gqa" in (F.scaled_dot_product_attention.__doc__ or "")


@ATTENTIONS.register("gqa", aliases=("mha",))
class GroupedQueryAttention(nn.Module):
    """Attention causale multi-têtes, avec regroupement clé/valeur optionnel.

    Args:
        d_model: dimension du modèle.
        n_heads: nombre de têtes de requête.
        n_kv_heads: nombre de têtes clé/valeur. Égal à ``n_heads`` donne de
            l'attention multi-têtes classique ; à 1, de la multi-query.
        head_dim: dimension par tête. Par défaut ``d_model // n_heads``.
        qk_norm: normalise q et k avant le produit scalaire.
        bias: biais sur les projections. Faux par défaut — les modèles récents
            les retirent, ils coûtent des paramètres sans rien apporter.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        head_dim: int | None = None,
        qk_norm: bool = True,
        bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = head_dim if head_dim is not None else d_model // n_heads
        self.dropout = dropout

        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) doit être un multiple de "
                f"n_kv_heads ({self.n_kv_heads}) — chaque groupe partage un couple clé/valeur"
            )

        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(d_model, q_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, kv_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=bias)
        self.o_proj = nn.Linear(q_dim, d_model, bias=bias)

        self.q_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else None

    @property
    def n_groups(self) -> int:
        """Têtes de requête par tête clé/valeur — le facteur de réduction du cache."""
        return self.n_heads // self.n_kv_heads

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq, _ = x.shape

        # Disposition (B, S, H, D) : on garde les tenseurs **contigus** tant qu'on
        # leur applique des opérations élément par élément. Mesuré sur M5 Pro :
        # RMSNorm est 2,54× plus lente et RoPE 2,00× plus lente sur un tenseur
        # transposé. La transposition n'intervient donc qu'au dernier moment,
        # juste avant SDPA — un noyau fusionné qui gère les strides lui-même.
        q = self.q_proj(x).view(batch, seq, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq, self.n_kv_heads, self.head_dim)

        # QK-norm avant RoPE : on borne l'amplitude, puis on encode la position.
        # L'ordre inverse annulerait la normalisation, la rotation modifiant la norme
        # de chaque paire de dimensions.
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)

        q = apply_rope(q, cos, sin).transpose(1, 2)
        k = apply_rope(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)

        if _SDPA_HAS_GQA:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                enable_gqa=self.n_groups > 1,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            # Repli : on matérialise les têtes clé/valeur répétées.
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )

        out = out.transpose(1, 2).reshape(batch, seq, self.n_heads * self.head_dim)
        return self.o_proj(out)

    def extra_repr(self) -> str:
        return (
            f"n_heads={self.n_heads}, n_kv_heads={self.n_kv_heads}, "
            f"head_dim={self.head_dim}, groups={self.n_groups}"
        )
