"""Dimensionnement — répondre « combien ça coûte ? » avant de construire.

Un modèle qu'on instancie pour compter ses paramètres, c'est trente secondes
d'allocation mémoire pour un nombre qu'une formule donne instantanément. À
l'échelle d'un balayage de configurations, la différence entre explorer dix
architectures et en explorer mille.

Ce module sert deux décisions ouvertes du projet :

- **La part de l'embedding.** À ``vocab = 32 k`` et ``d_model = 640``, la table
  pèse 20 M de paramètres. Sans partage entrée/sortie, ce serait 40 M — soit la
  moitié d'un modèle de 85 M consacrée à une table de correspondance.
- **Le budget de calcul.** Combien de tokens ce modèle peut-il voir dans le temps
  machine dont on dispose ? C'est la seule question qui décide de sa taille.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from thadeus.bench.flops import transformer_flops_per_token
from thadeus.model.blocks.ffn import swiglu_hidden_dim
from thadeus.model.config import ModelConfig

__all__ = ["Sizing", "estimate"]


@dataclass(frozen=True)
class Sizing:
    """Décomposition des paramètres et coût en calcul."""

    d_model: int
    n_layers: int
    vocab_size: int
    embedding: int
    attention: int
    ffn: int
    norms: int

    @property
    def non_embedding(self) -> int:
        """Le compte qui entre dans le calcul des FLOPs."""
        return self.attention + self.ffn + self.norms

    @property
    def total(self) -> int:
        return self.embedding + self.non_embedding

    @property
    def embedding_share(self) -> float:
        """Fraction des paramètres consacrée à la table d'embedding."""
        return self.embedding / self.total if self.total else 0.0

    def flops_per_token(self, seq_len: int) -> float:
        return transformer_flops_per_token(
            self.non_embedding,
            n_layers=self.n_layers,
            d_model=self.d_model,
            seq_len=seq_len,
        )

    def hours_for(self, n_tokens: float, *, seq_len: int, effective_tflops: float) -> float:
        """Heures de calcul pour voir ``n_tokens``, au débit effectif donné.

        ``effective_tflops`` est le débit **réel** en boucle d'entraînement, pas
        la crête matmul : compter 30-40 % de la crête. Utiliser la crête ici est
        l'erreur qui fait promettre trois jours pour un run qui en prend dix.
        """
        return (n_tokens * self.flops_per_token(seq_len)) / (effective_tflops * 1e12 * 3600)

    def to_dict(self) -> dict[str, Any]:
        return {
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "vocab_size": self.vocab_size,
            "total": self.total,
            "non_embedding": self.non_embedding,
            "embedding": self.embedding,
            "embedding_share": round(self.embedding_share, 4),
            "attention": self.attention,
            "ffn": self.ffn,
            "norms": self.norms,
        }

    def __str__(self) -> str:
        return (
            f"{self.total / 1e6:.1f} M paramètres "
            f"({self.non_embedding / 1e6:.1f} M hors embedding, "
            f"embedding {100 * self.embedding_share:.0f} %)"
        )


def estimate(cfg: ModelConfig) -> Sizing:
    """Compte les paramètres depuis la config, sans instancier le modèle."""
    d = cfg.d_model
    attention = cfg.attention if isinstance(cfg.attention, dict) else {"name": cfg.attention}
    ffn = cfg.ffn if isinstance(cfg.ffn, dict) else {"name": cfg.ffn}

    n_heads = attention.get("n_heads", 8)
    head_dim = attention.get("head_dim", d // n_heads)
    n_kv_heads = attention.get("n_kv_heads", n_heads)
    qk_norm = attention.get("qk_norm", True)
    bias = attention.get("bias", False)

    q_dim, kv_dim = n_heads * head_dim, n_kv_heads * head_dim
    per_layer_attn = d * q_dim + 2 * d * kv_dim + q_dim * d
    if bias:
        per_layer_attn += 2 * q_dim + 2 * kv_dim
    if qk_norm:
        per_layer_attn += 2 * head_dim

    if ffn.get("name", "swiglu") == "swiglu":
        hidden = ffn.get("hidden_dim") or swiglu_hidden_dim(d)
        per_layer_ffn = 3 * d * hidden
    else:
        hidden = ffn.get("hidden_dim") or 4 * d
        per_layer_ffn = 2 * d * hidden

    # Deux normalisations par couche, plus la finale.
    norms = (2 * cfg.n_layers + 1) * d

    embedding = cfg.vocab_size * d
    if not cfg.tie_embeddings:
        embedding *= 2

    return Sizing(
        d_model=d,
        n_layers=cfg.n_layers,
        vocab_size=cfg.vocab_size,
        embedding=embedding,
        attention=per_layer_attn * cfg.n_layers,
        ffn=per_layer_ffn * cfg.n_layers,
        norms=norms,
    )
