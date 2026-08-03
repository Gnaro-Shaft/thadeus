"""Schéma de configuration du modèle.

Le modèle ne lit **que** cet objet. Toute constante d'architecture qui
apparaîtrait ailleurs serait un paramètre qu'on ne pourrait plus faire varier
dans un A/B — donc une comparaison qu'on ne pourrait plus faire.

Les valeurs par défaut décrivent le modèle visé pour le premier vrai run : ~85 M
paramètres, vocabulaire de 32 k (hypothèse de travail arrêtée le 2026-08-03).
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from thadeus.core.config import Schema

__all__ = ["ModelConfig"]


class ModelConfig(Schema):
    """Architecture complète d'un modèle Thadeus.

    Args:
        vocab_size: doit correspondre **exactement** au tokenizer utilisé.
            Un écart silencieux ici produit des indices hors bornes en
            entraînement, ou pire, un modèle qui n'atteint jamais certains tokens.
        tie_embeddings: partage la table entre entrée et sortie. À notre échelle,
            c'est ~25 % des paramètres du modèle.
        norm, attention, ffn: specs de composants, résolues par les registres.
            Une chaîne seule (``"rmsnorm"``) ou un mapping avec la clé ``name``
            et des paramètres.
    """

    vocab_size: int = 32_000
    d_model: int = 640
    n_layers: int = 12
    max_seq_len: int = 1024
    rope_base: float = 10_000.0
    tie_embeddings: bool = True

    norm: str | dict[str, Any] = "rmsnorm"
    attention: str | dict[str, Any] = Field(
        default_factory=lambda: {"name": "gqa", "n_heads": 10, "n_kv_heads": 2, "head_dim": 64}
    )
    ffn: str | dict[str, Any] = Field(default_factory=lambda: {"name": "swiglu"})

    init_std: float = 0.02
    scale_residual_init: bool = True

    @model_validator(mode="after")
    def _check_coherence(self) -> ModelConfig:
        attention = self.attention if isinstance(self.attention, dict) else {}
        n_heads = attention.get("n_heads")
        head_dim = attention.get("head_dim")

        if n_heads and head_dim and n_heads * head_dim != self.d_model:
            # Autorisé (rien ne l'interdit mathématiquement, la projection de
            # sortie recolle), mais presque toujours une faute de frappe.
            raise ValueError(
                f"n_heads × head_dim = {n_heads * head_dim} ≠ d_model = {self.d_model}. "
                f"Si c'est voulu, retirer head_dim de la config pour le laisser déduire."
            )
        if n_heads and not head_dim and self.d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) doit être divisible par n_heads ({n_heads})"
            )
        return self
