"""Étage 3 — le modèle.

Transformer moderne (RMSNorm, SwiGLU, RoPE, GQA, QK-norm) assemblé depuis la
config. Chaque brique est enregistrée sous un nom et substituable en une ligne
de TOML : c'est la condition matérielle des A/B d'architecture de la Phase 6.
"""

from thadeus.model.config import ModelConfig
from thadeus.model.sizing import Sizing, estimate
from thadeus.model.transformer import Thadeus, TransformerBlock

__all__ = ["ModelConfig", "Sizing", "Thadeus", "TransformerBlock", "estimate"]
