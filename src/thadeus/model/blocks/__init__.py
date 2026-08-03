"""Briques interchangeables du modèle.

Chaque famille de composant a son registre. C'est ce qui rend possibles les A/B
d'architecture de la Phase 6 : passer de GQA à MLA doit être une ligne de config,
pas un refactoring — sinon on introduit des différences involontaires entre les
variantes qu'on prétend comparer.

    [model.attention]
    name = "gqa"          # ou "mla" en Phase 6
    n_kv_heads = 4
"""

from __future__ import annotations

from torch import nn

from thadeus.core.registry import Registry

NORMS: Registry[nn.Module] = Registry("norm")
ATTENTIONS: Registry[nn.Module] = Registry("attention")
FFNS: Registry[nn.Module] = Registry("ffn")

from thadeus.model.blocks import attention, ffn, norm, rope  # noqa: E402,F401

__all__ = ["ATTENTIONS", "FFNS", "NORMS", "attention", "ffn", "norm", "rope"]
