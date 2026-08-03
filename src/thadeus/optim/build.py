"""Assemblage de l'optimiseur.

Muon ne s'utilise **jamais seul** : il n'a de sens que sur les matrices de
couches cachées. Un entraînement réel combine donc deux optimiseurs sur des
familles de paramètres disjointes, et c'est cet assemblage que gère ce module.

Le résultat expose la même interface qu'un optimiseur PyTorch — ``step``,
``zero_grad``, ``param_groups``, ``state_dict`` — pour que la boucle
d'entraînement n'ait pas à savoir combien d'optimiseurs tournent réellement.

Chaque groupe porte un ``base_lr`` propre, et le planificateur agit
**multiplicativement** dessus. C'est nécessaire : Muon tolère des taux ~50 fois
plus élevés qu'AdamW, et leur appliquer un taux commun reviendrait à sous-régler
l'un ou faire diverger l'autre.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from thadeus.core.logs import get_logger
from thadeus.core.registry import Registry
from thadeus.model.init import parameter_groups
from thadeus.optim.muon import Muon

__all__ = ["OPTIMIZERS", "ChainedOptimizer", "build_optimizer"]

log = get_logger(__name__)

OPTIMIZERS: Registry = Registry("optimizer")


class ChainedOptimizer:
    """Plusieurs optimiseurs sur des paramètres disjoints, vus comme un seul.

    ``param_groups`` renvoie les groupes **des sous-optimiseurs eux-mêmes**, pas
    des copies : modifier ``group["lr"]`` depuis la boucle agit donc réellement
    sur l'optimiseur concerné.
    """

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        if not optimizers:
            raise ValueError("aucun optimiseur à chaîner")
        self.optimizers = optimizers

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [group for opt in self.optimizers for group in opt.param_groups]

    def step(self, closure=None) -> None:
        loss = closure() if closure is not None else None
        for opt in self.optimizers:
            opt.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return {"chained": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        saved = state["chained"]
        if len(saved) != len(self.optimizers):
            raise ValueError(
                f"checkpoint avec {len(saved)} optimiseurs, {len(self.optimizers)} attendus. "
                f"La configuration d'optimiseur a changé depuis la sauvegarde."
            )
        for opt, sub in zip(self.optimizers, saved, strict=True):
            opt.load_state_dict(sub)

    def __repr__(self) -> str:
        noms = " + ".join(type(o).__name__ for o in self.optimizers)
        return f"ChainedOptimizer({noms})"


@OPTIMIZERS.register("adamw")
def build_adamw(groups: list[dict[str, Any]], *, lr: float, betas, eps: float, **_: Any):
    """AdamW sur tous les groupes — la référence contre laquelle Muon se mesure."""
    for group in groups:
        group["base_lr"] = lr * group.get("lr_scale", 1.0)
        group["lr"] = group["base_lr"]
    return torch.optim.AdamW(groups, lr=lr, betas=tuple(betas), eps=eps)


@OPTIMIZERS.register("muon", aliases=("muon_adamw",))
def build_muon(
    groups: list[dict[str, Any]],
    *,
    lr: float,
    betas,
    eps: float,
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    muon_ns_steps: int = 5,
    **_: Any,
):
    """Muon sur les matrices cachées, AdamW sur le reste.

    Le partage est celui de :func:`~thadeus.model.init.classify_parameters` :
    orthogonaliser une table d'embedding n'aurait aucun sens, et les gains de
    normalisation sont des vecteurs.
    """
    hidden = [g for g in groups if g["kind"] == "hidden"]
    autres = [g for g in groups if g["kind"] != "hidden"]

    optimizers: list[torch.optim.Optimizer] = []

    if hidden:
        for group in hidden:
            group["base_lr"] = muon_lr * group.get("lr_scale", 1.0)
            group["lr"] = group["base_lr"]
        optimizers.append(
            Muon(
                hidden,
                lr=muon_lr,
                momentum=muon_momentum,
                ns_steps=muon_ns_steps,
                weight_decay=hidden[0].get("weight_decay", 0.0),
            )
        )
    if autres:
        for group in autres:
            group["base_lr"] = lr * group.get("lr_scale", 1.0)
            group["lr"] = group["base_lr"]
        optimizers.append(torch.optim.AdamW(autres, lr=lr, betas=tuple(betas), eps=eps))

    n_hidden = sum(len(g["params"]) for g in hidden)
    n_autres = sum(len(g["params"]) for g in autres)
    log.info("Muon sur %d matrices cachées · AdamW sur %d autres tenseurs", n_hidden, n_autres)
    return ChainedOptimizer(optimizers)


def build_optimizer(
    model: nn.Module,
    *,
    spec: Any,
    lr_scales: dict[str, float] | None = None,
):
    """Construit l'optimiseur depuis la config du run.

    Args:
        spec: un :class:`~thadeus.train.config.OptimSpec`.
        lr_scales: multiplicateurs par famille, fournis par muP.
    """
    groups = parameter_groups(model, weight_decay=spec.weight_decay, lr_scales=lr_scales)
    return OPTIMIZERS.build(
        {"name": spec.name},
        groups=groups,
        lr=spec.lr,
        betas=spec.betas,
        eps=spec.eps,
        muon_lr=getattr(spec, "muon_lr", 0.02),
        muon_momentum=getattr(spec, "muon_momentum", 0.95),
        muon_ns_steps=getattr(spec, "muon_ns_steps", 5),
    )
