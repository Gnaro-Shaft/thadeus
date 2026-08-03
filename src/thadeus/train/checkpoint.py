"""Sauvegarde et reprise.

Le module le plus important de la Phase 4, parce que **l'entraînement de Thadeus
est fractionné par construction** : des nuits sur le Mac, des tranches de crédits
sur le H100. Une reprise qui perd silencieusement de l'information transforme
plusieurs jours de calcul en résultat ininterprétable.

Trois précautions, chacune contre un mode de défaillance précis :

1. **Écriture atomique.** On écrit dans un fichier temporaire puis on renomme.
   Un ``rename`` est atomique sur le système de fichiers : une coupure de
   courant pendant la sauvegarde laisse l'ancien checkpoint intact, jamais un
   fichier à moitié écrit qui échouera au chargement six heures plus tard.

2. **Poids démêlés de ``torch.compile``.** Un modèle compilé préfixe toutes ses
   clés par ``_orig_mod.``. Sauvegarder tel quel produit un checkpoint qu'on ne
   peut plus recharger dans un modèle non compilé — piège d'autant plus vicieux
   que la compilation est notre mode par défaut.

3. **Le pas est dans le checkpoint.** Le chargeur étant sans état (voir
   :mod:`thadeus.train.tokens`), restaurer le numéro de pas suffit à retrouver
   exactement la même séquence de lots. Rien d'autre à sauvegarder côté données.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from thadeus.core.logs import get_logger

__all__ = ["Checkpoint", "CheckpointManager", "unwrap"]

log = get_logger(__name__)

LATEST = "latest.pt"
BEST = "best.pt"


def unwrap(model: nn.Module) -> nn.Module:
    """Retrouve le module d'origine sous les enveloppes de ``torch.compile``."""
    return getattr(model, "_orig_mod", model)


@dataclass
class Checkpoint:
    """Contenu d'un point de sauvegarde."""

    step: int
    model: dict[str, Any]
    optimizer: dict[str, Any] | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "model": self.model,
            "optimizer": self.optimizer,
            "metrics": self.metrics,
            "config": self.config,
        }


class CheckpointManager:
    """Gère les points de sauvegarde d'un run.

    Args:
        directory: où écrire.
        keep_last: nombre de sauvegardes horodatées à conserver, en plus de
            ``latest.pt`` et ``best.pt``. Une seule ne suffit pas : si la
            dernière est corrompue ou correspond à une divergence, il faut
            pouvoir revenir en arrière.
        monitor: métrique surveillée pour ``best.pt``.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        keep_last: int = 2,
        monitor: str = "val_loss",
        lower_is_better: bool = True,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.monitor = monitor
        self.lower_is_better = lower_is_better
        self.best_value: float | None = None

    def _atomic_save(self, payload: dict[str, Any], path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)  # atomique : jamais de fichier à moitié écrit

    def save(
        self,
        *,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        metrics: dict[str, float] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Path:
        """Sauvegarde l'état courant et met à jour ``latest`` et, si besoin, ``best``."""
        metrics = metrics or {}
        checkpoint = Checkpoint(
            step=step,
            model=unwrap(model).state_dict(),
            optimizer=optimizer.state_dict() if optimizer is not None else None,
            metrics=metrics,
            config=config or {},
        )
        payload = checkpoint.to_payload()

        path = self.directory / f"step-{step:08d}.pt"
        self._atomic_save(payload, path)
        self._atomic_save(payload, self.directory / LATEST)

        value = metrics.get(self.monitor)
        if value is not None and self._is_better(value):
            self.best_value = value
            self._atomic_save(payload, self.directory / BEST)
            log.info("Nouveau meilleur %s = %.4f (pas %d)", self.monitor, value, step)

        self._prune()
        return path

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        return value < self.best_value if self.lower_is_better else value > self.best_value

    def _prune(self) -> None:
        saved = sorted(self.directory.glob("step-*.pt"))
        for path in saved[: max(0, len(saved) - self.keep_last)]:
            path.unlink(missing_ok=True)

    def load(self, path: str | Path | None = None) -> Checkpoint | None:
        """Charge un checkpoint, ou ``None`` s'il n'y en a pas.

        Retourner ``None`` plutôt que lever est délibéré : le script
        d'entraînement appelle systématiquement ``load`` au démarrage, et
        l'absence de checkpoint est le cas normal d'un premier run.
        """
        target = Path(path) if path is not None else self.directory / LATEST
        if not target.is_file():
            return None
        payload = torch.load(target, map_location="cpu", weights_only=False)
        log.info("Reprise depuis %s (pas %d)", target.name, payload["step"])
        return Checkpoint(**payload)

    def restore(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        path: str | Path | None = None,
    ) -> int:
        """Restaure modèle et optimiseur, et retourne le pas atteint.

        Le chargement est **strict** : une clé manquante ou en trop lève. Un
        modèle rechargé partiellement s'entraîne sans erreur visible et donne
        des résultats faux — c'est exactement ce qu'il ne faut pas laisser
        passer en silence.
        """
        checkpoint = self.load(path)
        if checkpoint is None:
            return 0
        unwrap(model).load_state_dict(checkpoint.model, strict=True)
        if optimizer is not None and checkpoint.optimizer is not None:
            optimizer.load_state_dict(checkpoint.optimizer)
        self.best_value = checkpoint.metrics.get(self.monitor, self.best_value)
        return checkpoint.step

    def write_summary(self, payload: dict[str, Any]) -> Path:
        target = self.directory / "summary.json"
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return target
