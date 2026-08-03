"""Abstraction de l'accélérateur — le même code sur Mac et sur H100.

Thadeus tourne sur deux machines aux rôles distincts : le MacBook M5 Pro (MPS)
sert de laboratoire, le H100 (CUDA) d'usine. Une seule base de code couvre les
deux, et ce module est la frontière : partout ailleurs, on ne teste plus
``if backend == "mps"``.

Deux invariants mesurés le 2026-08-03 sur le M5 Pro et inscrits ici :

1. **bf16 partout sur le chemin chaud.** MPS passe de 29,5 à 7,4 TFLOPS en
   fp32 — un facteur 4 offert à personne. :func:`hot_path_dtype` retourne bf16
   sur les deux backends, et :func:`warn_if_fp32` sert de garde-fou.
2. **On est limité par le calcul, pas par la mémoire.** 51,8 Go alloués au GPU
   pour un modèle de 150 M paramètres qui en occupe 2 avec ses états
   d'optimiseur. Les fonctions mémoire ici servent à *surveiller*, pas à
   optimiser — l'optimisation mémoire est hors sujet dans ce projet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

__all__ = [
    "DeviceInfo",
    "Backend",
    "describe",
    "empty_cache",
    "hot_path_dtype",
    "peak_memory_gb",
    "reset_peak_memory",
    "resolve_device",
    "supports_bf16",
    "synchronize",
    "warn_if_fp32",
]

log = logging.getLogger(__name__)

Backend = Literal["cuda", "mps", "cpu"]

_GB = 1024**3


@dataclass(frozen=True)
class DeviceInfo:
    """Signalétique de l'accélérateur, telle qu'on la consigne dans les artefacts.

    Sert à comparer deux benchmarks : un résultat sans la machine qui l'a
    produit n'est pas comparable.
    """

    backend: Backend
    name: str
    torch_version: str
    bf16: bool
    total_memory_gb: float | None = None
    usable_memory_gb: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "name": self.name,
            "torch_version": self.torch_version,
            "bf16": self.bf16,
            "total_memory_gb": self.total_memory_gb,
            "usable_memory_gb": self.usable_memory_gb,
            **self.detail,
        }

    def __str__(self) -> str:
        mem = f", {self.usable_memory_gb:.1f} Go utilisables" if self.usable_memory_gb else ""
        return f"{self.name} [{self.backend}{mem}, bf16={'oui' if self.bf16 else 'non'}]"


def resolve_device(pref: str = "auto") -> torch.device:
    """Choisit l'accélérateur.

    ``"auto"`` prend CUDA s'il existe, sinon MPS, sinon CPU — c'est-à-dire le
    H100 sur Lightning AI et le GPU du Mac en local, sans rien changer à la
    config.
    """
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    log.warning("Aucun accélérateur détecté — repli sur le CPU (lent).")
    return torch.device("cpu")


def _backend(device: torch.device) -> Backend:
    kind = device.type
    if kind not in ("cuda", "mps", "cpu"):
        raise ValueError(f"backend non pris en charge : {kind!r}")
    return kind  # type: ignore[return-value]


def supports_bf16(device: torch.device) -> bool:
    """Le backend sait-il calculer en bfloat16 ?"""
    backend = _backend(device)
    if backend == "cuda":
        return torch.cuda.is_bf16_supported()
    if backend == "mps":
        # Disponible depuis macOS 14 ; on teste au lieu de supposer.
        try:
            torch.ones(2, dtype=torch.bfloat16, device=device) * 2
        except (RuntimeError, TypeError):
            return False
        return True
    return True  # le CPU émule, lentement mais correctement


def hot_path_dtype(device: torch.device) -> torch.dtype:
    """Le dtype des matmuls d'entraînement.

    Réponse unique et volontairement rigide : bfloat16 si disponible. Le fp32
    n'est acceptable que pour les accumulations sensibles (loss, normes,
    états d'optimiseur), jamais pour le corps du modèle.
    """
    return torch.bfloat16 if supports_bf16(device) else torch.float32


def warn_if_fp32(tensor: torch.Tensor, where: str) -> None:
    """Signale un fp32 sur un chemin qui devrait être en bf16.

    Garde-fou volontairement bavard : sur MPS, un fp32 égaré divise le débit
    par 4, et c'est le genre de régression qui passe inaperçue pendant des
    heures d'entraînement.
    """
    if tensor.dtype is torch.float32 and tensor.device.type in ("mps", "cuda"):
        log.warning(
            "fp32 détecté sur le chemin chaud (%s) — attendu bf16. "
            "Sur MPS c'est un facteur ~4 sur le débit.",
            where,
        )


def describe(device: torch.device) -> DeviceInfo:
    """Relève la signalétique complète de l'accélérateur."""
    backend = _backend(device)
    common = {"torch_version": torch.__version__, "bf16": supports_bf16(device)}

    if backend == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        total = props.total_memory / _GB
        return DeviceInfo(
            backend="cuda",
            name=props.name,
            total_memory_gb=total,
            usable_memory_gb=total,
            detail={
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
                "cuda_version": torch.version.cuda,
            },
            **common,
        )

    if backend == "mps":
        # `recommended_max_memory` est le plafond que Metal conseille de ne pas
        # dépasser ; c'est la vraie limite pratique, pas la RAM totale.
        usable = getattr(torch.mps, "recommended_max_memory", lambda: 0)() / _GB or None
        return DeviceInfo(
            backend="mps",
            name="Apple Silicon GPU (Metal)",
            total_memory_gb=torch.mps.driver_allocated_memory() / _GB if usable else None,
            usable_memory_gb=usable,
            detail={"unified_memory": True},
            **common,
        )

    return DeviceInfo(backend="cpu", name="CPU", **common)


def synchronize(device: torch.device) -> None:
    """Attend la fin des travaux GPU en vol.

    Indispensable avant toute mesure de temps : les deux backends sont
    asynchrones, et chronométrer sans synchroniser mesure la vitesse à laquelle
    on empile des ordres, pas celle à laquelle le GPU les exécute.
    """
    backend = _backend(device)
    if backend == "cuda":
        torch.cuda.synchronize(device)
    elif backend == "mps":
        torch.mps.synchronize()


def empty_cache(device: torch.device) -> None:
    """Rend au système la mémoire mise en cache par l'allocateur."""
    backend = _backend(device)
    if backend == "cuda":
        torch.cuda.empty_cache()
    elif backend == "mps":
        torch.mps.empty_cache()


def reset_peak_memory(device: torch.device) -> None:
    """Remet à zéro le compteur de pic mémoire (CUDA uniquement)."""
    if _backend(device) == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_gb(device: torch.device) -> float | None:
    """Pic d'occupation mémoire depuis la dernière remise à zéro.

    MPS n'expose pas de véritable compteur de pic : on retourne l'allocation
    courante du pilote, qui en est une approximation par excès.
    """
    backend = _backend(device)
    if backend == "cuda":
        return torch.cuda.max_memory_allocated(device) / _GB
    if backend == "mps":
        return torch.mps.driver_allocated_memory() / _GB
    return None
