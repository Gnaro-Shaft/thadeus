"""Banc de mesure des noyaux — l'étage 0 de la chaîne.

Ce module existe pour une raison précise : **le même code doit tourner à
l'identique sur le Mac et sur le H100**. C'est ce qui permet de dimensionner le
run final sur des mesures plutôt que sur des estimations, et de vérifier après
coup que le H100 rend bien ce qu'on avait supposé.

Chaque benchmark est enregistré sous un nom et sélectionné par la config, comme
tout composant interchangeable du projet :

    [[benchmarks]]
    name = "matmul"
    sizes = [4096, 8192]
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import torch

from thadeus.core.device import describe, hot_path_dtype, synchronize
from thadeus.core.logs import get_logger
from thadeus.core.registry import Registry

__all__ = ["BENCHMARKS", "run_benchmark", "timed"]

log = get_logger(__name__)

BENCHMARKS: Registry[list[dict[str, Any]]] = Registry("benchmark")

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _dtype(name: str) -> torch.dtype:
    try:
        return _DTYPES[name]
    except KeyError:
        raise KeyError(f"dtype inconnu : {name!r}. Connus : {', '.join(_DTYPES)}") from None


def timed(
    fn: Callable[[], Any],
    device: torch.device,
    *,
    warmup: int = 3,
    iters: int = 10,
) -> float:
    """Chronomètre ``fn``, en secondes par itération.

    Les deux précautions qui font la différence entre une mesure et un chiffre
    inventé :

    - **Chauffe** — la première exécution paie la compilation des shaders et
      l'allocation ; l'inclure peut doubler le temps mesuré.
    - **Synchronisation** — MPS et CUDA sont asynchrones. Sans
      :func:`synchronize`, on chronomètre la vitesse à laquelle Python empile
      des ordres, pas celle à laquelle le GPU les exécute. C'est l'erreur qui
      produit des « 400 TFLOPS » sur du matériel qui en fait 30.
    """
    for _ in range(warmup):
        fn()
    synchronize(device)

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) / iters


@BENCHMARKS.register("matmul")
def bench_matmul(
    device: torch.device,
    *,
    sizes: Sequence[int] = (2048, 4096, 8192),
    dtypes: Sequence[str] = ("bfloat16", "float16", "float32"),
    warmup: int = 3,
    iters: int = 10,
) -> list[dict[str, Any]]:
    """Débit du produit matriciel dense — la mesure qui compte le plus.

    Un Transformer passe l'essentiel de son temps dans des matmuls : ce chiffre
    est le plafond de tout le reste. Le balayage en dtype est ce qui a révélé,
    le 2026-08-03, que MPS s'effondre en fp32 (facteur 4) — d'où la règle du
    bf16 sur tout le chemin chaud.
    """
    results = []
    for dtype_name in dtypes:
        dt = _dtype(dtype_name)
        for n in sizes:
            a = torch.randn(n, n, device=device, dtype=dt)
            b = torch.randn(n, n, device=device, dtype=dt)
            # Liaison par argument par défaut : la lambda capture les tenseurs
            # au moment de sa création, pas au moment de son appel. Sans cela,
            # le `del` de fin de boucle la laisserait pointer dans le vide.
            seconds = timed(lambda a=a, b=b: a @ b, device, warmup=warmup, iters=iters)
            tflops = 2 * n**3 / seconds / 1e12
            results.append(
                {
                    "benchmark": "matmul",
                    "dtype": dtype_name,
                    "n": n,
                    "seconds": seconds,
                    "tflops": tflops,
                }
            )
            log.info("matmul %-9s n=%-5d %7.2f TFLOPS", dtype_name, n, tflops)
            del a, b
    return results


@BENCHMARKS.register("bandwidth")
def bench_bandwidth(
    device: torch.device,
    *,
    megabytes: int = 256,
    dtypes: Sequence[str] = ("bfloat16", "float32"),
    warmup: int = 3,
    iters: int = 10,
) -> list[dict[str, Any]]:
    """Bande passante mémoire sur une opération élémentaire.

    Complète le matmul : les normalisations, activations et mises à jour
    d'optimiseur ne font presque aucun calcul et sont bornées par la mémoire.
    Quand le MFU d'un entraînement est bas sans raison apparente, c'est très
    souvent que ces opérations-là dominent.
    """
    results = []
    for dtype_name in dtypes:
        dt = _dtype(dtype_name)
        n = (megabytes * 1024 * 1024) // torch.empty(0, dtype=dt).element_size()
        x = torch.randn(n, device=device, dtype=dt)
        seconds = timed(lambda x=x: x * 1.0001, device, warmup=warmup, iters=iters)
        gb_s = 2 * x.numel() * x.element_size() / seconds / 1e9  # une lecture + une écriture
        results.append(
            {
                "benchmark": "bandwidth",
                "dtype": dtype_name,
                "megabytes": megabytes,
                "seconds": seconds,
                "gb_per_s": gb_s,
            }
        )
        log.info("bandwidth %-9s %7.1f Go/s", dtype_name, gb_s)
        del x
    return results


@BENCHMARKS.register("attention")
def bench_attention(
    device: torch.device,
    *,
    batch: int = 8,
    heads: int = 12,
    seq_lens: Sequence[int] = (1024, 2048, 4096),
    head_dim: int = 64,
    warmup: int = 3,
    iters: int = 10,
) -> list[dict[str, Any]]:
    """Débit de l'attention causale (SDPA).

    Mesuré séparément du matmul parce que son coût croît en **s²** : c'est lui
    qui décide de la longueur de contexte qu'on peut se permettre, et c'est le
    terme que les variantes d'attention du pari « architecture » (MLA, GQA)
    cherchent à réduire. Sans mesure de référence ici, impossible de dire si
    une variante gagne vraiment.
    """
    dt = hot_path_dtype(device)
    results = []
    for seq in seq_lens:
        shape = (batch, heads, seq, head_dim)
        q = torch.randn(shape, device=device, dtype=dt)
        k = torch.randn(shape, device=device, dtype=dt)
        v = torch.randn(shape, device=device, dtype=dt)

        def run(q=q, k=k, v=v) -> torch.Tensor:
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

        seconds = timed(run, device, warmup=warmup, iters=iters)
        # QKᵀ puis ·V : 2 matmuls de 2·b·h·s²·d FLOPs, divisés par 2 pour le masque causal.
        flops = 2 * (2 * batch * heads * seq**2 * head_dim) / 2
        results.append(
            {
                "benchmark": "attention",
                "dtype": str(dt).removeprefix("torch."),
                "batch": batch,
                "heads": heads,
                "seq_len": seq,
                "head_dim": head_dim,
                "seconds": seconds,
                "tflops": flops / seconds / 1e12,
            }
        )
        log.info("attention seq=%-5d %7.2f TFLOPS", seq, flops / seconds / 1e12)
        del q, k, v
    return results


def run_benchmark(
    spec: str | dict[str, Any],
    device: torch.device,
    *,
    defaults: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Exécute un benchmark décrit par une spec de config, et l'annote de la machine.

    L'annotation n'est pas décorative : un débit sans la machine qui l'a produit
    n'est comparable à rien, et tout l'intérêt de ce module est de comparer Mac
    et H100.
    """
    as_dict = spec if isinstance(spec, dict) else {"name": spec}
    merged: dict[str, Any] = {**(defaults or {}), **as_dict}
    info = describe(device)
    rows = BENCHMARKS.build(merged, device=device)
    for row in rows:
        row["backend"] = info.backend
        row["machine"] = info.name
    return rows
