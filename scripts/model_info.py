#!/usr/bin/env python3
"""Décrit un modèle : paramètres, budget de calcul, et débit réel mesuré.

    python scripts/model_info.py --config model/small.toml
    python scripts/model_info.py --config model/small.toml --benchmark
    python scripts/model_info.py --config model/tiny.toml --overfit

Trois usages distincts :

- par défaut, le **dimensionnement** — calculé par formule, sans rien instancier ;
- ``--benchmark``, le **débit réel** d'un pas d'entraînement, et le MFU qui en
  découle. C'est le seul chiffre qui dit si le modèle exploite la machine ;
- ``--overfit``, la **preuve que la chaîne apprend** : un modèle jouet doit
  mémoriser un lot minuscule et faire tomber la perte à ~0. Tant que ce test
  échoue, tout résultat d'entraînement est ininterprétable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from thadeus.bench.flops import MEASURED_EFFECTIVE_TFLOPS, mfu  # noqa: E402
from thadeus.core.config import load_config  # noqa: E402
from thadeus.core.device import describe, hot_path_dtype, resolve_device, synchronize  # noqa: E402
from thadeus.core.logs import setup_logging  # noqa: E402
from thadeus.core.seeding import seed_everything  # noqa: E402
from thadeus.model import ModelConfig, Thadeus, estimate  # noqa: E402
from thadeus.model.init import parameter_groups  # noqa: E402


def benchmark(model: Thadeus, device: torch.device, *, batch: int, iters: int = 8) -> dict:
    """Mesure un pas d'entraînement complet : avant, arrière, mise à jour."""
    seq = model.cfg.max_seq_len
    optimizer = torch.optim.AdamW(parameter_groups(model), lr=1e-4)
    tokens = torch.randint(0, model.cfg.vocab_size, (batch, seq + 1), device=device)
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    dtype = hot_path_dtype(device)

    def step() -> None:
        with torch.autocast(device_type=device.type, dtype=dtype):
            _, loss = model(inputs, targets=targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    for _ in range(3):  # chauffe : compilation de shaders et allocations
        step()
    synchronize(device)

    start = time.perf_counter()
    for _ in range(iters):
        step()
    synchronize(device)
    seconds = (time.perf_counter() - start) / iters

    sizing = estimate(model.cfg)
    tokens_per_s = batch * seq / seconds
    return {
        "batch": batch,
        "seq_len": seq,
        "seconds_per_step": seconds,
        "tokens_per_second": tokens_per_s,
        "flops_per_token": sizing.flops_per_token(seq),
        # Crête bf16 mesurée en Phase 0 sur ce Mac ; sur H100 le banc doit être
        # rejoué pour obtenir la vraie référence locale.
        "mfu_vs_measured_peak": mfu(tokens_per_s, sizing.flops_per_token(seq), 30.0e12),
    }


def overfit(model: Thadeus, device: torch.device, *, steps: int = 200) -> dict:
    """Fait mémoriser un lot minuscule — le test qui prouve que la chaîne apprend.

    Si la perte ne s'effondre pas ici, le problème n'est ni les données ni les
    hyperparamètres : c'est le modèle, l'optimiseur ou le branchement des cibles.
    Aucun run long ne doit démarrer avant que ce test passe.
    """
    seq = min(64, model.cfg.max_seq_len)
    tokens = torch.randint(0, model.cfg.vocab_size, (2, seq + 1), device=device)
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    optimizer = torch.optim.AdamW(parameter_groups(model, weight_decay=0.0), lr=3e-3)

    losses = []
    for _ in range(steps):
        _, loss = model(inputs, targets=targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())

    return {"loss_initiale": losses[0], "loss_finale": losses[-1], "steps": steps}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="model/small.toml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="CLE=VAL")
    parser.add_argument("--benchmark", action="store_true", help="mesure le débit réel")
    parser.add_argument("--overfit", action="store_true", help="prouve que la chaîne apprend")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="désactive torch.compile — mesuré à 3x plus lent sur MPS, pour comparaison seulement",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    setup_logging("WARNING")
    seed_everything(1337)

    cfg = ModelConfig(**load_config(args.config, overrides=args.overrides))
    sizing = estimate(cfg)

    print(f"=== {args.config} ===")
    print(json.dumps(sizing.to_dict(), indent=2))
    print(f"\n{sizing}")
    print(f"  FLOPs/token (seq={cfg.max_seq_len}) : {sizing.flops_per_token(cfg.max_seq_len):.3e}")
    for budget, label in [(1.7e9, "1,7 Md tokens")]:
        # Débit Mac **mesuré** (compilé), pas une fraction supposée de la crête.
        mac = sizing.hours_for(
            budget,
            seq_len=cfg.max_seq_len,
            effective_tflops=MEASURED_EFFECTIVE_TFLOPS["m5_pro_20c"],
        )
        print(f"  {label} : {mac:.1f} h sur Mac (débit mesuré, compilé)")
        print("    H100 : à mesurer — rejouer ce banc avant d'engager des crédits")

    if not (args.benchmark or args.overfit):
        return 0

    device = resolve_device(args.device)
    print(f"\nMachine : {describe(device)}")
    model = Thadeus(cfg).to(device)
    actual = model.n_parameters()
    if not args.no_compile:
        # Facteur 3,00 mesuré sur MPS (Phase 3). Compilé par défaut : mesurer en
        # eager donnerait un débit qui ne correspond à aucun run réel.
        model = torch.compile(model)
    print(f"Paramètres réels : {actual / 1e6:.2f} M (estimation : {sizing.total / 1e6:.2f} M)")

    if args.overfit:
        print("\n=== surapprentissage d'un lot minuscule ===")
        print(json.dumps(overfit(model, device), indent=2))

    if args.benchmark:
        print("\n=== débit d'un pas d'entraînement ===")
        result = benchmark(model, device, batch=args.batch)
        print(json.dumps(result, indent=2))
        tps = result["tokens_per_second"]
        print(f"\n  {tps:.0f} tokens/s · MFU {100 * result['mfu_vs_measured_peak']:.1f} %")
        print(f"  1,7 Md tokens : {1.7e9 / tps / 3600:.1f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
