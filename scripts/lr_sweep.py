#!/usr/bin/env python3
"""Balaye le taux d'apprentissage à plusieurs largeurs — le test de muP.

    python scripts/lr_sweep.py --widths 128 256 512 --optim adamw
    python scripts/lr_sweep.py --widths 128 256 512 --optim adamw --mup
    python scripts/lr_sweep.py --widths 256 --optim adamw --optim muon

**Ce que ça vérifie.** En paramétrisation standard, le taux d'apprentissage
optimal **dérive** quand le modèle s'élargit : ce qui marche à 128 diverge à 512.
muP prétend supprimer cette dérive. On ne le croit pas sur parole — on entraîne
plusieurs largeurs à plusieurs taux, et on regarde si l'optimum bouge.

**Pourquoi c'est le test le plus rentable du projet.** Avec 15,24 crédits, on ne
peut pas chercher le taux d'apprentissage sur le modèle de 188 M. muP permet de
le chercher ici, sur des modèles de quelques millions de paramètres, gratuitement
et en quelques minutes. Si le transfert ne marche pas, il vaut infiniment mieux
le découvrir sur le Mac que sur des crédits.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from thadeus.core.device import hot_path_dtype, resolve_device  # noqa: E402
from thadeus.core.logs import setup_logging  # noqa: E402
from thadeus.core.seeding import derive_seed, seed_everything  # noqa: E402
from thadeus.model import ModelConfig, Thadeus  # noqa: E402
from thadeus.optim.build import build_optimizer  # noqa: E402
from thadeus.optim.mup import MupConfig, apply_mup, logit_scale, lr_scales  # noqa: E402
from thadeus.train.config import OptimSpec  # noqa: E402

VOCAB = 512
SEQ = 128

# Chaîne de Markov d'ordre 1 : une permutation fixe du vocabulaire, suivie avec
# probabilité `_SIGNAL`. Tirage figé pour que toutes les configurations comparées
# voient exactement la même tâche.
_SIGNAL = 0.8
_TRANSITION = np.random.default_rng(20260803).permutation(VOCAB)


def synthetic_batch(step: int, batch: int, device: torch.device) -> tuple:
    """Lot d'une tâche **réellement apprenable**, reproductible à partir du pas.

    Premier essai raté, et l'erreur mérite d'être expliquée : la tâche initiale
    appliquait une récurrence chaotique qui produisait des tokens uniformément
    distribués. La perte restait donc à `ln(V)` pour **tous** les taux et toutes
    les largeurs — et le balayage concluait fièrement « l'optimum transfère »,
    alors qu'il n'y avait simplement rien à apprendre. Trois lignes identiques
    ne démontrent pas un transfert.

    La tâche actuelle est une chaîne de Markov : le token suivant est
    `TRANSITION[courant]` avec probabilité 0,8, aléatoire sinon. La perte
    atteignable est donc bornée par en dessous (~0,2·ln(V) plus l'entropie du
    bruit), et un modèle de quelques millions de paramètres l'approche en une
    centaine de pas. C'est ce qui rend le balayage discriminant.
    """
    rng = np.random.default_rng(derive_seed(1337, "sweep", step))
    tokens = np.empty((batch, SEQ + 1), dtype=np.int64)
    tokens[:, 0] = rng.integers(0, VOCAB, size=batch)
    for t in range(1, SEQ + 1):
        suivant = _TRANSITION[tokens[:, t - 1]]
        bruit = rng.integers(0, VOCAB, size=batch)
        tokens[:, t] = np.where(rng.random(batch) < _SIGNAL, suivant, bruit)
    x = torch.from_numpy(tokens).to(device)
    return x[:, :-1], x[:, 1:]


def train_once(
    *,
    width: int,
    lr: float,
    optim_name: str,
    mup: MupConfig,
    steps: int,
    device: torch.device,
    adamw_lr: float = 1e-3,
) -> float:
    """Entraîne un modèle jouet et rend la perte moyenne des derniers pas.

    La moyenne sur une fenêtre finale plutôt que la dernière valeur : une perte
    unique est bruitée, et on compare des écarts faibles entre taux voisins.
    """
    seed_everything(1337)
    heads = max(1, width // 64)
    cfg = ModelConfig(
        vocab_size=VOCAB,
        d_model=width,
        n_layers=4,
        max_seq_len=SEQ,
        attention={
            "name": "gqa",
            "n_heads": heads,
            "n_kv_heads": max(1, heads // 2),
            "head_dim": 64,
        },
    )
    cfg = cfg.model_copy(update={"logit_scale": logit_scale(cfg, mup)})

    model = Thadeus(cfg).to(device)
    apply_mup(model, cfg, mup)

    # Muon ne pilote que les matrices cachées ; embeddings et vecteurs restent à
    # AdamW. Leur donner le taux qu'on balaye pour Muon (~30x le leur) fausserait
    # la comparaison en handicapant les runs Muon sur des paramètres qui ne sont
    # pas le sujet. On les fixe donc à leur propre optimum, mesuré séparément.
    spec = (
        OptimSpec(name="muon", lr=adamw_lr, muon_lr=lr, weight_decay=0.0)
        if optim_name == "muon"
        else OptimSpec(name="adamw", lr=lr, weight_decay=0.0)
    )
    optimizer = build_optimizer(model, spec=spec, lr_scales=lr_scales(cfg, mup))
    dtype = hot_path_dtype(device)

    model.train()
    recent: list[float] = []
    for step in range(steps):
        x, y = synthetic_batch(step, 8, device)
        with torch.autocast(device.type, dtype=dtype):
            _, loss = model(x, targets=y)
        value = loss.item()
        if not math.isfinite(value):
            return float("nan")  # divergence : on ne pollue pas la moyenne
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step >= steps - 20:
            recent.append(value)
    return sum(recent) / len(recent)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--widths", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--lrs", type=float, nargs="+", default=None)
    parser.add_argument("--optim", action="append", default=[], choices=["adamw", "muon"])
    parser.add_argument("--mup", action="store_true", help="active la paramétrisation muP")
    parser.add_argument("--base-width", type=int, default=128)
    parser.add_argument(
        "--adamw-lr",
        type=float,
        default=1e-3,
        help="taux d'AdamW pour embeddings et vecteurs quand Muon pilote les matrices cachées",
    )
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    setup_logging("ERROR")  # le balayage parle par son tableau, pas par ses journaux
    device = resolve_device("auto")
    optims = args.optim or ["adamw"]
    mup = MupConfig(enabled=args.mup, base_d_model=args.base_width)

    resultats: dict[str, dict[int, dict[float, float]]] = {}
    for optim_name in optims:
        lrs = args.lrs or ([3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
        resultats[optim_name] = {}
        titre = f"{optim_name}{' + muP' if args.mup else ''}"
        print(f"\n=== {titre} ===")
        print("largeur".ljust(9) + "".join(f"{lr:>10.0e}" for lr in lrs) + "   optimum")

        for width in args.widths:
            pertes: dict[float, float] = {}
            ligne = f"{width:<9}"
            for lr in lrs:
                debut = time.perf_counter()
                perte = train_once(
                    width=width,
                    lr=lr,
                    optim_name=optim_name,
                    mup=mup,
                    steps=args.steps,
                    device=device,
                    adamw_lr=args.adamw_lr,
                )
                pertes[lr] = perte
                ligne += f"{'div.':>10}" if math.isnan(perte) else f"{perte:>10.3f}"
                del debut
            valides = {lr: v for lr, v in pertes.items() if not math.isnan(v)}
            best = min(valides, key=valides.get) if valides else None
            ligne += f"   {best:.0e}" if best else "   —"
            print(ligne, flush=True)
            resultats[optim_name][width] = pertes

        # Le verdict : l'optimum est-il le même à toutes les largeurs ?
        optima = {
            w: min(
                ((lr, v) for lr, v in p.items() if not math.isnan(v)),
                key=lambda kv: kv[1],
                default=(None, None),
            )[0]
            for w, p in resultats[optim_name].items()
        }
        distincts = {o for o in optima.values() if o is not None}
        verdict = "TRANSFÈRE" if len(distincts) == 1 else "DÉRIVE"
        print(f"  -> optimum par largeur : {optima}  ==> {verdict}")

    if args.out:
        Path(args.out).write_text(json.dumps(resultats, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
