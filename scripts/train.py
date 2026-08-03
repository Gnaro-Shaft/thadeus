#!/usr/bin/env python3
"""Entraîne un modèle.

    python scripts/train.py --config train/smoke.toml
    python scripts/train.py --config train/small.toml
    python scripts/train.py --config train/small.toml --set total_steps=5000
    python scripts/train.py --config train/small.toml --fresh   # ignore les checkpoints

La reprise est le comportement **par défaut** : relancer la même commande après
une interruption repart exactement où l'on s'était arrêté, mêmes lots compris.
C'est ce qui rend praticable un entraînement fractionné en plusieurs nuits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thadeus.core.config import load_config  # noqa: E402
from thadeus.core.env import load_dotenv  # noqa: E402
from thadeus.core.logs import setup_logging  # noqa: E402
from thadeus.train.loop import train  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="train/smoke.toml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="CLE=VAL")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="repart de zéro au lieu de reprendre le dernier checkpoint",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    load_dotenv()

    cfg = load_config(args.config, overrides=args.overrides)
    artifact = train(cfg, resume=not args.fresh)
    print(f"\nArtefact : {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
