#!/usr/bin/env python3
"""Évalue un modèle : perplexité par source, sondes grammaticales, génération.

    python scripts/evaluate.py --config eval/default.toml
    python scripts/evaluate.py --set checkpoint=/chemin/best.pt

Sans checkpoint, évalue un modèle **non entraîné** — ce qui donne la ligne de
base : perplexité ≈ taille du vocabulaire, sondes ≈ 50 %. Tout run ultérieur se
compare à ces chiffres.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thadeus.core.config import load_config  # noqa: E402
from thadeus.core.env import load_dotenv  # noqa: E402
from thadeus.core.logs import setup_logging  # noqa: E402
from thadeus.eval.suite import evaluate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="eval/default.toml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="CLE=VAL")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    load_dotenv()
    artifact = evaluate(load_config(args.config, overrides=args.overrides), force=args.force)
    print(f"\nArtefact : {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
