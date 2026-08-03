#!/usr/bin/env python3
"""Lance une campagne de mesure des noyaux.

    python scripts/bench.py --config bench/quick.toml
    python scripts/bench.py --config bench/base.toml --set device=cpu
    python scripts/bench.py --config bench/base.toml --force

Un script = un étage, exécutable seul. Le script ne contient aucune logique :
il traduit une ligne de commande en appel de bibliothèque, et rien d'autre.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Permet d'exécuter le script sans avoir installé le paquet.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thadeus.bench.suite import run_suite, summarize  # noqa: E402
from thadeus.core.config import load_config  # noqa: E402
from thadeus.core.logs import setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="bench/base.toml", help="config, relative à configs/")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="CHEMIN=VALEUR",
        help="surcharge ponctuelle, ex. --set device=cpu (répétable)",
    )
    parser.add_argument("--force", action="store_true", help="rejoue même si l'artefact existe")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config, overrides=args.overrides)
    artifact = run_suite(cfg, force=args.force)

    rows = json.loads((artifact.path / "results.json").read_text(encoding="utf-8"))
    print(json.dumps(summarize(rows), indent=2, ensure_ascii=False))
    print(f"\nArtefact : {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
