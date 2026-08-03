#!/usr/bin/env python3
"""Construit un corpus.

    python scripts/build_corpus.py --config data/smoke.toml
    python scripts/build_corpus.py --config data/fr_first.toml
    python scripts/build_corpus.py --config data/fr_first.toml --set total_tokens=500_000_000

Après exécution, regarder `report.json` dans l'artefact : composition demandée
contre composition obtenue, et taux de rejet par filtre. C'est le seul endroit
où l'on voit ce que le corpus contient réellement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thadeus.core.config import load_config  # noqa: E402
from thadeus.core.env import load_dotenv  # noqa: E402
from thadeus.core.logs import setup_logging  # noqa: E402
from thadeus.data.pipeline import build_corpus, peek  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="data/smoke.toml", help="config, relative à configs/")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="CHEMIN=VALEUR",
        help="surcharge ponctuelle (répétable)",
    )
    parser.add_argument("--force", action="store_true", help="reconstruit même si présent")
    parser.add_argument("--peek", type=int, default=2, help="documents à afficher, pour vérifier")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    # HF_TOKEN et clés Lightning. Sans cet appel, les datasets sous licence
    # échouent avec « you must be authenticated » alors que le jeton est bien
    # dans le .env — le piège qui a fait échouer les sources de code.
    load_dotenv()
    cfg = load_config(args.config, overrides=args.overrides)
    artifact = build_corpus(cfg, force=args.force)

    report = json.loads((artifact.path / "report.json").read_text(encoding="utf-8"))
    print(json.dumps(report["mixture"], indent=2, ensure_ascii=False))

    # Regarder le corpus avec ses yeux attrape ce qu'aucune statistique ne montre.
    for doc in peek(artifact, args.peek):
        print(f"\n--- {doc.source} · {doc.lang} · {doc.n_words} mots · {doc.id}")
        print(doc.text[:400].replace("\n", " ⏎ "))

    print(f"\nArtefact : {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
