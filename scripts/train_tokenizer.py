#!/usr/bin/env python3
"""Entraîne un tokenizer sur notre corpus.

python scripts/train_tokenizer.py --config tokenizer/bpe32k.toml
python scripts/train_tokenizer.py --config tokenizer/bpe32k.toml --set corpus_label=smoke
python scripts/train_tokenizer.py --config tokenizer/bpe32k_gpt2pattern.toml   # témoin
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
from thadeus.tokenizer.codec import Codec  # noqa: E402
from thadeus.tokenizer.train import train_tokenizer  # noqa: E402

ECHANTILLONS = [
    "L'homme qu'il a rencontré aujourd'hui n'était pas celui qu'on attendait.",
    "En 1997, la production s'élevait à 1 234 567 unités.",
    "The quick brown fox jumps over the lazy dog.",
    "def calculer(x: int) -> int:\n    return sum(i**2 for i in range(x))",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="tokenizer/bpe32k.toml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="CLE=VAL")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    load_dotenv()  # HF_TOKEN : limites de débit et dépôts restreints

    cfg = load_config(args.config, overrides=args.overrides)
    artifact = train_tokenizer(cfg, force=args.force)

    print(json.dumps(json.loads((artifact.path / "stats.json").read_text()), indent=2))

    # Vérification à l'œil : c'est ici qu'on voit si les élisions sont bien
    # rattachées, et si l'aller-retour est exact.
    codec = Codec.load(artifact.path)
    print(f"\nVocabulaire : {codec.vocab_size}\n")
    for texte in ECHANTILLONS:
        ids = codec.encode(texte)
        pieces = [codec.decode([i]) for i in ids]
        print(f"  {texte[:60]!r}")
        print(f"    {len(ids):>3} tokens  {' · '.join(pieces[:18])}")
        assert codec.decode(ids) == texte, "aller-retour non exact"

    print(f"\nArtefact : {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
