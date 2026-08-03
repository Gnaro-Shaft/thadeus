#!/usr/bin/env python3
"""Compare notre tokenizer aux références publiques, sur *notre* corpus.

    python scripts/compare_tokenizers.py --corpus-label smoke
    python scripts/compare_tokenizers.py --ours bpe32k --ours bpe32k_gpt2pattern

Le seul chiffre qui nous concerne est mesuré sur les textes que Thadeus verra
réellement. Les fertilités publiées le sont sur des corpus généralistes anglais
et ne disent rien de notre cas.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thadeus.core.artifacts import ARTIFACT_ROOT  # noqa: E402
from thadeus.core.env import load_dotenv  # noqa: E402
from thadeus.core.logs import get_logger, setup_logging  # noqa: E402
from thadeus.data.shard import iter_documents  # noqa: E402
from thadeus.tokenizer.codec import Codec  # noqa: E402
from thadeus.tokenizer.metrics import (  # noqa: E402
    REFERENCE_TOKENIZERS,
    compare,
    format_comparison,
    load_reference,
)
from thadeus.tokenizer.train import find_corpus  # noqa: E402

log = get_logger(__name__)


def find_tokenizer(label: str) -> Path:
    candidates = [
        p
        for p in sorted((ARTIFACT_ROOT / "tokenizer").glob(f"{label}-*"))
        if (p / "meta.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"aucun tokenizer achevé nommé {label!r}")
    return max(candidates, key=lambda p: (p / "meta.json").stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--corpus-label", default="fr_first")
    parser.add_argument(
        "--ours", action="append", default=[], help="libellés à comparer (répétable)"
    )
    parser.add_argument("--sample", type=int, default=3_000, help="documents échantillonnés")
    parser.add_argument("--baseline", default="gpt2")
    parser.add_argument("--group-by", default="lang", choices=("lang", "source"))
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    setup_logging(args.log_level)
    load_dotenv()  # HF_TOKEN : nécessaire pour les tokenizers restreints (Llama, Mistral)

    docs = list(iter_documents(find_corpus(args.corpus_label), limit=args.sample))
    if not docs:
        print("corpus vide", file=sys.stderr)
        return 1

    counters = {}
    for label in args.ours or ["bpe32k"]:
        try:
            counters[f"thadeus/{label}"] = Codec.load(find_tokenizer(label)).count
        except FileNotFoundError as exc:
            print(f"⚠️  {exc}", file=sys.stderr)

    for name in REFERENCE_TOKENIZERS:
        counter = load_reference(name)
        if counter is not None:
            counters[name] = counter

    if not counters:
        print("aucun tokenizer à comparer", file=sys.stderr)
        return 1

    key = (lambda d: d.lang) if args.group_by == "lang" else (lambda d: d.source)
    result = compare(counters, docs, key=key, baseline=args.baseline)

    print(f"Échantillon : {len(docs)} documents du corpus {args.corpus_label!r}\n")
    print(format_comparison(result))

    out = ARTIFACT_ROOT / "tokenizer" / "comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDétail : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
