#!/usr/bin/env python3
"""Encode un corpus de documents en shards binaires de tokens.

    python scripts/tokenize_corpus.py --corpus-label smoke --tokenizer bpe32k
    python scripts/tokenize_corpus.py --corpus-label fr_first --tokenizer bpe32k

C'est le pont entre l'étage 2 (tokenizer) et l'étage 5 (entraînement). Chaque
document se termine par le token de fin de texte : sans ce séparateur, le modèle
apprendrait que la fin d'un article enchaîne naturellement sur le début d'un
autre.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thadeus.core.artifacts import ARTIFACT_ROOT, open_artifact  # noqa: E402
from thadeus.core.env import load_dotenv  # noqa: E402
from thadeus.core.logs import get_logger, setup_logging  # noqa: E402
from thadeus.data.schema import format_tokens  # noqa: E402
from thadeus.data.shard import iter_documents  # noqa: E402
from thadeus.tokenizer.codec import Codec  # noqa: E402
from thadeus.tokenizer.train import find_corpus  # noqa: E402
from thadeus.train.tokens import TokenShardWriter  # noqa: E402

log = get_logger(__name__)

BATCH = 512  # documents encodés d'un coup — le moteur Rust travaille en lot


def find_tokenizer(label: str) -> Path:
    candidates = [
        p
        for p in sorted((ARTIFACT_ROOT / "tokenizer").glob(f"{label}-*"))
        if (p / "meta.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"aucun tokenizer achevé nommé {label!r}. "
            f"L'entraîner avec : python scripts/train_tokenizer.py --config tokenizer/{label}.toml"
        )
    return max(candidates, key=lambda p: (p / "meta.json").stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--corpus-label", default="fr_first")
    parser.add_argument("--tokenizer", default="bpe32k")
    parser.add_argument("--label", default=None, help="libellé de sortie (défaut : corpus-label)")
    parser.add_argument("--limit", type=int, default=None, help="documents maximum")
    parser.add_argument("--tokens-per-shard", type=int, default=100_000_000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    load_dotenv()

    corpus = find_corpus(args.corpus_label)
    tokenizer_dir = find_tokenizer(args.tokenizer)
    codec = Codec.load(tokenizer_dir)
    label = args.label or args.corpus_label

    # Le hash de config nomme la sortie : changer de tokenizer ou de corpus
    # produit un répertoire distinct, jamais un écrasement silencieux.
    spec = {
        "corpus": str(corpus),
        "tokenizer": tokenizer_dir.name,
        "vocab_size": codec.vocab_size,
        "limit": args.limit,
    }
    artifact = open_artifact("tokens", label, spec)
    if artifact.exists() and not args.force:
        print(f"Déjà tokenisé : {artifact}")
        return 0

    log.info(
        "Corpus %s -> tokenizer %s (vocabulaire %d)", corpus, tokenizer_dir.name, codec.vocab_size
    )
    artifact.create()

    batch: list[str] = []
    with TokenShardWriter(
        artifact.path,
        vocab_size=codec.vocab_size,
        tokens_per_shard=args.tokens_per_shard,
        metadata=spec,
    ) as writer:

        def flush(texts: list[str]) -> None:
            for ids in codec.encode_batch(texts, add_eot=True):
                writer.write(ids)

        for index, doc in enumerate(iter_documents(corpus, limit=args.limit), start=1):
            batch.append(doc.text)
            if len(batch) >= BATCH:
                flush(batch)
                batch.clear()
            if index % 100_000 == 0:
                log.info("  %d documents · %s tokens", index, format_tokens(writer.n_tokens))
        if batch:
            flush(batch)

    artifact.write_json(
        "stats.json",
        {
            "documents": writer.n_documents,
            "tokens": writer.n_tokens,
            "shards": len(writer.shards),
            "size_gb": writer.n_tokens * writer.dtype.itemsize / 1024**3,
        },
    )
    artifact.write_meta(spec, n_tokens=writer.n_tokens, n_documents=writer.n_documents)

    print(json.dumps(json.loads((artifact.path / "stats.json").read_text()), indent=2))
    print(f"\nArtefact : {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
