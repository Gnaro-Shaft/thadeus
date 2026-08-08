#!/usr/bin/env python3
"""Interroge le Vault — récupération BM25 puis génération ancrée.

    python scripts/ask.py "quelle est la stratégie pour le lancement freelance ?"
    python scripts/ask.py --retrieve-only "muon optimiseur"
    python scripts/ask.py --model medium_mup "..."   # modèle avant fine-tuning

**Ce qu'il faut en attendre.** La récupération est fiable (94,5 % de rappel@1 sur
le protocole le plus conservateur) et utile par elle-même : c'est un moteur de
recherche sur le Vault. La génération, elle, vient d'un modèle de 188 M non
instruit — il *continue* du texte plutôt qu'il ne *répond*. Il reprendra le
vocabulaire des passages retrouvés, ce qui est déjà l'essentiel de l'ancrage,
mais ne raisonnera pas dessus de façon fiable.

C'est pourquoi les **sources sont toujours affichées** : la réponse se vérifie.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thadeus.core.device import hot_path_dtype, resolve_device  # noqa: E402
from thadeus.core.env import load_dotenv  # noqa: E402
from thadeus.core.logs import setup_logging  # noqa: E402
from thadeus.rag import BM25Index, answer, iter_vault_passages  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="+")
    parser.add_argument("--vault", default="~/dGnaro")
    parser.add_argument("--model", default="vault_ft")
    parser.add_argument("-k", type=int, default=3, help="passages récupérés")
    parser.add_argument("--tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--retrieve-only", action="store_true", help="pas de génération")
    args = parser.parse_args()

    setup_logging("ERROR")
    load_dotenv()
    question = " ".join(args.question)

    index = BM25Index()
    for p in iter_vault_passages(args.vault):
        index.add(p)
    index.build()

    if args.retrieve_only:
        print(f"« {question} »\n")
        for rang, (p, score) in enumerate(index.search(question, k=args.k), 1):
            print(f"{rang}. [{score:.2f}] {p.title}")
            print(f"   {p.source}")
            print(f"   {p.text[:220].strip()}\n")
        return 0

    from thadeus.eval.suite import EvalConfig, _find, _load_model
    from thadeus.tokenizer.codec import Codec

    device = resolve_device("auto")
    dtype = hot_path_dtype(device)
    codec = Codec.load(_find("tokenizer", "bpe32k"))
    model, _, step = _load_model(EvalConfig(run_label=args.model), device)

    res = answer(
        question,
        index=index,
        model=model,
        codec=codec,
        device=device,
        dtype=dtype,
        k=args.k,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
    )

    print(f"« {question} »\n")
    print(f"{res.text}\n")
    print(f"— sources ({res.prompt_tokens} tokens de contexte) —")
    for p, score in res.passages:
        print(f"  [{score:.2f}] {p.title}  ·  {p.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
