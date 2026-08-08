#!/usr/bin/env python3
"""Construit le corpus de réglage par instructions, avec masque de perte.

    python scripts/build_sft.py --limit 60000

Ce script ne passe **pas** par la chaîne de documents habituelle, et c'est
voulu : un exemple d'instruction n'est pas un document, c'est une **paire**
question/réponse dont seule la seconde moitié doit être apprise. Le masque se
calcule donc à la tokenisation, quand on connaît encore la frontière.

**Le format est le contrat le plus important de cette phase.** Le modèle
apprendra à réagir à ces marqueurs exacts ; toute inférence ultérieure devra les
reproduire au caractère près. On les garde donc textuels et proches du Markdown
que le modèle a vu en pré-entraînement, plutôt que des jetons réservés dont les
embeddings partiraient de zéro.

Le corpus mêle un **rejeu** du pré-entraînement : sans lui, 60 000 exemples
courts déplaceraient le modèle vers un registre télégraphique et lui feraient
oublier la prose longue.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thadeus.core.artifacts import open_artifact  # noqa: E402
from thadeus.core.env import load_dotenv  # noqa: E402
from thadeus.core.logs import get_logger, setup_logging  # noqa: E402
from thadeus.core.seeding import seed_everything  # noqa: E402
from thadeus.data.shard import iter_documents  # noqa: E402
from thadeus.tokenizer.codec import Codec  # noqa: E402
from thadeus.tokenizer.train import find_corpus  # noqa: E402
from thadeus.train.tokens import TokenShardWriter  # noqa: E402

log = get_logger(__name__)

# Les marqueurs que le modèle apprendra. À ne plus changer sans réentraîner :
# une inférence qui n'utiliserait pas exactement ces chaînes ne déclencherait pas
# le comportement appris.
INSTRUCTION = "### Instruction\n"
ENTREE = "\n### Entrée\n"
REPONSE = "\n### Réponse\n"


def formate(instruction: str, entree: str, sortie: str) -> tuple[str, str]:
    """Rend ``(question, réponse)`` — séparés, car seul le second est supervisé."""
    question = INSTRUCTION + instruction.strip()
    if entree and entree.strip():
        question += ENTREE + entree.strip()
    question += REPONSE
    return question, sortie.strip()


def exemples(limit: int):
    """Flux d'exemples (instruction, entrée, sortie) depuis Hugging Face."""
    from datasets import load_dataset

    ds = load_dataset(
        "jpacifico/French-Alpaca-dataset-Instruct-110K", split="train", streaming=True
    ).shuffle(seed=1337, buffer_size=10_000)
    produits = 0
    for row in ds:
        instruction = (row.get("instruction") or "").strip()
        sortie = (row.get("output") or "").strip()
        # Un exemple sans réponse n'apprend rien ; une réponse d'un mot non plus.
        if not instruction or len(sortie.split()) < 3:
            continue
        yield instruction, (row.get("input") or ""), sortie
        produits += 1
        if produits >= limit:
            return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=60_000, help="exemples d'instruction")
    parser.add_argument("--replay", type=int, default=8_000, help="documents de rejeu")
    parser.add_argument("--tokenizer", default="bpe32k")
    parser.add_argument("--corpus-label", default="thadeus_v1")
    parser.add_argument("--label", default="sft")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    setup_logging("INFO")
    load_dotenv()
    seed_everything(1337)

    from thadeus.eval.suite import _find

    codec = Codec.load(_find("tokenizer", args.tokenizer))
    spec = {
        "dataset": "jpacifico/French-Alpaca-dataset-Instruct-110K",
        "limit": args.limit,
        "replay": args.replay,
        "tokenizer": args.tokenizer,
        "format": [INSTRUCTION, ENTREE, REPONSE],
    }
    artifact = open_artifact("tokens", args.label, spec)
    if artifact.exists() and not args.force:
        print(f"Déjà construit : {artifact}")
        return 0
    artifact.create()

    n_instr = n_replay = 0
    with TokenShardWriter(artifact.path, vocab_size=codec.vocab_size, metadata=spec) as writer:
        for instruction, entree, sortie in exemples(args.limit):
            question, reponse = formate(instruction, entree, sortie)
            ids_q = codec.encode(question)
            ids_r = codec.encode(reponse, add_eot=True)
            if len(ids_q) + len(ids_r) > args.max_tokens:
                continue  # tronquer une réponse apprendrait au modèle à s'arrêter au hasard
            # 0 sur la question, 1 sur la réponse : le modèle apprend à
            # RÉPONDRE, pas à inventer des questions.
            writer.write(ids_q + ids_r, mask=[0] * len(ids_q) + [1] * len(ids_r))
            n_instr += 1
            if n_instr % 10_000 == 0:
                log.info("  %d exemples · %d tokens", n_instr, writer.n_tokens)

        # Rejeu : masque plein, c'est du pré-entraînement ordinaire.
        for doc in iter_documents(find_corpus(args.corpus_label), limit=args.replay):
            writer.write(codec.encode(doc.text, add_eot=True)[: args.max_tokens])
            n_replay += 1

    stats = {
        "exemples_instruction": n_instr,
        "documents_rejeu": n_replay,
        "tokens": writer.n_tokens,
        "tokens_supervises": writer.n_masked_tokens,
        "part_supervisee": round(writer.n_masked_tokens / writer.n_tokens, 3),
    }
    artifact.write_json("stats.json", stats)
    artifact.write_meta(spec, **stats)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nArtefact : {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
