#!/usr/bin/env python3
"""Compare deux modèles sur les mêmes mesures — le compromis d'un fine-tuning.

    python scripts/compare_models.py --base medium_mup --tuned vault_ft

Un fine-tuning **échange toujours** du général contre du spécifique. La seule
question qui vaille est : combien ? Ce script mesure les deux côtés du troc sur
exactement les mêmes textes.

Les notes de validation du Vault n'ont **jamais** été vues à l'entraînement
(split par hachage du chemin) : c'est ce qui distingue « a appris un style » de
« a mémorisé un corpus ».
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from thadeus.core.device import hot_path_dtype, resolve_device  # noqa: E402
from thadeus.core.env import load_dotenv  # noqa: E402
from thadeus.core.logs import setup_logging  # noqa: E402
from thadeus.data.shard import iter_documents  # noqa: E402
from thadeus.data.sources.obsidian import from_obsidian  # noqa: E402
from thadeus.eval.perplexity import evaluate_documents  # noqa: E402
from thadeus.eval.probes import ALL_PROBES, HARD_PROBES, PROBES, run_probes  # noqa: E402
from thadeus.eval.suite import EvalConfig, _find, _load_model  # noqa: E402
from thadeus.tokenizer.codec import Codec  # noqa: E402


def echantillonner(model, codec, prompts, device, *, seed=1337, n=70):
    """Génère un échantillon par invite, avec une graine **dérivée par invite**.

    Une graine commune ferait partager le même flux aléatoire à tous les
    échantillons : le même token sortirait au même rang dans des textes sans
    rapport, et ressemblerait à une pathologie du modèle. Voir le commentaire
    correspondant dans `thadeus.eval.suite`.
    """
    from thadeus.core.seeding import derive_seed, seed_everything

    sorties = []
    for index, prompt in enumerate(prompts):
        seed_everything(derive_seed(seed, "sample", index))
        ids = torch.tensor([codec.encode(prompt)], device=device)
        out = model.generate(
            ids, max_new_tokens=n, temperature=0.75, top_k=50, forbidden=codec.service_ids
        )
        sorties.append(codec.decode(out[0, ids.shape[1] :].tolist()))
    return sorties


def mesurer(model, codec, docs, device, dtype, seq_len=1024):
    return evaluate_documents(
        model, codec, docs, device=device, dtype=dtype, seq_len=seq_len, group_by="lang"
    ).overall


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="medium_mup")
    parser.add_argument("--tuned", default="vault_ft")
    parser.add_argument("--vault", default="${THADEUS_VAULT}")
    parser.add_argument("--corpus-label", default="thadeus_v1")
    parser.add_argument("--documents", type=int, default=150)
    args = parser.parse_args()

    setup_logging("ERROR")
    load_dotenv()
    device = resolve_device("auto")
    dtype = hot_path_dtype(device)
    codec = Codec.load(_find("tokenizer", "bpe32k"))

    # Les mêmes textes pour les deux modèles — sinon on ne compare rien.
    vault_val = list(from_obsidian(vault=args.vault, split="val"))
    corpus = list(iter_documents(_find("data", args.corpus_label) / "corpus", limit=args.documents))
    gutenberg_root = Path("${THADEUS_GUTENBERG}").expanduser()
    gut = (
        list(
            __import__("thadeus.data.sources.gutenberg", fromlist=["x"]).from_gutenberg(
                root=str(gutenberg_root), limit=80
            )
        )
        if gutenberg_root.is_dir()
        else []
    )

    resultats = {}
    for nom, label in (("base", args.base), ("fine-tuné", args.tuned)):
        model, _, step = _load_model(EvalConfig(run_label=label), device)
        r = {"pas": step}
        r["vault"] = mesurer(model, codec, vault_val, device, dtype)
        r["corpus"] = mesurer(model, codec, corpus, device, dtype)
        if gut:
            r["gutenberg"] = mesurer(model, codec, gut, device, dtype)
        for cle, jeu in (("sondes", ALL_PROBES), ("faciles", PROBES), ("dures", HARD_PROBES)):
            p = run_probes(model, codec, device=device, dtype=dtype, pairs=jeu)
            r[cle] = (sum(v.correct for v in p.values()), sum(v.total for v in p.values()))
        resultats[nom] = r
        del model
        torch.mps.empty_cache() if device.type == "mps" else None

    b, t = resultats["base"], resultats["fine-tuné"]
    print(f"\n{'mesure':<32}{'base':>11}{'fine-tuné':>12}{'écart':>11}")
    print("-" * 66)
    # Toutes ces mesures sont des perplexités : plus bas vaut mieux.
    for cle, titre in (
        (
            "vault",
            "Vault (102 notes JAMAIS vues)",
        ),
        (
            "corpus",
            "corpus général",
        ),
        (
            "gutenberg",
            "Gutenberg (français littéraire)",
        ),
    ):
        if cle not in b:
            continue
        pb, pt = b[cle].perplexity, t[cle].perplexity
        fleche = "↓ mieux" if pt < pb else "↑ pire"
        print(f"{titre:<32}{pb:>11.2f}{pt:>12.2f}{100 * (pt / pb - 1):>+9.1f}%  {fleche}")
    print("-" * 66)
    for cle, titre in (
        ("faciles", "sondes faciles"),
        ("dures", "sondes DIFFICILES"),
        ("sondes", "sondes (total)"),
    ):
        cb, tot = b[cle]
        ct, _ = t[cle]
        print(f"{titre:<32}{cb}/{tot:<9}{ct}/{tot:<10}{ct - cb:>+6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
