#!/usr/bin/env python3
"""Mesure le suivi d'instructions — ce qu'aucune perplexité ne dit.

    python scripts/eval_instruct.py --models medium_mup sft

Une perplexité qui baisse ne dit pas si le modèle **obéit**. On teste donc des
comportements observables, chacun vérifiable par un programme :

- **s'arrête-t-il ?** Un modèle non réglé continue jusqu'à la limite de tokens.
  Produire le jeton de fin est le comportement le plus élémentaire, et le plus
  visible : sans lui, toute réponse déborde sur du remplissage.
- **respecte-t-il une contrainte de forme ?** « en un mot », « une liste de
  trois éléments » — vérifiable sans juger le contenu.
- **reste-t-il en français ?**
- **répond-il, ou paraphrase-t-il la question ?**

Aucune de ces mesures ne juge la *justesse* des réponses : à 188 M, ce serait
mesurer autre chose que ce que le réglage par instructions apporte. Elles
mesurent l'**obéissance au format**, qui est exactement ce que le SFT installe.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from thadeus.core.device import resolve_device  # noqa: E402
from thadeus.core.env import load_dotenv  # noqa: E402
from thadeus.core.logs import setup_logging  # noqa: E402
from thadeus.core.seeding import derive_seed, seed_everything  # noqa: E402
from thadeus.data.clean.language import language_scores  # noqa: E402

# Le format exact appris au réglage. Une inférence qui s'en écarterait ne
# déclencherait pas le comportement installé.
GABARIT = "### Instruction\n{}\n\n### Réponse\n"

TESTS: tuple[dict, ...] = (
    {"q": "Cite trois fruits, séparés par des virgules.", "check": "liste3"},
    {"q": "Réponds par un seul mot : quelle est la capitale de la France ?", "check": "court"},
    {"q": "Explique en une phrase ce qu'est un réseau de neurones.", "check": "phrase"},
    {"q": "Traduis en français : the cat is on the table.", "check": "francais"},
    {"q": "Donne la définition du mot « corpus ».", "check": "francais"},
    {"q": "Écris une liste de trois conseils pour bien dormir.", "check": "liste3"},
    {"q": "Quel est le contraire de « rapide » ? Réponds en un mot.", "check": "court"},
    {
        "q": "Résume en une phrase : Paris est la capitale de la France depuis 987.",
        "check": "phrase",
    },
    {"q": "Nomme trois langages de programmation.", "check": "liste3"},
    {"q": "Complète : le soleil se lève à l'", "check": "court"},
)


def verifie(check: str, texte: str) -> bool:
    """Chaque vérification est programmatique — aucun jugement de contenu."""
    t = texte.strip()
    if not t:
        return False
    if check == "liste3":
        # Trois éléments, en virgules, tirets ou lignes.
        items = [x for x in re.split(r"[,\n]|^\s*[-•*]\s*", t, flags=re.MULTILINE) if x.strip()]
        return len(items) >= 3
    if check == "court":
        return len(t.split()) <= 8
    if check == "phrase":
        return 3 <= len(t.split()) <= 60
    if check == "francais":
        return language_scores(t).get("fr", 0.0) >= 0.05
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["medium_mup", "sft"])
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--show", type=int, default=4, help="réponses affichées par modèle")
    args = parser.parse_args()

    setup_logging("ERROR")
    load_dotenv()
    device = resolve_device("auto")

    from thadeus.eval.suite import EvalConfig, _find, _load_model
    from thadeus.tokenizer.codec import Codec

    codec = Codec.load(_find("tokenizer", "bpe32k"))
    eot = codec.eot_id

    resultats: dict[str, dict] = {}
    for label in args.models:
        model, _, step = _load_model(EvalConfig(run_label=label), device)
        scores = {"format": 0, "arret": 0, "francais": 0}
        exemples = []
        for i, test in enumerate(TESTS):
            seed_everything(derive_seed(1337, "instruct", i))
            prompt = GABARIT.format(test["q"])
            ids = codec.encode(prompt)
            sortie = model.generate(
                torch.tensor([ids], device=device),
                max_new_tokens=args.tokens,
                temperature=0.6,
                top_k=50,
                forbidden=codec.service_ids,
            )
            produits = sortie[0, len(ids) :].tolist()
            # Le modèle a-t-il produit le jeton de fin de lui-même ?
            arret = eot in produits
            if arret:
                produits = produits[: produits.index(eot)]
            texte = codec.decode(produits)
            # On coupe aussi sur un nouveau bloc : sans cela un modèle non réglé
            # enchaîne sur une instruction inventée et paraîtrait plus verbeux
            # qu'il ne l'est vraiment.
            texte = texte.split("###")[0].strip()

            scores["arret"] += arret
            scores["format"] += verifie(test["check"], texte)
            scores["francais"] += language_scores(texte).get("fr", 0.0) >= 0.05
            exemples.append((test["q"], texte))

        resultats[label] = {"step": step, "scores": scores, "exemples": exemples}
        del model

    n = len(TESTS)
    print(f"\n{'modèle':<16}{'format':>10}{'arrêt':>10}{'français':>11}")
    print("-" * 47)
    for label, r in resultats.items():
        s = r["scores"]
        print(f"{label:<16}{s['format']}/{n:<8}{s['arret']}/{n:<8}{s['francais']}/{n:<9}")
    print("\nformat = contrainte respectée · arrêt = jeton de fin produit spontanément")

    for label, r in resultats.items():
        print(f"\n=== {label} ===")
        for q, a in r["exemples"][: args.show]:
            print(f"  ▸ {q}\n    {a[:200] or '(vide)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
