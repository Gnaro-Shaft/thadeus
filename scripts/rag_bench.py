#!/usr/bin/env python3
"""Mesure la qualité de récupération sur le Vault.

    python scripts/rag_bench.py

**Deux protocoles, parce que le premier ne suffit pas.**

``--mode titre`` fabrique la requête à partir du **titre** de la note. Simple,
mais **circulaire** : les titres sont sur-pondérés dans l'index, donc on se note
en partie sur sa propre pondération. Mesuré : 86 % de rappel@1 avec un poids de
titre de 2, contre 16,5 % à poids nul. L'écart est la mesure du biais.

``--mode corps`` (défaut) tire la requête de **termes rares du corps** d'un
passage, jamais du titre. C'est plus proche d'un usage réel — on cherche ce dont
on se souvient du contenu — et la mesure ne dépend plus de la pondération.

Aucun des deux ne remplace l'autre : le premier dit si la note remonte quand on
sait comment elle s'appelle, le second si elle remonte quand on ne s'en souvient
que vaguement.

On rapporte le **rappel@k** — la note attendue est-elle parmi les k premiers
passages ? — et le **rang réciproque moyen** (MRR), qui pénalise une bonne
réponse arrivée en cinquième position.

Une récupération qui échoue rend la génération inutile : le modèle ne peut pas
citer ce qu'on ne lui a pas donné. C'est donc la mesure à regarder en premier.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thadeus.core.logs import setup_logging  # noqa: E402
from thadeus.rag import BM25Index, iter_vault_passages  # noqa: E402

_SEPARATEURS = re.compile(r"[—\-_·:]+")


def requete_depuis_titre(titre: str) -> str:
    """Transforme un titre de note en requête plausible."""
    return _SEPARATEURS.sub(" ", titre.rsplit("/", 1)[-1].removesuffix(".md")).strip()


def requete_depuis_corps(
    index, passages_de_la_note, rng, n_termes: int = 5, choix: str = "rare"
) -> str:
    """Fabrique une requête à partir des termes les plus **rares** du corps.

    Les termes rares sont ceux qui identifient réellement une note : « muon »,
    « ChromaDB », « SASU ». Les prendre revient à simuler quelqu'un qui se
    souvient de deux ou trois mots marquants — l'usage réel d'une recherche dans
    ses propres notes.

    On exclut explicitement les termes du titre : sans cela, on retomberait dans
    la circularité que ce mode est censé lever.
    """
    from thadeus.rag.index import tokenize

    passage = rng.choice(passages_de_la_note)
    interdits = set(tokenize(passage.title))
    candidats = [t for t in set(tokenize(passage.text)) if t not in interdits]
    if not candidats:
        return ""
    if choix == "rare":
        # Les plus rares d'abord. Biais assumé : ce sont les termes les plus
        # discriminants, donc la mesure est une BORNE HAUTE — elle simule
        # quelqu'un dont le souvenir tombe pile sur le mot le plus distinctif.
        candidats.sort(key=lambda t: len(index._postings.get(t, ())))
        return " ".join(candidats[:n_termes])
    # Termes tirés au hasard : borne BASSE, un souvenir quelconque du contenu.
    rng.shuffle(candidats)
    return " ".join(candidats[:n_termes])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", default="~/dGnaro")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--mode", default="corps", choices=("titre", "corps"))
    parser.add_argument(
        "--termes",
        default="rare",
        choices=("rare", "hasard"),
        help="rare = borne haute (souvenir du mot le plus distinctif) ; "
        "hasard = borne basse (souvenir quelconque)",
    )
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument(
        "--title-weight",
        type=int,
        default=2,
        help="poids du titre dans l'index ; 0 mesure la récupération sur le CORPS SEUL, "
        "ce qui lève la circularité du protocole (requêtes tirées des titres)",
    )
    args = parser.parse_args()

    setup_logging("ERROR")
    index = BM25Index(title_weight=args.title_weight)
    for p in iter_vault_passages(args.vault):
        index.add(p)
    index.build()

    # Une requête par note, pas par passage : c'est la note qu'on veut retrouver.
    par_note: dict[str, str] = {}
    passages_par_note: dict[str, list] = {}
    for p in index.passages:
        par_note.setdefault(p.source, p.title.split(" — ")[0])
        passages_par_note.setdefault(p.source, []).append(p)

    rng = random.Random(1337)
    echantillon = rng.sample(sorted(par_note), min(args.queries, len(par_note)))

    rangs: list[int | None] = []
    kmax = max(args.ks)
    for source in echantillon:
        q = (
            requete_depuis_titre(par_note[source])
            if args.mode == "titre"
            else requete_depuis_corps(index, passages_par_note[source], rng, choix=args.termes)
        )
        if not q:
            continue
        resultats = index.search(q, k=kmax)
        rang = next((i + 1 for i, (p, _) in enumerate(resultats) if p.source == source), None)
        rangs.append(rang)

    print(
        f"{len(index.passages)} passages · {len(par_note)} notes · {len(rangs)} requêtes"
        f" · mode « {args.mode}/{args.termes} » · poids titre {args.title_weight}\n"
    )
    print(f"{'mesure':<14}{'valeur':>9}")
    print("-" * 23)
    for k in args.ks:
        rappel = sum(1 for r in rangs if r is not None and r <= k) / len(rangs)
        print(f"{'rappel@' + str(k):<14}{100 * rappel:>8.1f}%")
    mrr = sum(1 / r for r in rangs if r is not None) / len(rangs)
    print(f"{'MRR':<14}{mrr:>9.3f}")
    manques = [s for s, r in zip(echantillon, rangs, strict=True) if r is None]
    print(f"{'jamais trouvé':<14}{len(manques):>9}")
    for s in manques[:5]:
        print(f"    · {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
