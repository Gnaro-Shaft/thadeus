"""Mesure ce qu'une collecte apporte de **nouveau**, pas ce qu'elle pèse.

**Pourquoi ce module existe.** Cinq collectes quotidiennes ont produit 10 Md de
tokens en cinq nuits. Chaque rapport annonçait fièrement « 2,03 Md ». Elles
étaient identiques entre elles à 98-99 %, et identiques à 98,8 % à un corpus
déjà intégré : cinq jours de téléchargement pour zéro document nouveau.

Rien ne l'a signalé, parce que rien ne le mesurait. Le volume était juste, la
déduplication interne était juste — mais personne ne comparait la récolte du
jour à ce qu'on possédait déjà. Une automatisation qui rapporte du volume sans
rapporter de la **nouveauté** est une boîte noire qui ment poliment.

**Comment on mesure.** Comparer deux corpus document par document coûterait des
heures. On échantillonne : quelques milliers de documents de chaque côté,
réduits à une empreinte courte de leur début de texte. Ce n'est pas une mesure
exacte du recouvrement — c'est un **détecteur de redondance massive**, et c'est
tout ce qu'on lui demande. À 98 % il crie ; à 3 % il se tait. Distinguer 3 % de
5 % n'aurait aucune conséquence pratique.

L'empreinte porte sur les premiers caractères : deux versions d'un même document
qui ne diffèrent que par leur fin comptent comme un doublon, ce qui est le
comportement voulu ici.
"""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path
from typing import Any

from thadeus.core.logs import get_logger

__all__ = ["compare_to_existing", "fingerprints", "overlap"]

log = get_logger(__name__)

# Au-delà, la collecte n'a pratiquement rien apporté et il faut le dire fort.
SEUIL_ALERTE = 0.50

# 4 000 documents suffisent : à 98 % de recouvrement l'échantillon le voit dès
# les premières centaines, et lire davantage ne changerait aucune décision.
ECHANTILLON = 4_000

# Longueur de texte retenue pour l'empreinte. Assez pour identifier un document,
# assez court pour rester insensible à une troncature de fin.
PREFIXE = 400


def fingerprints(corpus: Path, *, limit: int = ECHANTILLON) -> set[bytes]:
    """Empreintes des ``limit`` premiers documents d'un répertoire de corpus."""
    from thadeus.data.shard import iter_documents

    if not corpus.is_dir():
        return set()
    return {
        hashlib.blake2b(doc.text[:PREFIXE].encode("utf-8"), digest_size=8).digest()
        for doc in itertools.islice(iter_documents(corpus), limit)
    }


def overlap(a: set[bytes], b: set[bytes]) -> float:
    """Part de ``a`` déjà présente dans ``b``. 0 si l'un des deux est vide.

    Volontairement **asymétrique** : la question n'est pas « ces deux corpus se
    ressemblent-ils » mais « ce que je viens de collecter, le possédais-je
    déjà ». Un petit corpus entièrement contenu dans un grand a un recouvrement
    de 100 %, et c'est bien ce qu'on veut lire.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / len(a)


def compare_to_existing(
    corpus: Path,
    *,
    root: Path,
    exclude: str | None = None,
    limit: int = ECHANTILLON,
) -> dict[str, Any]:
    """Compare un corpus fraîchement construit à tous les autres déjà présents.

    Args:
        corpus: répertoire ``corpus`` de l'artefact à évaluer.
        root: racine des artefacts de données, où chercher les références.
        exclude: nom d'artefact à ignorer — le sien.

    Returns:
        Un dictionnaire prêt à être consigné dans ``report.json`` : recouvrement
        par artefact de référence, le maximum, et le verdict.
    """
    nouvelles = fingerprints(corpus, limit=limit)
    if not nouvelles:
        return {"echantillon": 0, "par_artefact": {}, "recouvrement_max": None}

    par_artefact: dict[str, float] = {}
    for repertoire in sorted(root.iterdir()) if root.is_dir() else []:
        if not repertoire.is_dir() or repertoire.name == exclude:
            continue
        # Seuls les artefacts achevés font référence : un corpus interrompu
        # n'est pas une possession fiable.
        if not (repertoire / "meta.json").is_file():
            continue
        connues = fingerprints(repertoire / "corpus", limit=limit)
        if connues:
            par_artefact[repertoire.name] = round(overlap(nouvelles, connues), 4)

    if not par_artefact:
        return {"echantillon": len(nouvelles), "par_artefact": {}, "recouvrement_max": None}

    pire = max(par_artefact.items(), key=lambda kv: kv[1])
    resultat = {
        "echantillon": len(nouvelles),
        "par_artefact": par_artefact,
        "recouvrement_max": pire[1],
        "artefact_le_plus_proche": pire[0],
    }

    if pire[1] >= SEUIL_ALERTE:
        log.warning(
            "REDONDANCE : %.1f %% de cette collecte est déjà dans %r. "
            "Elle n'apporte presque rien. Vérifier que la graine du corpus varie "
            "d'une collecte à l'autre — une graine figée reprélève la même tranche.",
            100 * pire[1],
            pire[0],
        )
    else:
        log.info(
            "Nouveauté : recouvrement maximal %.1f %% (avec %r) sur %d documents comparés.",
            100 * pire[1],
            pire[0],
            len(nouvelles),
        )
    return resultat
