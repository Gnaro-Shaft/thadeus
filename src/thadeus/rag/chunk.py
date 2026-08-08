"""Découpage des notes en passages récupérables.

**La contrainte qui décide de tout : le modèle n'a que 1024 tokens de contexte.**

Un prompt de RAG doit y loger la question, les passages retrouvés, et laisser la
place à la réponse. Le budget se répartit ainsi :

    1024 = ~200 (génération) + ~60 (structure et question) + ~760 (passages)

Soit **trois passages d'environ 250 tokens**, ou 150 mots. C'est peu, et ça
impose des passages courts — donc un découpage qui ne coupe pas au milieu d'une
idée, sinon un passage récupéré n'a plus de sens à la lecture.

D'où le découpage **par section Markdown** en priorité : dans des notes
Obsidian, un titre de section est une frontière sémantique posée par l'auteur.
On ne retombe sur un découpage par paragraphes que lorsqu'une section dépasse la
taille visée.

Chaque passage garde le **titre de sa note et celui de sa section** : ils servent
à la fois à l'indexation (une note nommée « Muon » doit remonter sur « muon »)
et à la citation, pour que la réponse soit traçable.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from thadeus.data.sources.obsidian import from_obsidian
from thadeus.rag.index import Passage

__all__ = ["chunk_note", "iter_vault_passages"]

_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def chunk_note(
    text: str,
    *,
    title: str,
    source: str,
    target_words: int = 150,
    min_words: int = 20,
) -> Iterator[Passage]:
    """Découpe une note en passages, aux frontières de section quand c'est possible.

    Args:
        target_words: ~150 mots ≈ 250 tokens, soit un tiers du budget de contexte.
        min_words: en dessous, un passage n'apporte pas assez pour occuper une
            place dans un contexte aussi contraint.
    """
    sections = _split_sections(text)
    index = 0
    for titre_section, corps in sections:
        for morceau in _split_by_words(corps, target_words):
            if len(morceau.split()) < min_words:
                continue
            yield Passage(
                id=f"{source}#{index}",
                text=morceau.strip(),
                source=source,
                # Le titre de section rejoint celui de la note : les deux sont
                # des mots que l'auteur a choisis, donc de bons signaux.
                title=f"{title} — {titre_section}" if titre_section else title,
            )
            index += 1


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Découpe aux titres Markdown. Sans titre, rend le texte entier."""
    positions = [(m.start(), m.group(2).strip()) for m in _HEADING.finditer(text)]
    if not positions:
        return [("", text)]

    sections: list[tuple[str, str]] = []
    if positions[0][0] > 0:
        preambule = text[: positions[0][0]].strip()
        if preambule:
            sections.append(("", preambule))

    bornes = [p for p, _ in positions] + [len(text)]
    for (debut, titre), fin in zip(positions, bornes[1:], strict=True):
        corps = text[debut:fin]
        corps = _HEADING.sub("", corps, count=1).strip()
        if corps:
            sections.append((titre, corps))
    return sections


def _split_by_words(text: str, target: int) -> list[str]:
    """Regroupe les paragraphes jusqu'à la taille visée, sans couper dedans.

    **Un plafond dur reste nécessaire.** Une section faite d'un seul paragraphe
    géant — un long tableau, un bloc de code, une liste sans ligne vide — ne se
    découpe sur aucune frontière naturelle. Sans plafond, elle produit un
    passage de plusieurs milliers de mots qui ne tiendra jamais dans le
    contexte : `build_prompt` le sauterait, et le passage le MIEUX CLASSÉ
    disparaîtrait de la réponse sans que rien ne le signale.

    Dans ce cas seulement, on coupe au mot. C'est moins bon qu'une frontière
    sémantique, mais infiniment mieux qu'un passage inutilisable.
    """
    plafond = 2 * target
    morceaux: list[str] = []
    courant: list[str] = []
    compte = 0

    for paragraphe in text.split("\n\n"):
        mots = paragraphe.split()
        if len(mots) > plafond:
            if courant:
                morceaux.append("\n\n".join(courant))
                courant, compte = [], 0
            for debut in range(0, len(mots), target):
                morceaux.append(" ".join(mots[debut : debut + target]))
            continue
        if compte + len(mots) > target and courant:
            morceaux.append("\n\n".join(courant))
            courant, compte = [], 0
        courant.append(paragraphe)
        compte += len(mots)

    if courant:
        morceaux.append("\n\n".join(courant))
    return morceaux


def iter_vault_passages(
    vault: str,
    *,
    split: str = "all",
    target_words: int = 150,
    exclude: tuple[str, ...] = (".obsidian", ".trash", "Templates"),
) -> Iterator[Passage]:
    """Parcourt un Vault et rend ses passages.

    ``split`` permet d'indexer uniquement les notes d'entraînement ou uniquement
    celles tenues à l'écart — utile pour mesurer la récupération sur des notes
    que le modèle n'a jamais vues à l'entraînement.
    """
    for doc in from_obsidian(vault=vault, split=split, min_words=20, exclude=exclude):
        chemin = doc.meta.get("path", doc.id)
        titre = chemin.rsplit("/", 1)[-1].removesuffix(".md")
        # Le frontmatter et les liens ont déjà été retirés par la source ; les
        # titres de section, eux, sont conservés — le découpage s'en sert comme
        # frontières sémantiques.
        yield from chunk_note(doc.text, title=titre, source=chemin, target_words=target_words)
