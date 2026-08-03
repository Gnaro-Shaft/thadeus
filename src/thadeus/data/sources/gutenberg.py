"""Source Project Gutenberg — romans en texte brut.

Deux traitements que la source générique :func:`local_files` ne sait pas faire,
et sans lesquels ces fichiers seraient inutilisables :

1. **Retirer l'encadrement légal.** Chaque livre Gutenberg est encadré d'une
   licence d'environ 400 mots, identique dans tous les fichiers. La garder
   apprendrait au modèle à réciter une licence — et la déduplication ne
   l'attraperait pas, puisqu'elle est noyée dans un livre par ailleurs unique.

2. **Découper en chapitres.** Un roman fait 500 000 mots là où le filtre de
   longueur plafonne à 100 000, et où le modèle ne voit que 1024 tokens à la
   fois. Un livre entier en un seul document serait rejeté par les filtres, et
   inexploitable même s'il passait.

Le découpage suit les marqueurs de chapitre quand ils existent, et retombe sur
un découpage par paragraphes sinon — un roman sans chapitres reste utilisable.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from thadeus.core.logs import get_logger
from thadeus.data.schema import Document
from thadeus.data.sources import SOURCES

log = get_logger(__name__)

# Marqueurs officiels encadrant le texte réel, en anglais dans tous les fichiers.
_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.IGNORECASE)
_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.IGNORECASE)

# Chapitres français et anglais, en début de ligne.
_CHAPTER = re.compile(
    r"^\s*(?:CHAPITRE|Chapitre|CHAPTER|Chapter|LIVRE|Livre|PARTIE|Partie)\s+"
    r"[IVXLCDM\d]+[.\s]*.*$",
    re.MULTILINE,
)


def strip_boilerplate(text: str) -> str:
    """Retire l'encadrement légal, en gardant le texte de l'œuvre.

    Si les marqueurs sont absents (fichier retraité, édition ancienne), on rend
    le texte tel quel : mieux vaut un peu de bruit qu'un livre perdu.
    """
    start = _START.search(text)
    if start:
        text = text[start.end() :]
    end = _END.search(text)
    if end:
        text = text[: end.start()]
    return text.strip()


def split_into_chunks(text: str, *, target_words: int = 1200) -> list[str]:
    """Découpe un livre en morceaux exploitables.

    Args:
        target_words: taille visée. ~1200 mots ≈ 1700 tokens, soit un peu plus
            que la fenêtre de contexte du modèle : chaque morceau la remplit
            sans qu'on ait à en concaténer deux.

    On coupe aux chapitres quand ils existent — une frontière de chapitre est
    une frontière sémantique réelle — puis on regroupe ou redécoupe par
    paragraphes pour approcher la taille visée.
    """
    parts = _split_on_chapters(text)
    chunks: list[str] = []
    for part in parts:
        if len(part.split()) <= target_words * 1.5:
            chunks.append(part)
        else:
            chunks.extend(_split_on_paragraphs(part, target_words))
    return [c for c in chunks if c.strip()]


def _split_on_chapters(text: str) -> list[str]:
    positions = [m.start() for m in _CHAPTER.finditer(text)]
    if len(positions) < 2:
        return [text]
    bounds = [*positions, len(text)]
    return [text[a:b].strip() for a, b in zip(bounds[:-1], bounds[1:], strict=True)]


def _split_on_paragraphs(text: str, target_words: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for paragraph in text.split("\n\n"):
        words = len(paragraph.split())
        if count + words > target_words and current:
            chunks.append("\n\n".join(current))
            current, count = [], 0
        current.append(paragraph)
        count += words
    if current:
        chunks.append("\n\n".join(current))
    return chunks


@SOURCES.register("gutenberg")
def from_gutenberg(
    *,
    label: str = "gutenberg",
    root: str,
    pattern: str = "*.txt",
    lang: str = "fr",
    target_words: int = 1200,
    min_words: int = 100,
    encoding: str = "utf-8",
    limit: int | None = None,
) -> Iterator[Document]:
    """Lit des livres Gutenberg et les rend découpés en morceaux.

    L'identifiant porte le nom du livre et l'indice du morceau
    (``gutenberg:germinal#12``) : stable entre exécutions, et lisible quand on
    inspecte le corpus.
    """
    base = Path(root).expanduser()
    if not base.is_dir():
        raise FileNotFoundError(f"répertoire Gutenberg introuvable : {base}")

    emitted = 0
    books = 0
    for path in sorted(base.glob(pattern)):
        try:
            raw = path.read_text(encoding=encoding, errors="replace")
        except OSError:
            log.warning("livre illisible ignoré : %s", path.name)
            continue

        text = strip_boilerplate(raw)
        if not text:
            continue
        books += 1
        titre = path.stem

        for index, chunk in enumerate(split_into_chunks(text, target_words=target_words)):
            if len(chunk.split()) < min_words:
                continue
            yield Document(
                id=f"{label}:{titre}#{index}",
                text=chunk,
                source=label,
                lang=lang,
                meta={"book": titre},
            )
            emitted += 1
            if limit is not None and emitted >= limit:
                log.info("Source %s : %d morceaux (limite atteinte)", label, emitted)
                return

    log.info("Source %s : %d morceaux issus de %d livres", label, emitted, books)
