"""Sources locales — fichiers texte et shards déjà produits.

Deux usages :

- ``local_files`` pour verser au corpus des documents qu'on possède déjà
  (exports, archives, textes récupérés à la main).
- ``shards`` pour relire la sortie d'un passage précédent, ce qui permet de
  rejouer les étapes aval (filtres, dédup, mélange) sans refaire la collecte —
  l'étape la plus lente de la chaîne.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from thadeus.core.logs import get_logger
from thadeus.data.schema import Document
from thadeus.data.shard import iter_documents
from thadeus.data.sources import SOURCES

log = get_logger(__name__)


@SOURCES.register("local_files")
def from_local_files(
    *,
    label: str,
    root: str,
    pattern: str = "**/*.txt",
    lang: str = "unknown",
    encoding: str = "utf-8",
    limit: int | None = None,
) -> Iterator[Document]:
    """Lit des fichiers texte bruts, un document par fichier."""
    base = Path(root).expanduser()
    if not base.is_dir():
        raise FileNotFoundError(f"répertoire introuvable : {base}")

    emitted = 0
    for path in sorted(base.glob(pattern)):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding=encoding)
        except (OSError, UnicodeDecodeError):
            log.warning("fichier illisible ignoré : %s", path)
            continue
        if not text.strip():
            continue

        yield Document(
            id=f"{label}:{path.relative_to(base).as_posix()}",
            text=text,
            source=label,
            lang=lang,
        )
        emitted += 1
        if limit is not None and emitted >= limit:
            break

    log.info("Source %s : %d fichiers lus", label, emitted)


@SOURCES.register("shards")
def from_shards(
    *,
    label: str | None = None,
    path: str,
    limit: int | None = None,
) -> Iterator[Document]:
    """Relit des shards produits par un passage antérieur.

    ``label`` réétiquette la source au passage ; laissé à ``None``, chaque
    document garde la sienne — c'est ce qu'on veut pour rejouer un corpus
    multi-sources sans écraser sa composition.
    """
    for doc in iter_documents(path, limit=limit):
        yield (
            doc
            if label is None
            else Document(id=doc.id, text=doc.text, source=label, lang=doc.lang, meta=doc.meta)
        )
