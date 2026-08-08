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

from thadeus.core.artifacts import ARTIFACT_ROOT
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
    path: str | None = None,
    artifact: str | None = None,
    source: str | None = None,
    limit: int | None = None,
) -> Iterator[Document]:
    """Relit des shards produits par un passage antérieur.

    Deux façons de les désigner :

    - ``path`` : un chemin direct.
    - ``artifact`` + ``source`` : le **libellé** d'un artefact de corpus et le
      nom d'une de ses sources. C'est la forme à préférer dans les configs :
      les artefacts sont nommés par un hash de config qu'on ne connaît pas au
      moment d'écrire le TOML, et qui change dès qu'un paramètre bouge.

    ``label`` réétiquette la source au passage ; laissé à ``None``, chaque
    document garde la sienne — c'est ce qu'on veut pour rejouer un corpus
    multi-sources sans écraser sa composition.
    """
    if path is None:
        if not artifact or not source:
            raise ValueError("préciser `path`, ou bien `artifact` et `source`")
        path = str(_resolve_shards(artifact, source))
    for doc in iter_documents(path, limit=limit):
        yield (
            doc
            if label is None
            else Document(id=doc.id, text=doc.text, source=label, lang=doc.lang, meta=doc.meta)
        )


def _resolve_shards(artifact: str, source: str) -> Path:
    """Localise les shards d'une source dans le dernier artefact **achevé**.

    Un répertoire sans ``meta.json`` est une collecte interrompue : le prendre
    pour un corpus valide ferait assembler des données tronquées sans que rien
    ne le signale.
    """
    base = ARTIFACT_ROOT / "data"
    # `source="corpus"` désigne le mélange final de l'artefact, et non l'une de
    # ses sources d'entrée. C'est ce qu'on veut pour rejouer un corpus complet.
    sous_chemin = Path("corpus") if source == "corpus" else Path("sources") / source
    candidats = [
        p
        for p in sorted(base.glob(f"{artifact}-*"))
        if (p / "meta.json").is_file() and (p / sous_chemin).is_dir()
    ]
    if not candidats:
        raise FileNotFoundError(
            f"aucun artefact achevé {artifact!r} contenant {sous_chemin} sous {base}"
        )
    return max(candidats, key=lambda p: (p / "meta.json").stat().st_mtime) / sous_chemin
