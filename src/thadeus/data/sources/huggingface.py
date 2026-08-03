"""Source Hugging Face, en streaming.

Les corpus visés se comptent en téraoctets (FineWeb-2 français à lui seul) alors
qu'on n'a besoin que de quelques milliards de tokens. Télécharger puis filtrer
serait absurde : on lit en flux et on s'arrête quand on a notre compte.

Le corollaire à ne pas oublier : **les documents arrivent dans l'ordre du
dataset**, qui n'est pas aléatoire. Prendre les 200 000 premiers articles de
Wikipédia, c'est prendre une tranche corrélée (ordre d'identifiant, donc
d'ancienneté de création). D'où ``shuffle_buffer``, qui rend la tranche
représentative au prix d'un tampon en mémoire.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from thadeus.core.logs import get_logger
from thadeus.data.schema import Document
from thadeus.data.sources import SOURCES

log = get_logger(__name__)

# Champs portant le texte, essayés dans cet ordre. `complete_text` a été ajouté
# après que PleIAs/French-PD-Books a produit **zéro document en silence** : le
# champ ne figurait pas ici, `_pick` rendait None, et la boucle passait au
# document suivant sans un mot. Une source entière disparue sans erreur.
_TEXT_FIELDS = ("text", "content", "raw_content", "code", "complete_text")
# Champ portant un identifiant d'origine, s'il existe.
_ID_FIELDS = ("id", "_id", "url", "path", "repo_name")


def _pick(row: dict[str, Any], candidates: tuple[str, ...]) -> str | None:
    for key in candidates:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None


@SOURCES.register("huggingface", aliases=("hf",))
def from_huggingface(
    *,
    label: str,
    dataset: str,
    config: str | None = None,
    split: str = "train",
    lang: str = "unknown",
    limit: int | None = None,
    shuffle_buffer: int = 10_000,
    seed: int = 1337,
    text_field: str | None = None,
    keep_meta: tuple[str, ...] = (),
) -> Iterator[Document]:
    """Lit un dataset Hugging Face en flux.

    Args:
        label: nom logique de la source dans notre corpus ("wikipedia_fr").
            Distinct de ``dataset`` : on peut tirer deux sources de mêmes
            identifiants de dataset avec des configs différentes.
        dataset: identifiant HF ("wikimedia/wikipedia").
        config: sous-configuration ("20231101.fr", "fra_Latn").
        lang: langue déclarée, que le filtre de langue vérifiera ensuite.
        limit: nombre maximal de documents à produire.
        shuffle_buffer: taille du tampon de mélange. À 0, on prend la tranche
            brute — plus rapide, mais biaisée.
        keep_meta: champs du dataset à conserver. Volontairement vide par
            défaut : chaque champ gardé est réécrit pour chaque document.
    """
    from datasets import load_dataset  # import tardif : dépendance lourde

    log.info("Source %s : %s%s (limite=%s)", label, dataset, f":{config}" if config else "", limit)
    stream = load_dataset(dataset, config, split=split, streaming=True)
    if shuffle_buffer:
        stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)

    emitted = 0
    skipped = 0
    for index, row in enumerate(stream):
        text = row.get(text_field) if text_field else _pick(row, _TEXT_FIELDS)
        if not text:
            skipped += 1
            if skipped == 1:
                # Un champ texte introuvable est une erreur de config, pas un
                # document malformé : on le dit au premier, une seule fois.
                log.error(
                    "Source %s : aucun champ texte exploitable. Champs présents : %s. "
                    "Préciser `text_field` dans la config.",
                    label,
                    sorted(row),
                )
            continue

        origin = _pick(row, _ID_FIELDS) or str(index)
        meta = {k: row[k] for k in keep_meta if k in row}
        yield Document(
            id=f"{label}:{origin}",
            text=text,
            source=label,
            lang=lang,
            meta=meta,
        )

        emitted += 1
        if limit is not None and emitted >= limit:
            break

    if emitted == 0:
        raise RuntimeError(
            f"source {label!r} ({dataset}) n'a produit aucun document sur "
            f"{skipped + emitted} lignes lues. Cause la plus probable : le champ "
            f"texte n'est pas reconnu — préciser `text_field` dans la config."
        )
    log.info("Source %s : %d documents produits (%d lignes sans texte)", label, emitted, skipped)
