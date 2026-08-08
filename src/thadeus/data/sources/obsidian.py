"""Source Obsidian — les notes personnelles.

Cette source est minuscule à l'échelle d'un pré-entraînement (~1 M de tokens
pour le Vault de Genaro) et c'est un fait à garder en tête plutôt qu'à
contourner : lui donner un poids important dans le mélange reviendrait à la
répéter des dizaines de fois, c'est-à-dire à la faire apprendre par cœur.
Sa vraie place est le fine-tuning (Phase 8). Voir
:func:`~thadeus.data.mix.plan_mixture`, qui refuse silencieusement d'inventer
des données et signale toute source sur-répétée.

Le nettoyage effectué ici est spécifique au Markdown d'Obsidian : frontmatter
YAML, liens ``[[wiki]]``, blocs de code. On retire ce qui est du balisage pur et
on garde ce qui est du texte.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from pathlib import Path

from thadeus.core.logs import get_logger
from thadeus.data.schema import Document
from thadeus.data.sources import SOURCES

log = get_logger(__name__)

_FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_WIKILINK = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_EMBED = re.compile(r"!\[\[[^\]]+\]\]")
_TAG_LINE = re.compile(r"^#[\w/-]+(?:\s+#[\w/-]+)*\s*$", re.MULTILINE)


def strip_markdown(text: str, *, keep_code: bool = True) -> str:
    """Retire le balisage Obsidian en préservant le texte.

    Les liens ``[[Note|libellé]]`` deviennent leur libellé : c'est le mot que
    l'auteur voulait lire, et le seul porteur de sens. Les images intégrées
    disparaissent — un nom de fichier n'apprend rien à un modèle de langue.
    """
    text = _FRONTMATTER.sub("", text)
    text = _EMBED.sub("", text)
    text = _WIKILINK.sub(lambda m: m.group(2) or m.group(1), text)
    text = _MD_LINK.sub(lambda m: m.group(1), text)
    text = _TAG_LINE.sub("", text)
    if not keep_code:
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    return text.strip()


@SOURCES.register("obsidian")
def from_obsidian(
    *,
    label: str = "obsidian",
    vault: str,
    lang: str = "fr",
    exclude: tuple[str, ...] = (".obsidian", ".trash", "Templates"),
    min_words: int = 30,
    keep_code: bool = True,
    split: str = "all",
    val_fraction: float = 0.15,
    limit: int | None = None,
) -> Iterator[Document]:
    """Lit les notes Markdown d'un Vault Obsidian.

    Args:
        vault: racine du Vault.
        exclude: fragments de chemin à ignorer. ``Templates`` en fait partie :
            des gabarits à trous sont du bruit structuré, exactement le genre de
            texte répétitif qu'on passe l'étage suivant à éliminer.
        min_words: seuil sous lequel une note est ignorée. Un Vault contient
            beaucoup de fragments d'une ligne sans valeur d'entraînement.
        split: ``"all"``, ``"train"`` ou ``"val"``. Le partage se fait par
            **hachage du chemin de la note**, donc il est stable entre
            exécutions et ne demande aucun fichier d'index. Ajouter des notes
            au Vault ne redistribue pas les anciennes.
        val_fraction: part réservée à la validation.

    **Pourquoi un split est indispensable ici.** Le fine-tuning porte sur ~1 M
    de tokens : à ce volume, un modèle peut *mémoriser* le corpus au lieu d'en
    apprendre le style. Sans notes tenues à l'écart, on ne peut pas distinguer
    les deux — et une perplexité qui s'effondre sur les données vues ressemble
    exactement à une réussite.
    """
    if split not in ("all", "train", "val"):
        raise ValueError(f"split inconnu : {split!r} (attendu 'all', 'train' ou 'val')")
    root = Path(vault).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Vault introuvable : {root}")

    emitted = 0
    for path in sorted(root.rglob("*.md")):
        relative = path.relative_to(root)
        if any(part in exclude for part in relative.parts):
            continue

        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            log.warning("note illisible ignorée : %s", relative)
            continue

        if split != "all" and _is_val(relative.as_posix(), val_fraction) != (split == "val"):
            continue

        text = strip_markdown(raw, keep_code=keep_code)
        if len(text.split()) < min_words:
            continue

        yield Document(
            id=f"{label}:{relative.as_posix()}",
            text=text,
            source=label,
            lang=lang,
            meta={"path": relative.as_posix()},
        )
        emitted += 1
        if limit is not None and emitted >= limit:
            break

    log.info("Source %s : %d notes retenues depuis %s (split=%s)", label, emitted, root, split)


def _is_val(chemin: str, fraction: float) -> bool:
    """Une note appartient-elle à la validation ?

    Décidé par hachage du chemin, jamais par tirage aléatoire : le partage doit
    être identique d'une exécution à l'autre, et rester stable quand de
    nouvelles notes s'ajoutent au Vault.
    """
    digest = hashlib.blake2b(chemin.encode("utf-8"), digest_size=8).digest()
    return (int.from_bytes(digest, "big") % 10_000) < fraction * 10_000
