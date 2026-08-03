"""Sources de documents — un fichier par famille de source.

Une source est un callable qui rend un flux de :class:`~thadeus.data.schema.Document`.
Elle est enregistrée sous un nom et déclarée en config :

    [[sources]]
    name = "huggingface"
    label = "wikipedia_fr"
    dataset = "wikimedia/wikipedia"
    config = "20231101.fr"
    limit = 200_000

Deux contraintes que toute source doit respecter :

- **Produire en flux, jamais en liste.** Certaines sources font des téraoctets ;
  on n'en veut qu'une tranche, et on doit pouvoir s'arrêter à tout moment.
- **Donner des identifiants stables.** Rejouer la collecte doit redonner les
  mêmes ``id``, sinon la déduplication et les reprises deviennent illusoires.
"""

from __future__ import annotations

from collections.abc import Iterator

from thadeus.core.registry import Registry
from thadeus.data.schema import Document

SOURCES: Registry[Iterator[Document]] = Registry("source")

from thadeus.data.sources import gutenberg, huggingface, local, obsidian  # noqa: E402,F401

__all__ = ["SOURCES", "gutenberg", "huggingface", "local", "obsidian"]
