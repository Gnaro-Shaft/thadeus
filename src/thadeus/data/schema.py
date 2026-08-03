"""Le document — unité de circulation de toute la chaîne de données.

Chaque source, aussi différente soit-elle (Wikipédia, du code GitHub, une note
Obsidian), produit des :class:`Document`. Tout ce qui vient après — filtres,
déduplication, mélange, tokenizer — ne connaît que ce type. C'est cette
uniformisation qui permet d'ajouter une source sans toucher au reste.

Le champ ``source`` est conservé jusqu'au bout et c'est délibéré : sans lui,
impossible de dire *quelle* source un filtre est en train de décimer, ni de
mesurer la composition réelle du corpus final par rapport à celle qu'on avait
demandée.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Self

__all__ = ["Document", "estimate_tokens", "format_tokens", "text_fingerprint"]

_WORD_RE = re.compile(r"\S+")

# Facteur mots -> tokens, avant d'avoir notre tokenizer (Phase 2).
# ~1,45 pour du français avec un BPE générique : le français est plus fragmenté
# que l'anglais, ce qui est précisément le gâchis que le tokenizer maison
# viendra corriger. Sert uniquement à dimensionner, jamais à décider.
TOKENS_PER_WORD = 1.45


@dataclass(frozen=True, slots=True)
class Document:
    """Un document normalisé.

    Args:
        id: identifiant stable ``"source:clé"``. Stable veut dire : rejouer la
            chaîne redonne le même identifiant, ce qui rend la déduplication et
            les reprises idempotentes.
        text: le contenu, déjà normalisé si le document a traversé les filtres.
        source: nom logique de la source ("wikipedia_fr", "obsidian").
        lang: "fr", "en", "code"... tel que déclaré par la source, puis
            éventuellement corrigé par le filtre de langue.
        meta: métadonnées libres, gardées légères — elles sont écrites pour
            chaque document et un champ inutile coûte des gigaoctets.
    """

    id: str
    text: str
    source: str
    lang: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_words(self) -> int:
        return len(_WORD_RE.findall(self.text))

    @property
    def n_chars(self) -> int:
        return len(self.text)

    def with_text(self, text: str) -> Self:
        """Copie avec un texte remplacé — les documents sont immuables.

        L'immuabilité n'est pas un dogme : elle garantit qu'un filtre ne peut
        pas modifier en douce un document que la statistique de l'étape
        précédente a déjà comptabilisé.
        """
        return replace(self, text=text)

    def to_json(self) -> str:
        payload: dict[str, Any] = {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "lang": self.lang,
        }
        if self.meta:
            payload["meta"] = self.meta
        return json.dumps(payload, ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> Document:
        payload = json.loads(line)
        return cls(
            id=payload["id"],
            text=payload["text"],
            source=payload["source"],
            lang=payload.get("lang", "unknown"),
            meta=payload.get("meta", {}),
        )


def estimate_tokens(n_words: int) -> int:
    """Estime un nombre de tokens à partir d'un nombre de mots.

    Approximation grossière et assumée : elle sert à dimensionner un corpus
    avant que le tokenizer existe. Le compte exact viendra en Phase 2, et
    l'écart entre les deux mesurera précisément ce que le tokenizer maison
    aura fait gagner.
    """
    return int(n_words * TOKENS_PER_WORD)


def format_tokens(count: float) -> str:
    """Formate un nombre de tokens dans l'unité qui le rend lisible.

    Un corpus de fumée de 3 M tokens affiché « 0.00 Md » n'informe personne, et
    un journal qu'on ne lit plus ne sert à rien.
    """
    if count >= 1e9:
        return f"{count / 1e9:.2f} Md"
    if count >= 1e6:
        return f"{count / 1e6:.2f} M"
    if count >= 1e3:
        return f"{count / 1e3:.1f} k"
    return str(int(count))


def text_fingerprint(text: str) -> str:
    """Empreinte d'un texte, insensible aux différences d'espacement et de casse.

    Première ligne de défense contre les doublons : deux copies d'un même
    article qui ne diffèrent que par la mise en forme donnent la même empreinte.
    Beaucoup moins coûteux que le MinHash, qui ne sert ensuite qu'aux
    quasi-doublons.
    """
    normalized = " ".join(text.lower().split())
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).hexdigest()
