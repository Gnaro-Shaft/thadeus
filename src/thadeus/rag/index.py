"""Index BM25 sur le Vault — retrouver les passages pertinents.

**Pourquoi BM25 et pas des plongements denses.** La recherche dense est meilleure
sur les requêtes formulées autrement que le texte cible (« comment j'ai réglé le
souci de lenteur » → une note intitulée « Optimisation des requêtes SQL »). Mais
elle exige un modèle d'embedding à télécharger, à faire tourner, et dont la
qualité en français est à vérifier — soit une dépendance de plus dont on ne
mesurerait pas le comportement.

BM25 ne dépend de rien, tient en deux cents lignes, et se comporte très bien sur
un corpus personnel où l'on cherche généralement des termes qu'on a soi-même
écrits. C'est le bon point de départ : il donne une référence mesurable, et la
recherche dense pourra être comparée **contre** lui plutôt qu'adoptée sur la foi
de sa réputation.

**Ce que BM25 fait, en une phrase.** Il note un passage sur la rareté des mots
de la requête qu'il contient (un mot rare vaut plus qu'un mot fréquent), pondérée
par leur nombre d'occurrences, le tout corrigé pour que les passages longs
n'écrasent pas les courts.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from thadeus.core.logs import get_logger

__all__ = ["BM25Index", "Passage", "normalize_query", "tokenize"]

log = get_logger(__name__)

_WORD_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)

# Mots-outils français et anglais. Les retirer de l'index n'est pas de
# l'optimisation : ils apparaissent dans presque tous les passages, donc leur
# IDF est quasi nul et ils n'apportent aucun pouvoir discriminant — mais ils
# gonflent la longueur des documents, ce qui fausse la normalisation de BM25.
_STOPWORDS = frozenset(
    [
        "le",
        "la",
        "les",
        "de",
        "des",
        "du",
        "un",
        "une",
        "et",
        "en",
        "dans",
        "pour",
        "que",
        "qui",
        "ne",
        "pas",
        "plus",
        "sur",
        "au",
        "aux",
        "ce",
        "cette",
        "ces",
        "son",
        "sa",
        "ses",
        "est",
        "sont",
        "ete",
        "etre",
        "avec",
        "par",
        "il",
        "elle",
        "ils",
        "elles",
        "nous",
        "vous",
        "je",
        "tu",
        "on",
        "mais",
        "ou",
        "donc",
        "or",
        "ni",
        "car",
        "comme",
        "si",
        "tout",
        "tous",
        "toute",
        "toutes",
        "leur",
        "leurs",
        "se",
        "dont",
        "ou",
        "quand",
        "aussi",
        "bien",
        "peut",
        "faire",
        "fait",
        "avoir",
        "ont",
        "etait",
        "sera",
        "meme",
        "tres",
        "ainsi",
        "alors",
        "depuis",
        "entre",
        "sans",
        "sous",
        "chaque",
        "plusieurs",
        "autre",
        "autres",
        "the",
        "of",
        "and",
        "to",
        "in",
        "is",
        "that",
        "for",
        "it",
        "as",
        "was",
        "with",
        "be",
        "by",
        "on",
        "not",
        "this",
        "are",
        "or",
        "from",
        "at",
        "which",
        "have",
        "has",
        "had",
        "but",
        "they",
        "you",
        "all",
        "were",
        "we",
        "their",
        "been",
        "more",
        "when",
        "there",
        "can",
        "if",
        "would",
        "about",
        "them",
        "then",
        "some",
        "her",
        "she",
        "will",
        "what",
        "so",
        "no",
        "out",
        "up",
        "than",
        "into",
    ]
)


def _fold(text: str) -> str:
    """Retire les accents et met en minuscules.

    Indispensable sur un corpus personnel : on écrit « référence » dans une note
    et « reference » dans une requête tapée vite. Sans repli, ce sont deux mots
    différents et la note ne remonte pas.
    """
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Découpe en termes indexables, accents repliés, mots-outils retirés."""
    mots = _WORD_RE.findall(_fold(text))
    if keep_stopwords:
        return mots
    return [m for m in mots if m not in _STOPWORDS and len(m) > 1]


def normalize_query(query: str) -> list[str]:
    """Termes d'une requête. Même traitement que l'index, sans quoi rien ne matche."""
    return tokenize(query)


@dataclass
class Passage:
    """Un fragment de note, avec de quoi le retrouver et le citer."""

    id: str
    text: str
    source: str
    title: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "source": self.source, "title": self.title}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Passage:
        return cls(**payload)


@dataclass
class BM25Index:
    """Index inversé et scores BM25.

    Args:
        k1: saturation de la fréquence d'un terme. Au-delà de quelques
            occurrences, en ajouter n'apporte presque plus rien — un passage qui
            répète dix fois un mot n'est pas dix fois plus pertinent.
        b: force de la normalisation par la longueur. À 0, les passages longs
            sont avantagés (ils contiennent plus de mots) ; à 1, la correction
            est complète. 0,75 est le compromis usuel.
    """

    k1: float = 1.5
    b: float = 0.75
    # Poids du titre dans l'index. Une note nommée « Muon » doit remonter sur
    # « muon » même si le mot n'apparaît qu'une fois dans le corps.
    #
    # **Attention à la mesure** : évaluer la récupération avec des requêtes
    # tirées des titres, alors que les titres sont sur-pondérés, est en partie
    # circulaire. `scripts/rag_bench.py --title-weight 0` donne la performance
    # sur le corps seul, qui est la borne honnête.
    title_weight: int = 2

    passages: list[Passage] = field(default_factory=list)
    _postings: dict[str, dict[int, int]] = field(default_factory=lambda: defaultdict(dict))
    _lengths: list[int] = field(default_factory=list)
    _avg_length: float = 0.0

    def add(self, passage: Passage) -> None:
        index = len(self.passages)
        self.passages.append(passage)
        # Le titre est indexé **en plus** du corps : une note s'appelant
        # « Muon » doit remonter sur « muon » même si le mot n'apparaît qu'une
        # fois dans le texte.
        termes = tokenize(passage.text) + tokenize(passage.title) * self.title_weight
        for terme, n in Counter(termes).items():
            self._postings[terme][index] = n
        self._lengths.append(len(termes))

    def build(self) -> BM25Index:
        """Fige l'index et calcule la longueur moyenne."""
        self._avg_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        log.info(
            "Index BM25 : %d passages · %d termes distincts · longueur moyenne %.0f",
            len(self.passages),
            len(self._postings),
            self._avg_length,
        )
        return self

    def _idf(self, terme: str) -> float:
        """Rareté d'un terme.

        La variante « lissée » : le ``+ 1`` final évite qu'un terme présent dans
        plus de la moitié des passages reçoive un score négatif, ce qui le
        rendrait *pénalisant* — un comportement absurde qu'on rencontre dans les
        implémentations naïves.
        """
        n = len(self._postings.get(terme, ()))
        if not n:
            return 0.0
        total = len(self.passages)
        return math.log(1 + (total - n + 0.5) / (n + 0.5))

    def search(self, query: str, *, k: int = 3) -> list[tuple[Passage, float]]:
        """Les ``k`` passages les plus pertinents, du meilleur au moins bon."""
        termes = normalize_query(query)
        if not termes or not self.passages:
            return []

        scores: dict[int, float] = defaultdict(float)
        for terme in termes:
            postings = self._postings.get(terme)
            if not postings:
                continue
            idf = self._idf(terme)
            for index, tf in postings.items():
                longueur = self._lengths[index]
                norme = self.k1 * (1 - self.b + self.b * longueur / (self._avg_length or 1))
                scores[index] += idf * (tf * (self.k1 + 1)) / (tf + norme)

        meilleurs = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [(self.passages[i], s) for i, s in meilleurs]

    def save(self, path: str | Path) -> Path:
        """Sérialise l'index. Les postings sont reconstruits au chargement.

        On ne sauvegarde que les passages : réindexer 600 notes prend une
        seconde, alors qu'un index sérialisé se désynchronise silencieusement du
        Vault dès qu'une note change.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {"k1": self.k1, "b": self.b, "passages": [p.to_dict() for p in self.passages]},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> BM25Index:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        index = cls(k1=payload["k1"], b=payload["b"])
        for p in payload["passages"]:
            index.add(Passage.from_dict(p))
        return index.build()
