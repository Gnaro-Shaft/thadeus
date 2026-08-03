"""Identification de langue par mots-outils.

Choix technique assumé, et son coût. Les sources retenues sont **déjà**
filtrées par langue en amont (Wikipédia français par construction, FineWeb-2
``fra_Latn`` par son propre classifieur). Le rôle de ce module n'est donc pas de
classer, mais de **vérifier** — attraper l'anglais résiduel, les pages
multilingues et les listes de noms propres qui traversent les filtres amont.

Pour cet usage, un classifieur par mots-outils suffit : les mots grammaticaux
sont les plus fréquents d'une langue, quasi absents des autres, et leur densité
est un signal robuste. Il ne demande aucune dépendance ni aucun modèle à
télécharger.

Ses deux limites, mesurées et à connaître avant de le réutiliser ailleurs :

- **Textes courts** — en dessous de ~30 mots, trop peu de mots-outils pour
  trancher. D'où :data:`_MIN_TOKENS`, qui fait répondre « je ne sais pas ».
- **Le code obtient un score anglais élevé** (~0,25 sur du Python courant), parce
  que ``for``, ``in`` et ``if`` sont à la fois des mots-clés de programmation et
  des mots-outils anglais. Un classifieur par mots-outils ne peut structurellement
  pas les distinguer. Sans conséquence dans notre chaîne — les sources de code
  n'appliquent aucun filtre de langue, et le score **français** du code est nul,
  ce qui protège le corpus « français d'abord ». Mais c'est un piège pour tout
  autre usage.
"""

from __future__ import annotations

import re

__all__ = ["detect_language", "language_scores"]

_TOKEN_RE = re.compile(r"[a-zà-öø-ÿœæ']+", re.IGNORECASE)

# Mots-outils les plus fréquents. Le recouvrement entre les deux listes est
# volontairement nul : un mot partagé (« a », « on », « son ») n'apporte aucune
# information discriminante et ne ferait qu'ajouter du bruit aux deux scores.
_STOPWORDS: dict[str, frozenset[str]] = {
    "fr": frozenset(
        """le la les des du une et en dans pour que qui ne pas plus sur aux
        cette ces sa ses est sont été être avec par il elle ils elles nous vous
        je mais ou donc car comme si tout tous toute toutes leur leurs se dont
        où quand aussi bien peut faire fait avoir ont était sera même très ainsi
        alors depuis entre sans sous chaque plusieurs autre autres celui celle
        ceux notamment lors afin dès déjà encore toujours jamais puis ensuite
        cependant néanmoins pourtant lorsque parce pendant avant après""".split()
    ),
    "en": frozenset(
        """the of and to in is that for it as was with be by not he this are
        have has had but they you all were we her from which their been more
        when there can if would about them then some she will what so no out
        up than into only other new could time these two may first any these
        such through because while during before after both each many most
        those where does did doing having its""".split()
    ),
}

_MIN_TOKENS = 20


def language_scores(text: str, *, max_tokens: int = 2000) -> dict[str, float]:
    """Densité de mots-outils par langue, dans ``[0, 1]``.

    Args:
        max_tokens: on ne lit que le début du document. Un préfixe de 2000 mots
            suffit largement à trancher, et éviter de parcourir des articles de
            50 000 mots change l'ordre de grandeur du temps de traitement sur
            plusieurs millions de documents.
    """
    tokens = _TOKEN_RE.findall(text.lower())[:max_tokens]
    if len(tokens) < _MIN_TOKENS:
        return dict.fromkeys(_STOPWORDS, 0.0)

    total = len(tokens)
    return {
        lang: sum(1 for tok in tokens if tok in words) / total
        for lang, words in _STOPWORDS.items()
    }


def detect_language(text: str) -> tuple[str, float]:
    """Langue la plus probable et son score.

    Retourne ``("unknown", 0.0)`` quand le texte est trop court ou qu'aucune
    langue ne se détache. Renvoyer « je ne sais pas » plutôt qu'un pari est
    délibéré : le filtre appelant décide alors de rejeter, là où une réponse
    inventée ferait entrer du bruit dans le corpus.
    """
    scores = language_scores(text)
    best = max(scores, key=lambda lang: scores[lang])
    return (best, scores[best]) if scores[best] > 0 else ("unknown", 0.0)
