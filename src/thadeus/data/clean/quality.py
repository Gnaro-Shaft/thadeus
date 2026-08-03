"""Filtres de qualité — l'un des trois paris du projet.

Rappel de la thèse : à budget de calcul minuscule, on gagne par la **densité
d'information par FLOP**. Chaque document de mauvaise qualité conservé est un
budget d'entraînement dépensé à apprendre du bruit. Ces filtres sont donc un
levier de performance, pas une opération de ménage.

Les heuristiques viennent de Gopher (DeepMind) et C4 (Google), adaptées au
français. Trois familles :

- **Longueur et forme** — un document trop court n'apprend rien, un document
  aux mots anormalement longs ou courts n'est pas du texte courant.
- **Répétition** — le symptôme le plus fiable du contenu généré, des pages de
  navigation et du spam. Un texte humain ne se répète pas.
- **Balisage résiduel** — menus, mentions légales, bandeaux de cookies.

Chaque filtre est réglable depuis la config, et chacun rend compte de ce qu'il
rejette (voir :class:`~thadeus.data.clean.CleaningPipeline`). Un seuil ne se
choisit pas dans l'absolu : on le règle en regardant le taux de rejet qu'il
produit sur *notre* corpus.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from thadeus.data.clean import FILTERS, Filter
from thadeus.data.clean.language import detect_language
from thadeus.data.schema import Document

__all__ = ["normalize_text"]

_WORD_RE = re.compile(r"\S+")
_ALPHA_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MANY_BLANKS_RE = re.compile(r"\n{3,}")
_SPACES_RE = re.compile(r"[ \t]{2,}")

# Apostrophes et guillemets typographiques -> forme ASCII.
# Choix important pour le français : « l'homme » avec U+2019 et avec U+0027 sont
# deux séquences distinctes pour un tokenizer BPE, qui apprendrait deux fois le
# même motif. Unifier, c'est rendre du vocabulaire disponible pour autre chose.
_TYPOGRAPHIC = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def normalize_text(text: str, *, unify_apostrophes: bool = True) -> str:
    """Normalise un texte sans en altérer le contenu.

    NFC recompose les caractères accentués : « é » peut être codé sur un ou deux
    points de code, et sans normalisation le tokenizer traiterait ces deux
    formes comme différentes — un gâchis pur sur un corpus français.
    """
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL_RE.sub("", text)
    if unify_apostrophes:
        text = text.translate(_TYPOGRAPHIC)
    text = _SPACES_RE.sub(" ", text)
    text = _MANY_BLANKS_RE.sub("\n\n", text)
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


@FILTERS.register("normalize")
def make_normalize(*, unify_apostrophes: bool = True) -> Filter:
    """Normalisation Unicode et espacement. À placer en tête de chaîne."""

    def step(doc: Document) -> Document | None:
        text = normalize_text(doc.text, unify_apostrophes=unify_apostrophes)
        return doc.with_text(text) if text else None

    return step


@FILTERS.register("word_count")
def make_word_count(*, min_words: int = 50, max_words: int = 100_000) -> Filter:
    """Borne la longueur en mots.

    Le plancher élimine les fragments sans contexte suffisant pour apprendre
    quoi que ce soit. Le plafond attrape les concaténations accidentelles et les
    vidages de base de données.
    """

    def step(doc: Document) -> Document | None:
        return doc if min_words <= doc.n_words <= max_words else None

    return step


@FILTERS.register("mean_word_length")
def make_mean_word_length(*, lo: float = 3.0, hi: float = 10.0) -> Filter:
    """Longueur moyenne des mots.

    En dessous du plancher : listes de sigles, tableaux de chiffres. Au-dessus
    du plafond : texte sans espaces, base64, identifiants. Le français tourne
    autour de 4,8 — sensiblement plus que l'anglais, d'où un plancher qui
    serait mal placé si on reprenait tel quel un seuil calibré sur l'anglais.
    """

    def step(doc: Document) -> Document | None:
        words = _WORD_RE.findall(doc.text)
        if not words:
            return None
        mean = sum(len(w) for w in words) / len(words)
        return doc if lo <= mean <= hi else None

    return step


@FILTERS.register("alpha_ratio")
def make_alpha_ratio(*, min_ratio: float = 0.75) -> Filter:
    """Fraction de mots contenant au moins une lettre.

    Rejette les tableaux de données, les listes de références et les journaux
    applicatifs, où la majorité des « mots » sont des nombres ou des symboles.
    """

    def step(doc: Document) -> Document | None:
        words = _WORD_RE.findall(doc.text)
        if not words:
            return None
        alpha = sum(1 for w in words if _ALPHA_RE.search(w))
        return doc if alpha / len(words) >= min_ratio else None

    return step


@FILTERS.register("symbol_ratio")
def make_symbol_ratio(*, max_ratio: float = 0.1, symbols: str = "#…") -> Filter:
    """Densité de symboles parasites (dièses de balisage, points de suspension).

    Une forte densité signale un texte tronqué par un aperçu (« lire la
    suite… ») ou du balisage non converti.
    """

    def step(doc: Document) -> Document | None:
        words = _WORD_RE.findall(doc.text)
        if not words:
            return None
        count = sum(doc.text.count(symbol) for symbol in symbols)
        count += doc.text.count("...")
        return doc if count / len(words) <= max_ratio else None

    return step


@FILTERS.register("stopword_density")
def make_stopword_density(*, lang: str = "fr", min_density: float = 0.05) -> Filter:
    """Exige une densité minimale de mots-outils.

    Le test de « est-ce de la prose ? » le plus efficace pour son coût. Une
    liste de produits, un nuage de mots-clés ou une table des matières
    contiennent des mots de la langue mais presque aucun mot grammatical.
    """
    from thadeus.data.clean.language import language_scores

    def step(doc: Document) -> Document | None:
        return doc if language_scores(doc.text).get(lang, 0.0) >= min_density else None

    return step


@FILTERS.register("language_is")
def make_language_is(*, lang: str = "fr", min_score: float = 0.05, margin: float = 1.2) -> Filter:
    """Vérifie la langue et corrige l'étiquette.

    Args:
        margin: facteur par lequel la langue attendue doit dominer la suivante.
            Sans cette marge, un texte franco-anglais passerait sur un écart
            négligeable — or c'est exactement ce qu'on veut écarter d'un corpus
            « français d'abord ».
    """
    from thadeus.data.clean.language import language_scores

    def step(doc: Document) -> Document | None:
        scores = language_scores(doc.text)
        target = scores.get(lang, 0.0)
        if target < min_score:
            return None
        others = [v for k, v in scores.items() if k != lang]
        if others and target < margin * max(others):
            return None
        return doc if doc.lang == lang else Document(
            id=doc.id, text=doc.text, source=doc.source, lang=lang, meta=doc.meta
        )

    return step


@FILTERS.register("duplicate_lines")
def make_duplicate_lines(*, max_ratio: float = 0.3, min_length: int = 10) -> Filter:
    """Fraction de lignes dupliquées dans le document.

    Signature des pages de navigation, des tableaux répétitifs et du contenu
    assemblé automatiquement. Les lignes très courtes sont ignorées : dans du
    texte structuré, quelques puces identiques sont normales.
    """

    def step(doc: Document) -> Document | None:
        lines = [line.strip() for line in doc.text.split("\n")]
        lines = [line for line in lines if len(line) >= min_length]
        if len(lines) < 4:
            return doc
        unique = len(set(lines))
        return doc if 1 - unique / len(lines) <= max_ratio else None

    return step


@FILTERS.register("repeated_ngrams")
def make_repeated_ngrams(*, n: int = 10, max_ratio: float = 0.2) -> Filter:
    """Fraction du texte couverte par le n-gramme le plus répété.

    Le filtre anti-répétition de Gopher, et le plus discriminant du lot. Un
    texte humain ne répète jamais une séquence de 10 mots. Quand ça arrive,
    c'est du contenu généré, du spam, ou un pied de page recopié à chaque
    section.
    """

    def step(doc: Document) -> Document | None:
        words = _WORD_RE.findall(doc.text.lower())
        if len(words) < 2 * n:
            return doc
        grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
        _, top = grams.most_common(1)[0]
        return doc if (top * n) / len(words) <= max_ratio else None

    return step


@FILTERS.register("boilerplate")
def make_boilerplate(
    *,
    terms: tuple[str, ...] = (
        "conditions générales",
        "politique de confidentialité",
        "tous droits réservés",
        "accepter les cookies",
        "javascript est désactivé",
        "veuillez activer javascript",
        "lorem ipsum",
    ),
    max_hits: int = 1,
) -> Filter:
    """Rejette les pages dominées par des mentions légales ou des bandeaux.

    ``max_hits`` à 1 tolère une occurrence isolée : un article *à propos* des
    cookies est du contenu légitime, une page qui n'est qu'un bandeau ne l'est
    pas.
    """
    lowered = tuple(t.lower() for t in terms)

    def step(doc: Document) -> Document | None:
        haystack = doc.text.lower()
        hits = sum(1 for term in lowered if term in haystack)
        return doc if hits <= max_hits else None

    return step


@FILTERS.register("drop_short_lines")
def make_drop_short_lines(*, min_words: int = 3, keep_ratio: float = 0.5) -> Filter:
    """Retire les lignes trop courtes, et rejette si le document n'en survit pas.

    Nettoie les restes de menus et de fils d'Ariane. Le garde-fou
    ``keep_ratio`` évite le piège du filtre trop zélé : si plus de la moitié du
    document part, ce n'était pas un article avec un menu, c'était un menu.
    """

    def step(doc: Document) -> Document | None:
        lines = doc.text.split("\n")
        kept = [line for line in lines if len(line.split()) >= min_words or not line.strip()]
        text = normalize_text("\n".join(kept))
        if not text:
            return None
        if len(text) < keep_ratio * len(doc.text):
            return None
        return doc.with_text(text)

    return step


@FILTERS.register("detect_lang")
def make_detect_lang(*, min_score: float = 0.0) -> Filter:
    """Étiquette la langue sans rejeter — utile pour inspecter un corpus."""

    def step(doc: Document) -> Document | None:
        lang, score = detect_language(doc.text)
        if score < min_score:
            return doc
        return Document(id=doc.id, text=doc.text, source=doc.source, lang=lang, meta=doc.meta)

    return step
