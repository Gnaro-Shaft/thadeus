"""Pré-tokenisation — là où se joue l'essentiel du gain français.

La pré-tokenisation découpe le texte **avant** le BPE, et fixe une frontière que
le BPE ne franchira jamais : deux morceaux séparés ici ne pourront jamais
fusionner en un seul token, quel que soit le vocabulaire appris. C'est donc une
borne inférieure sur le nombre de tokens, décidée par une expression régulière
avant même le début de l'entraînement.

Le motif de GPT-2 a été écrit pour l'anglais. Il traite explicitement les
contractions anglaises (``'s``, ``'t``, ``'re``…) mais ignore les **élisions
françaises**, qui sont pourtant parmi les séquences les plus fréquentes de la
langue :

    " l'homme"  →  GPT-2 : [" l", "'", "homme"]     — 3 morceaux, plancher à 3 tokens
                →  nôtre : [" l'", "homme"]         — 2 morceaux

Sur un texte français courant, une élision apparaît toutes les quelques mots
(``l'``, ``d'``, ``j'``, ``n'``, ``qu'``, ``jusqu'``…). Corriger ce seul point
récupère un pourcentage à deux chiffres de tokens — c'est-à-dire du budget de
calcul, sans toucher au modèle.

Second ajustement, repris des modèles récents : **les nombres sont découpés par
tranches de trois chiffres au plus**. GPT-2 groupe une suite de chiffres
arbitrairement longue, ce qui fabrique un token dédié pour « 1997 », un autre
pour « 1998 »… et gaspille le vocabulaire sur des motifs qui n'apprennent rien.
"""

from __future__ import annotations

import regex as re

__all__ = [
    "ELISIONS",
    "FRENCH_LONGNUM_PATTERN",
    "FRENCH_PATTERN",
    "GPT2_PATTERN",
    "PATTERNS",
    "compare",
    "split",
]

# Élisions du français, de la plus longue à la plus courte : l'alternance d'une
# regex prend la **première** branche qui matche, donc « jusqu' » doit être
# essayé avant « j' », sinon on découperait " j" + "usqu'".
ELISIONS: tuple[str, ...] = (
    "aujourd",
    "quelqu",
    "lorsqu",
    "puisqu",
    "quoiqu",
    "presqu",
    "jusqu",
    "entr",
    "qu",
    "l",
    "d",
    "j",
    "n",
    "m",
    "t",
    "s",
    "c",
)

# Le motif historique de GPT-2, gardé comme référence de comparaison.
GPT2_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?\p{L}+"
    r"| ?\p{N}+"
    r"| ?[^\s\p{L}\p{N}]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def _french_pattern() -> str:
    elisions = "|".join(ELISIONS)
    return (
        # Contractions anglaises : le corpus est bilingue, on ne les perd pas.
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
        # Élisions françaises, avec l'apostrophe rattachée au mot-outil.
        rf"| ?(?i:{elisions})'"
        r"| ?\p{L}+"
        # Tranches de 3 chiffres maximum, pour ne pas gaspiller le vocabulaire.
        r"| ?\p{N}{1,3}"
        r"| ?[^\s\p{L}\p{N}]+"
        r"|\s+(?!\S)"
        r"|\s+"
    )


FRENCH_PATTERN = _french_pattern()

# Variante de contrôle : élisions françaises, mais découpage des nombres à la
# GPT-2. Isole l'effet des élisions de celui des nombres — deux changements
# mesurés ensemble ne s'attribuent pas.
FRENCH_LONGNUM_PATTERN = FRENCH_PATTERN.replace(r"\p{N}{1,3}", r"\p{N}+")

PATTERNS: dict[str, str] = {
    "gpt2": GPT2_PATTERN,
    "french": FRENCH_PATTERN,
    "french_longnum": FRENCH_LONGNUM_PATTERN,
}


def split(text: str, pattern: str = FRENCH_PATTERN) -> list[str]:
    """Applique un motif de pré-tokenisation, en Python.

    Sert à **inspecter et comparer** les motifs — c'est ainsi qu'on mesure le
    gain d'une règle avant de lancer le moindre entraînement. Le découpage réel
    est fait par le moteur Rust de ``tokenizers``, à partir du même motif.

    On utilise le module ``regex`` et non ``re`` : lui seul comprend les classes
    Unicode ``\\p{L}`` et ``\\p{N}``. Traduire ces classes vers des équivalents
    de la bibliothèque standard paraît tentant mais casse silencieusement les
    classes **niées** — ``[^\\s\\p{L}\\p{N}]`` deviendrait une classe imbriquée
    invalide, l'apostrophe cesserait d'être capturée, et l'outil de comparaison
    sous-estimerait le découpage sans rien signaler.
    """
    return re.findall(pattern, text)


def compare(text: str) -> dict[str, list[str]]:
    """Découpage du même texte par chaque motif connu — pour l'œil humain."""
    return {name: split(text, pattern) for name, pattern in PATTERNS.items()}
