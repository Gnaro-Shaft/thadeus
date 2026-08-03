"""Sondes ciblées — ce qu'une perte moyenne ne distingue pas.

Une perplexité qui passe de 18,4 à 18,2 ne dit pas **ce que** le modèle a
appris. A-t-il progressé sur les accords ? Sur les élisions ? S'est-il dégradé
sur le code pendant qu'il s'améliorait sur la prose ? La moyenne ne répond à
aucune de ces questions.

Les sondes y répondent par des **paires minimales** : deux phrases identiques à
un détail près, l'une correcte, l'autre non. On demande au modèle laquelle est
la plus probable. Le résultat est un pourcentage de bonnes réponses par
catégorie grammaticale — interprétable, comparable entre modèles, et surtout
**diagnostique** : si le modèle échoue sur les accords au pluriel, on sait quoi
regarder.

Le hasard donne 50 %. Un modèle de 188 M sur 3,76 Md de tokens de français
devrait dépasser 80 % sur les accords simples et l'élision. Rester à 50 % sur
une catégorie signale un problème réel, invisible dans la courbe de perte.

**Précaution de méthode.** Les deux phrases d'une paire doivent être aussi
proches que possible en longueur : une phrase plus longue a mécaniquement une
probabilité totale plus faible, et on mesurerait alors la longueur plutôt que
la grammaire.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from thadeus.core.logs import get_logger

__all__ = ["MinimalPair", "PROBES", "ProbeResult", "run_probes"]

log = get_logger(__name__)


@dataclass(frozen=True)
class MinimalPair:
    """Deux phrases identiques sauf sur un point de grammaire."""

    category: str
    good: str
    bad: str


# Paires françaises. Chaque catégorie teste une compétence isolée, et le point
# de divergence est toujours situé **après** un contexte suffisant : mettre
# l'erreur au premier mot ne testerait rien, le modèle n'ayant encore rien lu.
PROBES: tuple[MinimalPair, ...] = (
    # Accord sujet-verbe au pluriel — la faute la plus révélatrice d'un modèle
    # qui n'a pas appris à porter l'information sur plusieurs mots.
    MinimalPair("accord_verbe", "Les enfants mangent une pomme.", "Les enfants mange une pomme."),
    MinimalPair(
        "accord_verbe", "Mes voisins partent en vacances.", "Mes voisins part en vacances."
    ),
    MinimalPair("accord_verbe", "Ces machines fonctionnent bien.", "Ces machines fonctionne bien."),
    MinimalPair(
        "accord_verbe", "Les résultats semblent corrects.", "Les résultats semble corrects."
    ),
    MinimalPair("accord_verbe", "Le chien dort sur le tapis.", "Le chien dorment sur le tapis."),
    # Accord de l'adjectif en genre et en nombre.
    MinimalPair("accord_adjectif", "une grande maison blanche", "une grand maison blanc"),
    MinimalPair("accord_adjectif", "des voitures rouges et neuves", "des voitures rouge et neuf"),
    MinimalPair(
        "accord_adjectif", "cette longue journée difficile", "cette long journée difficile"
    ),
    MinimalPair("accord_adjectif", "un petit village tranquille", "un petite village tranquille"),
    # Élision — le point que notre pré-tokenisation traite spécifiquement.
    # Si le modèle échoue ici, c'est que le travail de la Phase 2 n'a pas payé.
    MinimalPair(
        "elision", "L'homme qui arrive est mon frère.", "Le homme qui arrive est mon frère."
    ),
    MinimalPair("elision", "Il n'a pas encore répondu.", "Il ne a pas encore répondu."),
    MinimalPair("elision", "Je pense qu'elle viendra demain.", "Je pense que elle viendra demain."),
    MinimalPair("elision", "C'est l'histoire d'une famille.", "Ce est le histoire de une famille."),
    # Genre des déterminants.
    MinimalPair("genre", "le tableau est accroché au mur", "la tableau est accroché au mur"),
    MinimalPair("genre", "une décision difficile à prendre", "un décision difficile à prendre"),
    MinimalPair("genre", "cette question reste ouverte", "cet question reste ouverte"),
    # Pluriels irréguliers — mémorisation lexicale plutôt que règle.
    MinimalPair("pluriel", "Les chevaux courent dans le pré.", "Les chevals courent dans le pré."),
    MinimalPair("pluriel", "Il a acheté trois journaux.", "Il a acheté trois journals."),
    MinimalPair("pluriel", "Les travaux durent tout l'été.", "Les travails durent tout l'été."),
    # Ordre des mots — la syntaxe la plus élémentaire.
    MinimalPair("ordre", "Le chat noir dort paisiblement.", "Le noir chat dort paisiblement."),
    MinimalPair("ordre", "Elle lui a donné un livre.", "Elle a lui donné un livre."),
    # Prépositions et locutions figées.
    MinimalPair(
        "preposition", "Il habite à Paris depuis dix ans.", "Il habite en Paris depuis dix ans."
    ),
    MinimalPair("preposition", "Je vais au marché ce matin.", "Je vais à le marché ce matin."),
    # Code — syntaxe Python élémentaire.
    MinimalPair("code", "def somme(a, b):\n    return a + b", "def somme(a, b)\n    return a + b"),
    MinimalPair("code", "for i in range(10):\n    print(i)", "for i in range(10)\n    print i"),
    MinimalPair("code", "if x is not None:\n    x.close()", "if x is not None\n    x.close("),
)


@dataclass
class ProbeResult:
    """Résultat par catégorie."""

    category: str
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "correct": self.correct,
            "total": self.total,
            "accuracy": round(self.accuracy, 3),
        }


@torch.no_grad()
def sequence_logprob(model: torch.nn.Module, codec: Any, text: str, device, dtype) -> float:
    """Log-probabilité **totale** de la séquence.

    Totale et non moyenne par token : les deux phrases d'une paire minimale
    décrivent le même contenu, donc leurs probabilités totales sont directement
    comparables. Normaliser par la longueur avantagerait mécaniquement la phrase
    qui produit le plus de tokens — souvent la version fautive, justement parce
    qu'elle est mal découpée.
    """
    ids = codec.encode(text)
    if len(ids) < 2:
        return float("-inf")
    x = torch.tensor([ids[:-1]], device=device)
    y = torch.tensor([ids[1:]], device=device)
    with torch.autocast(device.type, dtype=dtype):
        logits, _ = model(x)
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    chosen = logprobs.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return float(chosen.sum())


def run_probes(
    model: torch.nn.Module,
    codec: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
    pairs: Sequence[MinimalPair] = PROBES,
) -> dict[str, ProbeResult]:
    """Évalue toutes les sondes et agrège par catégorie.

    Returns:
        Un résultat par catégorie. Le hasard donne 50 % ; en dessous de 60 %
        après un vrai entraînement, la compétence n'est pas acquise.
    """
    was_training = model.training
    model.eval()

    resultats: dict[str, ProbeResult] = {}
    for pair in pairs:
        bon = sequence_logprob(model, codec, pair.good, device, dtype)
        mauvais = sequence_logprob(model, codec, pair.bad, device, dtype)
        res = resultats.setdefault(pair.category, ProbeResult(pair.category, 0, 0))
        resultats[pair.category] = ProbeResult(
            pair.category, res.correct + int(bon > mauvais), res.total + 1
        )

    if was_training:
        model.train()
    return resultats


def format_probes(resultats: dict[str, ProbeResult]) -> str:
    """Rend les résultats lisibles, avec le total."""
    lignes = [f"{'sonde':<20}{'justes':>9}{'total':>7}{'taux':>9}", "-" * 45]
    for nom, res in sorted(resultats.items(), key=lambda kv: -kv[1].accuracy):
        marque = "  ← hasard" if res.accuracy <= 0.55 else ""
        lignes.append(f"{nom:<20}{res.correct:>9}{res.total:>7}{100 * res.accuracy:>8.0f}%{marque}")
    correct = sum(r.correct for r in resultats.values())
    total = sum(r.total for r in resultats.values())
    lignes.append("-" * 45)
    lignes.append(f"{'TOTAL':<20}{correct:>9}{total:>7}{100 * correct / total:>8.0f}%")
    lignes.append("\nle hasard donne 50 % · en dessous de 60 %, la compétence n'est pas acquise")
    return "\n".join(lignes)
