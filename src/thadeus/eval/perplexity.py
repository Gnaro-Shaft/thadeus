"""Perplexité — la mesure de base, et ses pièges.

La perplexité est l'exponentielle de la perte : ``exp(2,5) ≈ 12,2`` se lit
« le modèle hésite comme s'il choisissait entre 12 tokens équiprobables ». On
préfère souvent la perte pour l'optimisation, et la perplexité pour en parler.

**Trois pièges, tous rencontrés dans des projets réels.**

1. **Une perplexité globale ne veut rien dire.** Un modèle peut s'améliorer en
   moyenne tout en se dégradant sur le français, simplement parce que le code
   pèse plus dans le mélange. On ventile donc **par source** et **par langue**,
   toujours.

2. **Deux perplexités ne sont comparables que sur le même tokenizer.** Un
   tokenizer plus efficace produit moins de tokens pour le même texte, donc une
   perplexité par token plus élevée à qualité égale. C'est pourquoi on rapporte
   aussi les **bits par caractère**, qui sont indépendants du découpage — la
   seule métrique qui permette de comparer deux modèles à tokenizers différents.

3. **Une évaluation doit porter sur exactement les mêmes fenêtres à chaque
   fois.** Échantillonner au hasard mélange le progrès du modèle et la variance
   du tirage, et rend deux évaluations successives incomparables.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch

from thadeus.core.logs import get_logger

__all__ = ["Score", "evaluate_documents"]

log = get_logger(__name__)

_LN2 = math.log(2)


@dataclass
class Score:
    """Comptages bruts et métriques dérivées.

    On accumule des **sommes**, pas des moyennes : deux évaluations ne
    s'additionnent que sur leurs comptages. Faire la moyenne de moyennes
    pondère chaque lot également, quel que soit son nombre de tokens.
    """

    total_loss: float = 0.0
    n_tokens: int = 0
    n_chars: int = 0
    n_documents: int = 0

    @property
    def loss(self) -> float:
        """Perte moyenne par token — ce qu'on optimise."""
        return self.total_loss / self.n_tokens if self.n_tokens else float("nan")

    @property
    def perplexity(self) -> float:
        """Exponentielle de la perte. Bornée pour rester lisible en début de run."""
        return math.exp(min(self.loss, 20)) if self.n_tokens else float("nan")

    @property
    def bits_per_char(self) -> float:
        """Bits par caractère — **indépendant du tokenizer**.

        La seule métrique qui permette de comparer deux modèles dont les
        tokenizers diffèrent. Un tokenizer plus efficace gonfle mécaniquement la
        perplexité par token sans que le modèle soit moins bon ; les bits par
        caractère ne bougent pas.
        """
        if not self.n_chars:
            return float("nan")
        return (self.total_loss / _LN2) / self.n_chars

    def __add__(self, other: Score) -> Score:
        return Score(
            total_loss=self.total_loss + other.total_loss,
            n_tokens=self.n_tokens + other.n_tokens,
            n_chars=self.n_chars + other.n_chars,
            n_documents=self.n_documents + other.n_documents,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": self.n_documents,
            "tokens": self.n_tokens,
            "loss": round(self.loss, 4),
            "perplexity": round(self.perplexity, 2),
            "bits_per_char": round(self.bits_per_char, 4),
        }


@dataclass
class GroupedScore:
    """Scores ventilés, plus le total."""

    groups: dict[str, Score] = field(default_factory=dict)

    @property
    def overall(self) -> Score:
        return sum(self.groups.values(), Score())

    def add(self, group: str, score: Score) -> None:
        self.groups[group] = self.groups.get(group, Score()) + score

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall.to_dict(),
            "by_group": {k: v.to_dict() for k, v in sorted(self.groups.items())},
        }

    def format(self, *, title: str = "perplexité") -> str:
        lignes = [
            f"{title:<22}{'docs':>7}{'tokens':>10}{'perte':>9}{'ppl':>9}{'bits/car':>10}",
            "-" * 67,
        ]
        for nom, score in sorted(self.groups.items(), key=lambda kv: kv[1].loss):
            d = score.to_dict()
            lignes.append(
                f"{nom:<22}{d['documents']:>7}{d['tokens']:>10}"
                f"{d['loss']:>9.3f}{d['perplexity']:>9.2f}{d['bits_per_char']:>10.4f}"
            )
        d = self.overall.to_dict()
        lignes.append("-" * 67)
        lignes.append(
            f"{'GLOBAL':<22}{d['documents']:>7}{d['tokens']:>10}"
            f"{d['loss']:>9.3f}{d['perplexity']:>9.2f}{d['bits_per_char']:>10.4f}"
        )
        return "\n".join(lignes)


@torch.no_grad()
def evaluate_documents(
    model: torch.nn.Module,
    codec: Any,
    documents: Iterable[Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
    seq_len: int,
    group_by: str = "source",
    max_documents: int | None = None,
) -> GroupedScore:
    """Mesure la perplexité document par document, ventilée par groupe.

    Chaque document est encodé puis découpé en fenêtres de ``seq_len``. Les
    documents plus courts qu'une fenêtre sont évalués tels quels : les écarter
    biaiserait la mesure vers les textes longs, or les notes personnelles et les
    fichiers de code sont courts par nature.

    Le **dernier fragment incomplet** d'un document est ignoré. Il serait
    évalué sur moins de contexte que les autres, ce qui gonflerait artificiellement
    la perte — un artefact de découpage, pas une propriété du modèle.
    """
    was_training = model.training
    model.eval()

    scores = GroupedScore()
    seen = 0

    for doc in documents:
        ids = codec.encode(doc.text)
        if len(ids) < 2:
            continue

        groupe = getattr(doc, group_by, "inconnu")
        score = Score(n_documents=1, n_chars=len(doc.text))

        # Fenêtres disjointes. Un recouvrement donnerait une meilleure perplexité
        # (plus de contexte par token prédit) mais rendrait la mesure
        # incomparable à une perte d'entraînement, qui n'en a pas.
        for start in range(0, len(ids) - 1, seq_len):
            fenetre = ids[start : start + seq_len + 1]
            if len(fenetre) < 2:
                break
            x = torch.tensor([fenetre[:-1]], device=device)
            y = torch.tensor([fenetre[1:]], device=device)
            with torch.autocast(device.type, dtype=dtype):
                _, loss = model(x, targets=y)
            n = y.numel()
            score.total_loss += loss.item() * n
            score.n_tokens += n

        if score.n_tokens:
            scores.add(groupe, score)
        seen += 1
        if max_documents is not None and seen >= max_documents:
            break

    if was_training:
        model.train()
    return scores
