"""Assemblage du prompt et génération ancrée.

**Ce que le RAG apporte, et ce qu'il n'apporte pas — à dire avant de mesurer.**

Le fine-tuning a donné au modèle le *style* du Vault ; il ne pouvait pas lui
donner ses *faits* — un million de tokens, c'est trop peu pour installer des
connaissances dans des poids. Le RAG comble exactement ce manque : il met les
bons passages **dans le contexte**, où le modèle n'a plus qu'à les lire.

Deux limites structurelles qu'il faut annoncer, pas découvrir :

1. **Le modèle n'est pas instruit.** C'est un modèle de langue brut : il
   *continue* du texte, il ne *répond* pas à des questions. Le prompt est donc
   rédigé comme une continuation naturelle, jamais comme une conversation. Un
   format de chat produirait du charabia.

2. **À 188 M de paramètres, la synthèse reste faible.** Le modèle reprendra le
   vocabulaire et les tournures des passages retrouvés — ce qui est déjà
   l'essentiel de l'ancrage — mais il ne raisonnera pas dessus de façon fiable.

Autrement dit : **la récupération est immédiatement utile par elle-même** (c'est
un moteur de recherche sur le Vault), et la génération est un bonus dont il faut
juger la qualité sans complaisance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from thadeus.core.logs import get_logger
from thadeus.core.seeding import derive_seed, seed_everything
from thadeus.rag.index import BM25Index, Passage

__all__ = ["Answer", "answer", "build_prompt"]

log = get_logger(__name__)

# Réservé à la génération. Le reste du contexte accueille la question et les
# passages ; sans cette réserve, le prompt remplirait la fenêtre et il ne
# resterait plus de place pour répondre.
RESERVE_GENERATION = 220


def build_prompt(question: str, passages: list[Passage], *, codec, budget: int) -> str:
    """Assemble le prompt, en tronquant les passages pour tenir dans le budget.

    Le format est une **continuation** : « Extraits… Question… Réponse : ».
    Le modèle poursuit naturellement après « Réponse : », ce qu'il sait faire.
    Un format de dialogue supposerait un modèle instruit, ce qu'il n'est pas.

    Les passages sont ajoutés du plus pertinent au moins pertinent, et l'on
    s'arrête dès que le budget est atteint. Tronquer *au milieu* d'un passage
    serait pire que l'omettre : une phrase coupée introduit du bruit que le
    modèle tentera de continuer.
    """
    entete = "Extraits de mes notes :\n\n"
    queue = f"\nQuestion : {question}\nRéponse :"
    reste = budget - len(codec.encode(entete + queue))

    blocs: list[str] = []
    for p in passages:
        bloc = f"[{p.title}]\n{p.text}\n"
        cout = len(codec.encode(bloc))
        if cout > reste:
            continue  # on saute plutôt que de couper une phrase en deux
        blocs.append(bloc)
        reste -= cout

    return entete + "\n".join(blocs) + queue


@dataclass
class Answer:
    """Réponse générée, avec ses sources — pour qu'elle soit vérifiable."""

    question: str
    text: str
    passages: list[tuple[Passage, float]]
    prompt_tokens: int

    @property
    def sources(self) -> list[str]:
        return [p.source for p, _ in self.passages]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.text,
            "prompt_tokens": self.prompt_tokens,
            "sources": [
                {"source": p.source, "title": p.title, "score": round(s, 3)}
                for p, s in self.passages
            ],
        }


@torch.no_grad()
def answer(
    question: str,
    *,
    index: BM25Index,
    model,
    codec,
    device: torch.device,
    dtype: torch.dtype,
    k: int = 3,
    max_new_tokens: int = 120,
    temperature: float = 0.7,
    seed: int = 1337,
) -> Answer:
    """Récupère, assemble, génère.

    La graine est **dérivée de la question**, jamais fixée à une constante :
    deux questions différentes doivent tirer des flux aléatoires différents.
    Une graine commune corrèle les réponses entre elles — c'est le piège
    diagnostiqué en Phase 8, où un même mot ressortait au même rang dans cinq
    générations sans rapport.
    """
    retrouves = index.search(question, k=k)
    if not retrouves:
        return Answer(question, "", [], 0)

    budget = model.cfg.max_seq_len - RESERVE_GENERATION
    prompt = build_prompt(question, [p for p, _ in retrouves], codec=codec, budget=budget)
    ids = codec.encode(prompt)

    seed_everything(derive_seed(seed, "rag", question))
    entree = torch.tensor([ids], device=device)
    sortie = model.generate(
        entree,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=50,
        forbidden=codec.service_ids,
    )
    texte = codec.decode(sortie[0, len(ids) :].tolist())

    # Le modèle continue indéfiniment : on s'arrête au premier changement de
    # sujet marqué, sinon la réponse dérive sur un nouveau paragraphe.
    for marqueur in ("\nQuestion :", "\nExtraits", "\n[", "\n\n\n"):
        if marqueur in texte:
            texte = texte.split(marqueur)[0]
    return Answer(question, texte.strip(), retrouves, len(ids))
