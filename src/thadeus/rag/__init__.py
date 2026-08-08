"""Étage 9 — RAG sur le Vault.

    notes -> passages -> index BM25 -> récupération -> contexte -> génération

Le fine-tuning a donné le **style** du Vault ; il ne pouvait pas en donner les
**faits**. Le RAG comble ce manque en mettant les bons passages dans le contexte.

Contrainte dimensionnante : le modèle n'a que 1024 tokens de fenêtre. Tout le
découpage en découle — trois passages d'environ 150 mots, et pas davantage.
"""

from thadeus.rag.answer import Answer, answer, build_prompt
from thadeus.rag.chunk import chunk_note, iter_vault_passages
from thadeus.rag.index import BM25Index, Passage, tokenize

__all__ = [
    "Answer",
    "BM25Index",
    "Passage",
    "answer",
    "build_prompt",
    "chunk_note",
    "iter_vault_passages",
    "tokenize",
]
