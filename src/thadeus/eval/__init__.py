"""Étage 6 — l'évaluation.

Trois mesures, trois questions distinctes :

- **perplexité par source** : le modèle prédit-il bien ce corpus ?
- **sondes** : *qu'a-t-il* appris ? (accords, élisions, syntaxe, code)
- **génération** : que produit-il vraiment ?

Aucune ne remplace les autres. Une perplexité qui baisse de 0,2 ne dit pas si
le modèle a progressé sur les accords ou régressé sur le code ; une sonde à
50 % le dit immédiatement.
"""

from thadeus.eval.perplexity import GroupedScore, Score, evaluate_documents
from thadeus.eval.probes import PROBES, MinimalPair, run_probes
from thadeus.eval.suite import EvalConfig, evaluate

__all__ = [
    "PROBES",
    "EvalConfig",
    "GroupedScore",
    "MinimalPair",
    "Score",
    "evaluate",
    "evaluate_documents",
    "run_probes",
]
