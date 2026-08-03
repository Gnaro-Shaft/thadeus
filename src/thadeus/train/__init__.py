"""Étage 5 — l'entraînement.

    shards de tokens -> boucle -> checkpoints

Deux propriétés rendent la reprise fiable, et elles sont structurelles :
le chargeur est **sans état** (les lots dérivent du numéro de pas), et le
`meta.json` d'un checkpoint est écrit atomiquement. Sur un projet qui entraîne
par nuits et par tranches de crédits, ce n'est pas du confort.
"""

from thadeus.train.config import TrainConfig
from thadeus.train.loop import Trainer, TrainState, train
from thadeus.train.tokens import TokenShardWriter, TokenStore

__all__ = ["TokenShardWriter", "TokenStore", "TrainConfig", "TrainState", "Trainer", "train"]
