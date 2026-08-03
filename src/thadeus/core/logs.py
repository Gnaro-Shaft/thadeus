"""Journalisation — deux flux qui ne servent pas au même public.

- **Console** : lisible par un humain qui surveille un run.
- **JSONL** : une ligne = une mesure, destiné à être relu par un script.

Les séparer n'est pas cosmétique. Un entraînement produit des dizaines de
milliers de points de mesure ; les extraire ensuite d'un log texte à coups
d'expressions régulières est une perte de temps garantie, et fragile.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

__all__ = ["MetricWriter", "get_logger", "setup_logging"]

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(level: str | int = "INFO", *, stream: TextIO | None = None) -> None:
    """Configure la journalisation console du processus.

    Idempotent : appelable depuis n'importe quel script sans empiler les
    handlers (sinon chaque ligne apparaît en double, triple...).
    """
    root = logging.getLogger()
    root.setLevel(level)
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Logger nommé — à appeler avec ``__name__``."""
    return logging.getLogger(name)


class MetricWriter:
    """Écrit des mesures en JSONL, une ligne par point.

    Ouvert en mode ajout et vidé à chaque écriture : un run interrompu garde
    toutes ses mesures jusqu'à l'instant de l'interruption. Sur un entraînement
    fractionné en plusieurs nuits, c'est la différence entre diagnostiquer une
    divergence et repartir à l'aveugle.

        with MetricWriter(artifact.path / "metrics.jsonl") as metrics:
            metrics.log(step=100, loss=3.21, tokens_per_s=48_000)
    """

    def __init__(self, path: str | Path, *, context: Mapping[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._context = dict(context or {})
        self._fh = self.path.open("a", encoding="utf-8")

    def log(self, **fields: Any) -> None:
        """Écrit un point de mesure, horodaté et enrichi du contexte."""
        record = {
            "t": datetime.now(UTC).isoformat(timespec="milliseconds"),
            **self._context,
            **fields,
        }
        self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> MetricWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
