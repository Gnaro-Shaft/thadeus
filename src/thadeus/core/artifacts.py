"""Artefacts — le contrat entre étages.

Rappel du principe d'architecture de Thadeus : **le contrat d'entrée/sortie
d'un étage est un fichier sur disque, pas un appel de fonction**. L'étage
« tokenizer » ne reçoit pas un objet de l'étage « données », il lit des shards
qu'un autre processus a écrits, éventuellement il y a une semaine.

Un artefact est donc un répertoire nommé par l'étage, un libellé, et le hash de
la config qui l'a produit :

    artifacts/tokenizer/bpe32k-a1b2c3d4/
        meta.json      <- config complète, révision git, machine, date
        vocab.json     <- la sortie proprement dite

Le hash dans le nom fait tout le travail : deux variantes ne peuvent pas
s'écraser, et retrouver « avec quelle config ce fichier a-t-il été produit »
ne demande jamais de fouiller un historique.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from thadeus.core.config import config_hash

__all__ = ["Artifact", "ARTIFACT_ROOT", "git_revision", "open_artifact"]

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ROOT = Path(os.environ.get("THADEUS_ARTIFACTS", REPO_ROOT / "artifacts"))

_META = "meta.json"


def git_revision(repo: Path | None = None) -> str | None:
    """Révision git courante, avec un suffixe ``-dirty`` si l'arbre est modifié.

    Consignée dans chaque artefact : sans elle, un résultat de benchmark n'est
    pas rattachable au code qui l'a produit.
    """
    repo = repo or REPO_ROOT
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.SubprocessError, OSError):
        return None


def _machine_id() -> str:
    """Identifiant de machine **pseudonyme**, stable d'un run à l'autre.

    Le nom d'hôte brut est une donnée personnelle : sur un poste personnel il
    contient couramment le nom de son propriétaire (« MacBook-de-Untel »). Or
    les artefacts de banc sont versionnés volontairement — voir `.gitignore` —
    donc ce nom finirait dans un dépôt public à chaque mesure.

    Un condensé tronqué remplit exactement la même fonction : distinguer deux
    machines et suivre l'une d'elles dans le temps, sans jamais la nommer. Ce
    qui identifie réellement le matériel — `platform` et `device` — reste en
    clair, puisque c'est ce qu'on compare.
    """
    empreinte = hashlib.blake2b(socket.gethostname().encode("utf-8"), digest_size=4)
    return empreinte.hexdigest()


def _environment() -> dict[str, Any]:
    """Contexte machine, pour comparer un run Mac et un run H100."""
    return {
        "machine": _machine_id(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


@dataclass(frozen=True)
class Artifact:
    """Un répertoire de sortie versionné par le hash de sa config."""

    stage: str
    label: str
    hash: str
    root: Path

    @property
    def path(self) -> Path:
        return self.root / self.stage / f"{self.label}-{self.hash}"

    @property
    def meta_path(self) -> Path:
        return self.path / _META

    def exists(self) -> bool:
        """L'artefact a-t-il été produit *jusqu'au bout* ?

        On teste la présence des métadonnées, pas celle du répertoire : un
        répertoire peut exister à moitié rempli après une interruption, et le
        considérer comme valide ferait reprendre une chaîne sur des données
        tronquées.
        """
        return self.meta_path.is_file()

    def create(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path

    def write_meta(self, config: Mapping[str, Any], **extra: Any) -> Path:
        """Écrit ``meta.json``. À appeler **en dernier**, une fois la sortie complète.

        C'est ce qui donne son sens à :meth:`exists` : les métadonnées font
        office de marqueur d'achèvement.
        """
        self.create()
        payload = {
            "stage": self.stage,
            "label": self.label,
            "hash": self.hash,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git": git_revision(),
            "environment": _environment(),
            "config": dict(config),
            **extra,
        }
        self.meta_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        return self.meta_path

    def read_meta(self) -> dict[str, Any]:
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def write_json(self, name: str, payload: Any) -> Path:
        self.create()
        target = self.path / name
        target.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        return target

    def __str__(self) -> str:
        return str(self.path)


def open_artifact(
    stage: str,
    label: str,
    config: Mapping[str, Any],
    *,
    root: Path | None = None,
) -> Artifact:
    """Ouvre (sans créer) l'artefact correspondant à cette config.

    Args:
        stage: l'étage de la chaîne ("data", "tokenizer", "bench", ...).
        label: libellé lisible ("bpe32k", "matmul").
        config: la config qui produit cet artefact — son hash nomme le dossier.
    """
    return Artifact(
        stage=stage,
        label=label,
        hash=config_hash(config),
        root=(root or ARTIFACT_ROOT),
    )
