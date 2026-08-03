"""Chargement des secrets depuis ``.env``.

Le dépôt a besoin de deux jeux d'identifiants : ``HF_TOKEN`` pour Hugging Face
(datasets et tokenizers restreints, et surtout des limites de débit décentes en
collecte), et les clés Lightning AI pour le run H100 de la Phase 7.

Règle tenue par tout ce module : **on ne manipule jamais une valeur de secret,
seulement des noms de variables**. :func:`load_dotenv` retourne les noms chargés,
jamais leur contenu, et rien ici n'écrit un secret dans un journal, une
métadonnée d'artefact ou un message d'erreur. Un jeton recopié dans un
``meta.json`` versionné est une fuite définitive.

Le fichier lui-même n'est jamais versionné (voir ``.gitignore``).
"""

from __future__ import annotations

import os
from pathlib import Path

from thadeus.core.logs import get_logger

__all__ = ["has_secret", "load_dotenv", "require_secret"]

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV = REPO_ROOT / ".env"


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> list[str]:
    """Charge un fichier ``.env`` dans l'environnement du processus.

    Args:
        path: fichier à lire (par défaut ``.env`` à la racine du dépôt).
        override: écrase les variables déjà définies. Par défaut ``False`` :
            une variable exportée dans le shell l'emporte sur le fichier, ce qui
            permet de surcharger ponctuellement sans éditer ``.env``.

    Returns:
        Les **noms** des variables chargées, triés. Jamais les valeurs.
    """
    env_path = Path(path) if path is not None else DEFAULT_ENV
    if not env_path.is_file():
        return []

    loaded: list[str] = []
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        if key in os.environ and not override:
            continue
        # Les guillemets encadrants sont une convention d'écriture, pas la valeur.
        os.environ[key] = value.strip().strip("\"'")
        loaded.append(key)

    if loaded:
        log.debug("Variables chargées depuis %s : %s", env_path.name, ", ".join(sorted(loaded)))
    return sorted(loaded)


def has_secret(name: str) -> bool:
    """Le secret est-il disponible ? Ne révèle rien de sa valeur."""
    return bool(os.environ.get(name))


def require_secret(name: str, *, why: str) -> str:
    """Retourne un secret, ou lève une erreur qui explique quoi faire.

    Args:
        why: à quoi sert ce secret, pour que le message d'erreur soit
            actionnable plutôt que sibyllin.
    """
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} absent de l'environnement. Nécessaire pour : {why}. "
            f"L'ajouter dans {DEFAULT_ENV} (fichier non versionné) sous la forme "
            f"{name}=..., ou l'exporter dans le shell."
        )
    return value
