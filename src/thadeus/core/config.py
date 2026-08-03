"""Configuration en couches.

Contrat de ce module — une seule règle, tenue partout dans le projet :

    Aucune constante en dur dans le code. Tout ce qui peut varier vit dans un TOML.

Une config est un fichier TOML qui peut hériter d'autres configs via la clé
``extends``. La fusion est profonde et le dernier gagne :

    extends = ["model/base.toml", "optim/muon.toml"]   # gauche -> droite
    [model]
    n_layers = 12                                       # surcharge l'hérité

Les surcharges de ligne de commande (``--set model.n_layers=16``) passent en
tout dernier. On obtient ainsi trois niveaux — base, config du run, ligne de
commande — sans jamais dupliquer un fichier pour changer un paramètre.

Chaque config produit un **hash stable** (:func:`config_hash`) qui sert à
nommer les répertoires de sortie. Deux runs identiques écrivent au même
endroit ; changer un seul paramètre change le hash. C'est ce qui rend les
comparaisons A/B honnêtes : impossible d'écraser silencieusement le résultat
d'une variante avec celui d'une autre.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

__all__ = [
    "Schema",
    "config_hash",
    "deep_merge",
    "load_config",
    "parse_override",
    "set_by_path",
    "to_canonical_json",
]

# Racine des configs, résolue relativement au dépôt (src/thadeus/core/config.py -> ../../..)
CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"

_EXTENDS_KEY = "extends"
_MAX_DEPTH = 16


class Schema(BaseModel):
    """Base de tout schéma de config validé.

    ``extra="forbid"`` est volontaire et non négociable : une clé mal
    orthographiée dans un TOML doit lever une erreur, pas être ignorée en
    silence. Une faute de frappe qui passe inaperçue, c'est une expérience dont
    on croit connaître les paramètres alors qu'on tourne avec les valeurs par
    défaut — le pire mode de défaillance possible dans un projet d'entraînement.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


def deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """Fusionne ``over`` dans ``base`` en profondeur, sans muter les entrées.

    Deux dictionnaires fusionnent clé par clé ; tout le reste (listes incluses)
    est remplacé en bloc. Remplacer une liste plutôt que la concaténer est un
    choix : concaténer rendrait impossible de *retirer* un élément hérité.
    """
    out = dict(base)
    for key, value in over.items():
        current = out.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            out[key] = deep_merge(current, value)
        else:
            out[key] = value
    return out


def _resolve_path(ref: str, *, relative_to: Path, root: Path) -> Path:
    """Résout une référence de config.

    Une référence est cherchée d'abord à côté du fichier qui l'inclut, puis
    depuis la racine des configs. Cela permet d'écrire ``extends = "base.toml"``
    entre voisins et ``extends = "model/base.toml"`` depuis n'importe où.
    """
    candidate = (relative_to / ref).resolve()
    if candidate.is_file():
        return candidate
    candidate = (root / ref).resolve()
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"config introuvable : {ref!r} (cherchée dans {relative_to} puis {root})"
    )


def _load_raw(path: Path, *, root: Path, seen: tuple[Path, ...], depth: int) -> dict[str, Any]:
    """Charge un TOML et résout récursivement ses ``extends``."""
    if depth > _MAX_DEPTH:
        raise RecursionError(f"héritage de config trop profond (>{_MAX_DEPTH}) : {path}")
    if path in seen:
        chain = " -> ".join(p.name for p in (*seen, path))
        raise ValueError(f"cycle dans l'héritage de config : {chain}")

    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    parents = raw.pop(_EXTENDS_KEY, [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, Sequence):
        raise TypeError(f"{path}: '{_EXTENDS_KEY}' doit être une chaîne ou une liste")

    merged: dict[str, Any] = {}
    for ref in parents:
        parent_path = _resolve_path(ref, relative_to=path.parent, root=root)
        parent = _load_raw(parent_path, root=root, seen=(*seen, path), depth=depth + 1)
        merged = deep_merge(merged, parent)

    return deep_merge(merged, raw)


def parse_override(text: str) -> tuple[list[str], Any]:
    """Traduit ``"model.n_layers=16"`` en ``(["model", "n_layers"], 16)``.

    La valeur est interprétée avec la grammaire TOML, ce qui donne gratuitement
    les entiers, flottants, booléens, chaînes et listes — et surtout garantit
    que ``--set x=12`` produit l'entier 12, pas la chaîne "12".
    """
    if "=" not in text:
        raise ValueError(f"surcharge invalide : {text!r} (attendu 'chemin.cle=valeur')")
    path, _, value = text.partition("=")
    path = path.strip()
    if not path:
        raise ValueError(f"surcharge invalide : {text!r} (chemin vide)")
    try:
        parsed = tomllib.loads(f"v = {value.strip()}")["v"]
    except tomllib.TOMLDecodeError:
        parsed = value.strip()  # repli : chaîne nue non quotée
    return path.split("."), parsed


def set_by_path(cfg: dict[str, Any], keys: Sequence[str], value: Any) -> None:
    """Écrit ``value`` à l'emplacement ``keys`` dans ``cfg``, en place."""
    node = cfg
    for key in keys[:-1]:
        nxt = node.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            node[key] = nxt
        node = nxt
    node[keys[-1]] = value


def load_config(
    path: str | Path,
    *,
    overrides: Sequence[str] = (),
    root: Path | None = None,
) -> dict[str, Any]:
    """Charge une config : héritage résolu, surcharges appliquées.

    Args:
        path: chemin du TOML, absolu ou relatif à la racine des configs.
        overrides: surcharges ``"chemin.cle=valeur"``, appliquées en dernier.
        root: racine des configs (par défaut ``configs/`` du dépôt).

    Returns:
        Le dictionnaire fusionné, prêt à être validé par un :class:`Schema`.
    """
    root = (root or CONFIG_ROOT).resolve()
    path = Path(path)
    if not path.is_absolute():
        path = _resolve_path(str(path), relative_to=Path.cwd(), root=root)

    cfg = _load_raw(path.resolve(), root=root, seen=(), depth=0)
    for item in overrides:
        keys, value = parse_override(item)
        set_by_path(cfg, keys, value)
    return cfg


def to_canonical_json(cfg: Mapping[str, Any]) -> str:
    """Sérialise une config de façon déterministe.

    Clés triées et séparateurs compacts : deux configs sémantiquement
    identiques produisent exactement la même chaîne, donc le même hash, quel
    que soit l'ordre d'écriture dans les TOML.
    """
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(cfg: Mapping[str, Any], *, length: int = 8) -> str:
    """Empreinte courte et stable d'une config, pour nommer les sorties."""
    digest = hashlib.sha256(to_canonical_json(cfg).encode("utf-8")).hexdigest()
    return digest[:length]
