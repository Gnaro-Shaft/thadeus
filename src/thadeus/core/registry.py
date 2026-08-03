"""Registre de composants — le mécanisme qui rend les briques interchangeables.

Le pari « architecture » de Thadeus (MLA vs GQA, MoE, blocs récursifs) n'a de
valeur que si l'on peut comparer deux variantes **toutes choses égales par
ailleurs**. Si changer d'attention demande de toucher au code, deux choses se
produisent : on le fait rarement, et on introduit des différences involontaires
entre les variantes comparées.

D'où ce registre. Chaque composant s'enregistre sous un nom :

    ATTENTION = Registry("attention")

    @ATTENTION.register("gqa")
    class GroupedQueryAttention(nn.Module):
        ...

et la config le sélectionne :

    [model.attention]
    name = "mla"
    latent_dim = 128

Passer de GQA à MLA devient une ligne de TOML, et l'A/B est honnête par
construction.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Iterator, Mapping
from typing import Any

__all__ = ["Registry"]

_NAME_KEY = "name"


class Registry[T]:
    """Association nom -> fabrique, avec des erreurs qui aident.

    Args:
        kind: le type de composant enregistré ("attention", "optimizer", ...),
            utilisé uniquement dans les messages d'erreur.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, Callable[..., T]] = {}

    def register(
        self,
        name: str | None = None,
        *,
        aliases: tuple[str, ...] = (),
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Décorateur d'enregistrement.

        Sans ``name``, le nom de la classe ou fonction est utilisé tel quel.
        """

        def decorator(obj: Callable[..., T]) -> Callable[..., T]:
            primary = name or getattr(obj, "__name__", None)
            if not primary:
                raise ValueError(f"[{self.kind}] impossible de déduire un nom pour {obj!r}")
            for key in (primary, *aliases):
                if key in self._entries:
                    raise KeyError(f"[{self.kind}] nom déjà pris : {key!r}")
                self._entries[key] = obj
            return obj

        return decorator

    def get(self, name: str) -> Callable[..., T]:
        """Récupère une fabrique par son nom.

        Une faute de frappe ne doit pas coûter une session de débogage : on
        propose les noms proches et on liste les possibilités.
        """
        try:
            return self._entries[name]
        except KeyError:
            available = sorted(self._entries)
            hint = ""
            close = difflib.get_close_matches(name, available, n=1)
            if close:
                hint = f" Vouliez-vous dire {close[0]!r} ?"
            raise KeyError(
                f"[{self.kind}] composant inconnu : {name!r}.{hint} "
                f"Disponibles : {', '.join(available) or '(aucun)'}"
            ) from None

    def build(self, spec: str | Mapping[str, Any], **extra: Any) -> T:
        """Instancie un composant depuis une spec de config.

        ``spec`` est soit un nom nu (``"gqa"``), soit un mapping contenant la
        clé ``name`` — les autres clés sont passées au constructeur :

            {"name": "mla", "latent_dim": 128}  ->  MLA(latent_dim=128)

        Les arguments ``extra`` (fournis par le code appelant, pas par la
        config) sont fusionnés ; ils servent aux dépendances que la config ne
        peut pas connaître, comme ``d_model`` déduit du reste du modèle.
        """
        if isinstance(spec, str):
            name, params = spec, {}
        elif isinstance(spec, Mapping):
            if _NAME_KEY not in spec:
                raise KeyError(
                    f"[{self.kind}] la spec doit contenir la clé {_NAME_KEY!r} : {spec!r}"
                )
            name = spec[_NAME_KEY]
            params = {k: v for k, v in spec.items() if k != _NAME_KEY}
        else:
            raise TypeError(f"[{self.kind}] spec invalide : {spec!r} (attendu str ou mapping)")

        factory = self.get(name)
        try:
            return factory(**params, **extra)
        except TypeError as exc:
            raise TypeError(f"[{self.kind}] échec de construction de {name!r} : {exc}") from exc

    def names(self) -> list[str]:
        """Noms enregistrés, triés."""
        return sorted(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._entries))

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {len(self)} composants : {', '.join(self.names())})"
