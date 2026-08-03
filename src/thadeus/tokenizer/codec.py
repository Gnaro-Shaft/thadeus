"""Le codec — encoder, décoder, sauvegarder, recharger.

Deux garanties tenues par la construction byte-level, et il faut comprendre
pourquoi elles ne sont pas gratuites :

1. **Aucun token inconnu, jamais.** L'alphabet initial contient les 256 octets,
   donc n'importe quelle séquence d'octets est encodable — un caractère chinois,
   un émoji, un octet invalide au milieu d'un fichier corrompu. Un tokenizer qui
   produit ``<unk>`` perd de l'information de façon irréversible.
2. **Aller-retour exact.** ``decode(encode(t)) == t`` pour tout ``t``, y compris
   les espaces multiples et les caractères de contrôle. C'est ce qui permet de
   vérifier qu'un corpus tokenisé n'a rien perdu.

Les identifiants des tokens spéciaux sont figés à l'avance et **des créneaux
sont réservés**. Ajouter un token après l'entraînement du modèle obligerait à
redimensionner la table d'embedding, opération qui invalide les états
d'optimiseur. Réserver seize créneaux coûte 0,05 % du vocabulaire ; ne pas les
réserver coûtera une reprise d'entraînement en Phase 8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

from tokenizers import Tokenizer

__all__ = ["Codec", "SpecialTokens"]

TOKENIZER_FILE = "tokenizer.json"


@dataclass(frozen=True)
class SpecialTokens:
    """Tokens spéciaux, dans l'ordre où ils occupent les premiers identifiants."""

    end_of_text: str = "<|endoftext|>"
    pad: str = "<|pad|>"
    reserved: int = 16

    def as_list(self) -> list[str]:
        """La liste complète, ordre compris — c'est elle qui fixe les identifiants."""
        return [
            self.end_of_text,
            self.pad,
            *(f"<|reserved_{i}|>" for i in range(self.reserved)),
        ]

    def __len__(self) -> int:
        return 2 + self.reserved


@dataclass
class Codec:
    """Enveloppe autour d'un tokenizer entraîné."""

    tokenizer: Tokenizer
    special: SpecialTokens = field(default_factory=SpecialTokens)

    @classmethod
    def load(cls, path: str | Path, special: SpecialTokens | None = None) -> Self:
        """Charge depuis un fichier ``tokenizer.json`` ou un répertoire d'artefact."""
        target = Path(path)
        if target.is_dir():
            target = target / TOKENIZER_FILE
        return cls(tokenizer=Tokenizer.from_file(str(target)), special=special or SpecialTokens())

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        if target.is_dir() or not target.suffix:
            target.mkdir(parents=True, exist_ok=True)
            target = target / TOKENIZER_FILE
        self.tokenizer.save(str(target))
        return target

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @property
    def eot_id(self) -> int:
        """Identifiant du séparateur de documents."""
        return self.tokenizer.token_to_id(self.special.end_of_text)

    @property
    def pad_id(self) -> int:
        return self.tokenizer.token_to_id(self.special.pad)

    def encode(self, text: str, *, add_eot: bool = False) -> list[int]:
        """Encode un texte.

        ``add_eot`` ajoute le séparateur de fin de document. À activer lors de
        la constitution des shards de tokens : sans séparateur, le modèle
        apprendrait que la fin d'un article enchaîne naturellement sur le début
        d'un autre.
        """
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        return [*ids, self.eot_id] if add_eot else ids

    def encode_batch(self, texts: list[str], *, add_eot: bool = False) -> list[list[int]]:
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=False)
        if not add_eot:
            return [e.ids for e in encodings]
        eot = self.eot_id
        return [[*e.ids, eot] for e in encodings]

    def decode(self, ids: list[int], *, skip_special: bool = False) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special)

    def count(self, text: str) -> int:
        """Nombre de tokens, sans matérialiser la liste — pour mesurer un corpus."""
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)
