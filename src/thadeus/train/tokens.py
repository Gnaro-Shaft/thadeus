"""Shards de tokens — le contrat sur disque entre le tokenizer et l'entraînement.

Format volontairement pauvre : des entiers bruts, à la queue leu leu, dans des
fichiers ``part-00000.bin``, plus un ``meta.json`` qui dit comment les lire.
Aucune structure, aucun délimiteur — les frontières de documents sont marquées
par le token de fin de texte, dans le flux lui-même.

Cette pauvreté est le point. Elle permet la lecture par **projection mémoire**
(``memmap``) : le système d'exploitation pagine à la demande, on ne charge
jamais le corpus en RAM, et lire un lot revient à un accès tableau. Un corpus de
2 Md de tokens occupe 4 Go sur disque et zéro octet de RAM tant qu'on ne le lit
pas.

Le choix ``uint16`` n'est pas une micro-optimisation : à 32 k de vocabulaire, il
divise par deux la taille du corpus **et** la bande passante nécessaire pour le
lire. Sur une machine où la mémoire est déjà la seconde ressource la plus rare,
c'est un facteur deux gratuit.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np

from thadeus.core.logs import get_logger

__all__ = ["TokenShardWriter", "TokenStore", "dtype_for_vocab"]

log = get_logger(__name__)

# Volontairement **pas** `meta.json` : ce nom est déjà pris par le marqueur
# d'achèvement des artefacts (voir `core/artifacts.py`). Les deux cohabitent dans
# le même répertoire, et le second écrasait silencieusement le premier — un
# corpus tokenisé parfaitement valide devenait illisible sans aucune erreur à
# l'écriture. Deux rôles, deux fichiers.
META = "tokens.json"
SHARD_GLOB = "part-*.bin"
MASK_SUFFIX = ".mask"
_UINT16_MAX = 65_535


def dtype_for_vocab(vocab_size: int) -> np.dtype:
    """Le plus petit entier capable de coder ce vocabulaire.

    En dessous de 65 536 tokens, ``uint16`` suffit et divise par deux la taille
    du corpus. Au-delà, ``uint32``. Le dtype est écrit dans les métadonnées :
    le relire depuis le fichier plutôt que le redéduire évite qu'un changement
    de vocabulaire ne rende illisibles des shards existants.
    """
    return np.dtype(np.uint16 if vocab_size <= _UINT16_MAX + 1 else np.uint32)


class TokenShardWriter:
    """Écrit un flux de tokens en shards binaires de taille bornée.

    with TokenShardWriter("data/tokens", vocab_size=32_000) as writer:
        for ids in encoded_documents:
            writer.write(ids)
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        vocab_size: int,
        tokens_per_shard: int = 100_000_000,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.vocab_size = vocab_size
        self.dtype = dtype_for_vocab(vocab_size)
        self.tokens_per_shard = tokens_per_shard
        self.metadata = dict(metadata or {})

        self.n_tokens = 0
        self.n_documents = 0
        self.n_masked_tokens = 0
        self.shards: list[dict[str, Any]] = []
        self.has_mask = False

        self._handle = None
        self._mask_handle = None
        self._shard_tokens = 0
        self._buffer: list[np.ndarray] = []
        self._mask_buffer: list[np.ndarray] = []
        self._buffered = 0

    def _open_shard(self) -> None:
        name = f"part-{len(self.shards):05d}.bin"
        self._handle = (self.directory / name).open("wb")
        self._mask_handle = (self.directory / (name + MASK_SUFFIX)).open("wb")
        self.shards.append({"name": name, "n_tokens": 0})
        self._shard_tokens = 0

    def _flush(self) -> None:
        if not self._buffer:
            return
        assert self._handle is not None and self._mask_handle is not None
        self._handle.write(np.concatenate(self._buffer).tobytes())
        self._mask_handle.write(np.concatenate(self._mask_buffer).tobytes())
        self._buffer.clear()
        self._mask_buffer.clear()
        self._buffered = 0

    def _close_shard(self) -> None:
        if self._handle is None:
            return
        self._flush()
        self.shards[-1]["n_tokens"] = self._shard_tokens
        self._handle.close()
        self._handle = None
        if self._mask_handle is not None:
            self._mask_handle.close()
            self._mask_handle = None

    def write(self, ids: Iterable[int], mask: Iterable[int] | None = None) -> None:
        """Ajoute les tokens d'un document, avec un masque de perte optionnel.

        Le **masque** vaut 1 là où la perte doit être calculée, 0 ailleurs. Il
        n'a de sens qu'en réglage par instructions : on veut que le modèle
        apprenne à produire la **réponse**, pas à inventer la **question**.

        Sans masque, un modèle entraîné sur « Question… Réponse… » consacre une
        part de sa capacité à générer des questions — exactement la
        confabulation qu'on cherche à réduire.

        Le masque est **toujours écrit**, valant 1 partout par défaut. Un fichier
        parallèle systématique évite d'avoir deux formats de corpus dont l'un
        casserait le lecteur de l'autre ; le surcoût est d'un octet par token,
        et le fichier se compresse à presque rien quand il est uniforme.
        """
        array = np.asarray(list(ids), dtype=self.dtype)
        if array.size == 0:
            return

        if mask is None:
            m = np.ones(array.size, dtype=np.uint8)
        else:
            m = np.asarray(list(mask), dtype=np.uint8)
            if m.size != array.size:
                raise ValueError(
                    f"masque de {m.size} valeurs pour {array.size} tokens — "
                    f"un décalage rendrait la perte incohérente sans lever d'erreur"
                )
            self.has_mask = True

        if self._handle is None or self._shard_tokens >= self.tokens_per_shard:
            self._close_shard()
            self._open_shard()

        self._buffer.append(array)
        self._mask_buffer.append(m)
        self._buffered += array.size
        self._shard_tokens += array.size
        self.n_tokens += array.size
        self.n_masked_tokens += int(m.sum())
        self.n_documents += 1
        if self._buffered >= 1_000_000:
            self._flush()

    def close(self) -> None:
        """Ferme les shards et écrit les métadonnées **en dernier**.

        Même règle que pour les artefacts : les métadonnées font office de
        marqueur d'achèvement. Un corpus tokenisé sans ``meta.json`` est un run
        interrompu, pas un corpus.
        """
        self._close_shard()
        payload = {
            "dtype": self.dtype.name,
            "vocab_size": self.vocab_size,
            "n_tokens": self.n_tokens,
            "n_documents": self.n_documents,
            "n_masked_tokens": self.n_masked_tokens,
            "has_mask": self.has_mask,
            "shards": self.shards,
            **self.metadata,
        }
        (self.directory / META).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass
class TokenStore:
    """Corpus tokenisé, lu par projection mémoire.

    Fournit des **fenêtres** de ``seq_len + 1`` tokens : le modèle apprend à
    prédire le token suivant, donc une fenêtre d'entraînement contient une
    entrée et sa cible décalée d'un cran.

    Args:
        directory: répertoire produit par :class:`TokenShardWriter`.
        val_tokens: tokens réservés à la validation, pris à la **fin** du
            corpus. Le corpus étant mélangé à la construction (voir
            ``data/pipeline.py``), une tranche finale est un échantillon
            représentatif — et la réserver ainsi est trivialement reproductible.
    """

    directory: Path
    val_tokens: int = 0

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        meta_path = self.directory / META
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"corpus tokenisé introuvable ou incomplet : {meta_path}. "
                f"Le produire avec : python scripts/tokenize_corpus.py"
            )
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.dtype = np.dtype(self.meta["dtype"])
        self.vocab_size = self.meta["vocab_size"]

        self._arrays = [
            np.memmap(self.directory / shard["name"], dtype=self.dtype, mode="r")
            for shard in self.meta["shards"]
        ]
        # Masques de perte, projetés en mémoire comme les tokens. Absents des
        # corpus produits avant leur introduction : on retombe alors sur « tout
        # compte », ce qui est le comportement d'un pré-entraînement.
        self._masks = [
            np.memmap(self.directory / (shard["name"] + MASK_SUFFIX), dtype=np.uint8, mode="r")
            if (self.directory / (shard["name"] + MASK_SUFFIX)).is_file()
            else None
            for shard in self.meta["shards"]
        ]
        self.has_mask = bool(self.meta.get("has_mask")) and all(m is not None for m in self._masks)
        self._sizes = np.array([a.size for a in self._arrays], dtype=np.int64)
        self.n_tokens = int(self._sizes.sum())

        if self.val_tokens >= self.n_tokens:
            raise ValueError(
                f"val_tokens ({self.val_tokens}) >= taille du corpus ({self.n_tokens})"
            )

    @property
    def n_train_tokens(self) -> int:
        return self.n_tokens - self.val_tokens

    def _split_bounds(self, split: str) -> tuple[int, int]:
        if split == "train":
            return 0, self.n_train_tokens
        if split == "val":
            return self.n_train_tokens, self.n_tokens
        raise ValueError(f"split inconnu : {split!r} (attendu 'train' ou 'val')")

    def _read_mask(self, start: int, length: int) -> np.ndarray:
        """Masque correspondant à une fenêtre. Tout à 1 si le corpus n'en a pas."""
        if not self.has_mask:
            return np.ones(length, dtype=np.uint8)
        return self._read(start, length, source=self._masks, dtype=np.uint8)

    def _read(self, start: int, length: int, *, source=None, dtype=None) -> np.ndarray:
        """Lit ``length`` tokens depuis la position globale ``start``.

        Recolle les shards si la fenêtre chevauche une frontière, plutôt que de
        l'éviter : perdre systématiquement les tokens de bord biaiserait
        légèrement le corpus, et la fenêtre à cheval est un cas rare mais réel.
        """
        tableaux = self._arrays if source is None else source
        out = np.empty(length, dtype=dtype or self.dtype)
        written = 0
        offset = start
        for array, size in zip(tableaux, self._sizes, strict=True):
            if offset >= size:
                offset -= size
                continue
            take = min(int(size - offset), length - written)
            out[written : written + take] = array[offset : offset + take]
            written += take
            if written == length:
                return out
            offset = 0
        raise IndexError(f"lecture hors bornes : {start}+{length} > {self.n_tokens}")

    def windows(
        self,
        *,
        batch_size: int,
        seq_len: int,
        seed: int,
        split: str = "train",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Tire un lot de fenêtres aléatoires — **sans aucun état interne**.

        Rend ``(fenêtres, masques)``. Le masque vaut 1 là où la perte compte ;
        il vaut 1 partout sur un corpus de pré-entraînement.

        Le tirage ne dépend que de ``seed``, jamais d'un compteur interne. En
        dérivant la graine du numéro de pas, reprendre un entraînement au pas N
        redonne **exactement** les mêmes lots que la première exécution.

        C'est ce qui permet de ne rien sauvegarder du chargeur dans les
        checkpoints. Un chargeur à état est la source classique de reprises
        subtilement fausses : on restaure le modèle mais pas la position dans
        les données, et le modèle revoit les mêmes lots sans qu'aucune erreur
        ne se manifeste.
        """
        start, stop = self._split_bounds(split)
        span = stop - start - seq_len - 1
        if span <= 0:
            raise ValueError(
                f"split {split!r} trop court ({stop - start} tokens) pour seq_len={seq_len}"
            )

        rng = np.random.default_rng(seed)
        offsets = start + rng.integers(0, span, size=batch_size, dtype=np.int64)
        fenetres = np.stack([self._read(int(o), seq_len + 1) for o in offsets])
        masques = np.stack([self._read_mask(int(o), seq_len + 1) for o in offsets])
        return fenetres, masques

    def sequential_windows(
        self, *, batch_size: int, seq_len: int, split: str = "val", limit: int | None = None
    ) -> Iterator[np.ndarray]:
        """Parcourt un split de bout en bout, sans recouvrement.

        Pour l'évaluation : une perte de validation doit couvrir tout le split
        de façon déterministe, pas échantillonner au hasard — sinon deux
        évaluations successives ne sont pas comparables.
        """
        start, stop = self._split_bounds(split)
        window = seq_len + 1
        position, produced = start, 0
        while position + window * batch_size <= stop:
            yield (
                np.stack([self._read(position + i * window, window) for i in range(batch_size)]),
                np.stack(
                    [self._read_mask(position + i * window, window) for i in range(batch_size)]
                ),
            )
            position += window * batch_size
            produced += 1
            if limit is not None and produced >= limit:
                return

    def describe(self) -> dict[str, Any]:
        return {
            "n_tokens": self.n_tokens,
            "n_train_tokens": self.n_train_tokens,
            "n_val_tokens": self.val_tokens,
            "n_documents": self.meta.get("n_documents"),
            "vocab_size": self.vocab_size,
            "dtype": self.dtype.name,
            "shards": len(self._arrays),
            "size_gb": self.n_tokens * self.dtype.itemsize / 1024**3,
        }
