"""Déduplication — exacte puis approchée.

Pourquoi c'est important au-delà du ménage : un document vu deux fois est un
document **appris deux fois**. Sur un corpus web, les quasi-doublons sont
massifs (articles syndiqués, pages miroir, citations recopiées), et les laisser
passer revient à concentrer le budget d'entraînement sur une fraction du corpus
tout en croyant l'avoir couvert entièrement. C'est aussi la première cause de
mémorisation par cœur.

Deux étages, du moins cher au plus cher :

1. **Exact** — empreinte du texte normalisé. Attrape les copies conformes pour
   presque rien.
2. **Approché (MinHash + LSH)** — attrape les quasi-doublons, ceux qui diffèrent
   par un en-tête, une date ou un paragraphe.

Le second est en flux et garde le **premier** exemplaire vu. Conséquence à
connaître : le résultat dépend de l'ordre de passage des documents. C'est
acceptable parce que l'ordre est déterministe (graine fixée, sources dans
l'ordre de la config), donc rejouable — mais changer l'ordre des sources change
*quel* exemplaire survit, pas leur nombre.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np

from thadeus.core.logs import get_logger
from thadeus.data.schema import Document, text_fingerprint

__all__ = ["Deduplicator", "MinHashDeduplicator", "lsh_threshold", "minhash_signature"]

log = get_logger(__name__)

_MERSENNE_61 = (1 << 61) - 1
_POLY_BASE = np.uint32(0x01000193)  # base FNV, pour le hachage roulant des shingles


def lsh_threshold(bands: int, rows: int) -> float:
    """Seuil de similarité Jaccard effectif d'un découpage LSH.

    Deux documents sont candidats s'ils partagent au moins une bande entière.
    La probabilité vaut ``1 - (1 - s^rows)^bands`` : une sigmoïde dont le point
    d'inflexion est ``(1/bands)^(1/rows)``. C'est ce seuil-là qui compte, pas
    un paramètre qu'on croirait régler directement — d'où cette fonction, qui
    permet de vérifier ce qu'on a réellement demandé.
    """
    return (1.0 / bands) ** (1.0 / rows)


def _word_hashes(text: str, *, max_words: int) -> np.ndarray:
    """Hache chaque mot en uint32 stable entre exécutions et entre machines.

    ``crc32`` plutôt que ``hash()`` : le hachage natif de Python est randomisé
    par processus, ce qui rendrait la déduplication non reproductible — et donc
    un corpus non rejouable.
    """
    words = text.lower().split()[:max_words]
    return np.fromiter(
        (zlib.crc32(w.encode("utf-8")) for w in words), dtype=np.uint32, count=len(words)
    )


def _shingle_hashes(word_hashes: np.ndarray, size: int) -> np.ndarray:
    """Hachage roulant des n-grammes de mots, entièrement vectorisé.

    ``size`` passes de numpy au lieu d'une boucle Python sur des millions de
    shingles. Le débordement des ``uint32`` réalise le modulo 2³² gratuitement.
    """
    count = len(word_hashes) - size + 1
    if count <= 0:
        return np.empty(0, dtype=np.uint32)
    acc = np.zeros(count, dtype=np.uint32)
    for offset in range(size):
        acc = acc * _POLY_BASE + word_hashes[offset : offset + count]
    return acc


def minhash_signature(
    text: str,
    coefficients: np.ndarray,
    *,
    shingle_size: int = 5,
    max_words: int = 5000,
) -> np.ndarray | None:
    """Signature MinHash d'un texte, ou ``None`` s'il est trop court.

    Les hachages de shingles tiennent sur 32 bits et les coefficients aussi :
    ``a * h + b`` reste donc strictement sous 2⁶⁴ et le modulo de Mersenne se
    calcule sans débordement. C'est la raison du choix de 32 bits, pas un
    hasard.
    """
    shingles = _shingle_hashes(_word_hashes(text, max_words=max_words), shingle_size)
    if shingles.size == 0:
        return None
    a = coefficients[0][:, None].astype(np.uint64)
    b = coefficients[1][:, None].astype(np.uint64)
    hashed = (a * shingles[None, :].astype(np.uint64) + b) % _MERSENNE_61
    return hashed.min(axis=1)


@dataclass
class MinHashDeduplicator:
    """Détection de quasi-doublons en flux par MinHash + LSH.

    Args:
        bands, rows: découpage de la signature. Le produit donne le nombre de
            permutations ; le couple fixe le seuil effectif (:func:`lsh_threshold`).
        shingle_size: taille des n-grammes de mots. 5 est le standard : assez
            long pour que la coïncidence soit improbable, assez court pour
            rester robuste à une reformulation locale.
        max_words: on ne considère que le début des documents longs. Deux
            articles identiques le sont dès leurs premiers milliers de mots, et
            ce plafond borne le coût sur les documents pathologiques.

    Coût mémoire : environ ``bands`` entrées de table de hachage par document
    retenu, soit quelques centaines d'octets. Acceptable ici — la machine a 64 Go
    et le corpus vise quelques millions de documents.
    """

    bands: int = 16
    rows: int = 8
    shingle_size: int = 5
    max_words: int = 5000
    seed: int = 1337

    _seen: set[int] = field(default_factory=set, init=False, repr=False)
    _coefficients: np.ndarray = field(init=False, repr=False)
    n_seen: int = field(default=0, init=False)
    n_duplicates: int = field(default=0, init=False)
    n_too_short: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        size = self.bands * self.rows
        self._coefficients = np.stack(
            [
                rng.integers(1, 2**32, size=size, dtype=np.uint64),
                rng.integers(0, 2**32, size=size, dtype=np.uint64),
            ]
        )
        log.info(
            "MinHash : %d permutations (%d bandes x %d lignes), seuil Jaccard effectif ~%.2f",
            size,
            self.bands,
            self.rows,
            lsh_threshold(self.bands, self.rows),
        )

    def is_duplicate(self, text: str) -> bool:
        """Teste et enregistre en une passe : ``True`` si un quasi-doublon existe déjà."""
        self.n_seen += 1
        signature = minhash_signature(
            text, self._coefficients, shingle_size=self.shingle_size, max_words=self.max_words
        )
        if signature is None:
            self.n_too_short += 1
            return False

        keys = [
            hash((band, signature[band * self.rows : (band + 1) * self.rows].tobytes()))
            for band in range(self.bands)
        ]
        if any(key in self._seen for key in keys):
            self.n_duplicates += 1
            return True

        self._seen.update(keys)
        return False


@dataclass
class Deduplicator:
    """Enchaîne la déduplication exacte puis approchée.

    L'ordre compte pour le coût, pas pour le résultat : le test exact est
    presque gratuit et absorbe l'essentiel des doublons d'un corpus web, ce qui
    évite de calculer une signature MinHash pour rien.
    """

    minhash: MinHashDeduplicator | None = None
    exact: bool = True

    _fingerprints: set[str] = field(default_factory=set, init=False, repr=False)
    n_seen: int = field(default=0, init=False)
    n_exact: int = field(default=0, init=False)
    n_near: int = field(default=0, init=False)

    def keep(self, doc: Document) -> bool:
        """``True`` si le document doit être conservé."""
        self.n_seen += 1
        if self.exact:
            fingerprint = text_fingerprint(doc.text)
            if fingerprint in self._fingerprints:
                self.n_exact += 1
                return False
            self._fingerprints.add(fingerprint)

        if self.minhash is not None and self.minhash.is_duplicate(doc.text):
            self.n_near += 1
            return False
        return True

    def apply(self, documents: Iterable[Document]) -> Iterator[Document]:
        for doc in documents:
            if self.keep(doc):
                yield doc

    def stats(self) -> dict[str, float | int]:
        removed = self.n_exact + self.n_near
        return {
            "seen": self.n_seen,
            "exact_duplicates": self.n_exact,
            "near_duplicates": self.n_near,
            "kept": self.n_seen - removed,
            "removal_rate": removed / self.n_seen if self.n_seen else 0.0,
        }

    def log_summary(self) -> None:
        stats = self.stats()
        log.info(
            "Déduplication : %d retenus sur %d (%.1f %% retirés — %d exacts, %d quasi)",
            stats["kept"],
            stats["seen"],
            100 * float(stats["removal_rate"]),
            stats["exact_duplicates"],
            stats["near_duplicates"],
        )
