"""Shards de documents — le contrat sur disque de l'étage données.

Un corpus n'est jamais un seul fichier. Il est découpé en shards
``part-00000.jsonl.zst`` pour trois raisons, dans l'ordre d'importance :

1. **Reprise** — un shard achevé est un shard qu'on ne recalcule pas. Collecter
   plusieurs millions de documents prend des heures ; une interruption ne doit
   pas tout annuler.
2. **Parallélisme** — un shard est l'unité naturelle de travail pour la suite.
3. **Mémoire** — on lit et on écrit en flux, jamais le corpus entier d'un coup.

Le format est du JSONL compressé en zstd : lisible ligne à ligne, concaténable,
et à peu près trois fois plus compact que du gzip à vitesse comparable.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Self

import zstandard

from thadeus.core.logs import get_logger
from thadeus.data.schema import Document

__all__ = ["ShardWriter", "iter_documents", "shard_paths"]

log = get_logger(__name__)

SHARD_SUFFIX = ".jsonl.zst"
_PREFIX = "part"
_COMPRESSION_LEVEL = 6  # au-delà, on paie beaucoup de CPU pour peu de place


def shard_paths(directory: str | Path) -> list[Path]:
    """Les shards d'un répertoire, dans l'ordre."""
    return sorted(Path(directory).glob(f"{_PREFIX}-*{SHARD_SUFFIX}"))


class ShardWriter:
    """Écrit des documents en shards de taille bornée.

    Bascule sur un nouveau shard dès que le nombre de documents **ou** la taille
    compressée dépasse le seuil. Borner les deux est nécessaire : un corpus
    mélange des notes de 200 mots et des articles de 20 000, et ne borner que le
    nombre de documents produit des shards dont la taille varie d'un facteur 100.

        with ShardWriter("data/corpus") as writer:
            for doc in documents:
                writer.write(doc)
        print(writer.n_documents, writer.n_words)
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        max_documents: int = 100_000,
        max_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_documents = max_documents
        self.max_bytes = max_bytes

        self.n_documents = 0
        self.n_words = 0
        self.n_shards = 0
        self.per_source: dict[str, int] = {}
        self.words_per_source: dict[str, int] = {}

        self._compressor = zstandard.ZstdCompressor(level=_COMPRESSION_LEVEL)
        self._handle = None
        self._stream = None
        self._shard_documents = 0

    def _open_shard(self) -> None:
        path = self.directory / f"{_PREFIX}-{self.n_shards:05d}{SHARD_SUFFIX}"
        self._handle = path.open("wb")
        self._stream = self._compressor.stream_writer(self._handle)
        self._shard_documents = 0
        self.n_shards += 1

    def _close_shard(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def _should_rotate(self) -> bool:
        if self._shard_documents >= self.max_documents:
            return True
        return self._handle is not None and self._handle.tell() >= self.max_bytes

    def write(self, doc: Document) -> None:
        """Ajoute un document, en ouvrant ou faisant tourner le shard au besoin."""
        if self._stream is None:
            self._open_shard()
        elif self._should_rotate():
            self._close_shard()
            self._open_shard()

        assert self._stream is not None
        self._stream.write((doc.to_json() + "\n").encode("utf-8"))

        words = doc.n_words
        self._shard_documents += 1
        self.n_documents += 1
        self.n_words += words
        self.per_source[doc.source] = self.per_source.get(doc.source, 0) + 1
        self.words_per_source[doc.source] = self.words_per_source.get(doc.source, 0) + words

    def write_all(self, documents: Iterable[Document]) -> None:
        for doc in documents:
            self.write(doc)

    def close(self) -> None:
        self._close_shard()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def iter_documents(
    source: str | Path | Iterable[str | Path],
    *,
    limit: int | None = None,
) -> Iterator[Document]:
    """Relit des documents depuis un répertoire de shards ou une liste de shards.

    Args:
        source: un répertoire de shards, ou des chemins de shards.
        limit: s'arrête après ``limit`` documents — pratique pour inspecter un
            corpus sans le lire en entier.
    """
    if isinstance(source, str | Path):
        path = Path(source)
        paths = shard_paths(path) if path.is_dir() else [path]
    else:
        paths = [Path(p) for p in source]

    decompressor = zstandard.ZstdDecompressor()
    emitted = 0
    for path in paths:
        with path.open("rb") as handle, decompressor.stream_reader(handle) as raw:
            for line in _iter_lines(raw):
                if not line.strip():
                    continue
                try:
                    yield Document.from_json(line)
                except (json.JSONDecodeError, KeyError):
                    log.warning("ligne illisible ignorée dans %s", path.name)
                    continue
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


def _iter_lines(raw, *, chunk_size: int = 1 << 20) -> Iterator[str]:
    """Découpe un flux décompressé en lignes.

    Le décodage se fait sur les frontières de ligne, pas sur les frontières de
    bloc : un caractère UTF-8 multi-octets à cheval sur deux blocs serait sinon
    coupé en deux et lèverait une erreur de décodage.
    """
    buffer = b""
    while True:
        chunk = raw.read(chunk_size)
        if not chunk:
            break
        buffer += chunk
        *lines, buffer = buffer.split(b"\n")
        for line in lines:
            yield line.decode("utf-8")
    if buffer:
        yield buffer.decode("utf-8")
