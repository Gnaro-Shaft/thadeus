"""Entraînement du tokenizer BPE sur *notre* corpus.

Deuxième levier multiplicatif du projet. Le raisonnement tient en une ligne :
un tokenizer qui encode le même texte en 25 % de tokens en moins fait voir 25 %
de contenu en plus au modèle, à budget de calcul identique. Aucune modification
d'architecture ne donne un gain aussi direct pour aussi peu d'effort.

L'assemblage suit celui de GPT-2, avec deux différences assumées :

- le motif de découpage est adapté au français (voir :mod:`.pretokenize`) ;
- l'alphabet initial contient les 256 octets, ce qui garantit qu'aucun texte ne
  produira jamais de token inconnu.

Le choix de la taille de vocabulaire est un **arbitrage**, pas un réglage :
augmenter le vocabulaire réduit le nombre de tokens par texte (moins de calcul
par document) mais grossit la table d'embedding (plus de paramètres consacrés à
autre chose qu'au raisonnement). À 85 M paramètres, un vocabulaire de 32 k avec
``d_model = 512`` occupe déjà ~19 % du modèle. C'est pourquoi
:mod:`.metrics` permet de mesurer la fertilité à plusieurs tailles avant de
trancher — la question n'a pas de bonne réponse dans l'absolu, seulement sur
notre corpus.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from thadeus.core.artifacts import ARTIFACT_ROOT, Artifact, open_artifact
from thadeus.core.config import Schema
from thadeus.core.logs import get_logger
from thadeus.core.seeding import seed_everything
from thadeus.data.schema import Document, format_tokens
from thadeus.data.shard import iter_documents
from thadeus.tokenizer.codec import Codec, SpecialTokens
from thadeus.tokenizer.pretokenize import PATTERNS

__all__ = ["TokenizerConfig", "build_tokenizer", "find_corpus", "train_tokenizer"]

log = get_logger(__name__)


class TokenizerConfig(Schema):
    """Recette d'un tokenizer."""

    label: str = "bpe32k"
    vocab_size: int = 32_000
    pattern: str = "french"
    min_frequency: int = 2
    reserved_slots: int = 16
    corpus: str | None = None
    corpus_label: str = "fr_first"
    max_documents: int | None = 400_000
    seed: int = 1337


def find_corpus(label: str, *, root: Path | None = None) -> Path:
    """Localise les shards du corpus le plus récent portant ce libellé.

    Les artefacts sont nommés ``<label>-<hash de config>`` : le hash change dès
    qu'un paramètre change, donc plusieurs corpus du même libellé peuvent
    coexister. On prend le plus récemment **achevé** — un répertoire sans
    ``meta.json`` est un run interrompu, jamais un corpus valide.
    """
    base = (root or ARTIFACT_ROOT) / "data"
    candidates = [
        path
        for path in sorted(base.glob(f"{label}-*"))
        if (path / "meta.json").is_file() and (path / "corpus").is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"aucun corpus achevé nommé {label!r} sous {base}. "
            f"Le construire avec : python scripts/build_corpus.py --config data/{label}.toml"
        )
    return max(candidates, key=lambda p: (p / "meta.json").stat().st_mtime) / "corpus"


def build_tokenizer(
    *,
    pattern: str = "french",
    special: SpecialTokens | None = None,
) -> Tokenizer:
    """Assemble un tokenizer BPE byte-level, non entraîné.

    L'ordre des deux pré-tokeniseurs est ce qui compte :

    1. ``Split`` applique notre motif — c'est lui qui décide des frontières que
       le BPE ne pourra jamais franchir.
    2. ``ByteLevel`` convertit ensuite chaque morceau en octets représentables.
       ``use_regex=False`` est **indispensable** : sans ça, ``ByteLevel``
       réappliquerait le découpage de GPT-2 par-dessus le nôtre et annulerait
       tout le travail sur les élisions.
    """
    if pattern not in PATTERNS and pattern.strip() == "":
        raise ValueError("motif de pré-tokenisation vide")
    regex = PATTERNS.get(pattern, pattern)

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(regex), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def _corpus_texts(directory: Path, limit: int | None) -> Iterator[str]:
    """Flux de textes pour l'entraînement, avec une trace de progression."""
    for index, doc in enumerate(iter_documents(directory, limit=limit), start=1):
        if index % 100_000 == 0:
            log.info("  %d documents lus", index)
        yield doc.text


def train_tokenizer(raw_config: dict[str, Any], *, force: bool = False) -> Artifact:
    """Entraîne un tokenizer et retourne son artefact achevé."""
    cfg = TokenizerConfig(**raw_config)
    artifact = open_artifact("tokenizer", cfg.label, raw_config)
    if artifact.exists() and not force:
        log.info("Tokenizer déjà entraîné : %s", artifact)
        return artifact

    seed_everything(cfg.seed)
    corpus = Path(cfg.corpus).expanduser() if cfg.corpus else find_corpus(cfg.corpus_label)
    if not corpus.is_dir():
        raise FileNotFoundError(f"corpus introuvable : {corpus}")

    special = SpecialTokens(reserved=cfg.reserved_slots)
    tokenizer = build_tokenizer(pattern=cfg.pattern, special=special)

    log.info(
        "Entraînement %s : vocabulaire %d, motif %r, corpus %s",
        cfg.label,
        cfg.vocab_size,
        cfg.pattern,
        corpus,
    )
    trainer = trainers.BpeTrainer(
        vocab_size=cfg.vocab_size,
        min_frequency=cfg.min_frequency,
        special_tokens=special.as_list(),
        # Les 256 octets dans l'alphabet initial : la garantie « jamais d'inconnu ».
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(_corpus_texts(corpus, cfg.max_documents), trainer=trainer)

    codec = Codec(tokenizer=tokenizer, special=special)
    codec.save(artifact.path)

    stats = _training_stats(codec, corpus)
    artifact.write_json("stats.json", stats)
    artifact.write_meta(raw_config, corpus=str(corpus), vocab_size=codec.vocab_size, **stats)

    log.info(
        "Tokenizer %s : %d tokens de vocabulaire, fertilité %.3f tokens/mot",
        cfg.label,
        codec.vocab_size,
        stats["fertility_overall"],
    )
    return artifact


def _training_stats(codec: Codec, corpus: Path, *, sample: int = 2_000) -> dict[str, Any]:
    """Mesure immédiate de la fertilité, sur un échantillon du corpus d'entraînement.

    Volontairement calculée ici et rangée dans l'artefact : un tokenizer sans
    sa fertilité est un objet qu'on ne peut pas comparer, donc dont on ne peut
    pas justifier le choix.
    """
    from thadeus.tokenizer.metrics import measure, measure_by

    docs: list[Document] = list(iter_documents(corpus, limit=sample))
    overall = measure(codec.count, (d.text for d in docs))
    by_lang = measure_by(codec.count, docs, key=lambda d: d.lang)
    log.info("  échantillon : %s tokens sur %d documents", format_tokens(overall.tokens), len(docs))
    return {
        "sample_documents": len(docs),
        "fertility_overall": overall.tokens_per_word,
        "chars_per_token_overall": overall.chars_per_token,
        "by_language": {lang: f.to_dict() for lang, f in by_lang.items()},
    }
