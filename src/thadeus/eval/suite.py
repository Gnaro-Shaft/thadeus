"""Campagne d'évaluation complète : perplexité, sondes, génération.

Trois mesures qui répondent à trois questions différentes, et aucune ne
remplace les autres :

- **Perplexité par source** — le modèle prédit-il bien ce corpus ? Comparable
  entre checkpoints d'un même modèle.
- **Sondes** — *qu'a-t-il* appris ? Interprétable, diagnostique, et sensible là
  où la perplexité est plate.
- **Génération** — que produit-il vraiment ? Attrape des pathologies (boucles,
  token dominant, changement de langue) qu'aucun chiffre ne montre.

**Le rôle particulier de Gutenberg.** Les 18 romans français du domaine public
ne pèsent presque rien dans le corpus (0,08 %), mais ils font un excellent juge :
mesurer la perplexité sur *Germinal* ou *Madame Bovary* dit quelque chose de la
maîtrise de la langue qu'aucune perplexité sur du web ne dira. C'est du français
littéraire, cohérent, sans balisage — le contraire d'une page web.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from pydantic import Field

from thadeus.core.artifacts import ARTIFACT_ROOT, Artifact, open_artifact
from thadeus.core.config import Schema
from thadeus.core.device import describe, hot_path_dtype, resolve_device
from thadeus.core.logs import get_logger
from thadeus.core.seeding import seed_everything
from thadeus.eval.perplexity import GroupedScore, evaluate_documents
from thadeus.eval.probes import format_probes, run_probes

__all__ = ["EvalConfig", "evaluate"]

log = get_logger(__name__)

PROMPTS = (
    "Le principe fondamental de",
    "Il était une fois, dans un village",
    "Pour installer la bibliothèque, il faut",
    "def calculer_moyenne(valeurs):",
)


class EvalConfig(Schema):
    """Description d'une campagne d'évaluation."""

    label: str = "eval"
    seed: int = 1337
    device: str = "auto"
    checkpoint: str | None = None
    run_label: str = "medium_mup"
    model_config_path: str = "model/medium.toml"
    tokenizer_label: str = "bpe32k"

    corpus_label: str = "fr_first_v2"
    max_documents: int = 400
    seq_len: int = 1024

    gutenberg_root: str = "~/LLM_personelle/datasets/gutenberg"
    gutenberg_documents: int = 200

    probes: bool = True
    generate: bool = True
    max_new_tokens: int = 60
    prompts: list[str] = Field(default_factory=lambda: list(PROMPTS))


def _find(stage: str, label: str) -> Path:
    base = ARTIFACT_ROOT / stage
    candidats = [p for p in sorted(base.glob(f"{label}-*")) if (p / "meta.json").is_file()]
    if not candidats:
        raise FileNotFoundError(f"aucun artefact {stage} nommé {label!r} sous {base}")
    return max(candidats, key=lambda p: (p / "meta.json").stat().st_mtime)


def _load_model(cfg: EvalConfig, device: torch.device):
    """Charge le modèle depuis un checkpoint, ou le construit à neuf.

    Sans checkpoint, on évalue un modèle **non entraîné**. C'est utile : ses
    scores donnent la ligne de base — perplexité ≈ taille du vocabulaire, sondes
    ≈ 50 %. Tout run ultérieur se compare à ça.
    """
    from thadeus.core.config import load_config
    from thadeus.model import ModelConfig, Thadeus
    from thadeus.train.checkpoint import CheckpointManager

    model_cfg = ModelConfig(**load_config(cfg.model_config_path))
    model = Thadeus(model_cfg).to(device)

    chemin = Path(cfg.checkpoint).expanduser() if cfg.checkpoint else None
    if chemin is None:
        try:
            chemin = _find("train", cfg.run_label) / "checkpoints" / "best.pt"
        except FileNotFoundError:
            log.warning("Aucun checkpoint trouvé — évaluation d'un modèle NON ENTRAÎNÉ (référence)")
            return model, model_cfg, None

    if not chemin.is_file():
        log.warning("Checkpoint absent (%s) — évaluation d'un modèle NON ENTRAÎNÉ", chemin)
        return model, model_cfg, None

    manager = CheckpointManager(chemin.parent)
    step = manager.restore(model=model, path=chemin)
    log.info("Checkpoint chargé : %s (pas %d)", chemin, step)
    return model, model_cfg, step


def evaluate(raw_config: dict[str, Any], *, force: bool = False) -> Artifact:
    """Exécute la campagne et retourne son artefact."""
    from thadeus.data.shard import iter_documents
    from thadeus.data.sources.gutenberg import from_gutenberg
    from thadeus.tokenizer.codec import Codec

    cfg = EvalConfig(**raw_config)
    artifact = open_artifact("eval", cfg.label, raw_config)
    if artifact.exists() and not force:
        log.info("Évaluation déjà faite : %s", artifact)
        return artifact

    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    dtype = hot_path_dtype(device)
    log.info("Machine : %s", describe(device))

    codec = Codec.load(_find("tokenizer", cfg.tokenizer_label))
    model, model_cfg, step = _load_model(cfg, device)
    seq_len = min(cfg.seq_len, model_cfg.max_seq_len)

    rapport: dict[str, Any] = {"step": step, "vocab_size": codec.vocab_size}

    # 1. Perplexité sur le corpus, ventilée par source.
    try:
        corpus = _find("data", cfg.corpus_label) / "corpus"
        scores = evaluate_documents(
            model,
            codec,
            iter_documents(corpus, limit=cfg.max_documents),
            device=device,
            dtype=dtype,
            seq_len=seq_len,
            group_by="source",
            max_documents=cfg.max_documents,
        )
        rapport["corpus"] = scores.to_dict()
        print("\n" + scores.format(title="perplexité par source"))
    except FileNotFoundError as exc:
        log.warning("Corpus indisponible (%s) — perplexité par source ignorée", exc)

    # 2. Gutenberg : le juge du français littéraire.
    gutenberg = list(
        from_gutenberg(root=cfg.gutenberg_root, limit=cfg.gutenberg_documents, label="gutenberg")
    )
    if gutenberg:
        livres = GroupedScore()
        par_livre = evaluate_documents(
            model,
            codec,
            gutenberg,
            device=device,
            dtype=dtype,
            seq_len=seq_len,
            group_by="lang",
        )
        livres.groups.update(par_livre.groups)
        rapport["gutenberg"] = livres.to_dict()
        print(f"\nGutenberg ({len(gutenberg)} extraits) : {livres.overall.to_dict()}")

    # 3. Sondes grammaticales.
    if cfg.probes:
        resultats = run_probes(model, codec, device=device, dtype=dtype)
        rapport["probes"] = {k: v.to_dict() for k, v in resultats.items()}
        print("\n" + format_probes(resultats))

    # 4. Génération — ce qu'aucun chiffre ne montre.
    if cfg.generate:
        echantillons = []
        for prompt in cfg.prompts:
            ids = torch.tensor([codec.encode(prompt)], device=device)
            sortie = model.generate(
                ids,
                max_new_tokens=cfg.max_new_tokens,
                temperature=0.8,
                forbidden=codec.service_ids,
            )
            texte = codec.decode(sortie[0].tolist())
            echantillons.append({"prompt": prompt, "texte": texte})
            print(f"\n> {prompt}\n  {texte[len(prompt) :].strip()[:280]}")
        rapport["samples"] = echantillons

    artifact.write_json("report.json", rapport)
    artifact.write_meta(raw_config, step=step)
    return artifact
