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
from thadeus.core.seeding import derive_seed, seed_everything
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


def _find_run(label: str) -> Path | None:
    """Localise le meilleur checkpoint d'un run, **même s'il est encore en cours**.

    Contrairement à :func:`_find`, on ne cherche pas de ``meta.json`` : il n'est
    écrit qu'à la **fin** de l'entraînement. Un run en cours a pourtant des
    checkpoints parfaitement valides, et c'est précisément pendant qu'il tourne
    qu'on veut mesurer sa qualité — sinon l'évaluation ne sert qu'à constater
    après coup.

    Ce détail a d'abord produit une évaluation d'apparence normale portant en
    réalité sur un modèle **non entraîné** : l'artefact du run était jugé
    inexistant, et le repli sur un modèle neuf donnait un rapport complet et
    plausible. Le seul indice était un avertissement dans les journaux.
    """
    base = ARTIFACT_ROOT / "train"
    candidats = [
        p / "checkpoints" / "best.pt"
        for p in sorted(base.glob(f"{label}-*"))
        if (p / "checkpoints" / "best.pt").is_file()
    ]
    if not candidats:
        return None
    return max(candidats, key=lambda p: p.stat().st_mtime)


def _load_model(cfg: EvalConfig, device: torch.device):
    """Charge le modèle depuis un checkpoint, ou le construit à neuf.

    Sans checkpoint, on évalue un modèle **non entraîné**. C'est utile : ses
    scores donnent la ligne de base — perplexité ≈ taille du vocabulaire, sondes
    ≈ 50 %. Tout run ultérieur se compare à ça.
    """
    from thadeus.core.config import load_config
    from thadeus.model import ModelConfig, Thadeus
    from thadeus.optim.mup import MupConfig, logit_scale
    from thadeus.train.checkpoint import CheckpointManager

    model_cfg = ModelConfig(**load_config(cfg.model_config_path))

    chemin = Path(cfg.checkpoint).expanduser() if cfg.checkpoint else _find_run(cfg.run_label)

    if chemin is None or not chemin.is_file():
        # Évaluer un modèle neuf est légitime (c'est la ligne de base), mais ce
        # doit être un choix affiché, jamais un repli discret : un rapport complet
        # sur un modèle non entraîné ressemble à s'y méprendre à un vrai résultat.
        print("\n" + "!" * 68)
        print(f"!!  MODELE NON ENTRAINE - aucun checkpoint pour le run {cfg.run_label!r}")
        print("!!  Les chiffres ci-dessous sont la LIGNE DE BASE, pas un résultat.")
        print("!" * 68)
        log.warning("Aucun checkpoint pour %r — évaluation de la ligne de base", cfg.run_label)
        return Thadeus(model_cfg).to(device), model_cfg, None

    # **Rejouer la paramétrisation du run.** muP fixe un facteur de logits qui
    # fait partie de l'architecture, pas de la conduite : un modèle entraîné avec
    # `logit_scale = 1/8` et rechargé avec 1,0 produit des logits huit fois trop
    # grands. Les sondes, qui *comparent* deux séquences, survivent au facteur ;
    # la perplexité, non — elle explose. Ce piège a d'abord donné 8,5 millions de
    # perplexité sur un modèle dont la génération était visiblement correcte.
    payload = torch.load(chemin, map_location="cpu", weights_only=False)
    mup = MupConfig(**payload.get("config", {}).get("mup", {}))
    if mup.enabled:
        facteur = logit_scale(model_cfg, mup)
        model_cfg = model_cfg.model_copy(update={"logit_scale": facteur})
        log.info("muP du run rejoué : logit_scale = %.4f", facteur)

    model = Thadeus(model_cfg).to(device)
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
    #
    # **Une graine dérivée PAR invite**, jamais une graine commune.
    #
    # Réinitialiser la même graine avant chaque échantillon paraît rendre les
    # comparaisons « équitables ». C'est l'inverse : tous les échantillons
    # partagent alors le même flux de nombres aléatoires, donc les mêmes tirages
    # aux mêmes rangs. Avec des distributions de forme voisine — ce qu'elles sont
    # toutes, puisqu'il s'agit de prose française —, le même token finit par
    # sortir au même rang dans des textes sans aucun rapport.
    #
    # Constaté en Phase 8 : le mot « amusant » apparaissait au token 17 de CINQ
    # invites différentes, y compris « La météo de demain sera ». Cela ressemblait
    # à une pathologie du modèle ; c'était le générateur pseudo-aléatoire.
    # Aucune autre graine ne le produisait.
    if cfg.generate:
        echantillons = []
        for index, prompt in enumerate(cfg.prompts):
            seed_everything(derive_seed(cfg.seed, "sample", index))
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
