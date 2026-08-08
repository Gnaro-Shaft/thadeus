"""La boucle d'entraînement.

Volontairement mince : lire un lot, calculer une perte, mettre à jour les poids.
Tout le reste — journalisation, évaluation, sauvegarde, échantillonnage — vit
dans :mod:`thadeus.train.hooks`. C'est le seul endroit du projet où un bug se
paie en heures de calcul avant de se manifester ; il doit rester lisible d'un
seul coup d'œil.

Deux propriétés qui rendent la reprise fiable, et qui découlent de choix faits
ailleurs :

- **Le chargeur est sans état.** Les lots sont tirés d'une graine dérivée du
  numéro de pas. Restaurer le pas suffit à retrouver la séquence exacte.
- **Le pas compté est la mise à jour de poids**, pas le micro-lot. Avec
  accumulation de gradient, confondre les deux fausse le planificateur de taux
  d'apprentissage d'un facteur ``grad_accum``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from thadeus.bench.flops import mfu as compute_mfu
from thadeus.core.artifacts import ARTIFACT_ROOT, Artifact, open_artifact
from thadeus.core.config import load_config
from thadeus.core.device import describe, hot_path_dtype, resolve_device
from thadeus.core.logs import MetricWriter, get_logger
from thadeus.core.seeding import derive_seed, seed_everything
from thadeus.data.schema import format_tokens
from thadeus.model import ModelConfig, Thadeus, estimate
from thadeus.optim.build import build_optimizer
from thadeus.optim.mup import MupConfig, apply_mup, logit_scale, lr_scales
from thadeus.optim.schedules import SCHEDULES
from thadeus.train.checkpoint import CheckpointManager, unwrap
from thadeus.train.config import TrainConfig
from thadeus.train.hooks import CheckpointHook, Hook, default_hooks
from thadeus.train.interrupt import GracefulStop
from thadeus.train.tokens import TokenStore

__all__ = ["TrainState", "Trainer", "find_tokens", "train"]

log = get_logger(__name__)


@dataclass
class TrainState:
    """État courant, passé aux hooks."""

    step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    lr: float = 0.0
    grad_norm: float = 0.0
    tokens_seen: int = 0
    tokens_per_second: float = 0.0
    mfu: float = 0.0
    elapsed: float = 0.0
    val_loss: float | None = None


def find_tokens(label: str, *, root: Path | None = None) -> Path:
    """Localise le corpus tokenisé le plus récent portant ce libellé."""
    base = (root or ARTIFACT_ROOT) / "tokens"
    candidates = [p for p in sorted(base.glob(f"{label}-*")) if (p / "tokens.json").is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"aucun corpus tokenisé nommé {label!r} sous {base}. "
            f"Le produire avec : python scripts/tokenize_corpus.py --corpus-label {label}"
        )
    return max(candidates, key=lambda p: (p / "tokens.json").stat().st_mtime)


@dataclass
class Trainer:
    """Rassemble tout ce dont la boucle a besoin, et rien de plus."""

    cfg: TrainConfig
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    schedule: Any
    store: TokenStore
    device: torch.device
    dtype: torch.dtype
    seq_len: int
    flops_per_token: float
    peak_flops: float
    checkpoints: CheckpointManager
    metrics: MetricWriter
    raw_config: dict[str, Any]
    mup_factors: dict[str, float] = field(default_factory=dict)
    codec: Any = None
    hooks: list[Hook] = field(default_factory=list)

    def to_device(
        self, windows: np.ndarray, masks: np.ndarray | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Découpe une fenêtre en entrée et cible décalée d'un cran.

        La conversion vers ``int64`` est nécessaire : les tokens sont stockés en
        ``uint16`` pour diviser par deux la taille du corpus, mais l'indexation
        d'embedding et l'entropie croisée exigent des entiers signés larges.

        **Le masque devient un ``-100`` dans les cibles** — la valeur qu'ignore
        l'entropie croisée. Le décalage compte : le masque du token *t* dit si
        l'on veut prédire *t*, donc il s'aligne sur les cibles (``[1:]``), pas
        sur les entrées.
        """
        batch = torch.from_numpy(windows.astype(np.int64)).to(self.device, non_blocking=True)
        inputs, targets = batch[:, :-1], batch[:, 1:]
        if masks is not None:
            garde = torch.from_numpy(masks[:, 1:].astype(bool)).to(self.device)
            targets = targets.masked_fill(~garde, -100)
        return inputs, targets

    def micro_batch(self, step: int, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Tire un micro-lot, de façon reproductible à partir du pas.

        La graine dépend du pas **et** de l'indice d'accumulation : sans le
        second, les ``grad_accum`` micro-lots d'un même pas seraient identiques,
        et l'accumulation n'augmenterait pas la taille effective du lot — un
        bug qui ne produit aucune erreur, juste un modèle qui converge mal.
        """
        windows, masks = self.store.windows(
            batch_size=self.cfg.batch_size,
            seq_len=self.seq_len,
            seed=derive_seed(self.cfg.seed, "batch", step, index),
            split="train",
        )
        return self.to_device(windows, masks)


def build_trainer(raw_config: dict[str, Any], *, artifact: Artifact) -> Trainer:
    """Assemble un entraîneur depuis la config."""
    cfg = TrainConfig(**raw_config)
    seed_everything(cfg.seed)

    device = resolve_device(cfg.device)
    dtype = hot_path_dtype(device)
    log.info("Machine : %s", describe(device))

    tokens_dir = Path(cfg.tokens).expanduser() if cfg.tokens else find_tokens(cfg.tokens_label)
    store = TokenStore(tokens_dir, val_tokens=cfg.eval.val_tokens)
    log.info("Corpus : %s", store.describe())

    model_cfg = ModelConfig(**load_config(cfg.model_config_path))
    if model_cfg.vocab_size != store.vocab_size:
        raise ValueError(
            f"incohérence de vocabulaire : modèle {model_cfg.vocab_size}, "
            f"corpus tokenisé {store.vocab_size}. Un écart ici produit des indices "
            f"hors bornes, ou des tokens que le modèle n'atteindra jamais."
        )

    # muP réajuste l'initialisation et fournit les multiplicateurs de taux.
    # `logit_scale` doit être fixé AVANT de construire le modèle : il fait
    # partie de son architecture, pas de sa conduite.
    mup = MupConfig(**raw_config.get("mup", {}))
    model_cfg = model_cfg.model_copy(update={"logit_scale": logit_scale(model_cfg, mup)})

    sizing = estimate(model_cfg)
    model = Thadeus(model_cfg).to(device)
    mup_factors = apply_mup(model, model_cfg, mup)
    log.info("Modèle : %s", sizing)

    # Le tokenizer qui a produit ce corpus, pour pouvoir décoder les
    # échantillons générés pendant l'entraînement. On le retrouve depuis les
    # métadonnées du corpus plutôt que de le redemander en config : c'est le
    # seul tokenizer avec lequel ces tokens ont un sens.
    codec = _load_codec(store)

    optimizer = build_optimizer(model, spec=cfg.optim, lr_scales=lr_scales(model_cfg, mup))
    schedule_spec = (
        cfg.optim.schedule if isinstance(cfg.optim.schedule, dict) else {"name": cfg.optim.schedule}
    )
    schedule = SCHEDULES.build({**schedule_spec, "total_steps": cfg.total_steps})

    if cfg.compile:
        # ×3,00 mesuré sur MPS en Phase 3. Appliqué **après** la création de
        # l'optimiseur : celui-ci doit référencer les paramètres d'origine.
        model = torch.compile(model)

    return Trainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        schedule=schedule,
        store=store,
        device=device,
        dtype=dtype,
        seq_len=model_cfg.max_seq_len,
        flops_per_token=sizing.flops_per_token(model_cfg.max_seq_len),
        peak_flops=30.0e12,  # crête bf16 mesurée sur M5 Pro ; à remesurer sur H100
        checkpoints=CheckpointManager(artifact.path / "checkpoints", keep_last=cfg.keep_last),
        metrics=MetricWriter(artifact.path / "metrics.jsonl", context={"run": cfg.label}),
        raw_config=raw_config,
        mup_factors=mup_factors,
        codec=codec,
        hooks=default_hooks(cfg),
    )


def _load_codec(store: TokenStore):
    """Recharge le tokenizer d'origine, ou ``None`` s'il est introuvable.

    Ne pas trouver le tokenizer prive des échantillons de texte, pas de
    l'entraînement : on dégrade plutôt que d'échouer.
    """
    name = store.meta.get("tokenizer")
    if not name:
        return None

    # Les métadonnées portent tantôt le nom complet du répertoire d'artefact
    # (`bpe32k-b5c5528b`), tantôt le seul libellé (`bpe32k`) selon le script qui
    # les a écrites. On accepte les deux : exiger une forme unique ferait perdre
    # les échantillons de texte pour une raison purement cosmétique.
    base = ARTIFACT_ROOT / "tokenizer"
    chemin = base / name
    if not chemin.is_dir():
        candidats = [p for p in sorted(base.glob(f"{name}-*")) if (p / "tokenizer.json").is_file()]
        if not candidats:
            log.warning("Tokenizer %s introuvable — pas d'échantillons de texte", name)
            return None
        chemin = max(candidats, key=lambda p: p.stat().st_mtime)

    from thadeus.tokenizer.codec import Codec

    return Codec.load(chemin)


def train(raw_config: dict[str, Any], *, resume: bool = True) -> Artifact:
    """Entraîne un modèle et retourne son artefact."""
    cfg = TrainConfig(**raw_config)
    # L'artefact est nommé par l'**identité** du run, pas par sa config complète :
    # allonger un run doit le reprendre, pas en créer un nouveau. Voir
    # `TrainConfig.identity`.
    artifact = open_artifact("train", cfg.label, cfg.identity())
    artifact.create()

    trainer = build_trainer(raw_config, artifact=artifact)
    start_step = (
        trainer.checkpoints.restore(model=trainer.model, optimizer=trainer.optimizer)
        if resume
        else 0
    )
    if start_step:
        log.info("Reprise au pas %d", start_step)
    elif cfg.init_from:
        # Poids seulement, jamais l'état d'optimiseur : celui-ci porte un
        # momentum accumulé sur la distribution du pré-entraînement.
        source = Path(cfg.init_from).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"checkpoint d'initialisation introuvable : {source}")
        payload = torch.load(source, map_location="cpu", weights_only=False)
        unwrap(trainer.model).load_state_dict(payload["model"], strict=True)
        log.info(
            "Poids initialisés depuis %s (entraîné jusqu'au pas %d) — optimiseur neuf",
            source.name,
            payload.get("step", -1),
        )

    state = TrainState(step=start_step, total_steps=cfg.total_steps)
    tokens_per_step = cfg.effective_batch_tokens * trainer.seq_len
    state.tokens_seen = start_step * tokens_per_step

    log.info(
        "Lot effectif : %d séquences x %d tokens = %s tokens par mise à jour",
        cfg.effective_batch_tokens,
        trainer.seq_len,
        format_tokens(tokens_per_step),
    )
    log.info(
        "Objectif : %d pas = %s tokens",
        cfg.total_steps,
        format_tokens(cfg.total_steps * tokens_per_step),
    )

    trainer.model.train()
    started = time.perf_counter()
    last_time = started
    interrompu = False

    with GracefulStop() as stop:
        for step in range(start_step + 1, cfg.total_steps + 1):
            # Le planificateur agit **multiplicativement** sur le taux de base de
            # chaque groupe. Muon et AdamW ont des taux propres (facteur ~50) et des
            # multiplicateurs muP distincts : un taux commun sous-réglerait l'un ou
            # ferait diverger l'autre.
            factor = trainer.schedule(step - 1)
            for group in trainer.optimizer.param_groups:
                group["lr"] = group["base_lr"] * factor
            lr = trainer.optimizer.param_groups[0]["lr"]

            total_loss = 0.0
            for index in range(cfg.grad_accum):
                inputs, targets = trainer.micro_batch(step, index)
                with torch.autocast(trainer.device.type, dtype=trainer.dtype):
                    _, loss = trainer.model(inputs, targets=targets)
                # Diviser avant l'accumulation : sinon le gradient est `grad_accum`
                # fois trop grand, et le taux d'apprentissage effectif aussi.
                (loss / cfg.grad_accum).backward()
                total_loss += loss.item() / cfg.grad_accum

            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainer.model.parameters(), cfg.optim.grad_clip
            )
            trainer.optimizer.step()
            trainer.optimizer.zero_grad(set_to_none=True)

            now = time.perf_counter()
            state.step = step
            state.loss = total_loss
            state.lr = lr
            state.grad_norm = float(grad_norm)
            state.tokens_seen += tokens_per_step
            state.tokens_per_second = tokens_per_step / (now - last_time)
            state.mfu = compute_mfu(
                state.tokens_per_second, trainer.flops_per_token, trainer.peak_flops
            )
            state.elapsed = now - started
            last_time = now

            for hook in trainer.hooks:
                hook(trainer, state)

            # Deux façons de s'arrêter, consultées **entre deux pas** et jamais
            # pendant : le checkpoint écrit correspond ainsi toujours à un pas
            # achevé, avec un optimiseur cohérent.
            #
            # Le budget de temps est la voie normale d'une session planifiée —
            # le run connaît sa fenêtre. Le signal reste le filet : il rattrape
            # tout ce qui tue le processus sans le prévenir.
            if cfg.max_hours is not None and state.elapsed >= cfg.max_hours * 3600:
                log.info(
                    "Budget de %.2f h atteint au pas %d — arrêt propre.",
                    cfg.max_hours,
                    state.step,
                )
                interrompu = True
                break
            if stop.requested:
                interrompu = True
                break

    # Sauvegarde finale, quel que soit l'intervalle : un run qui se termine sans
    # checkpoint est un run perdu.
    CheckpointHook(every=1).save(trainer, state)
    trainer.metrics.close()

    if interrompu:
        # **`meta.json` n'est pas écrit.** C'est le marqueur d'achèvement d'un
        # artefact : l'écrire ici ferait passer un run tronqué pour terminé, et
        # la nuit suivante repartirait de zéro au lieu de reprendre.
        log.info(
            "Interrompu (%s) au pas %d/%d — checkpoint écrit, reprise possible.",
            stop.signal_name or "arrêt demandé",
            state.step,
            cfg.total_steps,
        )
        return artifact

    artifact.write_meta(
        raw_config,
        mup=trainer.mup_factors,
        final_step=state.step,
        final_loss=state.loss,
        final_val_loss=state.val_loss,
        tokens_seen=state.tokens_seen,
        elapsed_hours=state.elapsed / 3600,
    )
    log.info(
        "Terminé : %d pas, %s tokens, perte %.4f, %.2f h",
        state.step,
        format_tokens(state.tokens_seen),
        state.loss,
        state.elapsed / 3600,
    )
    return artifact
