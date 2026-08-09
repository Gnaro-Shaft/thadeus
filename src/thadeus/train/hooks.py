"""Hooks — tout ce qui n'est pas le pas d'entraînement lui-même.

La boucle (:mod:`thadeus.train.loop`) ne fait qu'une chose : lire un lot,
calculer une perte, mettre à jour les poids. Journalisation, évaluation,
sauvegarde et échantillonnage vivent ici, en périphérie.

Ce n'est pas un choix esthétique. Une boucle d'entraînement qui contient aussi
la logique d'évaluation et de sauvegarde devient impossible à modifier sans
risquer de casser l'entraînement lui-même — or c'est le seul endroit du projet
où un bug coûte des heures de calcul avant de se manifester.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import torch

from thadeus.core.logs import get_logger
from thadeus.core.seeding import derive_seed
from thadeus.data.schema import format_tokens

if TYPE_CHECKING:
    from thadeus.train.loop import Trainer, TrainState

__all__ = ["CheckpointHook", "EvalHook", "Hook", "LogHook", "SampleHook"]

log = get_logger(__name__)


class Hook(Protocol):
    """Appelé après chaque mise à jour de poids."""

    def __call__(self, trainer: Trainer, state: TrainState) -> None: ...


@dataclass
class LogHook:
    """Journalise la progression : console pour l'humain, JSONL pour les scripts.

    Le **MFU** est la métrique à surveiller en priorité. Une perte qui descend
    dit que le modèle apprend ; un MFU qui s'effondre dit qu'on paie dix fois
    trop cher pour l'apprendre, et c'est le seul indicateur qui le révèle avant
    la fin du run.
    """

    every: int = 20

    def __call__(self, trainer: Trainer, state: TrainState) -> None:
        if state.step % self.every != 0:
            return
        log.info(
            "pas %6d/%d · perte %.4f · lr %.2e · |g| %.2f · %s tok/s · MFU %4.1f %% · %s vus",
            state.step,
            state.total_steps,
            state.loss,
            state.lr,
            state.grad_norm,
            f"{state.tokens_per_second:,.0f}".replace(",", " "),
            100 * state.mfu,
            format_tokens(state.tokens_seen),
        )
        trainer.metrics.log(
            step=state.step,
            loss=state.loss,
            lr=state.lr,
            grad_norm=state.grad_norm,
            tokens_seen=state.tokens_seen,
            tokens_per_second=state.tokens_per_second,
            mfu=state.mfu,
        )


@dataclass
class EvalHook:
    """Mesure la perte sur le split de validation.

    **Deux exigences, longtemps confondues.** Les fenêtres doivent être les
    mêmes d'une évaluation à l'autre — sinon l'écart entre deux mesures mélange
    progrès du modèle et variance d'échantillonnage. Mais elles doivent aussi
    **représenter le split**, faute de quoi la mesure est stable et fausse.

    La première version lisait le split séquentiellement depuis son début. Avec
    ``batches = 20`` cela couvrait 163 840 tokens sur 10 millions, soit **1,6 %
    du split, toujours au même endroit** — et cet endroit contenait du code
    Python. La perte de validation valait donc 1,65 quand l'entraînement était à
    2,34 : non pas un écart de régime, mais deux jeux de données différents. Une
    dégradation du français serait passée totalement inaperçue.

    Les fenêtres sont désormais tirées **au hasard dans tout le split**, avec une
    graine dérivée de celle du run et **indépendante du pas**. Le tirage est
    donc rejoué à l'identique à chaque évaluation : on garde la comparabilité
    exacte, on gagne la représentativité, et on mesure enfin la même
    distribution que celle sur laquelle le modèle s'entraîne.

    Conséquence assumée : les ``val_loss`` d'avant ce changement ne sont pas
    comparables à celles d'après.
    """

    every: int = 500
    batches: int = 20

    def __call__(self, trainer: Trainer, state: TrainState) -> None:
        if self.every <= 0 or state.step % self.every != 0:
            return
        value = self.evaluate(trainer)
        if value is None:
            return
        state.val_loss = value
        log.info("pas %6d · perte de validation %.4f", state.step, value)
        trainer.metrics.log(step=state.step, val_loss=value)

    def evaluate(self, trainer: Trainer) -> float | None:
        if trainer.store.val_tokens <= 0:
            return None
        trainer.model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for index in range(self.batches):
                # La graine dépend du numéro de lot, jamais du pas : les mêmes
                # fenêtres sont donc retirées à chaque évaluation du run.
                windows, masks = trainer.store.windows(
                    batch_size=trainer.cfg.batch_size,
                    seq_len=trainer.seq_len,
                    seed=derive_seed(trainer.cfg.seed, "val", index),
                    split="val",
                )
                inputs, targets = trainer.to_device(windows, masks)
                with torch.autocast(trainer.device.type, dtype=trainer.dtype):
                    _, loss = trainer.model(inputs, targets=targets)
                total += loss.item()
                count += 1
        trainer.model.train()
        return total / count if count else None


@dataclass
class CheckpointHook:
    """Sauvegarde périodiquement, et systématiquement à la fin.

    ``val_loss`` est transmise pour que le gestionnaire tienne à jour
    ``best.pt`` : le dernier modèle n'est pas toujours le meilleur, surtout si
    un run diverge sur la fin.
    """

    every: int = 1_000

    def __call__(self, trainer: Trainer, state: TrainState) -> None:
        if self.every <= 0 or state.step % self.every != 0:
            return
        self.save(trainer, state)

    def save(self, trainer: Trainer, state: TrainState) -> None:
        metrics: dict[str, float] = {"loss": state.loss}
        if state.val_loss is not None:
            metrics["val_loss"] = state.val_loss
        trainer.checkpoints.save(
            step=state.step,
            model=trainer.model,
            optimizer=trainer.optimizer,
            metrics=metrics,
            config=trainer.raw_config,
        )


@dataclass
class SampleHook:
    """Génère un échantillon de texte, pour regarder ce que le modèle produit.

    Une perte de 3,2 ne dit pas si le modèle écrit du français. Lire trente
    tokens de sa production le dit immédiatement — et attrape des pathologies
    (répétition en boucle, un seul token dominant) qu'aucune courbe ne montre.
    """

    every: int = 0
    prompt: str = "Le principe fondamental de"
    max_new_tokens: int = 48

    def __call__(self, trainer: Trainer, state: TrainState) -> None:
        if self.every <= 0 or state.step % self.every != 0 or trainer.codec is None:
            return
        from thadeus.train.checkpoint import unwrap

        model = unwrap(trainer.model)
        ids = torch.tensor([trainer.codec.encode(self.prompt)], device=trainer.device)
        out = model.generate(
            ids,
            max_new_tokens=self.max_new_tokens,
            temperature=0.8,
            top_k=50,
            forbidden=trainer.codec.service_ids,
        )
        model.train()
        texte = trainer.codec.decode(out[0].tolist())
        log.info("pas %6d · échantillon : %s", state.step, texte.replace("\n", " ⏎ ")[:300])
        trainer.metrics.log(step=state.step, sample=texte)


def default_hooks(cfg: Any) -> list[Hook]:
    """Jeu de hooks standard, dérivé de la config."""
    return [
        LogHook(every=cfg.log_every),
        EvalHook(every=cfg.eval.every, batches=cfg.eval.batches),
        CheckpointHook(every=cfg.checkpoint_every),
        SampleHook(every=cfg.sample_every, prompt=cfg.sample_prompt),
    ]
