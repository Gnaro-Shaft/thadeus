"""Schéma de configuration d'un entraînement."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from thadeus.core.config import Schema

__all__ = ["EvalSpec", "OptimSpec", "TrainConfig"]


class OptimSpec(Schema):
    """Optimiseur et sa trajectoire de taux d'apprentissage.

    ``schedule`` est une spec de registre : passer à un autre planificateur est
    une ligne de config. Muon arrivera en Phase 5 de la même façon, sans toucher
    à la boucle.
    """

    name: str = "adamw"
    lr: float = 3e-4
    # Muon tolère des taux ~50x plus élevés qu'AdamW : l'orthogonalisation borne
    # la norme du pas, ce qu'Adam ne fait pas. Un taux commun sous-réglerait l'un
    # ou ferait diverger l'autre.
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float = 1.0
    schedule: str | dict[str, Any] = Field(
        default_factory=lambda: {"name": "wsd", "warmup_steps": 200, "decay_fraction": 0.1}
    )


class EvalSpec(Schema):
    every: int = 500
    batches: int = 20
    val_tokens: int = 5_000_000


class TrainConfig(Schema):
    """Description complète d'un run.

    Args:
        batch_size: séquences par micro-lot. Borné par la mémoire.
        grad_accum: micro-lots accumulés avant une mise à jour. Le produit
            ``batch_size × grad_accum × seq_len`` est le **lot effectif**, et
            c'est lui qui compte pour la stabilité — pas ``batch_size`` seul.
        compile: mesuré à ×3,00 sur MPS en Phase 3. Le désactiver ne se
            justifie que pour déboguer une erreur que la compilation masque.
        total_steps: durée du run. Avec le planificateur WSD, on peut s'arrêter
            avant sans gâcher le modèle — voir :mod:`thadeus.optim.schedules`.
    """

    label: str = "run"
    seed: int = 1337
    device: str = "auto"
    compile: bool = True

    model_config_path: str = "model/small.toml"
    # Checkpoint dont on **hérite les poids**, sans hériter de son état.
    #
    # À ne pas confondre avec la reprise : reprendre restaure poids, optimiseur
    # ET numéro de pas pour continuer le MÊME run. Initialiser depuis un
    # checkpoint démarre un run NOUVEAU — pas 0, optimiseur neuf, planificateur
    # neuf — à partir de poids déjà entraînés. C'est ce que demande un
    # fine-tuning : l'état d'optimiseur du pré-entraînement porte un momentum
    # accumulé sur une autre distribution, et le réutiliser ferait diverger les
    # premiers pas.
    init_from: str | None = None
    tokens: str | None = None
    tokens_label: str = "fr_first"

    batch_size: int = 12
    grad_accum: int = 1
    total_steps: int = 10_000

    # Budget de temps d'une session, en heures. Le run s'arrête **de lui-même**
    # à l'échéance, à la fin du pas en cours, en écrivant un checkpoint.
    #
    # C'est ce qui rend un entraînement nocturne planifié sûr : le run connaît
    # sa fenêtre au lieu de dépendre d'un signal extérieur pour l'apprendre.
    # `total_steps` reste la cible du planificateur ; ce budget ne fait que
    # découper le trajet en sessions, sans toucher au taux d'apprentissage.
    #
    # Comme `total_steps`, il relève de la **conduite** du run et non de son
    # identité : il est donc absent de `identity()`, et changer sa valeur
    # reprend le run au lieu d'en créer un nouveau.
    max_hours: float | None = None

    optim: OptimSpec = Field(default_factory=OptimSpec)
    eval: EvalSpec = Field(default_factory=EvalSpec)
    mup: dict[str, Any] = Field(default_factory=dict)

    log_every: int = 20
    checkpoint_every: int = 1_000
    keep_last: int = 2
    sample_every: int = 0
    sample_prompt: str = "Le principe fondamental de"

    @property
    def effective_batch_tokens(self) -> int:
        """Tokens par mise à jour de poids — la quantité qui gouverne la stabilité."""
        return self.batch_size * self.grad_accum

    def identity(self) -> dict[str, Any]:
        """Ce qui définit **l'expérience**, par opposition à sa conduite.

        C'est ce sous-ensemble qui nomme le répertoire d'artefact, et la
        distinction n'est pas théorique : elle a été introduite après qu'une
        reprise a échoué en silence. Allonger un run de 60 à 90 pas changeait le
        hash complet, donc le répertoire, donc le checkpoint devenait
        introuvable — et l'entraînement repartait de zéro sans le dire.

        Or décider quand s'arrêter *pendant* le run est précisément ce que
        permet le planificateur WSD, et ce dont on a besoin avec un budget H100
        encore inconnu.

        Sont donc exclus de l'identité tous les réglages qui ne changent pas
        *ce qu'on entraîne* : durée, intervalles de journalisation, fréquence de
        sauvegarde, échantillonnage. Restent le modèle, les données,
        l'optimiseur, la taille de lot et la graine.

        Contrepartie assumée : deux runs ne différant que par leur durée
        partagent leurs checkpoints. C'est voulu — ce sont le même run, repris.
        """
        schedule = (
            dict(self.optim.schedule)
            if isinstance(self.optim.schedule, dict)
            else {"name": self.optim.schedule}
        )
        return {
            "label": self.label,
            "seed": self.seed,
            "model_config_path": self.model_config_path,
            # Deux fine-tunings partant de checkpoints différents sont deux
            # expériences distinctes, même à config identique par ailleurs.
            "init_from": self.init_from,
            "tokens": self.tokens,
            "tokens_label": self.tokens_label,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "optim": {
                "name": self.optim.name,
                "lr": self.optim.lr,
                "weight_decay": self.optim.weight_decay,
                "betas": list(self.optim.betas),
                "eps": self.optim.eps,
                "grad_clip": self.optim.grad_clip,
                "muon_lr": self.optim.muon_lr,
                # Le nom du planificateur fait partie de l'identité, pas ses
                # paramètres de durée : passer de WSD à cosinus est une autre
                # expérience, allonger le palier ne l'est pas.
                "schedule": schedule.get("name", "wsd"),
            },
            "val_tokens": self.eval.val_tokens,
            # muP change l'initialisation et les taux : c'est une autre expérience.
            "mup": self.mup,
        }
