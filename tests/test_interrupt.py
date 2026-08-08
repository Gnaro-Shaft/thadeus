"""Arrêt propre d'un entraînement : signal, budget de temps, reprise.

Ce mécanisme protège une nuit entière de calcul. Les tests vont donc jusqu'à
lancer de vrais entraînements — modèle jouet, corpus fabriqué, CPU — plutôt que
de se contenter de vérifier que le drapeau se lève.

La propriété la plus importante n'est pas « ça s'arrête » mais **« ça ne se
déclare pas terminé »** : `meta.json` est le marqueur d'achèvement d'un artefact.
L'écrire sur un run tronqué ferait passer une nuit interrompue pour un run fini,
et la nuit suivante repartirait de zéro sans que rien ne le signale.
"""

from __future__ import annotations

import os
import signal
import threading

import numpy as np
import pytest

from thadeus.core import artifacts as artifacts_module
from thadeus.train.interrupt import GracefulStop
from thadeus.train.loop import train
from thadeus.train.tokens import TokenShardWriter

VOCAB = 512  # doit correspondre à `configs/model/tiny.toml`


@pytest.fixture
def corpus(tmp_path):
    """Corpus tokenisé jouet, au vocabulaire du modèle `tiny`."""
    rng = np.random.default_rng(0)
    directory = tmp_path / "tokens"
    with TokenShardWriter(directory, vocab_size=VOCAB, tokens_per_shard=25_000) as writer:
        for _ in range(200):
            writer.write(rng.integers(0, VOCAB, size=300).tolist())
    return directory


@pytest.fixture
def artefacts(tmp_path, monkeypatch):
    """Redirige la racine des artefacts hors du dépôt."""
    racine = tmp_path / "artifacts"
    monkeypatch.setattr(artifacts_module, "ARTIFACT_ROOT", racine)
    return racine


def config(corpus, **extra):
    """Config d'un run minuscule : quelques pas sur CPU, sans compilation."""
    return {
        "label": "interrupt",
        "device": "cpu",
        "compile": False,
        "model_config_path": "model/tiny.toml",
        "tokens": str(corpus),
        "batch_size": 2,
        "grad_accum": 1,
        "total_steps": 20,
        "log_every": 1_000,
        "checkpoint_every": 1_000,  # aucun checkpoint périodique ne doit se déclencher
        "sample_every": 0,
        "eval": {"every": 0, "val_tokens": 5_000, "batches": 1},
        "optim": {"lr": 1e-3},
        **extra,
    }


class TestGracefulStop:
    """Le drapeau lui-même — sans entraînement."""

    def test_pas_de_demande_au_repos(self):
        with GracefulStop() as stop:
            assert not stop.requested
            assert stop.signal_name is None

    def test_demande_programmatique(self):
        with GracefulStop() as stop:
            stop.request()
            assert stop.requested

    def test_le_signal_leve_le_drapeau(self):
        with GracefulStop() as stop:
            os.kill(os.getpid(), signal.SIGTERM)
            assert stop.requested
            assert stop.signal_name == "SIGTERM"

    def test_les_gestionnaires_sont_restaures(self):
        avant = signal.getsignal(signal.SIGTERM)
        with GracefulStop():
            assert signal.getsignal(signal.SIGTERM) is not avant
        assert signal.getsignal(signal.SIGTERM) is avant

    def test_restaures_meme_si_le_bloc_leve(self):
        avant = signal.getsignal(signal.SIGINT)
        with pytest.raises(RuntimeError), GracefulStop():
            raise RuntimeError("boum")
        assert signal.getsignal(signal.SIGINT) is avant

    def test_hors_thread_principal_ne_leve_pas(self):
        # `signal.signal` est interdit hors du thread principal. Le contexte doit
        # rester utilisable : c'est le cas dans un test parallélisé, et le code
        # appelant n'a pas à savoir où il tourne.
        erreurs: list[BaseException] = []

        def cible():
            try:
                with GracefulStop() as stop:
                    stop.request()
                    assert stop.requested
            except BaseException as exc:  # noqa: BLE001
                erreurs.append(exc)

        fil = threading.Thread(target=cible)
        fil.start()
        fil.join()
        assert not erreurs, f"le contexte a levé hors du thread principal : {erreurs}"


class TestBudgetDeTemps:
    """`max_hours` — la voie normale d'une session nocturne planifiée."""

    def test_sans_budget_le_run_va_au_bout(self, corpus, artefacts):
        artifact = train(config(corpus), resume=False)
        assert (artifact.path / "meta.json").is_file(), "un run achevé écrit son meta.json"

    def test_budget_nul_arrete_des_le_premier_pas(self, corpus, artefacts):
        artifact = train(config(corpus, max_hours=0.0, total_steps=5_000), resume=False)
        # Le run devait durer 5 000 pas ; il s'est arrêté immédiatement.
        assert (artifact.path / "checkpoints" / "latest.pt").is_file()

    def test_un_run_interrompu_n_ecrit_pas_son_meta(self, corpus, artefacts):
        # LA propriété critique. Sans elle, une nuit tronquée passe pour finie.
        artifact = train(config(corpus, max_hours=0.0, total_steps=5_000), resume=False)
        assert not (artifact.path / "meta.json").exists(), (
            "un run interrompu ne doit pas se déclarer terminé — "
            "meta.json est le marqueur d'achèvement"
        )

    def test_l_interruption_laisse_un_checkpoint_reprenable(self, corpus, artefacts):
        premier = train(config(corpus, max_hours=0.0, total_steps=5_000), resume=False)
        pas_atteint = _step_du_checkpoint(premier)
        assert pas_atteint >= 1, "le pas en cours doit être achevé avant la sauvegarde"

        # Reprise : même identité de run, donc même artefact, donc même checkpoint.
        second = train(config(corpus, total_steps=pas_atteint + 3), resume=True)
        assert second.path == premier.path, "la reprise doit retrouver le même artefact"
        assert _step_du_checkpoint(second) == pas_atteint + 3
        assert (second.path / "meta.json").is_file(), "achevé cette fois-ci"

    def test_le_budget_ne_change_pas_l_identite_du_run(self, corpus, artefacts):
        # Sinon changer la durée d'une session créerait un artefact neuf et
        # orphelinerait le checkpoint de la veille — la régression exacte que
        # `identity()` a été introduite pour empêcher.
        from thadeus.train.config import TrainConfig

        sans = TrainConfig(**config(corpus))
        avec = TrainConfig(**config(corpus, max_hours=8.0))
        assert sans.identity() == avec.identity()


class TestSignal:
    """Le filet : ce que reçoit le processus quand l'ordonnanceur le termine."""

    def test_sigterm_pendant_le_run_arrete_proprement(self, corpus, artefacts):
        # Le run vise 20 000 pas — hors d'atteinte dans le délai du minuteur —
        # afin que le signal tombe forcément *pendant* la boucle. S'il tombait
        # après, les gestionnaires seraient restaurés et le signal tuerait le
        # processus de test.
        minuteur = threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGTERM))
        minuteur.start()
        try:
            artifact = train(config(corpus, total_steps=20_000), resume=False)
        finally:
            minuteur.cancel()

        assert (artifact.path / "checkpoints" / "latest.pt").is_file()
        assert not (artifact.path / "meta.json").exists()
        assert 1 <= _step_du_checkpoint(artifact) < 20_000


def _step_du_checkpoint(artifact) -> int:
    import torch

    payload = torch.load(
        artifact.path / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False
    )
    return int(payload["step"])
