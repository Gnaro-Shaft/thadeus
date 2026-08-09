"""La perte de validation doit être à la fois stable et représentative.

Ces deux exigences ont longtemps été confondues. Le parcours séquentiel depuis
le début du split garantissait la stabilité — les mêmes fenêtres à chaque fois —
mais avec `batches = 20` il ne couvrait que 1,6 % du split, toujours au même
endroit. La mesure était parfaitement reproductible, et parfaitement fausse.

Le test central de ce fichier est donc celui de la **couverture** : il échoue si
l'on revient à un échantillonnage concentré en tête de split.
"""

from __future__ import annotations

import numpy as np
import torch

from thadeus.core.seeding import derive_seed
from thadeus.model import ModelConfig, Thadeus
from thadeus.train.config import TrainConfig
from thadeus.train.hooks import EvalHook
from thadeus.train.tokens import TokenShardWriter, TokenStore

VOCAB = 64
SEQ = 32
# Le split de validation est bâti en deux moitiés de contenus distincts : la
# première n'utilise que le jeton TETE, la seconde que le jeton QUEUE. Une
# lecture qui resterait en tête du split ne verrait jamais QUEUE.
TETE, QUEUE = 11, 22


class TrainerFactice:
    """Le minimum dont `EvalHook.evaluate` a besoin, sans monter un vrai run."""

    def __init__(self, store: TokenStore, cfg: TrainConfig):
        self.store = store
        self.cfg = cfg
        self.seq_len = SEQ
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.model = Thadeus(
            ModelConfig(
                vocab_size=VOCAB,
                d_model=32,
                n_layers=1,
                max_seq_len=SEQ + 1,
                attention={"name": "gqa", "n_heads": 2, "n_kv_heads": 1, "head_dim": 16},
            )
        )
        self.vues: list[np.ndarray] = []

    def to_device(self, windows, masks):
        self.vues.append(np.asarray(windows).copy())
        w = torch.from_numpy(np.asarray(windows)).long()
        return w[:, :-1], w[:, 1:]


def corpus_marque(tmp_path, tokens_entrainement: int, tokens_val: int):
    """Corpus dont le split de validation change de contenu à mi-parcours."""
    directory = tmp_path / "tokens"
    rng = np.random.default_rng(0)
    with TokenShardWriter(directory, vocab_size=VOCAB, tokens_per_shard=10_000_000) as writer:
        writer.write(rng.integers(0, 10, size=tokens_entrainement).tolist())
        writer.write([TETE] * (tokens_val // 2))
        writer.write([QUEUE] * (tokens_val - tokens_val // 2))
    return TokenStore(directory, val_tokens=tokens_val)


def _hook_et_trainer(tmp_path, batches: int = 20):
    store = corpus_marque(tmp_path, tokens_entrainement=20_000, tokens_val=8_000)
    cfg = TrainConfig(batch_size=4, seed=1337, eval={"batches": batches, "val_tokens": 8_000})
    return EvalHook(every=1, batches=batches), TrainerFactice(store, cfg)


class TestCouvertureDeLaValidation:
    def test_la_seconde_moitie_du_split_est_lue(self, tmp_path):
        """LE test de non-régression.

        Avec un parcours séquentiel limité à quelques lots, seule la tête du
        split était lue et ce test échouerait — c'est exactement le défaut qui a
        fait mesurer le modèle sur du code Python pendant tout un run.
        """
        hook, trainer = _hook_et_trainer(tmp_path)
        hook.evaluate(trainer)
        vus = np.concatenate([v.ravel() for v in trainer.vues])
        assert (vus == QUEUE).any(), "la validation ne lit que le début du split"
        assert (vus == TETE).any(), "la validation ne lit que la fin du split"

    def test_les_deux_moities_sont_lues_a_peu_pres_autant(self, tmp_path):
        hook, trainer = _hook_et_trainer(tmp_path, batches=40)
        hook.evaluate(trainer)
        vus = np.concatenate([v.ravel() for v in trainer.vues])
        part = (vus == QUEUE).sum() / (vus == TETE).sum()
        assert 0.5 < part < 2.0, f"couverture déséquilibrée entre les moitiés ({part:.2f})"

    def test_ne_deborde_jamais_sur_l_entrainement(self, tmp_path):
        """Un seul jeton d'entraînement dans la validation la rendrait complaisante."""
        hook, trainer = _hook_et_trainer(tmp_path)
        hook.evaluate(trainer)
        vus = np.concatenate([v.ravel() for v in trainer.vues])
        assert set(np.unique(vus)) <= {TETE, QUEUE}, "des tokens d'entraînement ont fui"


class TestStabiliteDeLaValidation:
    """La représentativité ne doit pas avoir été gagnée contre la comparabilité."""

    def test_deux_evaluations_lisent_les_memes_fenetres(self, tmp_path):
        hook, trainer = _hook_et_trainer(tmp_path)
        hook.evaluate(trainer)
        premier = np.concatenate([v.ravel() for v in trainer.vues])
        trainer.vues.clear()
        hook.evaluate(trainer)
        assert np.array_equal(premier, np.concatenate([v.ravel() for v in trainer.vues]))

    def test_la_graine_ne_depend_pas_du_pas(self):
        # Si la graine suivait le pas, chaque évaluation porterait sur d'autres
        # fenêtres et l'écart entre deux mesures mélangerait le progrès du
        # modèle avec la variance d'échantillonnage.
        assert derive_seed(1337, "val", 0) == derive_seed(1337, "val", 0)
        assert derive_seed(1337, "val", 0) != derive_seed(1337, "val", 1)

    def test_les_lots_ne_sont_pas_tous_identiques(self, tmp_path):
        hook, trainer = _hook_et_trainer(tmp_path, batches=5)
        hook.evaluate(trainer)
        assert not all(np.array_equal(trainer.vues[0], v) for v in trainer.vues[1:])
