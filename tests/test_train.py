"""Étage entraînement : tokens, planificateurs, checkpoints, boucle.

Tout tourne sur CPU avec un modèle jouet et un corpus fabriqué — aucun réseau,
aucun GPU requis.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from thadeus.core.config import load_config
from thadeus.model import ModelConfig, Thadeus
from thadeus.optim.schedules import SCHEDULES, cosine, wsd
from thadeus.train.checkpoint import CheckpointManager, unwrap
from thadeus.train.config import TrainConfig
from thadeus.train.tokens import TokenShardWriter, TokenStore, dtype_for_vocab

VOCAB = 128


@pytest.fixture
def corpus(tmp_path):
    """Corpus tokenisé jouet : 60 000 tokens répartis sur plusieurs shards."""
    rng = np.random.default_rng(0)
    directory = tmp_path / "tokens"
    with TokenShardWriter(directory, vocab_size=VOCAB, tokens_per_shard=25_000) as writer:
        for _ in range(200):
            writer.write(rng.integers(0, VOCAB, size=300).tolist())
    return directory


class TestFormatDeTokens:
    def test_uint16_sous_65536(self):
        # Divise par deux la taille du corpus **et** la bande passante de lecture.
        assert dtype_for_vocab(32_000) == np.uint16
        assert dtype_for_vocab(200_000) == np.uint32

    def test_aller_retour(self, tmp_path):
        ids = [1, 2, 3, 4, 5, 6, 7, 8]
        with TokenShardWriter(tmp_path / "t", vocab_size=VOCAB) as writer:
            writer.write(ids)
        store = TokenStore(tmp_path / "t")
        assert store.n_tokens == len(ids)
        assert store._read(0, len(ids)).tolist() == ids

    def test_rotation_en_plusieurs_shards(self, corpus):
        store = TokenStore(corpus)
        assert len(store.meta["shards"]) > 1
        assert store.n_tokens == 60_000

    def test_lecture_a_cheval_sur_deux_shards(self, corpus):
        # On recolle plutôt que d'éviter : écarter systématiquement les tokens
        # de bord biaiserait légèrement le corpus.
        store = TokenStore(corpus)
        frontiere = store._sizes[0]
        window = store._read(int(frontiere) - 5, 10)
        assert window.size == 10

    def test_meta_absent_donne_la_commande(self, tmp_path):
        (tmp_path / "vide").mkdir()
        with pytest.raises(FileNotFoundError, match="tokenize_corpus"):
            TokenStore(tmp_path / "vide")

    def test_metadonnees_ne_collisionnent_pas_avec_l_artefact(self, corpus):
        # Régression : `meta.json` est le marqueur d'achèvement des artefacts.
        # Le format de tokens utilisait le même nom, et l'artefact écrasait
        # silencieusement les métadonnées du corpus — illisible, sans erreur.
        assert (corpus / "tokens.json").is_file()
        assert not (corpus / "meta.json").exists()

    def test_lecture_hors_bornes_rejetee(self, corpus):
        store = TokenStore(corpus)
        with pytest.raises(IndexError):
            store._read(store.n_tokens - 2, 100)


class TestChargeur:
    def test_formes(self, corpus):
        store = TokenStore(corpus)
        windows = store.windows(batch_size=4, seq_len=32, seed=1)
        assert windows.shape == (4, 33)  # seq_len + 1 : entrée et cible décalée

    def test_sans_etat_donc_reprise_exacte(self, corpus):
        # La propriété qui permet de ne rien sauvegarder du chargeur : deux
        # appels avec la même graine donnent exactement les mêmes fenêtres.
        store = TokenStore(corpus)
        a = store.windows(batch_size=4, seq_len=32, seed=42)
        b = store.windows(batch_size=4, seq_len=32, seed=42)
        assert np.array_equal(a, b)

    def test_graines_differentes_lots_differents(self, corpus):
        store = TokenStore(corpus)
        a = store.windows(batch_size=4, seq_len=32, seed=1)
        b = store.windows(batch_size=4, seq_len=32, seed=2)
        assert not np.array_equal(a, b)

    def test_validation_disjointe_de_l_entrainement(self, corpus):
        store = TokenStore(corpus, val_tokens=10_000)
        assert store.n_train_tokens == 50_000
        train_start, train_stop = store._split_bounds("train")
        val_start, val_stop = store._split_bounds("val")
        assert train_stop == val_start, "les splits doivent être adjacents et disjoints"

    def test_parcours_sequentiel_deterministe(self, corpus):
        # Deux évaluations doivent porter sur les mêmes fenêtres, sinon leur
        # différence mélange progrès du modèle et variance d'échantillonnage.
        store = TokenStore(corpus, val_tokens=10_000)
        a = list(store.sequential_windows(batch_size=2, seq_len=16, limit=3))
        b = list(store.sequential_windows(batch_size=2, seq_len=16, limit=3))
        assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))

    def test_split_trop_court_rejete(self, corpus):
        store = TokenStore(corpus, val_tokens=100)
        with pytest.raises(ValueError, match="trop court"):
            store.windows(batch_size=1, seq_len=500, seed=1, split="val")

    def test_split_inconnu_rejete(self, corpus):
        with pytest.raises(ValueError, match="split inconnu"):
            TokenStore(corpus)._split_bounds("test")


class TestPlanificateurs:
    def test_tous_enregistres(self):
        assert {"cosine", "wsd", "constant"} <= set(SCHEDULES.names())

    def test_chauffe_monte_sans_partir_de_zero(self):
        # Un premier pas à taux nul est un pas perdu.
        factor = wsd(total_steps=1000, warmup_steps=100)
        assert 0 < factor(0) < factor(50) < factor(99)
        assert factor(0) == pytest.approx(0.01)

    def test_wsd_reste_au_maximum_pendant_le_palier(self):
        # La propriété qui rend WSD utile : on peut s'arrêter n'importe quand
        # pendant le palier et ne payer que la décroissance.
        factor = wsd(total_steps=1000, warmup_steps=100, decay_fraction=0.1)
        assert factor(200) == 1.0
        assert factor(800) == 1.0
        assert factor(950) < 1.0

    def test_wsd_termine_bas(self):
        factor = wsd(total_steps=1000, warmup_steps=10, decay_fraction=0.1, min_ratio=0.0)
        assert factor(1000) == pytest.approx(0.0, abs=1e-6)

    def test_cosinus_decroit_des_la_fin_de_la_chauffe(self):
        # La faiblesse du cosinus : la décroissance commence immédiatement, donc
        # arrêter tôt donne un modèle qui n'a pas fini sa trajectoire.
        factor = cosine(total_steps=1000, warmup_steps=100)
        assert factor(100) > factor(300) > factor(600) > factor(999)

    def test_cosinus_respecte_le_plancher(self):
        factor = cosine(total_steps=1000, warmup_steps=0, min_ratio=0.1)
        assert factor(1000) == pytest.approx(0.1, abs=1e-6)

    def test_constant_apres_chauffe(self):
        factor = SCHEDULES.build({"name": "constant", "warmup_steps": 10})
        assert factor(50) == 1.0 == factor(5000)


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return Thadeus(
        ModelConfig(
            vocab_size=VOCAB,
            d_model=64,
            n_layers=2,
            max_seq_len=32,
            attention={"name": "gqa", "n_heads": 4, "n_kv_heads": 2, "head_dim": 16},
        )
    )


class TestCheckpoints:
    def test_aller_retour_exact(self, tmp_path, tiny_model):
        manager = CheckpointManager(tmp_path)
        manager.save(step=7, model=tiny_model, metrics={"val_loss": 2.0})

        torch.manual_seed(99)
        autre = Thadeus(tiny_model.cfg)
        assert manager.restore(model=autre) == 7
        for a, b in zip(tiny_model.parameters(), autre.parameters(), strict=True):
            assert torch.equal(a, b)

    def test_optimiseur_restaure(self, tmp_path, tiny_model):
        manager = CheckpointManager(tmp_path)
        opt = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        ids = torch.randint(0, VOCAB, (2, 16))
        _, loss = tiny_model(ids, targets=ids)
        loss.backward()
        opt.step()

        manager.save(step=3, model=tiny_model, optimizer=opt)
        autre_opt = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        manager.restore(model=tiny_model, optimizer=autre_opt)
        assert autre_opt.state_dict()["state"], "l'état de moment doit être restauré"

    def test_absence_de_checkpoint_nest_pas_une_erreur(self, tmp_path, tiny_model):
        # Le cas normal d'un premier run.
        assert CheckpointManager(tmp_path).restore(model=tiny_model) == 0

    def test_meilleur_suivi_separement(self, tmp_path, tiny_model):
        manager = CheckpointManager(tmp_path, monitor="val_loss")
        manager.save(step=1, model=tiny_model, metrics={"val_loss": 3.0})
        manager.save(step=2, model=tiny_model, metrics={"val_loss": 5.0})
        assert manager.best_value == 3.0
        assert (tmp_path / "best.pt").is_file()
        assert torch.load(tmp_path / "best.pt", weights_only=False)["step"] == 1

    def test_purge_conserve_les_dernieres(self, tmp_path, tiny_model):
        manager = CheckpointManager(tmp_path, keep_last=2)
        for step in range(1, 6):
            manager.save(step=step, model=tiny_model)
        assert len(list(tmp_path.glob("step-*.pt"))) == 2

    def test_pas_de_fichier_temporaire_residuel(self, tmp_path, tiny_model):
        # L'écriture atomique ne doit jamais laisser de .tmp derrière elle.
        CheckpointManager(tmp_path).save(step=1, model=tiny_model)
        assert not list(tmp_path.glob("*.tmp"))

    def test_chargement_strict(self, tmp_path, tiny_model):
        # Un modèle rechargé partiellement s'entraîne sans erreur visible et
        # donne des résultats faux.
        manager = CheckpointManager(tmp_path)
        manager.save(step=1, model=tiny_model)
        plus_profond = Thadeus(ModelConfig(**{**tiny_model.cfg.model_dump(), "n_layers": 4}))
        with pytest.raises(RuntimeError):
            manager.restore(model=plus_profond)

    def test_unwrap_traverse_torch_compile(self, tiny_model):
        # Un modèle compilé préfixe ses clés par `_orig_mod.` : sauvegarder tel
        # quel produit un checkpoint irrécupérable en mode non compilé.
        class FausseEnveloppe:
            def __init__(self, module):
                self._orig_mod = module

        assert unwrap(FausseEnveloppe(tiny_model)) is tiny_model
        assert unwrap(tiny_model) is tiny_model


class TestConfiguration:
    def test_lot_effectif(self):
        cfg = TrainConfig(batch_size=12, grad_accum=8)
        assert cfg.effective_batch_tokens == 96

    def test_configs_du_depot_valides(self):
        for name in ("train/small.toml", "train/smoke.toml"):
            assert TrainConfig(**load_config(name)).total_steps > 0

    def test_smoke_herite_de_small(self):
        small = TrainConfig(**load_config("train/small.toml"))
        smoke = TrainConfig(**load_config("train/smoke.toml"))
        assert smoke.seed == small.seed
        assert smoke.total_steps < small.total_steps
        assert not smoke.compile, "la compilation coûterait plus que le run de fumée"

    def test_identite_ignore_la_duree_du_run(self):
        # Régression : allonger un run changeait le hash de config, donc le
        # répertoire d'artefact, donc le checkpoint devenait introuvable — et
        # l'entraînement repartait de zéro **sans le dire**. Or décider quand
        # s'arrêter pendant le run est tout l'intérêt du planificateur WSD.
        court = TrainConfig(total_steps=60)
        long = TrainConfig(total_steps=90)
        assert court.identity() == long.identity()

    def test_identite_ignore_les_reglages_de_conduite(self):
        base = TrainConfig()
        bavard = TrainConfig(log_every=1, checkpoint_every=10, sample_every=5, keep_last=9)
        assert base.identity() == bavard.identity()

    def test_identite_distingue_les_vraies_experiences(self):
        base = TrainConfig()
        for autre in (
            TrainConfig(optim={"lr": 1e-3}),
            TrainConfig(batch_size=24),
            TrainConfig(seed=7),
            TrainConfig(model_config_path="model/tiny.toml"),
            TrainConfig(optim={"schedule": {"name": "cosine"}}),
        ):
            assert autre.identity() != base.identity()

    def test_identite_ignore_la_duree_du_palier(self):
        # Allonger le palier WSD ne change pas l'expérience ; changer de
        # planificateur, si.
        a = TrainConfig(optim={"schedule": {"name": "wsd", "warmup_steps": 200}})
        b = TrainConfig(optim={"schedule": {"name": "wsd", "warmup_steps": 500}})
        assert a.identity() == b.identity()

    def test_cle_inconnue_rejetee(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TrainConfig(batchsize=12)  # type: ignore[call-arg]


class TestInitialisationDepuisCheckpoint:
    """`init_from` hérite des poids, jamais de l'état d'optimiseur."""

    def test_fait_partie_de_l_identite(self):
        # Deux fine-tunings partant de checkpoints différents sont deux
        # expériences distinctes, même à config identique par ailleurs.
        a = TrainConfig(init_from="/chemin/a.pt")
        b = TrainConfig(init_from="/chemin/b.pt")
        assert a.identity() != b.identity()

    def test_absent_par_defaut(self):
        assert TrainConfig().init_from is None

    def test_config_du_depot_valide(self):
        cfg = TrainConfig(**load_config("train/vault_ft.toml"))
        assert cfg.init_from and cfg.init_from.endswith(".pt")
        assert cfg.tokens_label == "vault_ft"
        # muP doit être identique au run de base, sinon les poids chargés
        # seraient réinterprétés avec un autre facteur de logits.
        base = TrainConfig(**load_config("train/medium_mup.toml"))
        assert cfg.mup == base.mup
        assert cfg.optim.lr < base.optim.lr, "un fine-tuning s'entraîne plus doucement"
