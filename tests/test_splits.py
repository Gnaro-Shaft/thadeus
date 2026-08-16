"""Étanchéité et représentativité du découpage train/validation.

Le test central de ce fichier est celui de la **fuite** : on fabrique un corpus
dont les zones de validation ne contiennent qu'un jeton reconnaissable, on tire
des dizaines de milliers de fenêtres d'entraînement, et on vérifie que ce jeton
n'apparaît jamais. Une fuite ici ne produirait aucune erreur — elle rendrait
seulement la validation complaisante, et donc muette le jour où le modèle se
dégrade.
"""

from __future__ import annotations

import numpy as np
import pytest

from thadeus.train.splits import decouper, parcourir, tirer_positions
from thadeus.train.tokens import TokenShardWriter, TokenStore

VOCAB = 64
MARQUEUR = 63  # n'apparaît QUE dans les zones de validation


class TestDecoupage:
    def test_les_zones_couvrent_tout_le_corpus_sans_trou(self):
        val, train = decouper(100_000, 10_000, 10)
        zones = sorted(val + train)
        assert zones[0][0] == 0
        for (d1, l1), (d2, _) in zip(zones, zones[1:], strict=False):
            assert d1 + l1 == d2, "trou ou chevauchement entre zones"
        assert zones[-1][0] + zones[-1][1] == 100_000

    def test_les_zones_ne_se_chevauchent_pas(self):
        val, train = decouper(100_000, 10_000, 10)
        occupe = np.zeros(100_000, dtype=np.int8)
        for d, longueur in val + train:
            occupe[d : d + longueur] += 1
        assert occupe.max() == 1, "un token appartient à deux zones"
        assert occupe.min() == 1, "un token n'appartient à aucune zone"

    def test_le_volume_de_validation_est_respecte(self):
        val, _ = decouper(1_000_000, 50_000, 50)
        assert sum(longueur for _, longueur in val) == 50_000

    def test_la_validation_est_repartie_et_non_en_queue(self):
        """LE point du correctif : ne pas tout prélever au même endroit."""
        val, _ = decouper(1_000_000, 10_000, 20)
        debuts = [d for d, _ in val]
        assert len(val) == 20
        assert min(debuts) < 100_000, "aucun bloc dans le premier dixième du corpus"
        assert max(debuts) > 900_000, "aucun bloc dans le dernier dixième du corpus"
        ecarts = np.diff(debuts)
        assert ecarts.std() < 1, "les blocs ne sont pas régulièrement espacés"

    def test_sans_validation_tout_est_entrainement(self):
        val, train = decouper(1_000, 0, 10)
        assert val == []
        assert train == [(0, 1_000)]

    def test_blocs_vides_refuses(self):
        with pytest.raises(ValueError, match="blocs vides"):
            decouper(1_000_000, 5, 10)

    def test_bloc_plus_long_que_son_troncon_refuse(self):
        with pytest.raises(ValueError, match="place à l'entraînement"):
            decouper(1_000, 1_800, 2)

    def test_un_ratio_absurde_reste_coherent(self):
        """`decouper` ne juge pas du ratio — il garantit la cohérence.

        Réserver 90 % du corpus à la validation est une erreur de configuration,
        mais pas une incohérence : les zones restent complémentaires et
        étanches. Ce qui protège réellement, c'est le refus au tirage quand plus
        aucune zone ne peut contenir une fenêtre — testé plus bas, et
        indépendant de tout seuil arbitraire.
        """
        val, train = decouper(1_000, 900, 2)
        assert sum(longueur for _, longueur in val) == 900
        assert sum(longueur for _, longueur in train) == 100
        with pytest.raises(ValueError, match="aucune zone"):
            tirer_positions(np.random.default_rng(0), 4, train, 200)


class TestTirage:
    def test_toutes_les_positions_tombent_dans_les_zones(self):
        _, train = decouper(100_000, 10_000, 10)
        rng = np.random.default_rng(0)
        pos = tirer_positions(rng, 5_000, train, 33)
        for p in pos:
            assert any(d <= p and p + 33 <= d + longueur for d, longueur in train)

    def test_deux_tirages_de_meme_graine_sont_identiques(self):
        _, train = decouper(100_000, 10_000, 10)
        a = tirer_positions(np.random.default_rng(7), 200, train, 33)
        b = tirer_positions(np.random.default_rng(7), 200, train, 33)
        assert np.array_equal(a, b)

    def test_des_graines_differentes_donnent_des_tirages_differents(self):
        _, train = decouper(100_000, 10_000, 10)
        a = tirer_positions(np.random.default_rng(1), 200, train, 33)
        b = tirer_positions(np.random.default_rng(2), 200, train, 33)
        assert not np.array_equal(a, b)

    def test_zone_trop_courte_pour_la_fenetre(self):
        with pytest.raises(ValueError, match="aucune zone"):
            tirer_positions(np.random.default_rng(0), 10, [(0, 5), (100, 5)], 50)

    def test_le_parcours_reste_dans_les_zones(self):
        val, _ = decouper(100_000, 10_000, 10)
        for p in parcourir(val, 100):
            assert any(d <= p and p + 100 <= d + longueur for d, longueur in val)


@pytest.fixture
def corpus_marque(tmp_path):
    """Corpus où le jeton MARQUEUR n'existe QUE dans les zones de validation."""
    n, val_tokens, blocs = 200_000, 20_000, 20
    zones_val, _ = decouper(n, val_tokens, blocs)
    tokens = np.zeros(n, dtype=np.uint16)
    rng = np.random.default_rng(0)
    tokens[:] = rng.integers(0, MARQUEUR, size=n)  # jamais MARQUEUR
    for d, longueur in zones_val:
        tokens[d : d + longueur] = MARQUEUR

    directory = tmp_path / "tokens"
    with TokenShardWriter(directory, vocab_size=VOCAB, tokens_per_shard=n) as w:
        w.write(tokens.tolist())
    return TokenStore(directory, val_tokens=val_tokens, val_blocks=blocs)


class TestEtancheite:
    def test_aucune_fenetre_d_entrainement_ne_touche_la_validation(self, corpus_marque):
        """LE test du correctif. Une fuite ici ne lèverait aucune erreur."""
        vus = []
        for graine in range(60):
            fenetres, _ = corpus_marque.windows(
                batch_size=16, seq_len=64, seed=graine, split="train"
            )
            vus.append(fenetres)
        tous = np.concatenate([v.ravel() for v in vus])
        assert len(tous) > 60_000, "échantillon trop petit pour conclure"
        assert (tous == MARQUEUR).sum() == 0, (
            f"{(tous == MARQUEUR).sum()} tokens de validation ont fui dans l'entraînement"
        )

    def test_la_validation_ne_lit_que_la_validation(self, corpus_marque):
        fenetres, _ = corpus_marque.windows(batch_size=16, seq_len=64, seed=0, split="val")
        assert (fenetres == MARQUEUR).all(), "la validation lit des tokens d'entraînement"

    def test_le_parcours_sequentiel_reste_dans_la_validation(self, corpus_marque):
        for fenetres, _ in corpus_marque.sequential_windows(
            batch_size=4, seq_len=64, split="val", limit=20
        ):
            assert (fenetres == MARQUEUR).all()

    def test_le_compte_de_tokens_d_entrainement_exclut_la_validation(self, corpus_marque):
        assert corpus_marque.n_train_tokens == corpus_marque.n_tokens - 20_000

    def test_la_validation_couvre_plusieurs_blocs(self, corpus_marque):
        """Sans cela, on aurait remplacé une tranche unique par une autre."""
        lots = list(corpus_marque.sequential_windows(batch_size=1, seq_len=64, split="val"))
        assert len(lots) > 20, "le parcours ne visite qu'une poignée de positions"
