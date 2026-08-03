"""Étage évaluation : perplexité, sondes, jetons de service."""

from __future__ import annotations

import math

import pytest
import torch

from thadeus.eval.perplexity import GroupedScore, Score, evaluate_documents
from thadeus.eval.probes import PROBES, MinimalPair, run_probes, sequence_logprob
from thadeus.model import ModelConfig, Thadeus

VOCAB = 128


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


class FauxCodec:
    """Codec caractère, suffisant pour tester la mécanique d'évaluation."""

    def encode(self, text: str) -> list[int]:
        return [min(ord(c), VOCAB - 1) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)

    @property
    def service_ids(self) -> list[int]:
        return [1, 2]


class FauxDoc:
    def __init__(self, text: str, source: str = "s", lang: str = "fr"):
        self.text, self.source, self.lang = text, source, lang


class TestScore:
    def test_perplexite_est_l_exponentielle_de_la_perte(self):
        s = Score(total_loss=2.0 * 100, n_tokens=100, n_chars=400, n_documents=1)
        assert s.loss == 2.0
        assert s.perplexity == pytest.approx(math.exp(2.0))

    def test_bits_par_caractere_independant_du_tokenizer(self):
        # La seule métrique comparable entre modèles à tokenizers différents :
        # elle ne dépend que du texte, pas de son découpage.
        peu = Score(total_loss=100.0, n_tokens=50, n_chars=400, n_documents=1)
        beaucoup = Score(total_loss=100.0, n_tokens=200, n_chars=400, n_documents=1)
        assert peu.bits_per_char == beaucoup.bits_per_char
        assert peu.loss != beaucoup.loss

    def test_scores_s_additionnent_sur_les_comptages(self):
        # Faire la moyenne de moyennes pondérerait chaque lot également,
        # quel que soit son nombre de tokens.
        total = Score(10.0, 10, 40, 1) + Score(90.0, 90, 360, 2)
        assert total.n_tokens == 100
        assert total.loss == 1.0

    def test_division_par_zero_evitee(self):
        assert math.isnan(Score().loss)

    def test_groupes_et_total(self):
        g = GroupedScore()
        g.add("a", Score(10.0, 10, 40, 1))
        g.add("a", Score(10.0, 10, 40, 1))
        g.add("b", Score(40.0, 20, 80, 1))
        assert g.groups["a"].n_tokens == 20
        assert g.overall.n_tokens == 40


class TestPerplexite:
    def test_ventile_par_source(self, tiny_model):
        docs = [
            FauxDoc("bonjour le monde " * 5, source="a"),
            FauxDoc("hello world " * 5, source="b"),
        ]
        scores = evaluate_documents(
            tiny_model,
            FauxCodec(),
            docs,
            device=torch.device("cpu"),
            dtype=torch.float32,
            seq_len=32,
            group_by="source",
        )
        assert set(scores.groups) == {"a", "b"}
        assert scores.overall.n_documents == 2

    def test_modele_non_entraine_donne_ln_du_vocabulaire(self, tiny_model):
        # Validation de la chaîne de mesure elle-même : un modèle non entraîné
        # prédit uniformément, donc sa perte vaut ln(V). Si ce test échoue, ce
        # n'est pas le modèle qui est en cause, c'est la mesure.
        scores = evaluate_documents(
            tiny_model,
            FauxCodec(),
            [FauxDoc("".join(chr(65 + i % 26) for i in range(200)))],
            device=torch.device("cpu"),
            dtype=torch.float32,
            seq_len=32,
        )
        assert abs(scores.overall.loss - math.log(VOCAB)) < 0.6

    def test_ne_laisse_pas_le_modele_en_mode_evaluation(self, tiny_model):
        # Un modèle laissé en `eval()` après une évaluation périodique
        # désactiverait le dropout pour tout le reste de l'entraînement.
        tiny_model.train()
        evaluate_documents(
            tiny_model,
            FauxCodec(),
            [FauxDoc("texte " * 20)],
            device=torch.device("cpu"),
            dtype=torch.float32,
            seq_len=32,
        )
        assert tiny_model.training


class TestSondes:
    def test_paires_bien_formees(self):
        for p in PROBES:
            assert p.good != p.bad, f"paire identique : {p.category}"
            # Une phrase nettement plus longue serait favorisée par sa longueur,
            # et on mesurerait la longueur au lieu de la grammaire.
            ecart = abs(len(p.good) - len(p.bad)) / max(len(p.good), len(p.bad))
            assert ecart < 0.25, f"{p.category} : longueurs trop différentes ({p.good!r})"

    def test_categories_couvrent_le_francais_et_le_code(self):
        categories = {p.category for p in PROBES}
        assert {"accord_verbe", "elision", "genre", "pluriel", "code"} <= categories

    def test_chaque_categorie_a_plusieurs_paires(self):
        from collections import Counter

        for cat, n in Counter(p.category for p in PROBES).items():
            assert n >= 2, f"{cat} n'a qu'une paire — trop peu pour un taux lisible"

    def test_execution(self, tiny_model):
        res = run_probes(
            tiny_model,
            FauxCodec(),
            device=torch.device("cpu"),
            dtype=torch.float32,
            pairs=[MinimalPair("test", "les chats dorment", "les chats dort")],
        )
        assert res["test"].total == 1
        assert res["test"].accuracy in (0.0, 1.0)

    def test_logprob_totale_et_non_moyenne(self, tiny_model):
        # Deux fois le même texte doit donner ~deux fois la log-probabilité :
        # la mesure est bien une somme, pas une moyenne.
        codec = FauxCodec()
        court = sequence_logprob(tiny_model, codec, "abcdefgh", torch.device("cpu"), torch.float32)
        long = sequence_logprob(
            tiny_model, codec, "abcdefgh" * 2, torch.device("cpu"), torch.float32
        )
        assert long < court


class TestGeneration:
    def test_jetons_de_service_interdits(self, tiny_model):
        # Un modèle ne doit jamais produire <|pad|> ni ses créneaux réservés.
        interdits = [5, 6, 7]
        sortie = tiny_model.generate(
            torch.randint(0, VOCAB, (1, 4)), max_new_tokens=40, forbidden=interdits
        )
        produits = set(sortie[0, 4:].tolist())
        assert not (produits & set(interdits))

    def test_sans_interdiction_tout_le_vocabulaire_est_atteignable(self, tiny_model):
        sortie = tiny_model.generate(torch.randint(0, VOCAB, (1, 4)), max_new_tokens=40)
        assert sortie.shape == (1, 44)
