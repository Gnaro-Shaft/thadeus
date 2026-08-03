"""Étage tokenizer : pré-tokenisation, codec, métriques.

L'entraînement d'un BPE réel est testé sur un corpus minuscule fabriqué ici —
aucun accès réseau.
"""

from __future__ import annotations

import pytest

from thadeus.data.schema import Document
from thadeus.tokenizer.codec import Codec, SpecialTokens
from thadeus.tokenizer.metrics import Fertility, compare, measure, measure_by
from thadeus.tokenizer.pretokenize import (
    ELISIONS,
    FRENCH_PATTERN,
    GPT2_PATTERN,
    PATTERNS,
    split,
)
from thadeus.tokenizer.train import TokenizerConfig, build_tokenizer, train_tokenizer

FR_ELISIONS = "L'homme qu'il a rencontré aujourd'hui n'était pas celui qu'on attendait."


class TestPreTokenisation:
    def test_elisions_rattachees_au_mot_outil(self):
        # Le cœur du gain français : " qu'" est un seul morceau, donc le BPE
        # pourra en faire un seul token. Avec le motif GPT-2 c'est impossible.
        assert " qu'" in split(FR_ELISIONS, FRENCH_PATTERN)
        assert " qu'" not in split(FR_ELISIONS, GPT2_PATTERN)

    def test_gain_mesurable_sur_du_francais(self):
        gpt2 = len(split(FR_ELISIONS, GPT2_PATTERN))
        french = len(split(FR_ELISIONS, FRENCH_PATTERN))
        assert french < gpt2
        assert 1 - french / gpt2 > 0.15, "on attend plus de 15 % de morceaux en moins"

    def test_aucune_regression_sur_l_anglais(self):
        # Le corpus est bilingue : le motif français ne doit rien coûter à l'anglais.
        anglais = "The dog's toy isn't there and they've gone."
        assert len(split(anglais, FRENCH_PATTERN)) == len(split(anglais, GPT2_PATTERN))

    def test_contractions_anglaises_preservees(self):
        assert "'s" in split("The dog's toy", FRENCH_PATTERN)

    @pytest.mark.parametrize("elision", ["l", "d", "j", "n", "qu", "jusqu", "aujourd"])
    def test_chaque_elision_est_un_morceau(self, elision):
        assert f" {elision}'" in split(f"et {elision}'abc", FRENCH_PATTERN)

    def test_elisions_longues_avant_les_courtes(self):
        # L'alternance prend la première branche qui matche : si "j" passait
        # avant "jusqu", " jusqu'" se découperait en " j" + "usqu'".
        assert " jusqu'" in split("et jusqu'à demain", FRENCH_PATTERN)
        assert ELISIONS.index("jusqu") < ELISIONS.index("j")

    def test_elisions_insensibles_a_la_casse(self):
        assert "L'" in split(FR_ELISIONS, FRENCH_PATTERN)

    def test_nombres_decoupes_par_tranches_de_trois(self):
        morceaux = split("En 1234567 unités", FRENCH_PATTERN)
        assert " 123" in morceaux and "456" in morceaux

    def test_variante_longnum_garde_les_nombres_entiers(self):
        # Variante témoin : isole l'effet des élisions de celui des nombres.
        assert " 1234567" in split("En 1234567 unités", PATTERNS["french_longnum"])
        # L'élision reste rattachée : seul le traitement des nombres change.
        assert " qu'" in split("dit qu'il", PATTERNS["french_longnum"])

    def test_apostrophe_capturee_meme_hors_elision(self):
        # Régression : une traduction naïve de \p{L} cassait les classes niées,
        # l'apostrophe disparaissait sans erreur et l'outil sous-comptait.
        assert "".join(split("abc'def", FRENCH_PATTERN)) == "abc'def"

    @pytest.mark.parametrize("nom", list(PATTERNS))
    def test_tout_motif_couvre_le_texte_sans_perte(self, nom):
        texte = "Été 2026 : l'IA, c'est 42 % — «citations» & symboles\n\tet tabulations."
        assert "".join(split(texte, PATTERNS[nom])) == texte


class TestSpecialTokens:
    def test_ordre_fixe_les_identifiants(self):
        tokens = SpecialTokens(reserved=3).as_list()
        assert tokens[0] == "<|endoftext|>"
        assert tokens[1] == "<|pad|>"
        assert tokens[2:] == ["<|reserved_0|>", "<|reserved_1|>", "<|reserved_2|>"]

    def test_creneaux_reserves_comptes(self):
        assert len(SpecialTokens(reserved=16)) == 18


@pytest.fixture(scope="module")
def petit_codec():
    """Entraîne un vrai BPE sur un corpus minuscule — pas de réseau."""
    from tokenizers import pre_tokenizers, trainers

    textes = [
        FR_ELISIONS,
        "Il n'y a pas d'autre solution que d'attendre qu'elle arrive.",
        "L'apprentissage automatique demande beaucoup de données d'entraînement.",
        "The quick brown fox jumps over the lazy dog repeatedly.",
        "def somme(x):\n    return sum(i for i in range(x))",
    ] * 40

    special = SpecialTokens(reserved=2)
    tokenizer = build_tokenizer(pattern="french", special=special)
    tokenizer.train_from_iterator(
        textes,
        trainer=trainers.BpeTrainer(
            vocab_size=600,
            min_frequency=1,
            special_tokens=special.as_list(),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    return Codec(tokenizer=tokenizer, special=special)


class TestCodec:
    def test_identifiants_speciaux_en_tete(self, petit_codec):
        assert petit_codec.eot_id == 0
        assert petit_codec.pad_id == 1

    @pytest.mark.parametrize(
        "texte",
        [
            FR_ELISIONS,
            "Texte  avec   espaces\tmultiples\n\net sauts.",
            "Émojis 🇫🇷🚀 et 中文 jamais vus à l'entraînement",
            "",
            "\n",
        ],
    )
    def test_aller_retour_exact(self, petit_codec, texte):
        # Garantie byte-level : tout texte est encodable et restitué à l'identique,
        # même composé de caractères absents du corpus d'entraînement.
        assert petit_codec.decode(petit_codec.encode(texte)) == texte

    def test_aucun_token_inconnu(self, petit_codec):
        # L'alphabet initial contient les 256 octets : rien ne peut échouer.
        ids = petit_codec.encode("龍 🐉 \x01\x02")
        assert ids and all(i < petit_codec.vocab_size for i in ids)

    def test_eot_ajoute_a_la_demande(self, petit_codec):
        assert petit_codec.encode("bonjour", add_eot=True)[-1] == petit_codec.eot_id
        assert petit_codec.encode("bonjour")[-1] != petit_codec.eot_id

    def test_encode_batch_coherent_avec_encode(self, petit_codec):
        textes = ["premier texte", "second texte"]
        assert petit_codec.encode_batch(textes) == [petit_codec.encode(t) for t in textes]

    def test_count_egale_la_longueur(self, petit_codec):
        assert petit_codec.count(FR_ELISIONS) == len(petit_codec.encode(FR_ELISIONS))

    def test_elision_apprise_comme_un_seul_token(self, petit_codec):
        # La preuve que la pré-tokenisation paie : " qu'" existe dans le vocabulaire.
        vocab = petit_codec.tokenizer.get_vocab()
        assert any(token.endswith("qu'") for token in vocab), "aucune élision fusionnée"

    def test_sauvegarde_et_rechargement(self, petit_codec, tmp_path):
        petit_codec.save(tmp_path)
        rechargé = Codec.load(tmp_path)
        assert rechargé.vocab_size == petit_codec.vocab_size
        assert rechargé.encode(FR_ELISIONS) == petit_codec.encode(FR_ELISIONS)


class TestMetriques:
    def test_fertilite_ratios(self):
        f = Fertility(tokens=150, words=100, chars=600, documents=2)
        assert f.tokens_per_word == 1.5
        assert f.chars_per_token == 4.0

    def test_fertilites_s_additionnent_sur_les_comptages(self):
        # Deux mesures ne s'additionnent jamais sur leurs moyennes.
        total = Fertility(10, 5, 40, 1) + Fertility(30, 20, 120, 2)
        assert (total.tokens, total.words, total.documents) == (40, 25, 3)
        assert total.tokens_per_word == 1.6

    def test_division_par_zero_evitee(self):
        assert Fertility(0, 0, 0, 0).tokens_per_word == 0.0

    def test_measure(self, petit_codec):
        f = measure(petit_codec.count, [FR_ELISIONS, "Un autre texte français."])
        assert f.documents == 2 and f.tokens > 0

    def test_measure_by_ventile(self, petit_codec):
        docs = [
            Document(id="a", text=FR_ELISIONS, source="s", lang="fr"),
            Document(id="b", text="The quick brown fox.", source="s", lang="en"),
        ]
        groupes = measure_by(petit_codec.count, docs, key=lambda d: d.lang)
        assert set(groupes) == {"fr", "en"}

    def test_compare_calcule_le_gain_relatif(self, petit_codec):
        docs = [Document(id="a", text=FR_ELISIONS, source="s", lang="fr")]
        result = compare(
            {"reference": lambda t: 100, "notre": petit_codec.count},
            docs,
            key=lambda d: d.lang,
            baseline="reference",
        )
        assert result["tokenizers"]["reference"]["tokens_saved_vs_baseline"] == 0.0
        assert result["tokenizers"]["notre"]["tokens_saved_vs_baseline"] != 0.0


class TestConfiguration:
    def test_motif_inconnu_traite_comme_regex_brute(self):
        # Permet d'expérimenter un motif sans l'inscrire dans le code.
        assert build_tokenizer(pattern=r"\S+|\s+") is not None

    def test_motif_vide_rejete(self):
        with pytest.raises(ValueError, match="vide"):
            build_tokenizer(pattern="   ")

    def test_config_rejette_une_cle_inconnue(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TokenizerConfig(vocab_sizee=32_000)  # type: ignore[call-arg]

    def test_corpus_introuvable_donne_la_commande_a_lancer(self, tmp_path, monkeypatch):
        monkeypatch.setattr("thadeus.core.artifacts.ARTIFACT_ROOT", tmp_path)
        with pytest.raises(FileNotFoundError, match="build_corpus"):
            train_tokenizer({"label": "x", "corpus_label": "absent"})
