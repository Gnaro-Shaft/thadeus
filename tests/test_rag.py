"""Étage RAG : découpage, index BM25, assemblage du prompt.

Aucun modèle chargé — la mécanique de récupération se teste seule.
"""

from __future__ import annotations

import pytest

from thadeus.rag.chunk import chunk_note
from thadeus.rag.index import BM25Index, Passage, tokenize


class TestTokenisation:
    def test_accents_replies(self):
        # On écrit « référence » dans une note et « reference » dans une
        # requête tapée vite : sans repli, la note ne remonte pas.
        assert tokenize("Référence") == tokenize("reference")

    def test_mots_outils_retires(self):
        # Ils sont dans presque tous les passages : IDF quasi nul, mais ils
        # gonflent la longueur et faussent la normalisation de BM25.
        assert "le" not in tokenize("le chat dort")
        assert "chat" in tokenize("le chat dort")

    def test_lettres_isolees_ecartees(self):
        assert tokenize("a b chat") == ["chat"]

    def test_nombres_conserves(self):
        # Une note « SASU 2026 » doit être retrouvable par son année.
        assert "2026" in tokenize("immatriculation SASU 2026")


class TestDecoupage:
    NOTE = (
        "Préambule de la note.\n\n"
        "## Première section\n\n" + "mot " * 200 + "\n\n"
        "## Deuxième section\n\n" + "autre " * 40
    )

    def test_decoupe_aux_sections(self):
        ps = list(chunk_note(self.NOTE, title="Test", source="t.md", target_words=150))
        titres = {p.title for p in ps}
        assert any("Première section" in t for t in titres)
        assert any("Deuxième section" in t for t in titres)

    def test_aucun_passage_intenable(self):
        # Le modèle n'a que 1024 tokens : un passage de plusieurs milliers de
        # mots serait silencieusement écarté du prompt — et ce serait le
        # passage LE MIEUX CLASSÉ qui disparaîtrait.
        geant = "## Bloc\n\n" + "mot " * 3000
        ps = list(chunk_note(geant, title="T", source="t.md", target_words=150))
        assert ps and all(len(p.text.split()) <= 300 for p in ps)

    def test_note_sans_titre_reste_exploitable(self):
        ps = list(chunk_note("mot " * 100, title="T", source="t.md"))
        assert len(ps) == 1

    def test_fragments_trop_courts_ecartes(self):
        # Dans un contexte aussi contraint, un passage de dix mots ne mérite
        # pas la place qu'il occuperait.
        ps = list(chunk_note("## S\n\ntrois mots ici", title="T", source="t.md", min_words=20))
        assert not ps

    def test_identifiants_uniques(self):
        ps = list(chunk_note(self.NOTE, title="T", source="t.md", target_words=50))
        assert len({p.id for p in ps}) == len(ps)


@pytest.fixture
def index():
    idx = BM25Index()
    for i, (titre, texte) in enumerate(
        [
            (
                "Optimiseur Muon",
                "Le momentum orthogonalisé par Newton-Schulz égalise les directions.",
            ),
            (
                "Corpus français",
                "Le corpus mélange Wikipédia, FineWeb et des livres du domaine public.",
            ),
            (
                "Tokenizer BPE",
                "Les élisions restent rattachées au mot-outil, ce qui économise des tokens.",
            ),
            ("Réunion budget", "Le budget prévisionnel de la SASU pour 2026 est arrêté."),
        ]
    ):
        idx.add(Passage(id=f"p{i}", text=texte, source=f"n{i}.md", title=titre))
    return idx.build()


class TestRecherche:
    def test_retrouve_par_le_corps(self, index):
        r = index.search("Newton-Schulz momentum", k=1)
        assert r and r[0][0].title == "Optimiseur Muon"

    def test_retrouve_par_le_titre(self, index):
        # Le titre est indexé en plus du corps : une note nommée « Muon » doit
        # remonter sur « muon » même absent du texte.
        r = index.search("muon", k=1)
        assert r and r[0][0].title == "Optimiseur Muon"

    def test_insensible_aux_accents(self, index):
        assert index.search("wikipedia", k=1)[0][0].title == "Corpus français"

    def test_ordonne_par_pertinence(self, index):
        # La requête doit toucher PLUSIEURS passages, sinon r[0] est r[-1] et
        # le test ne vérifie rien.
        r = index.search("élisions tokens corpus", k=4)
        assert len(r) >= 2, "requête trop spécifique pour tester un ordre"
        assert r[0][0].title == "Tokenizer BPE"
        assert r[0][1] > r[-1][1]

    def test_ne_renvoie_que_les_passages_touches(self, index):
        # BM25 ne classe pas tout le corpus : un passage sans aucun terme de la
        # requête n'a pas de score, et l'inclure serait du bruit.
        assert len(index.search("Newton-Schulz", k=4)) == 1

    def test_requete_sans_terme_utile(self, index):
        # Une requête faite uniquement de mots-outils ne doit rien renvoyer
        # plutôt qu'un classement arbitraire.
        assert index.search("le de la et", k=3) == []

    def test_index_vide(self):
        assert BM25Index().build().search("quoi que ce soit") == []

    def test_idf_jamais_negatif(self, index):
        # La variante lissée : un terme présent partout ne doit pas devenir
        # pénalisant, ce que produit l'IDF naïf.
        for terme in ("corpus", "budget", "muon"):
            assert index._idf(terme) >= 0

    def test_poids_du_titre_configurable(self):
        # Le poids sert au cas « je me souviens du titre » ; sa mesure doit
        # rester séparable, sous peine d'évaluation circulaire.
        sans = BM25Index(title_weight=0)
        sans.add(Passage(id="a", text="texte quelconque", source="a.md", title="muon"))
        sans.build()
        assert sans.search("muon") == []

    def test_aller_retour_disque(self, index, tmp_path):
        chemin = index.save(tmp_path / "index.json")
        recharge = BM25Index.load(chemin)
        assert len(recharge.passages) == len(index.passages)
        assert recharge.search("muon", k=1)[0][0].title == "Optimiseur Muon"


class TestPrompt:
    class FauxCodec:
        def encode(self, text):
            return list(range(len(text.split())))

    def test_tient_dans_le_budget(self, index):
        from thadeus.rag.answer import build_prompt

        codec = self.FauxCodec()
        passages = [p for p, _ in index.search("corpus", k=4)] or index.passages
        prompt = build_prompt("ma question", passages, codec=codec, budget=40)
        assert len(codec.encode(prompt)) <= 40

    def test_passage_trop_long_saute_plutot_que_coupe(self, index):
        # Couper une phrase en deux introduit du bruit que le modèle tentera de
        # continuer : mieux vaut omettre le passage.
        from thadeus.rag.answer import build_prompt

        long = Passage(id="x", text="mot " * 500, source="x.md", title="Long")
        prompt = build_prompt("q", [long], codec=self.FauxCodec(), budget=60)
        assert "mot mot" not in prompt

    def test_format_de_continuation(self, index):
        # Le modèle n'est pas instruit : il continue du texte. Un format de
        # dialogue produirait du charabia.
        from thadeus.rag.answer import build_prompt

        prompt = build_prompt("ma question", index.passages[:1], codec=self.FauxCodec(), budget=200)
        assert prompt.rstrip().endswith("Réponse :")
        assert "Question : ma question" in prompt
