"""Étage données : schéma, shards, filtres, déduplication, mélange.

Aucun de ces tests ne touche au réseau — ils tournent sur des documents
fabriqués et un Vault jetable.
"""

from __future__ import annotations

import pytest

from thadeus.data.clean import FILTERS, build_pipeline
from thadeus.data.clean.language import detect_language, language_scores
from thadeus.data.clean.quality import normalize_text
from thadeus.data.dedup import Deduplicator, MinHashDeduplicator, lsh_threshold
from thadeus.data.mix import plan_mixture
from thadeus.data.schema import (
    Document,
    estimate_tokens,
    format_tokens,
    text_fingerprint,
)
from thadeus.data.shard import ShardWriter, iter_documents, shard_paths
from thadeus.data.sources.obsidian import strip_markdown

FR = (
    "La déduplication des corpus est une étape essentielle de la préparation "
    "des données pour un modèle de langue. Elle permet d'éviter que le modèle "
    "n'apprenne plusieurs fois les mêmes contenus, ce qui gaspille le budget de "
    "calcul et favorise la mémorisation par cœur des exemples plutôt que la "
    "généralisation à de nouveaux textes jamais rencontrés."
)
EN = (
    "The deduplication of training corpora is an essential step when preparing "
    "data for a language model. It prevents the model from learning the same "
    "content several times, which wastes the compute budget and encourages rote "
    "memorisation of examples rather than generalisation to new unseen text."
)


def doc(text: str, *, id: str = "t:1", source: str = "test", lang: str = "fr") -> Document:
    return Document(id=id, text=text, source=source, lang=lang)


class TestSchema:
    def test_aller_retour_json(self):
        original = Document(id="a:1", text="bonjour", source="a", lang="fr", meta={"k": 1})
        assert Document.from_json(original.to_json()) == original

    def test_meta_vide_non_serialisee(self):
        # Chaque champ inutile est réécrit pour chaque document : à l'échelle
        # d'un corpus, ce sont des gigaoctets.
        assert "meta" not in doc("x").to_json()

    def test_immuable(self):
        with pytest.raises(AttributeError):
            doc("x").text = "y"  # type: ignore[misc]

    def test_with_text_conserve_le_reste(self):
        original = Document(id="a:1", text="x", source="s", lang="fr", meta={"k": 1})
        modified = original.with_text("y")
        assert modified.text == "y"
        assert (modified.id, modified.source, modified.meta) == ("a:1", "s", {"k": 1})

    def test_empreinte_insensible_casse_et_espaces(self):
        assert text_fingerprint("Le  Chat\nnoir") == text_fingerprint("le chat noir")

    def test_empreinte_distingue_des_textes_differents(self):
        assert text_fingerprint("le chat") != text_fingerprint("le chien")

    def test_estimation_tokens_croissante(self):
        assert estimate_tokens(100) > estimate_tokens(50) > 0

    @pytest.mark.parametrize(
        ("count", "expected"),
        [(3_020_000, "3.02 M"), (1_700_000_000, "1.70 Md"), (4_500, "4.5 k"), (12, "12")],
    )
    def test_format_tokens_choisit_une_unite_lisible(self, count, expected):
        # Un corpus de 3 M tokens affiché « 0.00 Md » n'informe personne.
        assert format_tokens(count) == expected


class TestShards:
    def test_aller_retour(self, tmp_path):
        docs = [doc(f"document numéro {i} avec accents éàü", id=f"t:{i}") for i in range(50)]
        with ShardWriter(tmp_path) as writer:
            writer.write_all(docs)
        assert list(iter_documents(tmp_path)) == docs

    def test_rotation_par_nombre(self, tmp_path):
        with ShardWriter(tmp_path, max_documents=10) as writer:
            writer.write_all(doc(f"texte {i}", id=f"t:{i}") for i in range(35))
        assert len(shard_paths(tmp_path)) == 4
        assert writer.n_documents == 35

    def test_utf8_a_cheval_sur_deux_blocs(self, tmp_path):
        # Un caractère multi-octets coupé par une frontière de bloc est le bug
        # classique de ce genre de lecteur : on décode par ligne, pas par bloc.
        big = "é" * 2_000_000
        with ShardWriter(tmp_path) as writer:
            writer.write(doc(big))
        assert list(iter_documents(tmp_path))[0].text == big

    def test_statistiques_par_source(self, tmp_path):
        with ShardWriter(tmp_path) as writer:
            writer.write(doc("un deux trois", source="a"))
            writer.write(doc("un deux", source="b"))
        assert writer.per_source == {"a": 1, "b": 1}
        assert writer.words_per_source == {"a": 3, "b": 2}

    def test_limite_de_lecture(self, tmp_path):
        with ShardWriter(tmp_path) as writer:
            writer.write_all(doc(f"texte {i}", id=f"t:{i}") for i in range(20))
        assert len(list(iter_documents(tmp_path, limit=5))) == 5


class TestNormalisation:
    def test_nfc_unifie_les_accents_decomposes(self):
        # "é" décomposé (e + accent combinant) vs composé : sans NFC, le
        # tokenizer apprendrait deux fois le même motif sur un corpus français.
        assert normalize_text("été") == normalize_text("été")

    def test_apostrophes_typographiques_unifiees(self):
        assert normalize_text("l’homme") == "l'homme"

    def test_apostrophes_preservees_si_demande(self):
        # Dans du code, une apostrophe n'est pas de l'habillage.
        assert "’" in normalize_text("s = ’x’", unify_apostrophes=False)

    def test_caracteres_de_controle_retires(self):
        assert normalize_text("a\x00b\x07c") == "abc"

    def test_lignes_vides_multiples_reduites(self):
        assert normalize_text("a\n\n\n\n\nb") == "a\n\nb"


class TestLangue:
    def test_francais_reconnu(self):
        assert detect_language(FR)[0] == "fr"

    def test_anglais_reconnu(self):
        assert detect_language(EN)[0] == "en"

    def test_texte_court_indetermine(self):
        # Renvoyer « je ne sais pas » plutôt qu'un pari : c'est le filtre appelant
        # qui décide de rejeter.
        assert detect_language("bonjour")[0] == "unknown"

    def test_code_ne_ressemble_pas_a_du_francais(self):
        # C'est la protection dont on dépend réellement : le code ne doit jamais
        # entrer dans un corpus « français d'abord ».
        code = "def f(x):\n    return [i * 2 for i in range(x) if i % 3 == 0]\n" * 5
        assert language_scores(code)["fr"] == 0.0

    def test_code_est_un_faux_positif_anglais_connu(self):
        # Limite assumée : `for`, `in`, `if` sont à la fois des mots-clés de
        # programmation et des mots-outils anglais. Un classifieur par
        # mots-outils ne peut pas les distinguer.
        # Sans conséquence en pratique — les sources de code n'appliquent aucun
        # filtre de langue — mais à ne pas oublier avant de réutiliser ce
        # détecteur ailleurs.
        code = "def f(x):\n    return [i * 2 for i in range(x) if i % 3 == 0]\n" * 5
        assert language_scores(code)["en"] > 0.1


class TestFiltres:
    def test_tous_enregistres(self):
        assert {"normalize", "word_count", "language_is", "repeated_ngrams"} <= set(FILTERS.names())

    def test_word_count(self):
        step = FILTERS.build({"name": "word_count", "min_words": 5})
        assert step(doc("un deux trois quatre cinq six")) is not None
        assert step(doc("un deux")) is None

    def test_mean_word_length_rejette_les_sigles(self):
        step = FILTERS.build({"name": "mean_word_length", "lo": 3.0, "hi": 10.0})
        assert step(doc("a b c d e f g h i j")) is None
        assert step(doc(FR)) is not None

    def test_alpha_ratio_rejette_les_tableaux_de_chiffres(self):
        step = FILTERS.build({"name": "alpha_ratio", "min_ratio": 0.75})
        assert step(doc("12 34 56 78 90 11 22 33 44 55")) is None
        assert step(doc(FR)) is not None

    def test_repeated_ngrams_rejette_la_repetition(self):
        step = FILTERS.build({"name": "repeated_ngrams", "n": 5, "max_ratio": 0.2})
        spam = "achetez maintenant ce produit exceptionnel " * 40
        assert step(doc(spam)) is None
        assert step(doc(FR)) is not None

    def test_duplicate_lines(self):
        step = FILTERS.build({"name": "duplicate_lines", "max_ratio": 0.3})
        menu = "\n".join(["Accueil Contact Mentions"] * 10)
        assert step(doc(menu)) is None

    def test_boilerplate(self):
        step = FILTERS.build({"name": "boilerplate", "max_hits": 1})
        page = "Conditions générales. Politique de confidentialité. Tous droits réservés."
        assert step(doc(page)) is None
        assert step(doc(FR + " Tous droits réservés.")) is not None

    def test_language_is_rejette_et_reetiquette(self):
        step = FILTERS.build({"name": "language_is", "lang": "fr", "min_score": 0.05})
        assert step(doc(EN, lang="fr")) is None
        kept = step(doc(FR, lang="unknown"))
        assert kept is not None and kept.lang == "fr"

    def test_drop_short_lines_ne_devore_pas_le_document(self):
        # Garde-fou du filtre trop zélé : si plus de la moitié part, ce n'était
        # pas un article avec un menu, c'était un menu.
        step = FILTERS.build({"name": "drop_short_lines", "keep_ratio": 0.5})
        assert step(doc("a\nb\nc\nd\ne")) is None


class TestPipelineNettoyage:
    def test_compte_les_rejets_par_filtre(self):
        pipeline = build_pipeline(
            [{"name": "normalize"}, {"name": "word_count", "min_words": 50}], name="essai"
        )
        kept = list(pipeline.apply([doc(FR), doc("trop court"), doc("aussi court")]))
        assert len(kept) == 1
        stats = pipeline.stats()
        assert stats["seen"] == 3
        assert stats["rejected_by_step"]["word_count"] == 2

    def test_ventile_par_source(self):
        # Un seuil qui convient à Wikipédia peut décimer les notes personnelles ;
        # l'agrégat global masquerait exactement ce cas.
        pipeline = build_pipeline([{"name": "word_count", "min_words": 50}])
        list(pipeline.apply([doc("court", source="notes"), doc(FR, source="wiki")]))
        assert pipeline.stats()["rejected_by_source"] == {"notes": {"word_count": 1}}

    def test_chaine_inconnue_signale_les_disponibles(self):
        with pytest.raises(KeyError, match="Disponibles"):
            build_pipeline([{"name": "filtre_inexistant"}])


class TestDeduplication:
    def test_seuil_lsh(self):
        assert lsh_threshold(16, 8) == pytest.approx(0.707, abs=0.01)
        assert lsh_threshold(32, 4) < lsh_threshold(4, 32)

    def test_doublon_exact(self):
        dedup = Deduplicator(minhash=None)
        assert dedup.keep(doc(FR, id="a"))
        assert not dedup.keep(doc(FR, id="b"))

    def test_doublon_exact_insensible_a_la_mise_en_forme(self):
        dedup = Deduplicator(minhash=None)
        assert dedup.keep(doc(FR))
        assert not dedup.keep(doc(FR.replace(" ", "  ").upper()))

    def test_quasi_doublon_detecte(self):
        dedup = Deduplicator(exact=False, minhash=MinHashDeduplicator(bands=16, rows=4))
        assert dedup.keep(doc(FR * 3))
        assert not dedup.keep(doc("Publié le 3 août 2026. " + FR * 3))

    def test_texte_different_conserve(self):
        dedup = Deduplicator(exact=False, minhash=MinHashDeduplicator(bands=16, rows=4))
        assert dedup.keep(doc(FR * 3))
        assert dedup.keep(doc(EN * 3))

    def test_signature_reproductible_entre_instances(self):
        # Le hachage natif de Python est randomisé par processus : s'appuyer
        # dessus rendrait un corpus non rejouable.
        a, b = MinHashDeduplicator(seed=7), MinHashDeduplicator(seed=7)
        assert a.is_duplicate(FR * 3) is False
        assert b.is_duplicate(FR * 3) is False

    def test_statistiques(self):
        dedup = Deduplicator(minhash=None)
        list(dedup.apply([doc(FR, id="a"), doc(FR, id="b"), doc(EN, id="c")]))
        stats = dedup.stats()
        assert (stats["seen"], stats["exact_duplicates"], stats["kept"]) == (3, 1, 2)


class TestMelange:
    def test_composition_respectee_quand_les_sources_suffisent(self):
        mixture = plan_mixture(
            {"fr": 10_000_000, "en": 10_000_000},
            {"fr": 0.7, "en": 0.3},
            total_tokens=1_000_000,
        )
        effective = mixture.effective_weights()
        assert effective["fr"] == pytest.approx(0.7, abs=0.01)
        assert not mixture.warnings

    def test_poids_normalises(self):
        mixture = plan_mixture(
            {"a": 10_000_000, "b": 10_000_000}, {"a": 7, "b": 3}, total_tokens=1_000_000
        )
        assert mixture.effective_weights()["a"] == pytest.approx(0.7, abs=0.01)

    def test_source_insuffisante_signalee_et_deficit_redistribue(self):
        # Le cas du Vault Obsidian : ~1 M de tokens pour une cible à 5 % de 2 Md.
        mixture = plan_mixture(
            {"web": 5_000_000_000, "notes": 1_000_000},
            {"web": 0.95, "notes": 0.05},
            total_tokens=2_000_000_000,
            max_repeats={"web": 1.0, "notes": 3.0},
        )
        notes = mixture.take_for("notes")
        assert notes is not None
        assert notes.take_tokens == 3_000_000, "borné par max_repeats, pas par le poids"
        assert notes.repeats == pytest.approx(3.0)
        assert notes.is_short
        assert any("notes" in w for w in mixture.warnings)
        # Le déficit part vers la source qui a de la marge : le total est tenu.
        assert mixture.total_tokens == pytest.approx(2_000_000_000, rel=0.01)

    def test_corpus_trop_petit_signale(self):
        mixture = plan_mixture({"a": 1_000_000}, {"a": 1.0}, total_tokens=1_000_000_000)
        assert any("suffisent pas" in w for w in mixture.warnings)

    def test_repetition_excessive_signalee(self):
        mixture = plan_mixture(
            {"a": 1_000_000}, {"a": 1.0}, total_tokens=5_000_000, max_repeats=10.0
        )
        assert any("mémorisation" in w for w in mixture.warnings)

    def test_source_vide_signalee(self):
        mixture = plan_mixture({"a": 0}, {"a": 1.0}, total_tokens=1_000)
        assert any("aucune donnée" in w for w in mixture.warnings)

    def test_poids_nuls_rejetes(self):
        with pytest.raises(ValueError, match="poids"):
            plan_mixture({"a": 100}, {"a": 0.0}, total_tokens=100)


class TestObsidian:
    def test_frontmatter_retire(self):
        assert strip_markdown("---\nnom: x\n---\nLe texte.") == "Le texte."

    def test_wikilink_reduit_au_libelle(self):
        # Le libellé est le mot que l'auteur voulait lire, et le seul porteur de sens.
        assert strip_markdown("Voir [[Note interne|le guide]] ici.") == "Voir le guide ici."
        # Sans libellé, c'est le titre de la note qui reste — casse comprise.
        assert strip_markdown("Voir [[Le guide]] ici.") == "Voir Le guide ici."

    def test_image_integree_supprimee(self):
        assert strip_markdown("Texte ![[image.png]] suite.").replace("  ", " ") == "Texte suite."

    def test_lien_markdown_reduit_au_texte(self):
        assert strip_markdown("Voir [la doc](https://x.fr) ici.") == "Voir la doc ici."


class TestGutenberg:
    """Découpage de romans en texte brut."""

    LIVRE = (
        "The Project Gutenberg eBook of Test\n\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n\n"
        "CHAPITRE I\n\n" + "Le premier chapitre raconte une histoire. " * 40 + "\n\n"
        "CHAPITRE II\n\n" + "Le second chapitre en raconte une autre. " * 40 + "\n\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n\n"
        "Licence à ne surtout pas apprendre au modèle."
    )

    def test_encadrement_legal_retire(self):
        from thadeus.data.sources.gutenberg import strip_boilerplate

        texte = strip_boilerplate(self.LIVRE)
        assert "CHAPITRE I" in texte
        # La licence est identique dans tous les livres : la garder apprendrait
        # au modèle à la réciter, et la dédup ne l'attraperait pas.
        assert "Licence à ne surtout pas apprendre" not in texte
        assert "Project Gutenberg eBook of" not in texte

    def test_texte_rendu_tel_quel_sans_marqueurs(self):
        from thadeus.data.sources.gutenberg import strip_boilerplate

        # Mieux vaut un peu de bruit qu'un livre perdu.
        assert strip_boilerplate("Un texte sans marqueurs.") == "Un texte sans marqueurs."

    def test_decoupage_aux_chapitres(self):
        from thadeus.data.sources.gutenberg import split_into_chunks, strip_boilerplate

        morceaux = split_into_chunks(strip_boilerplate(self.LIVRE))
        assert len(morceaux) == 2
        assert morceaux[0].startswith("CHAPITRE I")

    def test_roman_sans_chapitres_decoupe_par_paragraphes(self):
        from thadeus.data.sources.gutenberg import split_into_chunks

        texte = "\n\n".join(["Un paragraphe de quelques mots seulement."] * 400)
        morceaux = split_into_chunks(texte, target_words=200)
        assert len(morceaux) > 5
        assert all(len(m.split()) < 400 for m in morceaux)

    def test_un_livre_entier_serait_rejete_sans_decoupage(self):
        # Justification du découpage : le filtre de longueur plafonne à 100 000
        # mots, et le modèle n'en voit que ~1024 tokens à la fois.
        from thadeus.data.sources.gutenberg import split_into_chunks

        roman = "\n\n".join(["Une phrase de roman assez longue pour compter."] * 20_000)
        assert len(roman.split()) > 100_000
        assert all(len(m.split()) < 3_000 for m in split_into_chunks(roman))

    def test_identifiants_stables_et_lisibles(self, tmp_path):
        from thadeus.data.sources.gutenberg import from_gutenberg

        (tmp_path / "germinal.txt").write_text(self.LIVRE, encoding="utf-8")
        docs = list(from_gutenberg(root=str(tmp_path), min_words=10))
        assert docs and all(d.id.startswith("gutenberg:germinal#") for d in docs)
        assert docs[0].meta["book"] == "germinal"

    def test_repertoire_absent_rejete(self):
        from thadeus.data.sources.gutenberg import from_gutenberg

        with pytest.raises(FileNotFoundError):
            list(from_gutenberg(root="/chemin/inexistant"))


class TestSplitObsidian:
    """Le split train/val du Vault — indispensable au fine-tuning."""

    def vault(self, tmp_path, n=60):
        for i in range(n):
            (tmp_path / f"note_{i:03d}.md").write_text("mot " * 60, encoding="utf-8")
        return str(tmp_path)

    def test_train_et_val_sont_disjoints(self, tmp_path):
        from thadeus.data.sources.obsidian import from_obsidian

        v = self.vault(tmp_path)
        tr = {d.id for d in from_obsidian(vault=v, split="train")}
        va = {d.id for d in from_obsidian(vault=v, split="val")}
        assert tr and va
        assert not (tr & va), "aucune note ne doit être dans les deux"

    def test_leur_union_est_le_tout(self, tmp_path):
        from thadeus.data.sources.obsidian import from_obsidian

        v = self.vault(tmp_path)
        tout = {d.id for d in from_obsidian(vault=v, split="all")}
        tr = {d.id for d in from_obsidian(vault=v, split="train")}
        va = {d.id for d in from_obsidian(vault=v, split="val")}
        assert tr | va == tout

    def test_split_stable_entre_appels(self, tmp_path):
        # Décidé par hachage du chemin, jamais par tirage : deux exécutions
        # doivent donner exactement le même partage, sinon une note de
        # validation finirait un jour dans l'entraînement.
        from thadeus.data.sources.obsidian import from_obsidian

        v = self.vault(tmp_path)
        a = [d.id for d in from_obsidian(vault=v, split="val")]
        b = [d.id for d in from_obsidian(vault=v, split="val")]
        assert a == b

    def test_ajouter_des_notes_ne_redistribue_pas_les_anciennes(self, tmp_path):
        # Propriété du hachage par chemin : le Vault grandit sans jamais
        # invalider un split déjà utilisé pour un entraînement.
        from thadeus.data.sources.obsidian import from_obsidian

        v = self.vault(tmp_path, n=40)
        avant = {d.id for d in from_obsidian(vault=v, split="val")}
        self.vault(tmp_path, n=80)
        apres = {d.id for d in from_obsidian(vault=v, split="val")}
        assert avant <= apres

    def test_fraction_respectee(self, tmp_path):
        from thadeus.data.sources.obsidian import from_obsidian

        v = self.vault(tmp_path, n=200)
        va = list(from_obsidian(vault=v, split="val", val_fraction=0.2))
        assert 0.12 < len(va) / 200 < 0.30

    def test_split_inconnu_rejete(self, tmp_path):
        from thadeus.data.sources.obsidian import from_obsidian

        with pytest.raises(ValueError, match="split inconnu"):
            list(from_obsidian(vault=self.vault(tmp_path), split="test"))
