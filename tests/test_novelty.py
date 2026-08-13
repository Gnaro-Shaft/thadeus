"""Une collecte doit être jugée sur ce qu'elle apporte, pas sur ce qu'elle pèse.

Ces tests encodent l'incident qui a motivé le module : cinq collectes
quotidiennes annonçant chacune 2,03 Md de tokens, identiques entre elles à
98-99 %, sans qu'aucune mesure ne le signale.
"""

from __future__ import annotations

import json

import pytest

from thadeus.data.novelty import SEUIL_ALERTE, compare_to_existing, fingerprints, overlap
from thadeus.data.schema import Document
from thadeus.data.shard import ShardWriter


def ecrire(repertoire, textes):
    """Écrit un corpus jouet et rend son répertoire."""
    with ShardWriter(repertoire, max_documents=1_000) as writer:
        writer.write_all(
            Document(id=f"d{i}", text=t, source="test", lang="fr") for i, t in enumerate(textes)
        )
    return repertoire


def artefact(base, nom, textes, *, acheve=True):
    """Artefact de données complet — `meta.json` marque l'achèvement."""
    chemin = base / nom
    ecrire(chemin / "corpus", textes)
    if acheve:
        (chemin / "meta.json").write_text(json.dumps({"stage": "data", "label": nom}))
    return chemin


PHRASES = [f"Le document numéro {i} parle de sujets variés et bien distincts." for i in range(60)]
AUTRES = [f"Texte sans rapport {i}, portant sur toute autre chose entièrement." for i in range(60)]


class TestRecouvrement:
    def test_identiques_donnent_cent_pour_cent(self, tmp_path):
        a = fingerprints(ecrire(tmp_path / "a", PHRASES))
        b = fingerprints(ecrire(tmp_path / "b", PHRASES))
        assert overlap(a, b) == 1.0

    def test_disjoints_donnent_zero(self, tmp_path):
        a = fingerprints(ecrire(tmp_path / "a", PHRASES))
        b = fingerprints(ecrire(tmp_path / "b", AUTRES))
        assert overlap(a, b) == 0.0

    def test_la_mesure_est_asymetrique(self, tmp_path):
        # La question est « ce que je viens de collecter, l'avais-je déjà ? »,
        # pas « ces deux corpus se ressemblent-ils ». Un petit corpus contenu
        # dans un grand est entièrement redondant, l'inverse est faux.
        petit = fingerprints(ecrire(tmp_path / "p", PHRASES[:10]))
        grand = fingerprints(ecrire(tmp_path / "g", PHRASES))
        assert overlap(petit, grand) == 1.0
        assert overlap(grand, petit) < 0.25

    def test_corpus_vide_ne_leve_pas(self, tmp_path):
        assert overlap(set(), fingerprints(ecrire(tmp_path / "a", PHRASES))) == 0.0
        assert fingerprints(tmp_path / "inexistant") == set()


class TestComparaisonAuxArtefacts:
    def test_detecte_la_redondance_massive(self, tmp_path):
        """LE test de l'incident : une collecte identique à une précédente."""
        artefact(tmp_path, "hier", PHRASES)
        aujourdhui = artefact(tmp_path, "aujourdhui", PHRASES)
        r = compare_to_existing(aujourdhui / "corpus", root=tmp_path, exclude="aujourdhui")
        assert r["recouvrement_max"] == 1.0
        assert r["artefact_le_plus_proche"] == "hier"
        assert r["recouvrement_max"] >= SEUIL_ALERTE

    def test_une_vraie_nouveaute_passe_sous_le_seuil(self, tmp_path):
        artefact(tmp_path, "hier", PHRASES)
        neuf = artefact(tmp_path, "neuf", AUTRES)
        r = compare_to_existing(neuf / "corpus", root=tmp_path, exclude="neuf")
        assert r["recouvrement_max"] < SEUIL_ALERTE

    def test_ignore_les_artefacts_inacheves(self, tmp_path):
        # Un corpus sans meta.json est une collecte interrompue : le prendre
        # pour référence ferait passer une vraie nouveauté pour un doublon.
        artefact(tmp_path, "interrompu", PHRASES, acheve=False)
        aujourdhui = artefact(tmp_path, "aujourdhui", PHRASES)
        r = compare_to_existing(aujourdhui / "corpus", root=tmp_path, exclude="aujourdhui")
        assert r["par_artefact"] == {}
        assert r["recouvrement_max"] is None

    def test_ne_se_compare_pas_a_lui_meme(self, tmp_path):
        seul = artefact(tmp_path, "seul", PHRASES)
        r = compare_to_existing(seul / "corpus", root=tmp_path, exclude="seul")
        assert "seul" not in r["par_artefact"]

    def test_classe_les_references_par_proximite(self, tmp_path):
        artefact(tmp_path, "proche", PHRASES)
        artefact(tmp_path, "lointain", AUTRES)
        aujourdhui = artefact(tmp_path, "aujourdhui", PHRASES)
        r = compare_to_existing(aujourdhui / "corpus", root=tmp_path, exclude="aujourdhui")
        assert r["par_artefact"]["proche"] > r["par_artefact"]["lointain"]

    def test_alerte_journalisee_au_dela_du_seuil(self, tmp_path, caplog):
        artefact(tmp_path, "hier", PHRASES)
        aujourdhui = artefact(tmp_path, "aujourdhui", PHRASES)
        with caplog.at_level("WARNING"):
            compare_to_existing(aujourdhui / "corpus", root=tmp_path, exclude="aujourdhui")
        assert any("REDONDANCE" in m for m in caplog.messages), (
            "une collecte redondante doit être signalée bruyamment, "
            "sinon l'incident se reproduit à l'identique"
        )


@pytest.mark.parametrize("seuil", [SEUIL_ALERTE])
def test_le_seuil_reste_raisonnable(seuil):
    # Trop bas, l'alerte crie sur des recouvrements normaux et on cesse de la
    # lire ; trop haut, elle laisse passer une collecte à moitié inutile.
    assert 0.3 <= seuil <= 0.8
