"""Le registre est ce qui rend un A/B d'architecture honnête : si changer
d'attention demande de toucher au code, on introduit des différences
involontaires entre les variantes comparées.
"""

from __future__ import annotations

import pytest

from thadeus.core.registry import Registry


@pytest.fixture
def reg():
    r: Registry = Registry("attention")

    @r.register("gqa", aliases=("grouped",))
    class GQA:
        def __init__(self, heads: int = 12, d_model: int | None = None):
            self.heads, self.d_model = heads, d_model

    @r.register("mla")
    def make_mla(latent_dim: int, d_model: int | None = None):
        return {"kind": "mla", "latent_dim": latent_dim, "d_model": d_model}

    return r


class TestEnregistrement:
    def test_nom_et_alias(self, reg):
        assert reg.get("gqa") is reg.get("grouped")

    def test_nom_deduit_si_absent(self):
        r: Registry = Registry("optim")

        @r.register()
        def muon():
            return "muon"

        assert "muon" in r

    def test_doublon_rejete(self, reg):
        with pytest.raises(KeyError, match="déjà pris"):

            @reg.register("gqa")
            def autre(): ...

    def test_listing(self, reg):
        assert reg.names() == ["gqa", "grouped", "mla"]


class TestConstruction:
    def test_depuis_un_nom_nu(self, reg):
        assert reg.build("gqa").heads == 12

    def test_depuis_une_spec_avec_parametres(self, reg):
        # La forme exacte que produit un TOML : {name = "mla", latent_dim = 128}
        assert reg.build({"name": "mla", "latent_dim": 128})["latent_dim"] == 128

    def test_extra_fusionne_avec_la_spec(self, reg):
        # d_model n'est pas dans la config : il vient du reste du modèle.
        assert reg.build({"name": "gqa", "heads": 8}, d_model=768).d_model == 768

    def test_spec_sans_nom_rejetee(self, reg):
        with pytest.raises(KeyError, match="name"):
            reg.build({"heads": 8})

    def test_parametre_inconnu_rejete(self, reg):
        with pytest.raises(TypeError, match="échec de construction"):
            reg.build({"name": "gqa", "hedas": 8})


class TestErreurs:
    def test_suggere_le_nom_proche(self, reg):
        # Une faute de frappe dans un TOML ne doit pas coûter une session de débogage.
        with pytest.raises(KeyError, match="gqa"):
            reg.get("gqua")

    def test_liste_les_disponibles(self, reg):
        with pytest.raises(KeyError, match="Disponibles"):
            reg.get("inexistant")
