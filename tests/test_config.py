"""Le socle de config doit tenir trois promesses : l'héritage fusionne
correctement, les fautes de frappe échouent bruyamment, et le hash est stable.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from thadeus.core.config import (
    Schema,
    config_hash,
    deep_merge,
    load_config,
    parse_override,
    set_by_path,
)


def write(tmp_path, name: str, body: str):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


class TestDeepMerge:
    def test_fusionne_en_profondeur(self):
        base = {"model": {"layers": 12, "d": 768}, "seed": 1}
        over = {"model": {"layers": 24}}
        assert deep_merge(base, over) == {"model": {"layers": 24, "d": 768}, "seed": 1}

    def test_ne_mute_pas_les_entrees(self):
        base = {"a": {"b": 1}}
        deep_merge(base, {"a": {"b": 2}})
        assert base == {"a": {"b": 1}}

    def test_les_listes_sont_remplacees_pas_concatenees(self):
        # Choix assumé : concaténer rendrait impossible de retirer un élément hérité.
        assert deep_merge({"x": [1, 2, 3]}, {"x": [9]}) == {"x": [9]}


class TestHeritage:
    def test_extends_simple(self, tmp_path):
        write(tmp_path, "base.toml", "seed = 1\n[model]\nlayers = 12\nd = 768\n")
        write(tmp_path, "run.toml", 'extends = "base.toml"\n[model]\nlayers = 24\n')
        cfg = load_config(tmp_path / "run.toml", root=tmp_path)
        assert cfg == {"seed": 1, "model": {"layers": 24, "d": 768}}

    def test_extends_multiple_le_dernier_gagne(self, tmp_path):
        write(tmp_path, "a.toml", "x = 1\ny = 1\n")
        write(tmp_path, "b.toml", "y = 2\n")
        write(tmp_path, "run.toml", 'extends = ["a.toml", "b.toml"]\n')
        assert load_config(tmp_path / "run.toml", root=tmp_path) == {"x": 1, "y": 2}

    def test_extends_transitif(self, tmp_path):
        write(tmp_path, "grand.toml", "a = 1\n")
        write(tmp_path, "parent.toml", 'extends = "grand.toml"\nb = 2\n')
        write(tmp_path, "run.toml", 'extends = "parent.toml"\nc = 3\n')
        assert load_config(tmp_path / "run.toml", root=tmp_path) == {"a": 1, "b": 2, "c": 3}

    def test_cycle_detecte(self, tmp_path):
        write(tmp_path, "a.toml", 'extends = "b.toml"\n')
        write(tmp_path, "b.toml", 'extends = "a.toml"\n')
        with pytest.raises(ValueError, match="cycle"):
            load_config(tmp_path / "a.toml", root=tmp_path)

    def test_reference_introuvable(self, tmp_path):
        write(tmp_path, "run.toml", 'extends = "absent.toml"\n')
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "run.toml", root=tmp_path)


class TestSurcharges:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("a.b=16", (["a", "b"], 16)),
            ("lr=0.001", (["lr"], 0.001)),
            ("flag=true", (["flag"], True)),
            ('name="mla"', (["name"], "mla")),
            ("sizes=[1, 2]", (["sizes"], [1, 2])),
        ],
    )
    def test_types_preserves(self, text, expected):
        # Le point de la grammaire TOML : --set x=12 doit donner l'entier 12,
        # pas la chaîne "12" — sinon le schéma rejette ou, pire, accepte.
        assert parse_override(text) == expected

    def test_chaine_nue_acceptee(self):
        assert parse_override("device=cpu") == (["device"], "cpu")

    def test_sans_egal_rejete(self):
        with pytest.raises(ValueError, match="surcharge invalide"):
            parse_override("device")

    def test_appliquee_apres_heritage(self, tmp_path):
        write(tmp_path, "base.toml", "[model]\nlayers = 12\n")
        write(tmp_path, "run.toml", 'extends = "base.toml"\n[model]\nlayers = 24\n')
        cfg = load_config(tmp_path / "run.toml", overrides=["model.layers=48"], root=tmp_path)
        assert cfg["model"]["layers"] == 48

    def test_cree_les_niveaux_manquants(self):
        cfg: dict = {}
        set_by_path(cfg, ["a", "b", "c"], 7)
        assert cfg == {"a": {"b": {"c": 7}}}


class TestHash:
    def test_stable_entre_appels(self):
        cfg = {"model": {"layers": 12}}
        assert config_hash(cfg) == config_hash(cfg)

    def test_independant_de_l_ordre_d_ecriture(self):
        # Deux TOML sémantiquement identiques doivent viser le même artefact.
        assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})

    def test_change_avec_le_moindre_parametre(self):
        # C'est ce qui empêche une variante d'écraser silencieusement une autre.
        assert config_hash({"lr": 0.001}) != config_hash({"lr": 0.002})


class TestSchema:
    def test_cle_inconnue_rejetee(self):
        class Cfg(Schema):
            layers: int = 12

        with pytest.raises(ValidationError):
            Cfg(layerz=24)  # type: ignore[call-arg]

    def test_immuable(self):
        class Cfg(Schema):
            layers: int = 12

        with pytest.raises(ValidationError):
            Cfg().layers = 24  # type: ignore[misc]
