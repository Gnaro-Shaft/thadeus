"""Étage optimiseur : Muon, muP, assemblage hybride.

Tout tourne sur CPU avec des modèles jouets.
"""

from __future__ import annotations

import pytest
import torch

from thadeus.model import ModelConfig, Thadeus
from thadeus.model.init import classify_parameters
from thadeus.optim.build import ChainedOptimizer, build_optimizer
from thadeus.optim.muon import Muon, orthogonalize
from thadeus.optim.mup import MupConfig, apply_mup, logit_scale, lr_scales, width_multiplier
from thadeus.train.config import OptimSpec

TINY = {
    "vocab_size": 128,
    "d_model": 64,
    "n_layers": 2,
    "max_seq_len": 32,
    "attention": {"name": "gqa", "n_heads": 4, "n_kv_heads": 2, "head_dim": 16},
}


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return Thadeus(ModelConfig(**TINY))


class TestOrthogonalisation:
    def test_egalise_les_valeurs_singulieres(self):
        # Le cœur de Muon : un gradient dont quelques directions dominent
        # devient une mise à jour qui avance également dans toutes.
        torch.manual_seed(0)
        g = torch.randn(128, 64) @ torch.diag(torch.linspace(1, 20, 64))
        avant = torch.linalg.svdvals(g)
        apres = torch.linalg.svdvals(orthogonalize(g).float())
        assert (avant.max() / avant.min()) > 10
        assert (apres.max() / apres.min()) < 3

    def test_la_masse_des_valeurs_singulieres_approche_un(self):
        # Limite réelle de l'approximation, mesurée : cinq itérations de
        # Newton-Schulz ramènent la **grande majorité** des valeurs singulières
        # autour de 1, mais ne relèvent pas celles qui partaient de très près de
        # zéro — une direction quasi dégénérée le reste. Sans conséquence pour
        # Muon, dont l'objet est d'égaliser les directions qui portent du signal.
        torch.manual_seed(0)
        s = torch.linalg.svdvals(orthogonalize(torch.randn(64, 64)).float())
        assert s.max() < 1.5
        assert (s > 0.5).float().mean() > 0.95, "au moins 95 % des directions égalisées"

    def test_fonctionne_dans_les_deux_orientations(self):
        # L'itération opère sur X @ X.T : on transpose quand c'est plus haut que
        # large, pour que le coût suive la plus petite dimension.
        torch.manual_seed(0)
        for shape in [(128, 32), (32, 128), (64, 64)]:
            out = orthogonalize(torch.randn(*shape))
            assert out.shape == shape
            assert torch.isfinite(out).all()

    def test_tenseur_non_2d_rejete(self):
        with pytest.raises(ValueError, match="2D"):
            orthogonalize(torch.randn(4, 4, 4))

    def test_deterministe(self):
        g = torch.randn(32, 16)
        assert torch.equal(orthogonalize(g), orthogonalize(g))


class TestMuon:
    def test_refuse_les_tenseurs_non_matriciels(self, tiny_model):
        # Orthogonaliser un gain de normalisation n'a aucun sens : mieux vaut
        # lever que produire silencieusement des mises à jour absurdes.
        vecteurs = classify_parameters(tiny_model)["vector"]
        with pytest.raises(ValueError, match="2D"):
            Muon(vecteurs)

    def test_refuse_un_taux_negatif(self, tiny_model):
        with pytest.raises(ValueError, match="lr"):
            Muon(classify_parameters(tiny_model)["hidden"], lr=-1)

    def test_fait_descendre_la_perte(self, tiny_model):
        # Test minimal mais non négociable : l'optimiseur optimise.
        hidden = classify_parameters(tiny_model)["hidden"]
        autres = [p for k, v in classify_parameters(tiny_model).items() if k != "hidden" for p in v]
        muon = Muon(hidden, lr=0.02)
        adam = torch.optim.AdamW(autres, lr=1e-3)
        ids = torch.randint(0, 128, (4, 16))

        pertes = []
        for _ in range(30):
            _, loss = tiny_model(ids, targets=ids)
            pertes.append(loss.item())
            loss.backward()
            muon.step()
            adam.step()
            muon.zero_grad(set_to_none=True)
            adam.zero_grad(set_to_none=True)
        assert pertes[-1] < pertes[0] * 0.7

    def test_etat_de_momentum_sauvegarde(self, tiny_model):
        hidden = classify_parameters(tiny_model)["hidden"]
        muon = Muon(hidden, lr=0.02)
        for p in hidden:
            p.grad = torch.randn_like(p)
        muon.step()
        assert muon.state_dict()["state"], "le momentum doit être restaurable"


class TestMup:
    def test_desactive_ne_change_rien(self):
        cfg = ModelConfig(**TINY)
        mup = MupConfig(enabled=False)
        assert lr_scales(cfg, mup) == {"hidden": 1.0, "embedding": 1.0, "vector": 1.0}
        assert logit_scale(cfg, mup) == 1.0

    def test_multiplicateur_de_largeur(self):
        cfg = ModelConfig(
            **{**TINY, "d_model": 512, "attention": {"name": "gqa", "n_heads": 8, "head_dim": 64}}
        )
        assert width_multiplier(cfg, MupConfig(base_d_model=128)) == 4.0

    def test_seules_les_matrices_cachees_voient_leur_taux_reduit(self):
        cfg = ModelConfig(
            **{**TINY, "d_model": 512, "attention": {"name": "gqa", "n_heads": 8, "head_dim": 64}}
        )
        scales = lr_scales(cfg, MupConfig(enabled=True, base_d_model=128))
        assert scales["hidden"] == pytest.approx(0.25)
        assert scales["embedding"] == 1.0
        assert scales["vector"] == 1.0

    def test_a_la_largeur_de_reference_rien_ne_bouge(self):
        # Le modèle sur lequel on règle les hyperparamètres ne doit pas être
        # modifié par muP — sinon on réglerait autre chose que ce qu'on transfère.
        cfg = ModelConfig(**TINY)
        mup = MupConfig(enabled=True, base_d_model=cfg.d_model)
        assert lr_scales(cfg, mup) == {"hidden": 1.0, "embedding": 1.0, "vector": 1.0}
        assert logit_scale(cfg, mup) == 1.0

    def test_initialisation_cachee_reduite_avec_la_largeur(self):
        torch.manual_seed(0)
        cfg = ModelConfig(
            **{**TINY, "d_model": 256, "attention": {"name": "gqa", "n_heads": 4, "head_dim": 64}}
        )
        modele = Thadeus(cfg)
        avant = modele.blocks[0].attn.q_proj.weight.std().item()
        apply_mup(modele, cfg, MupConfig(enabled=True, base_d_model=64))
        apres = modele.blocks[0].attn.q_proj.weight.std().item()
        assert apres == pytest.approx(avant * 0.5, rel=0.05)  # m = 4 -> 1/√4

    def test_embeddings_non_touches(self):
        torch.manual_seed(0)
        cfg = ModelConfig(
            **{**TINY, "d_model": 256, "attention": {"name": "gqa", "n_heads": 4, "head_dim": 64}}
        )
        modele = Thadeus(cfg)
        avant = modele.embedding.weight.clone()
        apply_mup(modele, cfg, MupConfig(enabled=True, base_d_model=64))
        assert torch.equal(modele.embedding.weight, avant)

    def test_facteur_de_logits_applique_par_le_modele(self):
        ids = torch.randint(0, 128, (2, 16))
        torch.manual_seed(0)
        normal = Thadeus(ModelConfig(**TINY))
        torch.manual_seed(0)
        reduit = Thadeus(ModelConfig(**{**TINY, "logit_scale": 0.25}))
        a, _ = normal(ids)
        b, _ = reduit(ids)
        assert torch.allclose(b, a * 0.25, atol=1e-5)


class TestAssemblage:
    def test_adamw_seul_par_defaut(self, tiny_model):
        opt = build_optimizer(tiny_model, spec=OptimSpec(name="adamw"))
        assert isinstance(opt, torch.optim.AdamW)

    def test_muon_route_les_matrices_cachees(self, tiny_model):
        opt = build_optimizer(tiny_model, spec=OptimSpec(name="muon"))
        assert isinstance(opt, ChainedOptimizer)
        types = {type(o).__name__ for o in opt.optimizers}
        assert types == {"Muon", "AdamW"}

        muon = next(o for o in opt.optimizers if isinstance(o, Muon))
        assert all(p.ndim == 2 for g in muon.param_groups for p in g["params"])
        # Aucun embedding chez Muon.
        embeddings = {id(p) for p in classify_parameters(tiny_model)["embedding"]}
        assert not any(id(p) in embeddings for g in muon.param_groups for p in g["params"])

    def test_taux_de_base_distincts(self, tiny_model):
        # Muon tolère ~50x le taux d'AdamW : un taux commun casserait l'un des deux.
        opt = build_optimizer(tiny_model, spec=OptimSpec(name="muon", lr=1e-3, muon_lr=0.05))
        par_kind = {g["kind"]: g["base_lr"] for g in opt.param_groups}
        assert par_kind["hidden"] == 0.05
        assert par_kind["embedding"] == 1e-3

    def test_multiplicateurs_mup_repercutes(self, tiny_model):
        opt = build_optimizer(
            tiny_model, spec=OptimSpec(name="muon", muon_lr=0.04), lr_scales={"hidden": 0.5}
        )
        hidden = next(g for g in opt.param_groups if g["kind"] == "hidden")
        assert hidden["base_lr"] == pytest.approx(0.02)

    def test_param_groups_modifiables_depuis_la_boucle(self, tiny_model):
        # La boucle règle le taux à chaque pas : les groupes exposés doivent être
        # ceux des sous-optimiseurs, pas des copies.
        opt = build_optimizer(tiny_model, spec=OptimSpec(name="muon"))
        for group in opt.param_groups:
            group["lr"] = 0.123
        assert all(g["lr"] == 0.123 for o in opt.optimizers for g in o.param_groups)

    def test_aller_retour_d_etat(self, tiny_model):
        opt = build_optimizer(tiny_model, spec=OptimSpec(name="muon"))
        ids = torch.randint(0, 128, (2, 16))
        _, loss = tiny_model(ids, targets=ids)
        loss.backward()
        opt.step()

        autre = build_optimizer(tiny_model, spec=OptimSpec(name="muon"))
        autre.load_state_dict(opt.state_dict())
        assert autre.state_dict()["chained"][0]["state"]

    def test_etat_incompatible_rejete(self, tiny_model):
        # Changer d'optimiseur entre deux runs ne doit pas charger en silence.
        muon = build_optimizer(tiny_model, spec=OptimSpec(name="muon"))
        with pytest.raises(ValueError, match="optimiseurs"):
            muon.load_state_dict({"chained": [{}]})
