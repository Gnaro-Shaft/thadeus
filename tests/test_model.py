"""Étage modèle : briques, assemblage, dimensionnement.

Tout tourne sur CPU en fp32 avec des modèles jouets — rapide, déterministe, et
indépendant de la machine.
"""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from thadeus.core.config import load_config
from thadeus.model import ModelConfig, Thadeus, estimate
from thadeus.model.blocks import ATTENTIONS, FFNS, NORMS
from thadeus.model.blocks.ffn import swiglu_hidden_dim
from thadeus.model.blocks.norm import RMSNorm
from thadeus.model.blocks.rope import RotaryEmbedding, apply_rope
from thadeus.model.init import parameter_groups

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


class TestRegistres:
    def test_composants_enregistres(self):
        assert {"rmsnorm", "layernorm"} <= set(NORMS.names())
        assert "gqa" in ATTENTIONS.names()
        assert {"swiglu", "mlp"} <= set(FFNS.names())

    def test_substitution_par_config(self):
        # La condition matérielle des A/B de Phase 6 : changer une brique est
        # une ligne de config, pas un refactoring.
        model = Thadeus(ModelConfig(**{**TINY, "norm": "layernorm", "ffn": {"name": "mlp"}}))
        assert type(model.final_norm).__name__ == "LayerNorm"
        assert type(model.blocks[0].ffn).__name__ == "MLP"


class TestNormalisation:
    def test_calcule_en_fp32_mais_rend_le_dtype_d_entree(self):
        # Le compromis de Phase 0 : stabilité numérique sans payer le fp32 sur
        # un chemin qui n'est pas un produit matriciel.
        norm = RMSNorm(16)
        assert norm(torch.randn(2, 16, dtype=torch.bfloat16)).dtype is torch.bfloat16

    def test_normalise_la_norme_quadratique(self):
        out = RMSNorm(64)(torch.randn(8, 64) * 100)
        assert out.pow(2).mean(-1).allclose(torch.ones(8), atol=1e-3)

    def test_invariante_a_l_echelle(self):
        norm, x = RMSNorm(32), torch.randn(4, 32)
        assert torch.allclose(norm(x), norm(x * 50), atol=1e-4)


class TestRoPE:
    def test_head_dim_impair_rejete(self):
        with pytest.raises(ValueError, match="pair"):
            RotaryEmbedding(15)

    def test_dispose_en_batch_seq_heads_dim(self):
        # Disposition (B, S, H, D) et non (B, H, S, D) : c'est celle où les
        # tenseurs sont contigus, et les opérations élément par élément y sont
        # 2× plus rapides (mesuré sur M5 Pro).
        rope = RotaryEmbedding(16)
        cos, sin = rope(8, device=torch.device("cpu"), dtype=torch.float32)
        assert cos.shape == (1, 8, 1, 16)

    def test_preserve_la_norme(self):
        # Une rotation ne change pas la longueur d'un vecteur.
        rope = RotaryEmbedding(16)
        cos, sin = rope(8, device=torch.device("cpu"), dtype=torch.float32)
        x = torch.randn(1, 8, 2, 16)  # (B, S, H, D) — la disposition contiguë
        assert torch.allclose(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), atol=1e-4)

    def test_produit_scalaire_ne_depend_que_de_l_ecart(self):
        # La propriété qui fait tout l'intérêt de RoPE : deux paires de
        # positions séparées du même écart donnent le même produit scalaire.
        rope = RotaryEmbedding(16)
        cos, sin = rope(16, device=torch.device("cpu"), dtype=torch.float32)
        q = torch.randn(1, 1, 1, 16).expand(1, 16, 1, 16).contiguous()
        k = torch.randn(1, 1, 1, 16).expand(1, 16, 1, 16).contiguous()
        qr, kr = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        produits = qr[0, :, 0] @ kr[0, :, 0].T
        assert produits[2, 5].isclose(produits[7, 10], atol=1e-4)

    def test_cache_s_etend_au_dela_de_la_longueur_prevue(self):
        # Une évaluation à contexte plus long que l'entraînement ne doit pas planter.
        rope = RotaryEmbedding(16, max_seq_len=8)
        cos, _ = rope(64, device=torch.device("cpu"), dtype=torch.float32)
        assert cos.shape[1] == 64

    def test_buffers_non_persistants(self):
        # Les stocker figerait la longueur de contexte du modèle sauvegardé.
        assert not any("rope.cos" in k for k in Thadeus(ModelConfig(**TINY)).state_dict())


class TestAttention:
    def test_regroupement_incoherent_rejete(self):
        with pytest.raises(ValueError, match="multiple"):
            ATTENTIONS.build({"name": "gqa", "n_heads": 6, "n_kv_heads": 4}, d_model=64)

    def test_facteur_de_reduction_du_cache(self):
        attn = ATTENTIONS.build(
            {"name": "gqa", "n_heads": 10, "n_kv_heads": 2, "head_dim": 16}, d_model=64
        )
        assert attn.n_groups == 5

    def test_causalite(self, tiny_model):
        # Le test qui compte : modifier un token futur ne doit rien changer aux
        # sorties passées. Une fuite ici donnerait un modèle qui semble
        # excellent à l'entraînement et incapable de générer.
        tiny_model.eval()
        ids = torch.randint(0, 128, (1, 16))
        with torch.no_grad():
            avant, _ = tiny_model(ids)
            modifie = ids.clone()
            modifie[0, -1] = (modifie[0, -1] + 1) % 128
            apres, _ = tiny_model(modifie)
        assert torch.allclose(avant[0, :-1], apres[0, :-1], atol=1e-5)


class TestFFN:
    def test_dimension_cachee_alignee(self):
        # Une dimension mal alignée laisse des unités de calcul inutilisées.
        assert swiglu_hidden_dim(640) % 64 == 0

    def test_swiglu_a_budget_comparable_au_mlp(self):
        # SwiGLU utilise 3 matrices au lieu de 2, compensé par une dimension
        # cachée réduite à 8/3·d : le nombre de paramètres doit rester proche.
        d = 640
        swiglu = sum(p.numel() for p in FFNS.build({"name": "swiglu"}, d_model=d).parameters())
        mlp = sum(p.numel() for p in FFNS.build({"name": "mlp"}, d_model=d).parameters())
        assert abs(swiglu / mlp - 1) < 0.1


class TestModele:
    def test_formes_de_sortie(self, tiny_model):
        logits, loss = tiny_model(torch.randint(0, 128, (2, 16)))
        assert logits.shape == (2, 16, 128)
        assert loss is None

    def test_perte_calculee_avec_cibles(self, tiny_model):
        ids = torch.randint(0, 128, (2, 16))
        _, loss = tiny_model(ids, targets=ids)
        assert loss.ndim == 0 and loss.item() > 0

    def test_perte_initiale_proche_du_hasard(self):
        # ln(vocab_size) est la perte d'un modèle qui prédit uniformément.
        # S'en écarter à l'initialisation signale une init cassée.
        # Les cibles doivent être **indépendantes** des entrées : voir le test
        # suivant, qui documente pourquoi.
        import math

        torch.manual_seed(0)
        model = Thadeus(ModelConfig(**TINY))
        ids = torch.randint(0, 128, (4, 16))
        cibles = torch.randint(0, 128, (4, 16))
        _, loss = model(ids, targets=cibles)
        assert abs(loss.item() - math.log(128)) < 0.5

    def test_le_partage_d_embeddings_facilite_la_copie_a_l_init(self):
        # Effet réel du partage entrée/sortie, à connaître : à l'initialisation,
        # le flux résiduel est dominé par l'embedding d'entrée, donc les logits
        # favorisent le token d'entrée lui-même. Prédire l'identité est alors
        # bien plus facile que le hasard (perte ~3,7 contre ln(128) = 4,85).
        # Sans conséquence à l'entraînement, où les cibles sont décalées — mais
        # de quoi rendre incompréhensible toute courbe de perte mesurée sur une
        # tâche d'identité.
        import math

        torch.manual_seed(0)
        model = Thadeus(ModelConfig(**TINY))
        ids = torch.randint(0, 128, (4, 16))
        _, identite = model(ids, targets=ids)
        assert identite.item() < math.log(128) - 0.5

    def test_sequence_trop_longue_rejetee(self, tiny_model):
        with pytest.raises(ValueError, match="max_seq_len"):
            tiny_model(torch.randint(0, 128, (1, 64)))

    def test_embeddings_partages(self, tiny_model):
        assert tiny_model.lm_head.weight is tiny_model.embedding.weight

    def test_embeddings_separes_si_demande(self):
        model = Thadeus(ModelConfig(**{**TINY, "tie_embeddings": False}))
        assert model.lm_head.weight is not model.embedding.weight

    def test_compte_hors_embedding(self, tiny_model):
        total = tiny_model.n_parameters()
        hors = tiny_model.n_parameters(embeddings=False)
        assert hors == total - tiny_model.embedding.weight.numel()

    def test_gradients_partout(self, tiny_model):
        ids = torch.randint(0, 128, (2, 16))
        _, loss = tiny_model(ids, targets=ids)
        loss.backward()
        sans_gradient = [n for n, p in tiny_model.named_parameters() if p.grad is None]
        assert not sans_gradient, f"paramètres sans gradient : {sans_gradient}"

    def test_generation(self, tiny_model):
        out = tiny_model.generate(torch.randint(0, 128, (1, 4)), max_new_tokens=6)
        assert out.shape == (1, 10)

    def test_generation_deterministe_a_temperature_nulle(self, tiny_model):
        ids = torch.randint(0, 128, (1, 4))
        a = tiny_model.generate(ids, max_new_tokens=5, temperature=0.0)
        b = tiny_model.generate(ids, max_new_tokens=5, temperature=0.0)
        assert torch.equal(a, b)


class TestInitialisation:
    def test_projections_residuelles_attenuees(self):
        # Sans cette mise à l'échelle, la variance croît avec la profondeur et
        # un modèle profond démarre saturé.
        torch.manual_seed(0)
        avec = Thadeus(ModelConfig(**{**TINY, "n_layers": 8}))
        torch.manual_seed(0)
        sans = Thadeus(ModelConfig(**{**TINY, "n_layers": 8, "scale_residual_init": False}))
        assert avec.blocks[0].attn.o_proj.weight.std() < sans.blocks[0].attn.o_proj.weight.std()

    def test_groupes_separent_matrices_et_vecteurs(self, tiny_model):
        # Régulariser les gains de normalisation revient à les désactiver
        # progressivement — bug silencieux, jamais une erreur.
        groups = parameter_groups(tiny_model, weight_decay=0.1)
        assert groups[0]["weight_decay"] == 0.1
        assert groups[1]["weight_decay"] == 0.0
        assert all(p.dim() >= 2 for p in groups[0]["params"])
        assert all(p.dim() < 2 for p in groups[1]["params"])

    def test_tous_les_parametres_classes(self, tiny_model):
        groups = parameter_groups(tiny_model)
        classes = sum(len(g["params"]) for g in groups)
        assert classes == len(list(tiny_model.parameters()))


class TestDimensionnement:
    @pytest.mark.parametrize(
        "surcharge",
        [
            {},
            {"tie_embeddings": False},
            {"norm": "layernorm", "ffn": {"name": "mlp"}},
            {"attention": {"name": "gqa", "n_heads": 4, "n_kv_heads": 4, "head_dim": 16}},
            {
                "attention": {
                    "name": "gqa",
                    "n_heads": 4,
                    "n_kv_heads": 1,
                    "head_dim": 16,
                    "qk_norm": False,
                }
            },
        ],
    )
    def test_formule_egale_le_modele_reel(self, surcharge):
        # Le test le plus important de ce fichier : si la formule dérive du
        # modèle, tout budget de calcul calculé à l'avance est faux.
        cfg = ModelConfig(**{**TINY, **surcharge})
        assert estimate(cfg).total == Thadeus(cfg).n_parameters()

    def test_configs_du_depot_valides(self):
        for name in (
            "model/small.toml",
            "model/tiny.toml",
            "model/small_mha.toml",
            "model/small_mlp.toml",
        ):
            cfg = ModelConfig(**load_config(name))
            assert estimate(cfg).total > 0

    def test_part_de_l_embedding_decroit_avec_la_largeur(self):
        # Le fait qui a dimensionné small.toml : à 32 k de vocabulaire, d_model
        # a un plancher imposé par la table d'embedding.
        commun = {"vocab_size": 32_000, "n_kv_heads": 2, "head_dim": 64}
        etroit = estimate(
            ModelConfig(
                vocab_size=commun["vocab_size"],
                d_model=512,
                n_layers=8,
                attention={"name": "gqa", "n_heads": 8, "n_kv_heads": 2, "head_dim": 64},
            )
        )
        large = estimate(
            ModelConfig(
                vocab_size=commun["vocab_size"],
                d_model=768,
                n_layers=12,
                attention={"name": "gqa", "n_heads": 12, "n_kv_heads": 4, "head_dim": 64},
            )
        )
        assert etroit.embedding_share > 0.40, "à 512, la table écrase le modèle"
        assert large.embedding_share < 0.30

    def test_heures_croissent_avec_la_taille(self):
        petit = estimate(ModelConfig(**TINY))
        grand = estimate(ModelConfig(**{**TINY, "n_layers": 8}))
        assert grand.hours_for(1e9, seq_len=32, effective_tflops=10) > petit.hours_for(
            1e9, seq_len=32, effective_tflops=10
        )


class TestValidationConfig:
    def test_incoherence_tetes_dimension_rejetee(self):
        with pytest.raises(ValidationError, match="d_model"):
            ModelConfig(d_model=640, attention={"name": "gqa", "n_heads": 8, "head_dim": 64})

    def test_d_model_non_divisible_rejete(self):
        with pytest.raises(ValidationError, match="divisible"):
            ModelConfig(d_model=100, attention={"name": "gqa", "n_heads": 8})

    def test_cle_inconnue_rejetee(self):
        with pytest.raises(ValidationError):
            ModelConfig(n_layer=12)  # type: ignore[call-arg]
