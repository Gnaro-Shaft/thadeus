"""Device, graines et artefacts : les trois briques qui doivent se comporter
identiquement sur le Mac et sur le H100.
"""

from __future__ import annotations

import torch

from thadeus.bench.flops import mfu, tokens_for_budget, training_flops, transformer_flops_per_token
from thadeus.core.artifacts import open_artifact
from thadeus.core.device import (
    describe,
    hot_path_dtype,
    resolve_device,
    supports_bf16,
    synchronize,
)
from thadeus.core.seeding import derive_seed, new_generator, seed_everything

CPU = torch.device("cpu")


class TestDevice:
    def test_resolve_auto_donne_un_device_utilisable(self):
        device = resolve_device("auto")
        assert device.type in ("cuda", "mps", "cpu")
        torch.zeros(2, device=device)  # doit passer sur les trois backends

    def test_resolve_explicite(self):
        assert resolve_device("cpu").type == "cpu"

    def test_describe_est_serialisable(self):
        # Les métadonnées d'artefact passent par json.dumps.
        payload = describe(resolve_device("auto")).to_dict()
        assert payload["backend"] in ("cuda", "mps", "cpu")
        assert isinstance(payload["bf16"], bool)

    def test_bf16_sur_le_chemin_chaud(self):
        device = resolve_device("auto")
        expected = torch.bfloat16 if supports_bf16(device) else torch.float32
        assert hot_path_dtype(device) is expected

    def test_synchronize_ne_leve_pas_sur_cpu(self):
        synchronize(CPU)


class TestSeeding:
    def test_meme_graine_meme_tirage(self):
        seed_everything(42)
        a = torch.randn(64)
        seed_everything(42)
        assert torch.equal(a, torch.randn(64))

    def test_graines_differentes_tirages_differents(self):
        seed_everything(1)
        a = torch.randn(64)
        seed_everything(2)
        assert not torch.equal(a, torch.randn(64))

    def test_derive_seed_est_deterministe(self):
        assert derive_seed(42, "dataloader", 3) == derive_seed(42, "dataloader", 3)

    def test_derive_seed_separe_les_etages(self):
        # Ajouter un tirage dans le chargeur ne doit pas décaler l'init du modèle,
        # sinon deux variantes comparées ne partent plus du même point.
        assert derive_seed(42, "dataloader") != derive_seed(42, "model")

    def test_generateur_explicite(self):
        a = torch.randn(8, generator=new_generator(7))
        b = torch.randn(8, generator=new_generator(7))
        assert torch.equal(a, b)


class TestArtifacts:
    def test_le_hash_nomme_le_repertoire(self, tmp_path):
        art = open_artifact("bench", "kernels", {"seed": 1}, root=tmp_path)
        assert art.path.name.startswith("kernels-")
        assert art.hash in art.path.name

    def test_deux_configs_ne_se_marchent_pas_dessus(self, tmp_path):
        a = open_artifact("bench", "k", {"lr": 0.001}, root=tmp_path)
        b = open_artifact("bench", "k", {"lr": 0.002}, root=tmp_path)
        assert a.path != b.path

    def test_exists_faux_tant_que_meta_absent(self, tmp_path):
        # Un répertoire à moitié rempli après interruption ne doit jamais être
        # pris pour un artefact valide.
        art = open_artifact("bench", "k", {"seed": 1}, root=tmp_path)
        art.create()
        art.write_json("results.json", [1, 2, 3])
        assert not art.exists()
        art.write_meta({"seed": 1})
        assert art.exists()

    def test_meta_contient_config_et_environnement(self, tmp_path):
        art = open_artifact("bench", "k", {"seed": 1}, root=tmp_path)
        art.write_meta({"seed": 1}, note="essai")
        meta = art.read_meta()
        assert meta["config"] == {"seed": 1}
        assert meta["note"] == "essai"
        assert "python" in meta["environment"]


class TestFlops:
    def test_backward_coute_trois_fois_le_forward(self):
        kwargs = {"n_layers": 12, "d_model": 768, "seq_len": 1024}
        fwd = transformer_flops_per_token(100_000_000, backward=False, **kwargs)
        bwd = transformer_flops_per_token(100_000_000, backward=True, **kwargs)
        assert bwd == 3 * fwd

    def test_attention_croit_avec_le_contexte(self):
        base = {"n_layers": 12, "d_model": 768}
        court = transformer_flops_per_token(1e8, seq_len=1024, **base)
        long = transformer_flops_per_token(1e8, seq_len=8192, **base)
        assert long > court

    def test_chinchilla_6nd(self):
        assert training_flops(100_000_000, 2_000_000_000) == 6 * 1e8 * 2e9

    def test_budget_reproduit_le_cadrage_du_projet(self):
        # 10 TFLOPS effectifs pendant 24 h sur un modèle de 85 M paramètres
        # doivent redonner ~1,7 Md de tokens (chiffre inscrit dans le Vault).
        tokens = tokens_for_budget(85_000_000, hours=24, effective_tflops=10)
        assert 1.5e9 < tokens < 2.0e9

    def test_mfu_borne(self):
        assert mfu(tokens_per_second=1000, flops_per_token=1e9, peak_flops=1e13) == 0.1
