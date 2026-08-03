"""Le Transformer, assemblé depuis la config.

Aucune constante d'architecture n'est écrite ici : normalisation, attention et
feed-forward sont choisis par leur nom dans les registres. C'est la condition
matérielle des A/B de la Phase 6 — comparer GQA et MLA doit se faire en changeant
une ligne de TOML, pas en éditant ce fichier.

**Précision sur les dtypes**, qui découle directement des mesures de Phase 0 :
les paramètres vivent en fp32, les produits matriciels s'exécutent en bf16 via
``autocast``. Ce n'est pas une demi-mesure — c'est le seul réglage qui donne à
la fois le débit du bf16 (facteur 4 sur MPS) et la stabilité numérique des mises
à jour d'optimiseur, qui accumulent des incréments minuscules qu'un bf16 à 8 bits
de mantisse arrondirait à zéro.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from thadeus.model.blocks import ATTENTIONS, FFNS, NORMS
from thadeus.model.blocks.rope import RotaryEmbedding
from thadeus.model.config import ModelConfig

__all__ = ["Thadeus", "TransformerBlock"]


class TransformerBlock(nn.Module):
    """Un bloc : attention puis feed-forward, chacun en pré-normalisation résiduelle.

    **Pré-normalisation** (normaliser l'entrée du sous-bloc) et non
    post-normalisation : le chemin résiduel reste alors une somme non normalisée
    de la couche 0 à la sortie, ce qui laisse le gradient remonter sans
    atténuation. C'est ce qui permet d'entraîner en profondeur sans warmup
    acrobatique — un luxe qu'on ne peut pas se payer avec notre budget.
    """

    def __init__(self, cfg: ModelConfig, *, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.attn_norm = NORMS.build(cfg.norm, dim=cfg.d_model)
        self.attn = ATTENTIONS.build(cfg.attention, d_model=cfg.d_model)
        self.ffn_norm = NORMS.build(cfg.norm, dim=cfg.d_model)
        self.ffn = FFNS.build(cfg.ffn, d_model=cfg.d_model)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.ffn(self.ffn_norm(x))


class Thadeus(nn.Module):
    """Le modèle de langue.

    Args:
        cfg: la configuration validée. Le modèle ne lit **jamais** ailleurs.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(cfg, layer_index=i) for i in range(cfg.n_layers)
        )
        self.final_norm = NORMS.build(cfg.norm, dim=cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # RoPE partagé par toutes les couches : le recalculer par couche
        # gaspillerait de la bande passante sans rien changer au résultat.
        self.rope = RotaryEmbedding(
            self.blocks[0].attn.head_dim,
            base=cfg.rope_base,
            max_seq_len=cfg.max_seq_len,
        )

        if cfg.tie_embeddings:
            # À 32 k de vocabulaire et d_model = 640, la table pèse 20 M de
            # paramètres. La partager entre entrée et sortie en économise autant,
            # soit ~25 % d'un modèle de 85 M — arbitrage décisif à notre échelle.
            self.lm_head.weight = self.embedding.weight

        from thadeus.model.init import initialize

        initialize(self, cfg)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Passe avant.

        Args:
            input_ids: entiers ``(batch, seq)``.
            targets: cibles décalées ``(batch, seq)``. Fournies, la perte est
                calculée ici — ce qui évite de matérialiser les logits en fp32
                dans la boucle d'entraînement.

        Returns:
            ``(logits, loss)``, ``loss`` valant ``None`` sans cibles.
        """
        _, seq = input_ids.shape
        if seq > self.cfg.max_seq_len:
            raise ValueError(
                f"séquence de {seq} tokens au-delà de max_seq_len={self.cfg.max_seq_len}"
            )

        x = self.embedding(input_ids)
        cos, sin = self.rope(seq, device=x.device, dtype=x.dtype)

        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        if self.cfg.logit_scale != 1.0:
            logits = logits * self.cfg.logit_scale

        loss = None
        if targets is not None:
            # Perte en fp32 : l'entropie croisée additionne des exponentielles
            # sur tout le vocabulaire, et le faire en bf16 perd la queue de
            # distribution — précisément les tokens rares qu'on veut apprendre.
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int | None = 50,
        forbidden: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Génération autorégressive naïve, sans cache clé/valeur.

        Volontairement simple : elle sert à **regarder ce que le modèle produit**
        pendant le développement, pas à servir des requêtes. Une génération
        efficace (cache KV, lots) viendra quand il y aura un modèle qui vaudra
        la peine d'être servi.

        Args:
            forbidden: identifiants interdits à l'échantillonnage. Sert aux
                jetons de service (`<|pad|>`, créneaux réservés) : ils occupent
                des identifiants du vocabulaire, donc un modèle peu entraîné les
                tire comme les autres, et ils polluent la sortie. Les interdire
                est une décision d'inférence, pas d'entraînement — le modèle
                continue de les voir en apprentissage.
        """
        self.eval()
        interdits = torch.tensor(list(forbidden), device=input_ids.device) if forbidden else None
        for _ in range(max_new_tokens):
            window = input_ids[:, -self.cfg.max_seq_len :]
            logits, _ = self(window)
            logits = logits[:, -1, :].float()
            if interdits is not None:
                logits[:, interdits] = float("-inf")

            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

            input_ids = torch.cat((input_ids, next_token), dim=1)
        return input_ids

    def n_parameters(self, *, embeddings: bool = True) -> int:
        """Nombre de paramètres.

        ``embeddings=False`` donne le compte **non-embedding**, celui qui entre
        dans le calcul des FLOPs : une table d'embedding est une lecture
        indexée, pas un produit matriciel.
        """
        total = sum(p.numel() for p in self.parameters())
        if embeddings:
            return total
        embed = self.embedding.weight.numel()
        # Poids partagés : la table n'est comptée qu'une fois par `parameters()`.
        if not self.cfg.tie_embeddings:
            embed += self.lm_head.weight.numel()
        return total - embed
