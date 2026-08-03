"""Muon — descente de gradient à momentum **orthogonalisé**.

L'intuition, avant les formules. Le gradient d'une matrice de poids est
lui-même une matrice, et ses valeurs singulières sont très inégales : quelques
directions dominent, les autres sont écrasées. Une mise à jour brute avance donc
beaucoup dans deux ou trois directions et presque pas dans les centaines
d'autres — alors que rien ne dit que ces directions-là méritent ce traitement.

Muon **égalise** : il remplace la mise à jour par la matrice orthogonale la plus
proche, c'est-à-dire la même matrice avec toutes ses valeurs singulières ramenées
à 1. Chaque direction reçoit le même pas. Empiriquement, cela fait converger en
nettement moins de tokens — ce qui, pour un projet limité par le calcul, est
exactement la bonne monnaie.

**Ce que Muon ne touche pas.** L'orthogonalisation n'a de sens que pour les
matrices d'une couche cachée, où lignes et colonnes jouent des rôles
symétriques. Les tables d'embedding (des vecteurs indexés, pas une
transformation), les gains de normalisation et les biais restent à AdamW. Muon
est donc toujours utilisé **en binôme**, jamais seul.

**Coût.** L'orthogonalisation passe par cinq itérations de Newton-Schulz, soit
~10 produits matriciels par matrice et par pas. Sur un modèle de 188 M, c'est
mesurable mais faible devant la passe avant/arrière — et c'est précisément ce
que le banc doit vérifier plutôt que supposer.

Référence : Keller Jordan, *Muon: An optimizer for hidden layers in neural
networks* (2024).
"""

from __future__ import annotations

import torch

__all__ = ["Muon", "orthogonalize"]

# Coefficients du polynôme quintique de Newton-Schulz. Ils ne sont pas choisis
# pour converger vite vers l'orthogonalisation exacte, mais pour amener
# **toutes** les valeurs singulières dans un voisinage de 1 en cinq itérations.
# Une orthogonalisation exacte (SVD) serait plus précise et bien trop lente ;
# l'approximation suffit, seule compte l'égalisation des directions.
_NS_COEFFS = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def orthogonalize(matrix: torch.Tensor, *, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approche la matrice orthogonale la plus proche, par itération de Newton-Schulz.

    Le calcul se fait en bfloat16 : l'itération est une suite de produits
    matriciels, et on cherche une **approximation** dont la précision fine n'a
    aucune importance — seul compte le fait de ramener les valeurs singulières
    autour de 1. Sur MPS, le bf16 vaut un facteur 4 sur ces produits (Phase 0).

    On transpose quand la matrice est plus haute que large : l'itération opère
    sur ``X @ X.T``, dont le coût dépend de la plus petite dimension.
    """
    if matrix.ndim != 2:
        raise ValueError(f"orthogonalize attend une matrice 2D, reçu {matrix.ndim}D")

    a, b, c = _NS_COEFFS
    x = matrix.bfloat16()
    x = x / (x.norm() + eps)

    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T

    for _ in range(steps):
        gram = x @ x.T
        poly = b * gram + c * (gram @ gram)
        x = a * x + poly @ x

    if transposed:
        x = x.T
    return x.to(matrix.dtype)


class Muon(torch.optim.Optimizer):
    """Momentum orthogonalisé, pour les matrices de couches cachées uniquement.

    Args:
        params: **uniquement** des matrices 2D de couches cachées. Y passer une
            table d'embedding ou un gain de normalisation est une erreur, et
            l'optimiseur la refuse plutôt que de produire silencieusement des
            mises à jour absurdes.
        lr: taux d'apprentissage. Muon tolère des valeurs nettement plus élevées
            qu'AdamW, l'orthogonalisation bornant la taille du pas.
        momentum: coefficient du momentum.
        nesterov: momentum de Nesterov — regarde le gradient après le pas de
            momentum plutôt qu'avant. Recommandé par l'auteur.
        ns_steps: itérations de Newton-Schulz. 5 est le compromis retenu ;
            en dessous l'égalisation est incomplète, au-dessus on paie sans gain.
        weight_decay: décroissance **découplée**, appliquée au poids et non au
            gradient — sinon elle passerait par l'orthogonalisation, qui
            détruirait son échelle.
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr doit être strictement positif, reçu {lr}")
        if not 0 <= momentum < 1:
            raise ValueError(f"momentum doit être dans [0, 1), reçu {momentum}")
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon n'accepte que des matrices 2D de couches cachées, reçu un "
                        f"tenseur {p.ndim}D de forme {tuple(p.shape)}. Les embeddings, "
                        f"gains de normalisation et biais doivent aller à AdamW."
                    )

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D102
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                update = grad.add(buf, alpha=momentum) if group["nesterov"] else buf
                update = orthogonalize(update, steps=group["ns_steps"])

                # Compense l'asymétrie de forme. Une matrice très rectangulaire
                # a beaucoup plus de lignes que de directions indépendantes ;
                # sans ce facteur, la norme du pas dépendrait de la forme plutôt
                # que du taux d'apprentissage demandé.
                update = update * max(1.0, p.size(0) / p.size(1)) ** 0.5

                if group["weight_decay"]:
                    p.mul_(1 - lr * group["weight_decay"])
                p.add_(update, alpha=-lr)

        return loss
