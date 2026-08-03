"""Comptabilité des FLOPs — l'unité de compte du projet.

Thadeus est **limité par le calcul**. Le FLOP est donc la monnaie : dimensionner
un run, comparer deux architectures, décider si les crédits H100 suffisent —
tout se ramène ici.

Deux usages distincts, à ne pas confondre :

- :func:`training_flops` répond à « combien coûte cet entraînement ? » (budget,
  avant de lancer).
- :func:`mfu` répond à « quelle fraction de la machine est réellement
  utilisée ? » (efficacité, pendant le run). Un MFU qui s'effondre signale un
  goulot — chargement de données, synchronisation, un fp32 égaré — et c'est le
  seul indicateur qui le révèle rapidement.
"""

from __future__ import annotations

__all__ = [
    "MEASURED_PEAK_TFLOPS",
    "mfu",
    "tokens_for_budget",
    "training_flops",
    "transformer_flops_per_token",
]

# Crêtes bf16 **mesurées**, pas annoncées par le constructeur (voir bench/kernels.py).
# Une crête de fiche technique conduit systématiquement à surestimer ce qu'on
# peut faire ; on ne compare qu'à ce qu'on a soi-même chronométré.
MEASURED_PEAK_TFLOPS: dict[str, float] = {
    "m5_pro_20c": 30.0,  # MacBook M5 Pro, 20 cœurs GPU — mesuré le 2026-08-03
}

# Débit **réellement atteint** en boucle d'entraînement complète (avant, arrière,
# pas d'optimiseur), à distinguer soigneusement de la crête matmul ci-dessus.
#
# Mesuré en Phase 3 sur un modèle de 80,7 M, batch 8, seq 1024 :
#
#   eager           3,1 TFLOPS   MFU 10 %    <- ne jamais dimensionner un run là-dessus
#   torch.compile   8,3 TFLOPS   MFU 28 %    <- le seul mode à utiliser
#
# **`torch.compile` vaut un facteur 3,00 sur MPS.** Ce n'est pas une optimisation
# de confort : c'est la différence entre 72 h et 27 h pour 1,7 Md de tokens sur
# le Mac. Le coût de compilation (~15 s) est négligeable sur un run long.
#
# La leçon de méthode, elle, a coûté du temps : on a d'abord traqué des gains de
# quelques pourcents sur les opérations élément par élément **avant** d'avoir
# établi la bonne référence. Un facteur 3 attendait dans un drapeau. Toujours
# épuiser les réglages globaux avant d'optimiser le détail.
#
# Règle qui en découle : **on ne dimensionne un run qu'avec un débit mesuré sur
# la machine visée, dans le mode où le run tournera**. Sur H100, ces deux chiffres
# devront être remesurés avant d'engager le moindre crédit.
MEASURED_EFFECTIVE_TFLOPS: dict[str, float] = {
    "m5_pro_20c": 8.3,  # avec torch.compile — le mode par défaut du projet
    "m5_pro_20c_eager": 3.1,  # sans compilation, pour référence
}


def transformer_flops_per_token(
    n_params_non_embed: int,
    *,
    n_layers: int,
    d_model: int,
    seq_len: int,
    backward: bool = True,
) -> float:
    """FLOPs par token pour un Transformer dense.

    Deux termes de nature différente :

    - ``6 * N`` — les matmuls des poids. Croît avec la taille du modèle.
    - ``12 * L * d * s`` — les produits d'attention (QKᵀ puis ·V). Ceux-ci ne
      dépendent d'aucun poids mais du **carré** de la longueur de contexte, et
      deviennent dominants en contexte long. C'est précisément ce terme que les
      variantes d'attention (MLA, GQA) cherchent à réduire — d'où l'importance
      de le compter séparément quand on les compare.

    Le facteur 6 (au lieu de 2) vient de la rétropropagation : une passe avant
    plus deux passes arrière, gradients d'entrée et de poids.

    Args:
        n_params_non_embed: paramètres hors table d'embedding. On l'exclut car
            un embedding est une lecture indexée, pas un produit matriciel — la
            compter gonflerait le coût estimé sans correspondre à du calcul.
        backward: mettre ``False`` pour l'inférence (rapport 3 avec l'entraînement).
    """
    factor = 6.0 if backward else 2.0
    weights = factor * n_params_non_embed
    attention = (2.0 * factor) * n_layers * d_model * seq_len
    return weights + attention


def training_flops(n_params: int, n_tokens: int) -> float:
    """Coût total approximé d'un entraînement : ``6 · N · D``.

    L'approximation de Chinchilla — elle ignore le terme d'attention, donc
    sous-estime en contexte long. Suffisante pour dimensionner un budget,
    insuffisante pour comparer deux architectures : utiliser
    :func:`transformer_flops_per_token` dans ce cas.
    """
    return 6.0 * n_params * n_tokens


def tokens_for_budget(n_params: int, hours: float, effective_tflops: float) -> float:
    """Combien de tokens un budget horaire permet de voir, à taille de modèle fixée.

    ``effective_tflops`` est le débit **réel** en boucle d'entraînement, pas la
    crête matmul : compter environ 30-40 % de la crête. Utiliser la crête ici
    est l'erreur qui fait promettre trois jours pour un run qui en prend dix.
    """
    return (effective_tflops * 1e12 * hours * 3600.0) / (6.0 * n_params)


def mfu(tokens_per_second: float, flops_per_token: float, peak_flops: float) -> float:
    """Model FLOPs Utilization — fraction de la crête effectivement exploitée.

    Repères : 0,35-0,50 est bon, en dessous de 0,20 il y a un goulot à trouver
    avant d'envisager quoi que ce soit d'autre. Optimiser une architecture
    pendant qu'on tourne à 10 % de MFU, c'est régler la voile d'un bateau dont
    l'ancre est jetée.
    """
    if peak_flops <= 0:
        raise ValueError("peak_flops doit être strictement positif")
    return (tokens_per_second * flops_per_token) / peak_flops
