# Thadeus

Un LLM entraîné from scratch, du corpus au modèle.

Journal du projet (source de vérité) : `dGnaro/02 - Projets/Thadeus/Thadeus - Vue d'ensemble.md`

## Le principe d'architecture

La chaîne est découpée en étages, et **le contrat entre deux étages est un fichier
sur disque, pas un appel de fonction** :

```
sources → [1 data] → shards → [2 tokenizer] → tokens ─┐
                                                       ├─→ [5 train] → ckpt → [6 eval]
                          [3 model] + [4 optim] ───────┘
```

Deux conséquences voulues :

- On rejoue un étage sans refaire les précédents.
- On compare deux variantes d'un étage toutes choses égales par ailleurs — ce
  qui est la condition pour que les A/B d'architecture veuillent dire quelque chose.

Second principe : **config + registre**. Aucune constante en dur. Chaque composant
interchangeable s'enregistre sous un nom et se sélectionne depuis un TOML, donc
passer de GQA à MLA est une ligne de config, pas un refactoring.

## Deux machines, deux rôles

| | **Mac M5 Pro (MPS)** | **H100 (CUDA, Lightning AI)** |
|---|---|---|
| Rôle | laboratoire | usine |
| Coût | gratuit, illimité | crédits finis |
| Sert à | pipeline, tokenizer, modèles jouets, A/B, réglage muP | le run scalé |

Une seule base de code couvre les deux : `thadeus.core.device` est la seule
frontière où le backend est visible.

## Démarrage

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

Mesurer la machine (à rejouer tel quel sur le H100 — c'est ce qui permet de
dimensionner le run final sur des mesures et non sur des estimations) :

```bash
.venv/bin/python scripts/bench.py --config bench/base.toml
```

Vérifications :

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

## Mesures de référence — MacBook M5 Pro, 20 cœurs GPU (2026-08-03)

| Mesure | Valeur |
|---|---|
| matmul bf16 (crête, n=4096) | **30,0 TFLOPS** |
| matmul fp16 | 29,7 TFLOPS |
| matmul fp32 | 7,4 TFLOPS — **÷4** |
| attention SDPA causale | **6,9 TFLOPS** — 23 % de la crête matmul |
| bande passante | 258 Go/s |
| mémoire GPU allouable | 51,8 Go sur 64 |

Trois règles qui en découlent et qui s'appliquent partout dans le code :

1. **bf16 sur tout le chemin chaud.** Le fp32 coûte un facteur 4 sur MPS.
   `core.device.warn_if_fp32` sert de garde-fou.
2. **Aucune optimisation mémoire.** Un modèle de 150 M paramètres occupe ~2 Go
   sur 51,8 disponibles : on est limité par le calcul, pas par la mémoire.
3. **L'attention est le vrai goulot sur MPS**, pas les matmuls. Toute variante
   qui réduit son coût (MLA, GQA) se juge contre les 6,9 TFLOPS ci-dessus.

## Structure

```
configs/          TOML, avec héritage (extends) — tout ce qui peut varier vit ici
scripts/          1 script = 1 étage, exécutable seul
src/thadeus/
  core/           config, registry, device, seeding, artifacts, logs
  bench/          étage 0 — mesure de la machine, identique Mac et H100
  data/           étage 1 — sources, nettoyage, dédup, mélange, shards
  tokenizer/      étage 2 — BPE byte-level entraîné sur notre corpus
  model/          étage 3 — Transformer assemblé par config
  optim/          étage 4 — Muon, muP, planificateurs
  train/          étage 5 — boucle, checkpoints, reprise
  eval/           étage 6 — perplexité, sondes, génération
tests/
artifacts/        sorties versionnées par hash de config (non versionné en git)
```

## Le corpus

```bash
.venv/bin/python scripts/build_corpus.py --config data/smoke.toml    # ~1 min, vérifie la chaîne
.venv/bin/python scripts/build_corpus.py --config data/fr_first.toml # plusieurs heures
```

Composition visée : **55 % français · 25 % technique anglais · 15 % code · 5 % notes**.
À ~85 M paramètres, viser le français *et* le code à la fois est le meilleur moyen
de n'avoir ni l'un ni l'autre ; le code se rattrape en fine-tuning sur une base
linguistique solide, l'inverse ne marche pas.

Après chaque exécution, lire `report.json` dans l'artefact : composition demandée
contre composition obtenue, et taux de rejet **par filtre et par source**. La
ventilation est le seul niveau interprétable — sur le run de fumée, Wikipédia FR
perd 26,7 % quand FineWeb-Edu perd 0,2 %, et c'est cet écart qui informe.

## Le tokenizer

```bash
.venv/bin/python scripts/train_tokenizer.py --config tokenizer/bpe32k.toml
.venv/bin/python scripts/compare_tokenizers.py --ours bpe32k
```

BPE byte-level entraîné sur notre corpus, avec un motif de pré-tokenisation
adapté au français : les élisions (`l'`, `qu'`, `jusqu'`…) restent rattachées au
mot-outil, là où le motif de GPT-2 les coupe systématiquement.

Fertilité mesurée (tokens/mot, plus bas = plus efficace), corpus de fumée :

| Tokenizer | Vocab | Global | FR | EN |
|---|---|---|---|---|
| **thadeus** | **16 k** | **1,690** | **1,747** | 1,558 |
| qwen2.5 | 150 k | 1,712 | 1,879 | 1,319 |
| mistral | 32 k | 1,934 | 2,131 | 1,472 |
| gpt2 | 50 k | 1,950 | 2,218 | 1,320 |

**−21 % de tokens sur le français** vs GPT-2, −13,3 % sur le mélange complet.
Notre vocabulaire de 16 k bat celui de Qwen2.5 à 150 k sur le français : un
vocabulaire dédié à une distribution bat un vocabulaire dix fois plus gros
partagé entre cent langues.

Le troc est assumé et mesuré : **on est 18 % moins efficace que GPT-2 sur
l'anglais**, parce que le vocabulaire est dépensé en français.

Deux gains distincts, à ne pas confondre : entraîner le vocabulaire sur notre
corpus vaut **−21 %**, le motif de pré-tokenisation française **−2 %** de plus.

## Le modèle

```bash
.venv/bin/python scripts/model_info.py --config model/medium.toml
.venv/bin/python scripts/model_info.py --config model/small.toml --benchmark
.venv/bin/python scripts/model_info.py --config model/tiny.toml --overfit
```

Transformer moderne — RMSNorm, SwiGLU, RoPE, GQA, QK-norm — assemblé depuis la
config. Chaque brique est enregistrée sous un nom et substituable en une ligne
de TOML : c'est la condition matérielle des A/B d'architecture.

`medium.toml` (188 M paramètres) est **dérivé du budget, pas choisi** : 15,24
crédits ≈ 4,2 EFLOPs, et l'optimum de Chinchilla place le modèle là.

Deux mesures qui décident de tout :

| | |
|---|---|
| `torch.compile` sur MPS | **×3,00** — 27 h au lieu de 72 h pour 1,7 Md de tokens |
| Débit effectif réel | 8,3 TFLOPS compilé · 3,1 en eager |

**Ne jamais dimensionner un run sur une fraction supposée de la crête matmul.**
Le mode compilé n'est pas une optimisation de confort, c'est la différence entre
un budget tenu et un budget faux d'un facteur 3.

## L'optimiseur

```bash
.venv/bin/python scripts/lr_sweep.py --widths 128 256 512 --optim adamw
.venv/bin/python scripts/lr_sweep.py --widths 128 256 512 --optim adamw --mup
```

| Configuration | Optimum par largeur (128/256/512) | Transfert |
|---|---|---|
| AdamW standard | 1e-3 → 3e-4 → 3e-4 | ❌ dérive |
| **AdamW + muP** | **1e-3 → 1e-3 → 1e-3** | ✅ transfère |

**muP n'est pas un accélérateur, c'est une assurance.** Il ne fait pas converger
plus vite ; il garantit que le taux d'apprentissage trouvé sur un modèle jouet
gratuit vaut encore sur le modèle final. À budget de crédits fini, éviter un run
gaspillé vaut plus qu'un gain de vitesse.

Muon est implémenté mais **non confirmé** : il perd de 2 % sur notre banc, dont
la tâche est cependant un mauvais proxy — elle est dominée par les couches qui
vont à AdamW dans les deux cas. À départager sur du texte réel.

## L'entraînement

```bash
.venv/bin/python scripts/tokenize_corpus.py --corpus-label thadeus_v1 --tokenizer bpe32k
.venv/bin/python scripts/train.py --config train/medium_mup.toml
```

Relancer la même commande après une interruption **reprend exactement où l'on
s'était arrêté, mêmes lots compris**. Deux propriétés le garantissent :

- **Le chargeur est sans état** : les lots dérivent du numéro de pas, donc
  restaurer le pas suffit. La meilleure façon de ne pas perdre un état est de
  ne pas en avoir.
- **L'artefact est nommé par l'identité du run**, pas par sa config complète.
  Durée, verbosité et fréquence de sauvegarde en sont exclues — allonger un run
  le reprend au lieu d'en créer un nouveau.

Le planificateur est **WSD** et non cosinus : palier à taux constant, puis
décroissance sur les derniers 10 %. On peut donc arrêter le palier quand on
veut et ne payer que la décroissance — ce qui permet d'entraîner par nuits sur
le Mac et de finir au cloud.

## L'évaluation

```bash
.venv/bin/python scripts/evaluate.py --config eval/default.toml
```

Trois mesures pour trois questions distinctes :

- **Perplexité par source** — le modèle prédit-il bien ce corpus ?
- **Sondes** (26 paires minimales) — *qu'a-t-il* appris ? *« Les enfants
  mangent »* contre *« Les enfants mange »* : laquelle juge-t-il plus probable ?
  Le hasard donne 50 %.
- **Génération** — attrape les pathologies qu'aucun chiffre ne montre.

Les **bits par caractère** sont la seule métrique comparable entre modèles à
tokenizers différents : un tokenizer plus efficace gonfle mécaniquement la
perplexité par token sans que le modèle soit moins bon.

Sans checkpoint, l'évaluation porte sur un modèle non entraîné et donne la ligne
de base — perte ≈ `ln(V)`, sondes ≈ 50 %. Si ce chiffre dévie, c'est la mesure
qu'il faut suspecter, pas le modèle.
