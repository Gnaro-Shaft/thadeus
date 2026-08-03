# Thadeus

Un LLM entraîné from scratch, du corpus au modèle.

Journal du projet (source de vérité) : `dGnaro/02 - Projets/Thadeus/Thadeus - Vue d'ensemble.md`

## Le principe d'architecture

La chaîne est découpée en étages, et **le contrat entre deux étages est un fichier
sur disque, pas un appel de fonction** :

```
corpus → [1 data] → shards → [2 tokenizer] → tokens → [3 model] ↘
                                                                  [5 train] → ckpt → [6 eval]
                                                        [4 optim] ↗
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
