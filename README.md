# Thadeus

**Un LLM de 188 M de paramètres, entraîné de zéro sur un MacBook.** Collecte du
corpus, tokenizer, architecture, entraînement, évaluation, réglage fin et RAG :
toute la chaîne est ici, et chaque chiffre annoncé a été mesuré sur cette machine.

Le modèle est orienté **français**. Il a vu 3,74 milliards de tokens en 110 heures
de calcul local, pour un coût matériel de zéro euro.

## Ce que ce dépôt est — et ce qu'il n'est pas

C'est un projet d'apprentissage mené jusqu'au bout, pas un modèle à mettre en
production. À 188 M de paramètres, **Thadeus écrit un français correct mais ne
raisonne pas et n'est pas une source fiable de faits** : il invente des dates,
des noms et des références avec le même aplomb qu'il conjugue juste. Les limites
sont détaillées en fin de page, chiffres à l'appui.

Ce qui vaut le détour, à mon sens, tient plutôt dans la **méthode** : une chaîne
découpée en étages rejouables, des décisions prises sur des mesures plutôt que
sur des réputations, et les erreurs consignées plutôt qu'effacées — plusieurs
sections ci-dessous documentent une estimation que la mesure a démentie.

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

Corpus final `thadeus_v1` : **4,53 millions de documents, 4,00 Md de tokens**,
répartis sur 91 shards.

| | Visé | Obtenu |
|---|---|---|
| Français | 55 % | **65,5 %** |
| Anglais technique | 25 % | 19,1 % |
| Code | 15 % | **15,3 %** |
| Notes personnelles | 5 % | 0,1 % |

L'écart n'est pas une dérive silencieuse : `plan_mixture` **refuse de combler un
déficit** en répétant des documents, et consigne chaque manque dans `report.json`.
Le français dépasse sa cible parce que les sources anglaises se sont épuisées les
premières ; les notes personnelles ne pesaient de toute façon que 1 681 documents.

À cette échelle, viser le français *et* le code à la fois est le meilleur moyen
de n'avoir ni l'un ni l'autre ; le code se rattrape en réglage fin sur une base
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

Fertilité mesurée (tokens/mot, plus bas = plus efficace) sur 3 000 documents du
corpus final :

| Tokenizer | Vocab | Global | FR | EN | Code | Gain vs GPT-2 |
|---|---|---|---|---|---|---|
| bloom | 250 k | **1,524** | **1,442** | 1,308 | 5,097 | **31,5 %** |
| **thadeus/bpe32k** | **32 k** | **1,622** | **1,522** | 1,489 | 5,630 | **27,2 %** |
| qwen2.5 | 150 k | 1,854 | 1,812 | 1,324 | **4,627** | 16,8 % |
| mistral | 32 k | 2,089 | 2,025 | 1,475 | 5,914 | 6,2 % |
| gpt2 | 50 k | 2,227 | 2,189 | **1,316** | 5,751 | 0,0 % |
| smollm2 | 49 k | 2,241 | 2,197 | 1,347 | 5,942 | −0,6 % |

**−30,5 % de tokens sur le français** face à GPT-2, −27,2 % sur le mélange
complet. À taille de vocabulaire égale, on devance Mistral de **22,4 %** ; on
devance aussi Qwen2.5 de **12,5 %** alors que son vocabulaire est presque cinq
fois plus gros. Un vocabulaire dédié à une distribution bat un vocabulaire bien
plus gros partagé entre cent langues.

**Mais BLOOM nous devance de 6 %**, avec un vocabulaire de 250 k pensé pour le
multilingue. C'est le résultat honnête : à 32 k on ne bat pas tout le monde, on
bat les vocabulaires de taille comparable et les gros vocabulaires généralistes.

Le troc est assumé et mesuré. On paie ce gain **deux fois** : 13 % moins
efficace que GPT-2 sur l'anglais, et 22 % moins efficace que Qwen2.5 sur le
code. Le vocabulaire dépensé en français ne l'est pas ailleurs.

Deux gains distincts, à ne pas confondre : entraîner le vocabulaire sur notre
corpus fait l'essentiel, le motif de pré-tokenisation française apporte **2 %**
de plus.

*(Llama 3 manque au tableau : son dépôt est sous licence à acceptation manuelle.)*

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

## Résultats du run principal

**38 000 pas · 3,74 Md de tokens · 110 heures sur le Mac · MFU ~35 %.**

| Mesure | Valeur |
|---|---|
| Perplexité globale | **22,7** |
| Bits par caractère | **1,116** |
| Sondes grammaticales | **26/26** |
| Sondes difficiles | 13/16 |

Ventilé par source, ce qui est plus parlant qu'un chiffre unique :

| Source | Perplexité |
|---|---|
| code (Python) | **4,32 – 4,60** |
| Wikipédia FR | **10,97** |
| Gutenberg (littéraire) | 21,66 |
| FineWeb-Edu (EN) | 21,43 |
| FineWeb FR | 24,01 |
| Livres FR | 27,24 |

Le code est de loin le mieux prédit — il est répétitif et structuré. La prose
littéraire est la plus dure. L'écart entre Wikipédia (11,0) et FineWeb FR (24,0)
mesure la différence entre une écriture encyclopédique régulière et le web tout
venant.

Les 26 sondes de grammaire sont toutes passées, dont **l'élision à 4/4** — le
point que le motif de pré-tokenisation français visait spécifiquement, et qui
était à 50 % en début d'entraînement.

## Le réglage fin sur un corpus personnel

```bash
.venv/bin/python scripts/train.py --config train/vault_ft.toml
.venv/bin/python scripts/compare_models.py --base medium_mup --tuned vault_ft
```

Mesuré sur **102 notes jamais vues à l'entraînement** :

| | Base | Réglé | Écart |
|---|---|---|---|
| Notes personnelles inédites | 17,37 | **13,39** | **−22,9 %** |
| Corpus général | 23,78 | 21,74 | −8,6 % |
| Gutenberg | 17,86 | 18,25 | **+2,2 %** |
| Sondes difficiles | 13/16 | 12/16 | **−1** |

Le gain sur le domaine visé est net, et il ne se paie pas *seulement* sur ce
domaine : le corpus général s'améliore aussi. Mais le français littéraire régresse
et une sonde difficile se perd — **l'oubli est réel, il est simplement petit**.
C'est exactement ce qu'un mélange de rappel est censé contenir sans l'annuler.

## RAG : retrouver avant de répondre

```bash
.venv/bin/python scripts/rag_bench.py --mode corps --title-weight 0
```

Sur 8 485 passages issus de 595 notes, **protocole non circulaire** (requêtes
tirées du corps, jamais des titres, titres non pondérés dans l'index) :

| | Termes rares | Termes au hasard |
|---|---|---|
| Rappel@1 | 96,5 % | 98,5 % |
| Rappel@3 | **100 %** | **100 %** |
| MRR | 0,983 | 0,992 |

Le premier protocole écrit était **circulaire** : il fabriquait les requêtes
depuis les titres alors que l'index sur-pondère les titres. L'écart mesuré était
massif — 86 % à pondération 2 contre 16,5 % à pondération 0. Le tableau ci-dessus
est la version honnête, et le script conserve les deux protocoles pour que
l'écart reste visible.

**La récupération marche. La génération, beaucoup moins** : le modèle reprend
bien le vocabulaire des passages fournis, mais synthétise mal. À 188 M, la
limite est le générateur, pas le moteur de recherche.

## Le réglage par instructions

```bash
.venv/bin/python scripts/build_sft.py && .venv/bin/python scripts/train.py --config train/sft.toml
.venv/bin/python scripts/eval_instruct.py --models medium_mup sft
```

Dix tests vérifiables par programme, avant et après 650 pas de réglage :

| | Base | Après SFT |
|---|---|---|
| Contrainte de format respectée | 4/10 | 4/10 |
| **Jeton de fin produit spontanément** | **0/10** | **8/10** |
| Réponse en français | 7/10 | **3/10** |

Un gain franc, un statu quo, une régression — et c'est le résultat, pas un
échec de mesure. Le réglage apprend au modèle **à s'arrêter**, ce qu'il ne
savait pas faire du tout. Il n'améliore pas le respect des consignes. Et il
dégrade la qualité de langue, parce que 650 pas sur un jeu d'instructions étroit
érodent ce que 38 000 pas de français avaient installé.

Concrètement, sur « Réponds par un seul mot : quelle est la capitale de la
France ? », la base part en boucle incohérente ; le modèle réglé répond « La
capitale de la France est Paris. » — **juste, mais pas en un mot**. Le progrès
est réel et la métrique de format ne le voit pas.

## Ce que le modèle ne sait pas faire

À énoncer clairement, parce que la fluidité d'un texte donne une impression de
compétence que les chiffres ci-dessus ne soutiennent pas :

- **Il n'est pas une source de faits.** Il invente dates, noms et références avec
  le même aplomb qu'il conjugue juste. Rien de ce qu'il affirme ne doit être cru
  sans vérification.
- **Il ne raisonne pas.** Pas d'arithmétique, pas de déduction en plusieurs
  étapes, pas de logique fiable.
- **Il ne suit pas les consignes** : 4/10 au mieux sur des contraintes de format
  élémentaires.
- **Il n'a aucun garde-fou.** Aucun alignement, aucun filtrage de sortie. Il
  reproduira les biais de son corpus — web francophone, Wikipédia, code public.
- **Son contexte est de 1 024 tokens**, soit environ 700 mots, question et
  réponse comprises.

Ce sont les limites structurelles d'un modèle de 188 M de paramètres entraîné sur
3,74 Md de tokens. Elles ne se corrigent pas par le réglage : elles se corrigent
en changeant d'échelle.

## Licence

MIT — voir [LICENSE](LICENSE). Le corpus assemblé et les poids ne sont pas
distribués ici ; les sources de données sont déclarées dans `configs/data/` et
restent soumises à leurs licences respectives.
