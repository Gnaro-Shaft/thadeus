#!/usr/bin/env bash
# Enchaîne tout ce qui sépare les collectes du premier entraînement.
#
#     bash scripts/prepare.sh
#
# Quatre étapes, chacune reprenant là où la précédente s'arrête :
#
#   1. assemblage    mélange les quatre collectes (aucun téléchargement)
#   2. tokenizer     BPE 32 k entraîné sur le corpus final
#   3. tokenisation  encodage en shards binaires memmap
#   4. ligne de base évaluation d'un modèle NON entraîné
#
# L'étape 4 n'est pas décorative : elle donne les chiffres auxquels tout run
# ultérieur se comparera (perte ≈ ln(32000) = 10,4 · sondes ≈ 50 %). Sans elle,
# on ne saurait pas distinguer « le modèle a appris » de « la mesure est fausse ».
#
# L'ENTRAÎNEMENT N'EST PAS ENCHAÎNÉ ICI. C'est un engagement de plusieurs heures
# ou plusieurs nuits, et il se lance en connaissance de cause :
#     .venv/bin/python scripts/train.py --config train/medium_mup.toml

set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG="prepare.log"

etape() { printf '\n\033[1m=== %s ===\033[0m\n' "$1" | tee -a "$LOG"; }

: > "$LOG"

etape "1/4 · assemblage du corpus"
$PY scripts/build_corpus.py --config data/assemble.toml --peek 2 2>&1 | tee -a "$LOG"

etape "2/4 · tokenizer BPE 32 k"
$PY scripts/train_tokenizer.py --config tokenizer/bpe32k.toml \
    --set corpus_label=\"thadeus_v1\" 2>&1 | tee -a "$LOG"

etape "3/4 · tokenisation du corpus"
$PY scripts/tokenize_corpus.py --corpus-label thadeus_v1 --tokenizer bpe32k 2>&1 | tee -a "$LOG"

etape "4/4 · ligne de base (modèle non entraîné)"
$PY scripts/evaluate.py --config eval/default.toml \
    --set label=\"baseline\" --set corpus_label=\"thadeus_v1\" 2>&1 | tee -a "$LOG"

etape "terminé"
cat <<'FIN'
Le corpus est prêt et la ligne de base est mesurée.

Comparer le tokenizer aux références publiques :
    .venv/bin/python scripts/compare_tokenizers.py --corpus-label thadeus_v1 --ours bpe32k

Lancer le premier entraînement (plusieurs heures — reprend automatiquement
après une interruption, mêmes lots compris) :
    .venv/bin/python scripts/train.py --config train/medium_mup.toml
FIN
