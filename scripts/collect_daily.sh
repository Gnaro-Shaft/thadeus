#!/usr/bin/env bash
# Collecte quotidienne — élargit le corpus pendant la journée, sans GPU.
#
#     bash scripts/collect_daily.sh
#
# Réseau et disque uniquement : le Mac reste utilisable, et cette tâche ne
# concurrence jamais l'entraînement nocturne. C'est l'intérêt du découpage —
# les deux ressources rares du projet ne sont pas les mêmes.
#
# **Comment on obtient des documents NOUVEAUX.** La source Hugging Face lit en
# streaming et n'a pas de curseur : relancer la même config re-lirait le même
# début de flux. On fait donc varier la **graine de mélange** avec le jour, ce
# qui change l'ordre des shards parcourus et donc la tranche obtenue.
#
# Le recouvrement n'est pas nul pour autant — deux tranches tirées au hasard
# d'un même corpus se croisent. C'est assumé : la déduplication de l'assemblage
# (MinHash + LSH) élimine les doublons, et `report.json` chiffre exactement
# combien ont été jetés. Si ce taux monte, c'est le signal qu'il faut élargir
# les sources plutôt que retirer davantage des mêmes.
#
# Chaque exécution écrit son propre artefact daté. L'assemblage hebdomadaire les
# recolle via la source `shards`, exactement comme `fr_first_resume.toml` a
# récupéré 1,24 Md de tokens d'une collecte interrompue.

set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
CONFIG="${THADEUS_COLLECT_CONFIG:-data/extend_fr.toml}"
JOUR=$(date +%Y%m%d)
LABEL="daily_${JOUR}"
LOG="logs/collect-${JOUR}.log"

mkdir -p logs

# Ne jamais empiéter sur la fenêtre d'entraînement : la collecte écrit beaucoup
# sur le disque, et l'entraînement y lit ses tokens.
if pgrep -f "scripts/train.py" > /dev/null; then
    echo "Un entraînement tourne — collecte reportée." | tee -a "$LOG"
    exit 0
fi

if [ -d "artifacts/data/${LABEL}"-* ] 2>/dev/null; then
    echo "Collecte du jour déjà faite (${LABEL})." | tee -a "$LOG"
    exit 0
fi

# La graine dérive du jour : même jour = même tranche (donc relancer après une
# coupure ne redemande pas un autre échantillon), jour suivant = autre tranche.
GRAINE=$((10#$JOUR % 100000))

{
    echo "════════════════════════════════════════════════════════════"
    echo "Collecte du $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  config : $CONFIG"
    echo "  label  : $LABEL   graine : $GRAINE"
    echo "════════════════════════════════════════════════════════════"
} | tee -a "$LOG"

# `nice` : la collecte ne doit jamais rendre la machine désagréable à utiliser.
nice -n 10 $PY scripts/build_corpus.py \
    --config "$CONFIG" \
    --set "label=\"$LABEL\"" \
    --set "seed=$GRAINE" 2>&1 | tee -a "$LOG"

echo "Collecte terminée à $(date '+%H:%M:%S')." | tee -a "$LOG"
