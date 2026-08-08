#!/usr/bin/env bash
# Session d'entraînement nocturne — reprend, s'arrête à l'heure, rend la machine.
#
#     bash scripts/nightly.sh              # jusqu'à 08:00
#     bash scripts/nightly.sh 06:30        # jusqu'à une autre heure
#
# Conçu pour être lancé par launchd à 00:00 (voir `deploy/launchd/`), mais
# parfaitement utilisable à la main.
#
# **Le budget se calcule à partir de l'heure réelle de démarrage**, pas d'une
# durée fixe. Si la machine était en veille et que le run démarre à 02:15, il
# s'arrête quand même à 08:00 — une durée fixe de 8 h le ferait déborder sur la
# journée et rendrait le Mac inutilisable au réveil.
#
# **`caffeinate -i`** empêche la mise en veille pendant le run, sans forcer
# l'écran allumé. Sans lui, le Mac s'endort et le GPU s'arrête : on croirait
# avoir entraîné huit heures pour en obtenir vingt minutes.
#
# L'arrêt est propre dans tous les cas : à l'échéance le run s'arrête de
# lui-même à la fin d'un pas, et un SIGTERM extérieur produit le même effet.
# Un checkpoint est toujours écrit ; `meta.json` ne l'est pas, ce qui garde le
# run reprenable — voir `thadeus.train.interrupt`.

set -euo pipefail
cd "$(dirname "$0")/.."

FIN="${1:-08:00}"
CONFIG="${THADEUS_NIGHTLY_CONFIG:-train/continuous.toml}"
PY=.venv/bin/python
LOG="logs/nightly-$(date +%Y%m%d).log"

mkdir -p logs

# --- Un seul entraînement à la fois ------------------------------------------
# Deux runs concurrents sur le même artefact s'écraseraient mutuellement leurs
# checkpoints, et le second lirait un fichier à moitié écrit.
if pgrep -f "scripts/train.py" > /dev/null; then
    echo "Un entraînement tourne déjà — session annulée." | tee -a "$LOG"
    exit 0
fi

# --- Budget restant jusqu'à l'heure de fin -----------------------------------
# `date -v` est la forme BSD (macOS). Si l'heure de fin est déjà passée, elle
# désigne demain.
maintenant=$(date +%s)
fin=$(date -v"${FIN%%:*}"H -v"${FIN##*:}"M -v0S +%s)
[ "$fin" -le "$maintenant" ] && fin=$((fin + 86400))
heures=$(echo "scale=3; ($fin - $maintenant) / 3600" | bc)

# Sous ce seuil, démarrer coûte plus (chargement, compilation ~2 min) que ça ne
# rapporte.
if (( $(echo "$heures < 0.5" | bc -l) )); then
    echo "Moins de 30 min avant $FIN — session sautée." | tee -a "$LOG"
    exit 0
fi

{
    echo "════════════════════════════════════════════════════════════"
    echo "Session du $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  config  : $CONFIG"
    echo "  fin     : $FIN  (budget ${heures} h)"
    echo "════════════════════════════════════════════════════════════"
} | tee -a "$LOG"

# `caffeinate -i` : pas de veille tant que la commande tourne.
caffeinate -i $PY scripts/train.py \
    --config "$CONFIG" \
    --set "max_hours=$heures" 2>&1 | tee -a "$LOG"

echo "Session terminée à $(date '+%H:%M:%S')." | tee -a "$LOG"
