#!/usr/bin/env bash
# Installe (ou retire) les deux agents launchd du dispositif automatique.
#
#     bash deploy/launchd/install.sh            # installe et active
#     bash deploy/launchd/install.sh --retirer  # désactive et supprime
#     bash deploy/launchd/install.sh --etat     # dit ce qui est chargé
#
# Ce que ça met en place :
#
#     00:00   entraînement, jusqu'à 08:00, arrêt propre garanti
#     10:00   collecte, en tâche de fond et `nice`, sans GPU
#
# **À lancer soi-même, en connaissance de cause.** Ces agents créent une
# configuration persistante : la machine se mettra à travailler toutes les nuits
# jusqu'à ce qu'on les retire. Le GPU tournera à pleine charge pendant huit
# heures — chaleur, ventilation, et Mac inutilisable pendant ce temps.
#
# Les gabarits ne contiennent aucun chemin absolu ; ce script les substitue au
# moment de l'installation, dans une copie placée sous ~/Library/LaunchAgents.

set -euo pipefail

RACINE="$(cd "$(dirname "$0")/../.." && pwd)"
CIBLE="$HOME/Library/LaunchAgents"
AGENTS=(com.thadeus.nightly com.thadeus.collect)

etat() {
    # `launchctl print` interroge **un** agent et fait autorité.
    #
    # La forme évidente — `launchctl list | grep -q "$nom"` — est piégeuse ici :
    # `grep -q` sort dès la première correspondance et ferme le tube, `launchctl`
    # reçoit un SIGPIPE, et `set -o pipefail` transforme ce succès en échec. Le
    # résultat dépend alors de la position de l'agent dans la sortie : le premier
    # listé était rapporté « absent » alors qu'il était parfaitement chargé.
    # Un rapport faux, sans erreur visible.
    for nom in "${AGENTS[@]}"; do
        if launchctl print "gui/$(id -u)/$nom" > /dev/null 2>&1; then
            printf "  %-24s chargé\n" "$nom"
        else
            printf "  %-24s absent\n" "$nom"
        fi
    done
}

retirer() {
    for nom in "${AGENTS[@]}"; do
        launchctl unload "$CIBLE/$nom.plist" 2>/dev/null || true
        rm -f "$CIBLE/$nom.plist"
        echo "  $nom retiré"
    done
}

case "${1:-}" in
    --etat)    echo "État des agents :"; etat; exit 0 ;;
    --retirer) echo "Retrait :"; retirer; exit 0 ;;
esac

mkdir -p "$CIBLE" "$RACINE/logs"

echo "Installation depuis $RACINE"
for nom in "${AGENTS[@]}"; do
    source_plist="$RACINE/deploy/launchd/$nom.plist"
    [ -f "$source_plist" ] || { echo "  gabarit manquant : $source_plist" >&2; exit 1; }

    # Recharger : `unload` avant `load`, sinon une ancienne définition persiste
    # et l'agent continue de pointer vers l'ancien chemin.
    launchctl unload "$CIBLE/$nom.plist" 2>/dev/null || true
    sed "s|__THADEUS_ROOT__|$RACINE|g" "$source_plist" > "$CIBLE/$nom.plist"
    launchctl load "$CIBLE/$nom.plist"
    echo "  $nom installé"
done

echo
echo "État :"
etat
echo
echo "Vérifier l'avancement à tout moment :  .venv/bin/python scripts/status.py"
echo "Tout arrêter :                         bash deploy/launchd/install.sh --retirer"
