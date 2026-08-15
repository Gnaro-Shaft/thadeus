#!/usr/bin/env bash
# Enchaîne l'évaluation dès que la décroissance est terminée. Jetable.
cd "/Users/dgnaro/projects/10-en-cours/projet_Thadeus"
while pgrep -f "scripts/train.py" > /dev/null; do sleep 60; done
sleep 30   # laisse le checkpoint final se refermer
.venv/bin/python scripts/evaluate.py --config eval/default.toml \
  --set 'label="anneal_13252"' \
  --set 'checkpoint="/Users/dgnaro/projects/10-en-cours/projet_Thadeus/artifacts/train/continuous-9e0687da/checkpoints/latest.pt"' --force
