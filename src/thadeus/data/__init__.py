"""Étage 1 — le corpus.

    sources -> nettoyage -> déduplication -> mélange -> shards

Le pari « données » du projet vit ici : à budget de calcul minuscule, chaque
document médiocre conservé est du budget d'entraînement dépensé à apprendre du
bruit. Le nettoyage n'est pas du ménage, c'est un levier de performance.
"""
