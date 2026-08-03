"""Socle du projet : config, registre, device, graines, artefacts, logs.

Aucun module de `core` ne dépend d'un étage métier. C'est la contrainte qui
garantit qu'on peut réutiliser le socle depuis n'importe quel étage sans
créer de dépendance circulaire.
"""
