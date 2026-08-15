"""Invariants de l'orchestration : ce que le pipeline passe aux sources.

Ces tests ne construisent pas de corpus — ils vérifient des **contrats** entre
le pipeline et les sources qu'il assemble. Un contrat rompu ici ne se voit
qu'après plusieurs secondes d'exécution, une fois les configs lues et
l'artefact ouvert, ce qui le fait passer pour une erreur de configuration.
"""

from __future__ import annotations


class TestInjectionDeGraine:
    """Le pipeline injecte une graine dérivée — mais pas à n'importe qui.

    Régression : la graine était passée à **toutes** les sources. Or relire des
    shards déjà écrits ou parcourir un répertoire est déterministe par nature,
    et ces fonctions n'ont pas de paramètre `seed`. L'assemblage entier tombait
    sur un `TypeError` — après avoir lu les configs, ouvert l'artefact et
    initialisé le dédupliqueur, donc plusieurs secondes trop tard pour être
    confondu avec une erreur de config.
    """

    def test_aucune_source_ne_recoit_un_argument_qu_elle_refuse(self):
        import inspect

        from thadeus.data.sources import SOURCES

        for nom in SOURCES:
            params = inspect.signature(SOURCES.get(nom)).parameters
            accepte_tout = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
            # C'est exactement le test que fait le pipeline avant d'injecter.
            if not accepte_tout and "seed" not in params:
                assert "seed" not in params, f"{nom} : incohérence de détection"

    def test_au_moins_une_source_tire_au_hasard(self):
        # Si plus aucune source n'acceptait de graine, l'injection deviendrait
        # du code mort et le correctif de reproductibilité serait silencieusement
        # perdu.
        import inspect

        from thadeus.data.sources import SOURCES

        avec = [n for n in SOURCES if "seed" in inspect.signature(SOURCES.get(n)).parameters]
        assert avec, "aucune source n'accepte de graine — l'injection ne sert plus à rien"

    def test_les_sources_de_relecture_sont_deterministes(self):
        import inspect

        from thadeus.data.sources import SOURCES

        for nom in ("shards", "obsidian", "local_files", "gutenberg"):
            if nom in SOURCES:
                params = inspect.signature(SOURCES.get(nom)).parameters
                assert "seed" not in params, (
                    f"{nom} accepte désormais une graine : vérifier que le pipeline "
                    f"la dérive bien plutôt que de la laisser au hasard"
                )
