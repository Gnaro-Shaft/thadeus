"""Arrêt propre d'un entraînement sur signal.

**Pourquoi ce module existe.** Un entraînement nocturne planifié est tué à heure
fixe par l'ordonnanceur. Sans interception, `SIGTERM` termine le processus au
milieu d'un pas : tout ce qui a été calculé depuis le dernier checkpoint est
perdu, et sur un intervalle de sauvegarde de 500 pas cela représente plusieurs
minutes de GPU par nuit, toutes les nuits.

**Ce que « propre » veut dire ici.** On ne s'arrête pas *pendant* un pas — un pas
interrompu laisserait des gradients accumulés à moitié et un optimiseur dans un
état incohérent. Le signal ne fait que lever un drapeau ; la boucle le consulte
entre deux pas et sort d'elle-même, ce qui garantit que le checkpoint écrit
correspond à un pas réellement terminé.

**Le second signal ne doit jamais être avalé.** Si le premier `Ctrl-C` demande
l'arrêt et que la boucle est bloquée, un utilisateur qui insiste doit obtenir un
arrêt immédiat. Le second signal rétablit donc le comportement par défaut, sans
quoi on aurait fabriqué un processus impossible à tuer autrement qu'au `KILL`.
"""

from __future__ import annotations

import signal
import threading
from types import FrameType

from thadeus.core.logs import get_logger

__all__ = ["GracefulStop"]

log = get_logger(__name__)

# SIGTERM : ce qu'envoie un ordonnanceur (launchd, systemd, `kill`).
# SIGINT  : Ctrl-C au clavier.
_SIGNAUX = (signal.SIGTERM, signal.SIGINT)


class GracefulStop:
    """Demande l'arrêt à la fin du pas courant plutôt qu'immédiatement.

    Utilisation ::

        with GracefulStop() as stop:
            for step in ...:
                entrainer_un_pas()
                if stop.requested:
                    break

    En dehors du thread principal — dans un test, par exemple — l'installation
    de gestionnaires de signaux est impossible. Le contexte reste alors
    utilisable et `requested` peut être piloté par `request()` : le code appelant
    n'a pas à savoir où il tourne.
    """

    def __init__(self) -> None:
        self._requested = False
        self._signal: int | None = None
        self._precedents: dict[int, object] = {}
        self._installe = False

    @property
    def requested(self) -> bool:
        """Vrai dès qu'un arrêt a été demandé."""
        return self._requested

    @property
    def signal_name(self) -> str | None:
        """Nom du signal reçu, pour le journal. ``None`` si arrêt programmatique."""
        return signal.Signals(self._signal).name if self._signal is not None else None

    def request(self) -> None:
        """Demande l'arrêt sans passer par un signal (tests, arrêt piloté)."""
        self._requested = True

    def _handler(self, signum: int, frame: FrameType | None) -> None:
        if self._requested:
            # Deuxième signal : on rend la main au comportement par défaut et on
            # se le renvoie, plutôt que de laisser l'utilisateur sans recours.
            log.warning("Second signal reçu — arrêt immédiat.")
            signal.signal(signum, signal.SIG_DFL)
            signal.raise_signal(signum)
            return
        self._requested = True
        self._signal = signum
        log.warning(
            "%s reçu — arrêt à la fin du pas en cours, checkpoint puis sortie.",
            signal.Signals(signum).name,
        )

    def __enter__(self) -> GracefulStop:
        if threading.current_thread() is not threading.main_thread():
            # `signal.signal` lève ValueError hors du thread principal. Ce n'est
            # pas une erreur pour autant : le contexte reste fonctionnel.
            log.debug("Hors du thread principal — pas d'interception de signal.")
            return self
        for sig in _SIGNAUX:
            self._precedents[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handler)
        self._installe = True
        return self

    def __exit__(self, *_exc: object) -> None:
        if not self._installe:
            return
        for sig, precedent in self._precedents.items():
            signal.signal(sig, precedent)  # type: ignore[arg-type]
        self._installe = False
