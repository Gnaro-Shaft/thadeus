"""Mesure de la fertilité — la métrique qui décide du tokenizer.

**La fertilité est un nombre de tokens par mot.** Plus elle est basse, moins il
faut de tokens pour dire la même chose, donc moins il faut de calcul pour lire
le même corpus. C'est une mesure d'efficacité, pas de qualité linguistique.

L'enjeu chiffré : si notre tokenizer atteint 1,45 token/mot là où un tokenizer
anglo-générique en demande 1,85, on lit **21 % de texte en plus** pour le même
budget de FLOPs. À l'échelle du projet, cela équivaut à trois jours de H100
offerts.

Deux précautions de méthode, sans lesquelles la comparaison ne veut rien dire :

- **Mesurer par langue et par source.** Un tokenizer peut gagner 25 % sur le
  français et perdre 10 % sur le code ; la moyenne globale masquerait les deux.
- **Comparer sur *notre* corpus.** Les fertilités publiées le sont sur des
  corpus généralistes anglais. Le seul chiffre qui nous concerne est celui
  mesuré sur les textes que Thadeus verra réellement.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from thadeus.core.logs import get_logger

__all__ = [
    "REFERENCE_TOKENIZERS",
    "Fertility",
    "compare",
    "load_reference",
    "measure",
    "measure_by",
]

log = get_logger(__name__)

# Tokenizers de référence. Choisis pour couvrir trois régimes :
#   - gpt2      : l'anglo-centrique historique, la borne haute qu'on veut battre
#   - qwen2.5   : multilingue moderne, vocabulaire 150 k — la borne basse
#   - smollm2   : petit modèle récent, vocabulaire 49 k — le comparable direct
#   - bloom     : explicitement multilingue français inclus
# Les modèles restreints (Llama, Mistral, Gemma) demandent un HF_TOKEN et une
# acceptation de licence ; ils sont ajoutés ici mais échouent proprement sans.
REFERENCE_TOKENIZERS: dict[str, str] = {
    "gpt2": "gpt2",
    "qwen2.5": "Qwen/Qwen2.5-0.5B",
    "smollm2": "HuggingFaceTB/SmolLM2-135M",
    "bloom": "bigscience/bloom-560m",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "llama3": "meta-llama/Meta-Llama-3-8B",
}


@dataclass(frozen=True)
class Fertility:
    """Comptages bruts et ratios dérivés.

    On conserve les comptages plutôt que les seuls ratios : deux mesures ne
    s'additionnent que sur leurs comptages, jamais sur leurs moyennes.
    """

    tokens: int
    words: int
    chars: int
    documents: int

    @property
    def tokens_per_word(self) -> float:
        """La fertilité. Plus bas = plus efficace."""
        return self.tokens / self.words if self.words else 0.0

    @property
    def chars_per_token(self) -> float:
        """Densité d'information par token — la même chose vue à l'envers."""
        return self.chars / self.tokens if self.tokens else 0.0

    def __add__(self, other: Fertility) -> Fertility:
        return Fertility(
            tokens=self.tokens + other.tokens,
            words=self.words + other.words,
            chars=self.chars + other.chars,
            documents=self.documents + other.documents,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "tokens": self.tokens,
            "words": self.words,
            "chars": self.chars,
            "tokens_per_word": round(self.tokens_per_word, 4),
            "chars_per_token": round(self.chars_per_token, 3),
        }


def measure(count_tokens: Callable[[str], int], texts: Iterable[str]) -> Fertility:
    """Mesure la fertilité d'un tokenizer sur un ensemble de textes."""
    tokens = words = chars = documents = 0
    for text in texts:
        tokens += count_tokens(text)
        words += len(text.split())
        chars += len(text)
        documents += 1
    return Fertility(tokens=tokens, words=words, chars=chars, documents=documents)


def measure_by[T](
    count_tokens: Callable[[str], int],
    items: Sequence[T],
    *,
    key: Callable[[T], str],
    text: Callable[[T], str] = lambda item: item.text,  # type: ignore[attr-defined]
) -> dict[str, Fertility]:
    """Mesure ventilée par groupe (langue, source…).

    C'est la ventilation qui informe : un gain global de 15 % peut cacher
    +25 % sur le français et −10 % sur le code.
    """
    groups: dict[str, Fertility] = {}
    for item in items:
        group = key(item)
        one = measure(count_tokens, [text(item)])
        groups[group] = groups[group] + one if group in groups else one
    return groups


def load_reference(name: str) -> Callable[[str], int] | None:
    """Charge un tokenizer de référence, ou ``None`` s'il est indisponible.

    Retourner ``None`` plutôt que lever : un modèle restreint sans jeton
    d'accès, ou une absence de réseau, ne doit pas faire échouer toute une
    campagne de comparaison. On mesure ce qu'on peut, et on dit ce qui manque.
    """
    from tokenizers import Tokenizer

    repo = REFERENCE_TOKENIZERS.get(name, name)
    try:
        tokenizer = Tokenizer.from_pretrained(repo)
    except Exception as exc:  # noqa: BLE001 — indisponibilité réseau, licence, 404…
        log.warning("Tokenizer de référence %s indisponible (%s)", name, type(exc).__name__)
        return None

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    return count


def compare[T](
    counters: dict[str, Callable[[str], int]],
    items: Sequence[T],
    *,
    key: Callable[[T], str],
    baseline: str | None = None,
) -> dict[str, Any]:
    """Compare plusieurs tokenizers sur le même échantillon.

    Args:
        counters: nom -> fonction de comptage.
        baseline: référence pour le calcul des gains relatifs. Le gain est
            exprimé en **tokens économisés**, c'est-à-dire directement en
            budget de calcul économisé.
    """
    results: dict[str, Any] = {}
    for name, counter in counters.items():
        by_group = measure_by(counter, items, key=key)
        overall = sum(by_group.values(), Fertility(0, 0, 0, 0))
        results[name] = {
            "overall": overall.to_dict(),
            "by_group": {group: f.to_dict() for group, f in by_group.items()},
        }

    if baseline and baseline in results:
        reference = results[baseline]["overall"]["tokens"]
        for entry in results.values():
            tokens = entry["overall"]["tokens"]
            entry["tokens_saved_vs_baseline"] = (
                round(1 - tokens / reference, 4) if reference else 0.0
            )

    return {"baseline": baseline, "tokenizers": results}


def format_comparison(comparison: dict[str, Any]) -> str:
    """Rend la comparaison lisible dans un terminal."""
    tokenizers = comparison["tokenizers"]
    groups = sorted({g for entry in tokenizers.values() for g in entry["by_group"]})
    header = f"{'tokenizer':<14}{'global':>9}" + "".join(f"{g:>9}" for g in groups) + f"{'gain':>9}"
    lines = [header, "-" * len(header)]

    for name, entry in sorted(
        tokenizers.items(), key=lambda kv: kv[1]["overall"]["tokens_per_word"]
    ):
        row = f"{name:<14}{entry['overall']['tokens_per_word']:>9.3f}"
        for group in groups:
            value = entry["by_group"].get(group)
            row += f"{value['tokens_per_word']:>9.3f}" if value else f"{'—':>9}"
        saved = entry.get("tokens_saved_vs_baseline")
        row += f"{100 * saved:>8.1f}%" if saved is not None else f"{'—':>9}"
        lines.append(row)

    lines.append("")
    lines.append("fertilité = tokens/mot (plus bas = plus efficace)")
    if comparison.get("baseline"):
        lines.append(f"gain = tokens économisés vs {comparison['baseline']}")
    return "\n".join(lines)
