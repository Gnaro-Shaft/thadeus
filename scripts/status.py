#!/usr/bin/env python
"""Le point régulier — où en est le dispositif, en une page.

    python scripts/status.py
    python scripts/status.py --depuis 7        # ce qui a bougé en 7 jours

Une automatisation qui tourne seule devient une boîte noire : on ne sait plus si
elle progresse, si elle piétine, ou si elle a cessé sans rien dire. Ce script
répond à trois questions, et rien d'autre :

    1. Le corpus grossit-il ? De combien, et les doublons montent-ils ?
    2. L'entraînement avance-t-il ? À quel rythme, et les nuits passent-elles ?
    3. Le modèle s'améliore-t-il ? Sur quoi, mesuré quand ?

Tout est lu depuis les artefacts. Ce script n'entraîne rien, ne télécharge rien
et ne modifie rien — on doit pouvoir le lancer sans y réfléchir.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thadeus.core.artifacts import ARTIFACT_ROOT  # noqa: E402
from thadeus.data.schema import format_tokens  # noqa: E402

PARAMETRES = 188_000_000  # modèle `medium` ; sert au ratio tokens/paramètre


def _charge(chemin: Path) -> dict | None:
    try:
        return json.loads(chemin.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _lignes_metriques(chemin: Path) -> list[dict]:
    if not chemin.is_file():
        return []
    out = []
    for ligne in chemin.read_text().splitlines():
        if ligne.strip():
            try:
                out.append(json.loads(ligne))
            except json.JSONDecodeError:
                continue
    return out


def _age(horodatage: str) -> str:
    """« il y a 3 j » — plus lisible qu'une date quand on suit un rythme."""
    try:
        quand = datetime.fromisoformat(horodatage)
    except ValueError:
        return "?"
    if quand.tzinfo is None:
        quand = quand.replace(tzinfo=UTC)
    delta = datetime.now(UTC) - quand
    if delta < timedelta(hours=1):
        return f"il y a {int(delta.total_seconds() // 60)} min"
    if delta < timedelta(days=1):
        return f"il y a {int(delta.total_seconds() // 3600)} h"
    return f"il y a {delta.days} j"


def titre(texte: str) -> None:
    print(f"\n\033[1m{texte}\033[0m")
    print("─" * len(texte))


def section_corpus(depuis_jours: int) -> None:
    titre("1. Le corpus")

    base = ARTIFACT_ROOT / "data"
    if not base.is_dir():
        print("  aucun artefact de données")
        return

    acheves, en_cours = [], []
    for repertoire in sorted(base.iterdir()):
        if not repertoire.is_dir():
            continue
        meta = _charge(repertoire / "meta.json")
        (acheves if meta else en_cours).append((repertoire, meta))

    if not acheves:
        print("  aucune collecte achevée")
        return

    total = 0
    recents = []
    limite = datetime.now(UTC) - timedelta(days=depuis_jours)
    for repertoire, meta in acheves:
        rapport = _charge(repertoire / "report.json") or {}
        tokens = (rapport.get("corpus") or {}).get("tokens_estimated", 0)
        total += tokens
        cree = meta.get("created_at", "")
        try:
            if datetime.fromisoformat(cree) >= limite:
                recents.append((repertoire.name, tokens, cree))
        except ValueError:
            pass

    # **Ne pas additionner les collectes.** Un corpus assemblé contient déjà les
    # collectes qui l'ont produit : les sommer compterait les mêmes tokens deux
    # fois et afficherait une croissance imaginaire. Le seul chiffre qui compte
    # est celui du corpus **tokenisé**, puisque c'est lui que l'entraînement lit.
    print(f"  collectes achevées : {len(acheves)}")
    print(f"  (somme brute {format_tokens(total)}, recouvrements compris — non significative)")
    _corpus_tokenise()
    if en_cours:
        print(f"  ⏳ en cours ou interrompues : {', '.join(p.name for p, _ in en_cours)}")

    if recents:
        print(f"\n  Nouveau depuis {depuis_jours} jours :")
        for nom, tokens, cree in sorted(recents, key=lambda r: r[2]):
            print(f"    {nom:32s} {format_tokens(tokens):>10s}   {_age(cree)}")
    else:
        print(f"\n  ⚠️  aucune collecte nouvelle depuis {depuis_jours} jours")

    # Le taux de doublons est le signal qui dit quand cesser de puiser aux mêmes
    # sources : il monte quand on retire des tranches de plus en plus proches.
    dernier = max(acheves, key=lambda r: (r[1] or {}).get("created_at", ""))
    rapport = _charge(dernier[0] / "report.json") or {}
    dedup = rapport.get("dedup") or {}
    if dedup:
        vus = dedup.get("seen") or dedup.get("total") or 0
        jetes = dedup.get("removed") or dedup.get("duplicates") or 0
        if vus:
            taux = 100 * jetes / vus
            print(f"\n  Doublons sur la dernière collecte : {taux:.1f} % ({dernier[0].name})")


def _corpus_tokenise() -> None:
    """Le corpus réellement lisible par l'entraînement, et sa couverture.

    La couverture n'est pas « tokens vus / taille du corpus » : le chargeur tire
    des fenêtres **au hasard avec remise**, il ne parcourt pas le corpus de bout
    en bout. La proportion réellement visitée suit ``1 - exp(-vus / taille)`` —
    à 3,74 Md de tokens traités sur 5,40 Md, la moitié du corpus n'a jamais été
    lue, et des répétitions ont déjà eu lieu.
    """
    import math

    base = ARTIFACT_ROOT / "tokens"
    if not base.is_dir():
        return
    corpus = [
        (p, _charge(p / "tokens.json"))
        for p in sorted(base.iterdir())
        if (p / "tokens.json").is_file()
    ]
    corpus = [(p, m) for p, m in corpus if m and m.get("n_tokens")]
    if not corpus:
        return

    chemin, meta = max(corpus, key=lambda c: c[1]["n_tokens"])
    taille = meta["n_tokens"]
    print(f"\n  Corpus tokenisé : {chemin.name}   {format_tokens(taille)} tokens")

    vus = _tokens_traites()
    if vus:
        couverture = 1 - math.exp(-vus / taille)
        print(
            f"    couverture ≈ {100 * couverture:.0f} %"
            f"   ·   jamais lu ≈ {format_tokens(taille * (1 - couverture))}"
        )


def _pas_des_checkpoints(run: Path) -> int:
    """Pas le plus avancé d'après les noms de fichiers (``step-00000007.pt``).

    Lire le nom plutôt que le contenu évite de charger torch pour un simple
    rapport — un point d'étape doit être instantané, sinon on cesse de le faire.
    """
    pas = [
        int(p.stem.split("-")[-1])
        for p in (run / "checkpoints").glob("step-*.pt")
        if p.stem.split("-")[-1].isdigit()
    ]
    return max(pas, default=0)


def _tokens_traites() -> int:
    """Total de tokens traités, tous runs d'entraînement confondus."""
    base = ARTIFACT_ROOT / "train"
    if not base.is_dir():
        return 0
    total = 0
    for run in base.iterdir():
        pertes = [m for m in _lignes_metriques(run / "metrics.jsonl") if "tokens_seen" in m]
        if pertes:
            total += pertes[-1]["tokens_seen"]
    return total


def section_entrainement(label: str) -> None:
    titre("2. L'entraînement")

    base = ARTIFACT_ROOT / "train"
    runs = sorted(base.glob(f"{label}-*")) if base.is_dir() else []
    if not runs:
        print(f"  aucun run nommé {label!r} — l'entraînement continu n'a pas encore démarré")
        return

    run = max(runs, key=lambda p: p.stat().st_mtime)
    metriques = _lignes_metriques(run / "metrics.jsonl")
    pertes = [m for m in metriques if "loss" in m]
    if not pertes:
        # Les métriques ne sont écrites que tous les `log_every` pas : une
        # session courte peut avoir réellement progressé sans avoir rien
        # journalisé. Annoncer « aucun pas » serait faux. Les checkpoints, eux,
        # portent leur pas dans leur nom — on le lit sans charger torch.
        pas = _pas_des_checkpoints(run)
        etat = f"pas {pas:,}" if pas else "aucun pas"
        print(f"  {run.name} : {etat}, rien encore journalisé (log_every non atteint)")
        return

    dernier = pertes[-1]
    pas = dernier["step"]
    tokens = dernier.get("tokens_seen", 0)
    acheve = (run / "meta.json").is_file()

    print(f"  run        : {run.name}   {'(achevé)' if acheve else '(reprenable)'}")
    print(f"  pas        : {pas:,}")
    # Le ratio tokens/paramètre porte sur ce que **le modèle** a vu, pas sur ce
    # que ce run-ci a traité. Un run lancé par `init_from` hérite des tokens de
    # son ancêtre : les ignorer afficherait 1,5 token par paramètre pour un
    # modèle qui en a vu vingt, et suggérerait un modèle très sous-entraîné
    # alors qu'il est à l'optimum de Chinchilla.
    cumul = _tokens_traites()
    print(f"  ce run     : {format_tokens(tokens)}")
    print(
        f"  cumul      : {format_tokens(cumul)}   ·   {cumul / PARAMETRES:.1f} tokens par paramètre"
    )
    print(f"  perte      : {dernier['loss']:.4f}   ·   dernier point {_age(dernier.get('t', ''))}")
    print(
        f"  débit      : {dernier.get('tokens_per_second', 0):,.0f} tok/s"
        f"   ·   MFU {100 * dernier.get('mfu', 0):.1f} %"
    )

    vals = [m for m in metriques if "val_loss" in m]
    if vals:
        print(f"  validation : {vals[-1]['val_loss']:.4f} au pas {vals[-1]['step']:,}")

    # Rythme : ce que rapporte réellement une nuit, mesuré et non estimé.
    _rythme(pertes)


def _rythme(pertes: list[dict]) -> None:
    """Progression par jour civil — une ligne par session effective."""
    par_jour: dict[str, list[dict]] = {}
    for m in pertes:
        jour = str(m.get("t", ""))[:10]
        if jour:
            par_jour.setdefault(jour, []).append(m)
    if len(par_jour) < 2:
        return

    print("\n  Sessions :")
    precedent = None
    for jour in sorted(par_jour)[-7:]:
        points = par_jour[jour]
        pas_faits = points[-1]["step"] - (precedent if precedent is not None else points[0]["step"])
        toks = points[-1].get("tokens_seen", 0) - points[0].get("tokens_seen", 0)
        print(
            f"    {jour}   {pas_faits:>7,} pas   {format_tokens(max(toks, 0)):>9s}"
            f"   perte {points[-1]['loss']:.4f}"
        )
        precedent = points[-1]["step"]


def section_evaluation() -> None:
    titre("3. Le modèle")

    base = ARTIFACT_ROOT / "eval"
    if not base.is_dir():
        print("  aucune évaluation")
        return

    rapports = []
    for repertoire in sorted(base.iterdir()):
        meta = _charge(repertoire / "meta.json")
        rapport = _charge(repertoire / "report.json")
        if meta and rapport and rapport.get("step"):
            rapports.append((meta.get("created_at", ""), repertoire.name, rapport))

    if not rapports:
        print("  aucune évaluation exploitable")
        return

    rapports.sort()
    print(f"  {'évaluation':22s} {'pas':>8s} {'ppl':>8s} {'bpc':>7s} {'sondes':>8s}")
    for cree, nom, rapport in rapports[-5:]:
        apercu = (rapport.get("corpus") or {}).get("overall") or {}
        sondes = rapport.get("probes") or {}
        ok = sum(v["correct"] for v in sondes.values() if isinstance(v, dict) and "correct" in v)
        tot = sum(v["total"] for v in sondes.values() if isinstance(v, dict) and "total" in v)
        print(
            f"  {nom[:22]:22s} {rapport['step']:>8,} {apercu.get('perplexity', 0):>8.2f}"
            f" {apercu.get('bits_per_char', 0):>7.4f} {f'{ok}/{tot}':>8s}   {_age(cree)}"
        )

    if len(rapports) >= 2:
        avant = (rapports[-2][2].get("corpus") or {}).get("overall", {}).get("perplexity")
        apres = (rapports[-1][2].get("corpus") or {}).get("overall", {}).get("perplexity")
        if avant and apres:
            ecart = 100 * (apres - avant) / avant
            sens = "mieux" if ecart < 0 else "moins bien"
            print(f"\n  Écart sur les deux dernières : {ecart:+.1f} %  ({sens})")


def section_sessions() -> None:
    """Les nuits ont-elles réellement tourné ? Un journal absent est un signal."""
    titre("4. Les sessions planifiées")

    journaux = sorted(Path("logs").glob("nightly-*.log")) if Path("logs").is_dir() else []
    if not journaux:
        print("  aucun journal de session — l'automatisation n'a pas encore tourné")
        return
    for journal in journaux[-7:]:
        taille = journal.stat().st_size
        quand = datetime.fromtimestamp(journal.stat().st_mtime, UTC).isoformat()
        # Un journal court n'est pas forcément un échec : une session
        # délibérément sautée (fenêtre trop courte, entraînement déjà en cours)
        # écrit une ligne et sort. Juger sur la taille seule criait à l'alerte
        # sur un comportement parfaitement normal — et une alerte qui se
        # déclenche à tort finit par ne plus être lue du tout.
        texte = journal.read_text(errors="replace")
        if "session annulée" in texte or "session sautée" in texte:
            etat = "sautée"
        elif "Session terminée" in texte:
            etat = f"{taille // 1024} Ko"
        elif taille < 200:
            etat = "vide ⚠️"
        else:
            etat = f"{taille // 1024} Ko ⚠️ inachevée"
        print(f"    {journal.name:28s} {etat:>18s}   {_age(quand)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depuis", type=int, default=7, help="fenêtre en jours (défaut : 7)")
    parser.add_argument("--run-label", default="continuous", help="run à suivre")
    args = parser.parse_args()

    print(f"\n\033[1mThadeus — point du {datetime.now().strftime('%Y-%m-%d %H:%M')}\033[0m")

    section_corpus(args.depuis)
    section_entrainement(args.run_label)
    section_evaluation()
    section_sessions()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
