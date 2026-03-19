"""
bim_summary.py
──────────────
Génère un résumé intelligent en français à partir du JSON BIM.
100% local — aucune API, aucun LLM externe requis.

Produit des phrases variées et contextuelles selon les données.
"""

import random


def generate_summary(bim_json: dict, ai_result: dict = None) -> str:
    """
    Génère un résumé textuel intelligent du modèle BIM analysé.

    Paramètres :
    - bim_json   : dict retourné par bim_json_builder.build_bim_json()
    - ai_result  : dict retourné par bim_ai_v1.run_anomaly_detection() (optionnel)

    Retourne une string en français.
    """

    discipline  = bim_json.get("discipline", {}).get("primary", "Inconnu")
    scores      = bim_json.get("scores", {})
    errors      = bim_json.get("errors", {})
    objects     = bim_json.get("objects", {})
    top_issues  = bim_json.get("top_issues", [])
    benchmark   = bim_json.get("benchmark")
    ifc_name    = bim_json.get("meta", {}).get("file", "this model")

    score_global = scores.get("global", 0)
    score_metier = scores.get("metier", 0)
    score_data   = scores.get("data_bim", 0)
    interp       = scores.get("interpretation", "")

    critical = errors.get("critical", 0)
    major    = errors.get("major", 0)
    minor    = errors.get("minor", 0)
    total_err = errors.get("total", 0)
    ratio    = errors.get("ratio_per_100_objects", 0)

    lines = []

    # ── Phrase d'intro ────────────────────────────────────
    intro_options = [
        f"Analysis of model **{ifc_name}** ({discipline}) completed.",
        f"Model **{ifc_name}** analyzed as {discipline} model.",
        f"Here are the QA analysis results for **{ifc_name}** ({discipline}).",
    ]
    lines.append(random.choice(intro_options))

    # ── Score global ──────────────────────────────────────
    score_phrase = _score_phrase(score_global, score_metier, score_data, interp)
    lines.append(score_phrase)

    # ── Erreurs ───────────────────────────────────────────
    if total_err == 0:
        lines.append("No anomaly detected — the model complies with all configured rules.")
    else:
        err_parts = []
        if critical > 0:
            err_parts.append(f"**{critical} erreur(s) critique(s)**")
        if major > 0:
            err_parts.append(f"**{major} majeure(s)**")
        if minor > 0:
            err_parts.append(f"{minor} mineure(s)")

        err_str = ", ".join(err_parts)
        lines.append(f"Detected {err_str} — {total_err} anomaly(-ies) total "
                     f"(ratio: {ratio} per 100 objects).")

    # ── Top problèmes ─────────────────────────────────────
    if top_issues:
        critical_issues = [i for i in top_issues if i.get("priority") == "Critical"]
        major_issues    = [i for i in top_issues if i.get("priority") == "Major"]

        if critical_issues:
            msgs = list({i["message"] for i in critical_issues[:3]})
            lines.append(
                "Critical issues: " +
                " ; ".join(f"*{m}*" for m in msgs) + "."
            )
        if major_issues:
            msgs = list({i["message"] for i in major_issues[:3]})
            lines.append(
                "Major issues include: " +
                " ; ".join(f"*{m}*" for m in msgs) + "."
            )

    # ── Benchmark ─────────────────────────────────────────
    if benchmark:
        delta    = benchmark.get("delta_vs_mean", 0)
        position = benchmark.get("position", "")
        moyenne  = benchmark.get("score_moyen", 0)
        n        = benchmark.get("nb_models", 0)

        if position == "above_average":
            bench_options = [
                f"Compared to {n} models in the park, this project is **above average** (mean: {moyenne}).",
                f"This model scores {abs(delta):.1f} points above the park average ({moyenne}).",
            ]
        else:
            bench_options = [
                f"Compared to {n} models in the park, this project is **below average** (mean: {moyenne}).",
                f"This model is {abs(delta):.1f} points below the park average ({moyenne}). Corrections recommended.",
            ]
        lines.append(random.choice(bench_options))

    # ── IA V1 ─────────────────────────────────────────────
    if ai_result and ai_result.get("available"):
        label      = ai_result.get("label", "")
        confidence = ai_result.get("confidence", "")
        ai_score   = ai_result.get("anomaly_score", 0)

        if label == "Normal":
            ai_phrase = (
                f"L'analyse IA indique que ce modèle est **dans la norme** du parc "
                f"(indice d'atypie : {ai_score:.2f}/1.00, confiance {confidence})."
            )
        elif label == "Atypique":
            ai_phrase = (
                f"L'analyse IA détecte un profil **atypique** pour ce modèle "
                f"(indice : {ai_score:.2f}/1.00, confiance {confidence}). "
                "A detailed review is advised."
            )
        else:
            ai_phrase = (
                f"⚠️ L'analyse IA signale un profil **très atypique** "
                f"(indice : {ai_score:.2f}/1.00, confiance {confidence}). "
                "This model stands out significantly from the park."
            )
        lines.append(ai_phrase)
    elif ai_result and not ai_result.get("available"):
        lines.append(f"*AI not available: {ai_result.get('reason', '')}*")

    # ── Recommandation finale ─────────────────────────────
    if critical > 0:
        lines.append(
            "**Priorité recommandée :** corriger les anomalies critiques en premier "
            "— elles pénalisent fortement le score métier."
        )
    elif major > 0:
        lines.append(
            "**Recommandation :** traiter les anomalies majeures pour améliorer "
            "significativement la qualité du modèle."
        )
    elif total_err > 0:
        lines.append(
            "Les anomalies détectées sont mineures. Une revue rapide suffit "
            "pour atteindre un niveau de qualité excellent."
        )

    return "\n\n".join(lines)


def _score_phrase(score_global, score_metier, score_data, interp):
    """Génère une phrase contextuelle sur le score."""

    if score_metier < score_data - 15:
        domination = "The conformity score is significantly lower than the Data BIM score"
        detail = (
            f"({score_metier:.1f} vs {score_data:.1f}) — les problèmes concernent "
            "principalement la conformité des éléments et non la structuration des données."
        )
    elif score_data < score_metier - 15:
        domination = "The Data BIM score is significantly lower than the conformity score"
        detail = (
            f"({score_data:.1f} vs {score_metier:.1f}) — les problèmes sont surtout "
            "liés à la classification, aux types et à la complétude des données BIM."
        )
    else:
        domination = "Conformity and Data BIM scores are relatively balanced"
        detail = f"({score_metier:.1f} and {score_data:.1f} respectively)."

    options = [
        f"The global score is **{score_global:.1f}/100** (level: {interp}). {domination} {detail}",
        f"With a global score of **{score_global:.1f}/100**, this model is rated **{interp}**. {domination} {detail}",
    ]
    return random.choice(options)


def _disc_fr(discipline: str) -> str:
    labels = {
        "Architecture": "architecture",
        "MEP":          "MEP / CVC",
        "Structure":    "structure",
        "Interior":     "interior",
        "Unknown":      "unknown discipline",
    }
    return labels.get(discipline, discipline.lower())
