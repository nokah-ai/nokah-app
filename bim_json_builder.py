"""
bim_json_builder.py
───────────────────
Construit le JSON structuré officiel du moteur BIM QA.
C'est la "vérité" du moteur — utilisée par l'IA et le résumé.
"""

import json
import datetime


def build_bim_json(
    ifc_name: str,
    discipline: dict,
    score_map: dict,
    counts: dict,
    df_all,
    df_top,
    df_dataset=None,
) -> dict:
    """
    Construit un dict JSON structuré à partir des résultats d'analyse.

    Paramètres :
    - ifc_name   : nom du fichier IFC
    - discipline : dict retourné par detect_discipline()
    - score_map  : dict scores {'Global', 'Metier', 'Data BIM', 'Accepted_Check'}
    - counts     : dict comptage objets
    - df_all     : DataFrame complet des résultats
    - df_top     : DataFrame top anomalies
    - df_dataset : DataFrame dataset historique (pour benchmark)

    Retourne un dict JSON sérialisable.
    """

    # ── Scores ────────────────────────────────────────────
    score_global  = round(score_map.get("Global", 0), 2)
    score_metier  = round(score_map.get("Metier", 0), 2)
    score_data    = round(score_map.get("Data BIM", 0), 2)

    # ── Comptage erreurs ──────────────────────────────────
    errors_critical = errors_major = errors_minor = 0
    if df_all is not None and not df_all.empty:
        fail_df = df_all[df_all["Status"] == "FAIL"]
        errors_critical = int((fail_df["Priority"] == "Critical").sum())
        errors_major    = int((fail_df["Priority"] == "Major").sum())
        errors_minor    = int((fail_df["Priority"] == "Minor").sum())
    errors_total = errors_critical + errors_major + errors_minor

    # ── Objets ────────────────────────────────────────────
    # Objets archi de base
    archi_total = (counts.get("walls", 0) + counts.get("doors", 0) +
                   counts.get("windows", 0) + counts.get("slabs", 0) +
                   counts.get("railings", 0))

    # Objets MEP (préfixe mep_)
    mep_total = (counts.get("mep_count_mep", 0) +
                 counts.get("mep_count_ducts", 0) +
                 counts.get("mep_count_pipes", 0) +
                 counts.get("mep_count_terminals", 0) +
                 counts.get("mep_count_equip", 0))

    # Objets structure (préfixe str_)
    str_total = counts.get("str_count_structural", 0)

    # Total tous objets connus (dédupliqué au mieux possible)
    total_objects = max(archi_total, 1) if archi_total > 0 else max(mep_total + str_total, 1)
    if archi_total > 0 and (mep_total + str_total) > 0:
        total_objects = archi_total + mep_total + str_total

    ratio_erreurs = round(errors_total / max(total_objects, 1) * 100, 2)

    # ── Top anomalies ─────────────────────────────────────
    top_issues = []
    if df_top is not None and not df_top.empty:
        for _, row in df_top.head(10).iterrows():
            top_issues.append({
                "rule_id":  row.get("RuleID", ""),
                "priority": row.get("Priority", ""),
                "category": row.get("Category", ""),
                "message":  row.get("Message", ""),
                "suggestion": row.get("Suggestion", ""),
                "ifc_type": row.get("IFC_Type", ""),
                "storey":   row.get("Storey", ""),
            })

    # ── Benchmark ─────────────────────────────────────────
    benchmark = None
    if df_dataset is not None and not df_dataset.empty and len(df_dataset) > 1:
        score_moyen = round(df_dataset["score_global"].mean(), 2)
        score_min   = round(df_dataset["score_global"].min(), 2)
        score_max   = round(df_dataset["score_global"].max(), 2)
        delta       = round(score_global - score_moyen, 2)
        position    = "above_average" if delta >= 0 else "below_average"
        rank        = int(
            df_dataset["score_global"]
            .rank(ascending=False, method="min")
            .iloc[df_dataset.index[df_dataset["score_global"] == score_global].tolist()[-1]
                  if score_global in df_dataset["score_global"].values else -1]
        ) if score_global in df_dataset["score_global"].values else None

        benchmark = {
            "nb_models":    len(df_dataset),
            "score_moyen":  score_moyen,
            "score_min":    score_min,
            "score_max":    score_max,
            "delta_vs_mean": delta,
            "position":     position,
        }

    # ── Discipline ────────────────────────────────────────
    discipline_info = {
        "primary":   discipline.get("primary", "Unknown"),
        "secondary": discipline.get("secondary", []),
        "is_mixed":  discipline.get("is_mixed", False),
        "scores": {
            k: v for k, v in discipline.get("scores", {}).items() if v > 0
        }
    }

    # ── Assemblage final ──────────────────────────────────
    bim_json = {
        "meta": {
            "file":      ifc_name,
            "date":      datetime.datetime.now().isoformat(),
            "version":   "1.0"
        },
        "discipline": discipline_info,
        "scores": {
            "global":        score_global,
            "metier":        score_metier,
            "data_bim":      score_data,
            "interpretation": _interpret_score(score_global)
        },
        "errors": {
            "critical": errors_critical,
            "major":    errors_major,
            "minor":    errors_minor,
            "total":    errors_total,
            "ratio_per_100_objects": ratio_erreurs
        },
        "objects": {
            "walls":    counts.get("walls", 0),
            "doors":    counts.get("doors", 0),
            "windows":  counts.get("windows", 0),
            "slabs":    counts.get("slabs", 0),
            "railings": counts.get("railings", 0),
            "mep":      mep_total,
            "structure": str_total,
            "total":    total_objects,
        },
        "top_issues":  top_issues,
        "benchmark":   benchmark,
    }

    return bim_json


def _interpret_score(score: float) -> str:
    if score >= 90:
        return "excellent"
    elif score >= 75:
        return "bon"
    elif score >= 60:
        return "moyen"
    elif score >= 40:
        return "faible"
    return "critique"


def bim_json_to_string(bim_json: dict) -> str:
    """Sérialise le JSON en string formaté (pour debug ou export)."""
    return json.dumps(bim_json, ensure_ascii=False, indent=2)
