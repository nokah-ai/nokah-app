"""
bim_ai_v1.py
────────────
IA V1 — Détection d'anomalies BIM par IsolationForest.
100% locale, aucune dépendance externe.

Détecte si un modèle BIM est "atypique" par rapport au parc connu.
Affiche un indice d'atypie + les variables les plus contributives.
"""

import numpy as np
import pandas as pd

# Colonnes utilisées pour l'analyse IA
FEATURES = [
    "score_global",
    "score_metier",
    "score_data",
    "errors_critical",
    "errors_major",
    "errors_minor",
    "errors_total",
    "ratio_erreurs_100",   # calculé à la volée si absent
]

# Seuil de score d'anomalie pour classifier "atypique"
# IsolationForest : plus le score est négatif, plus c'est anormal
ANOMALY_THRESHOLD = -0.05

MIN_MODELS_FOR_AI = 3   # minimum de modèles pour que l'IA soit utile


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prépare et normalise les features pour l'IA."""
    df = df.copy()

    # Calcul du ratio si absent
    if "ratio_erreurs_100" not in df.columns:
        df["ratio_erreurs_100"] = (
            df["errors_total"] / df["total_objects"].replace(0, np.nan) * 100
        ).fillna(0).round(2)

    # Remplissage des valeurs manquantes
    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0.0

    return df[FEATURES]


def run_anomaly_detection(df_dataset: pd.DataFrame, bim_json: dict) -> dict:
    """
    Lance l'IA V1 sur le dataset et évalue le modèle courant.

    Paramètres :
    - df_dataset : DataFrame historique (bim_dataset.csv)
    - bim_json   : JSON structuré du modèle courant (bim_json_builder)

    Retourne un dict :
    {
        "available": bool,
        "reason": str,
        "anomaly_score": float,
        "label": str,           # "Normal" | "Atypique" | "Très atypique"
        "confidence": str,      # "low" | "medium" | "good"
        "top_contributors": list[dict],
        "interpretation": str,
    }
    """

    if df_dataset is None or df_dataset.empty:
        return _unavailable("No dataset available.")

    if len(df_dataset) < MIN_MODELS_FOR_AI:
        return _unavailable(
            f"Dataset too small ({len(df_dataset)} model(s)). "
            f"Minimum required: {MIN_MODELS_FOR_AI}."
        )

    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return _unavailable("scikit-learn not installed. Run: pip install scikit-learn")

    # ── Préparation des données ───────────────────────────
    df_feat = prepare_features(df_dataset)
    X = df_feat.values

    # ── Entraînement IsolationForest ──────────────────────
    contamination = min(0.3, 1.0 / len(df_dataset))   # adaptatif
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42
    )
    model.fit(X)

    # ── Vecteur du modèle courant ─────────────────────────
    current = _bim_json_to_vector(bim_json)
    current_df = pd.DataFrame([current], columns=FEATURES)
    current_arr = current_df.values

    # ── Score d'anomalie ──────────────────────────────────
    raw_score = float(model.score_samples(current_arr)[0])

    # Normalisation robuste : on étend la plage avec le modèle courant inclus
    # pour éviter que le score dépasse 1.0 quand le modèle est hors dataset
    all_scores = model.score_samples(X)
    combined_scores = np.append(all_scores, raw_score)
    score_min = combined_scores.min()
    score_max = combined_scores.max()
    score_range = score_max - score_min if score_max != score_min else 1.0

    # Normalisation 0→1 (0=très normal, 1=très anormal) + clip strict
    anomaly_score_normalized = float(
        np.clip(1 - (raw_score - score_min) / score_range, 0.0, 1.0)
    )
    anomaly_score_normalized = round(anomaly_score_normalized, 3)

    # ── Label ─────────────────────────────────────────────
    prediction = model.predict(current_arr)[0]  # 1=normal, -1=anormal
    if prediction == 1:
        label = "Normal"
    elif anomaly_score_normalized > 0.80:
        label = "Very atypical"
    else:
        label = "Atypical"

    # ── Confiance (basée sur taille dataset) ─────────────
    n = len(df_dataset)
    if n < 5:
        confidence = "low"
    elif n < 15:
        confidence = "medium"
    else:
        confidence = "good"

    # ── Top contributors ──────────────────────────────────
    # Feature-by-feature comparison vs park average
    means = df_feat.mean()
    stds  = df_feat.std().replace(0, 1)
    current_series = current_df.iloc[0]

    contributors = []
    for col in FEATURES:
        val     = float(current_series[col])
        mean    = float(means[col])
        std     = float(stds[col])
        z_score = abs((val - mean) / std) if std > 0 else 0
        if z_score > 0.3:
            direction = "high" if val > mean else "low"
            contributors.append({
                "feature":   col,
                "value":     round(val, 2),
                "mean":      round(mean, 2),
                "z_score":   round(z_score, 2),
                "direction": direction,
                "label":     _feature_label(col),
            })

    contributors.sort(key=lambda x: -x["z_score"])
    top_contributors = contributors[:5]

    # ── Interprétation textuelle ──────────────────────────
    interpretation = _build_interpretation(
        label, anomaly_score_normalized, top_contributors,
        bim_json, len(df_dataset)
    )

    return {
        "available":        True,
        "reason":           "",
        "anomaly_score":    anomaly_score_normalized,
        "raw_score":        raw_score,
        "label":            label,
        "confidence":       confidence,
        "top_contributors": top_contributors,
        "interpretation":   interpretation,
        "n_models":         n,
    }


def _bim_json_to_vector(bim_json: dict) -> dict:
    """Convertit le JSON BIM en vecteur de features pour l'IA."""
    errors  = bim_json.get("errors", {})
    scores  = bim_json.get("scores", {})
    objects = bim_json.get("objects", {})
    total   = max(objects.get("total", 1), 1)

    return {
        "score_global":     scores.get("global", 0),
        "score_metier":     scores.get("metier", 0),
        "score_data":       scores.get("data_bim", 0),
        "errors_critical":  errors.get("critical", 0),
        "errors_major":     errors.get("major", 0),
        "errors_minor":     errors.get("minor", 0),
        "errors_total":     errors.get("total", 0),
        "ratio_erreurs_100": errors.get("ratio_per_100_objects", 0),
    }


def _feature_label(col: str) -> str:
    labels = {
        "score_global": "Global score",
        "score_metier": "Conformity score",
        "score_data": "Data BIM score",
        "errors_critical": "Critical errors",
        "errors_major": "Major errors",
        "errors_minor": "Minor errors",
        "errors_total": "Total errors",
        "ratio_erreurs_100": "Errors per 100 objects",
    }
    return labels.get(col, col)


def _build_interpretation(label, score, contributors, bim_json, n_models):
    """Génère une interprétation textuelle de l'analyse IA."""
    discipline = bim_json.get("discipline", {}).get("primary", "Inconnu")
    score_global = bim_json.get("scores", {}).get("global", 0)

    lines = []

    if label == "Normal":
        lines.append(
            f"This {discipline} model is within the norm of the analyzed park "
            f"({n_models} model(s)). Its behavior is consistent with other projects."
        )
    elif label == "Atypique":
        lines.append(
            f"This {discipline} model has an atypical profile compared to "
            f"{n_models} other model(s) in the park."
        )
    else:
        lines.append(
            f"This {discipline} model is very atypical. Its profile stands out "
            f"significantly from the {n_models} other analyzed model(s)."
        )

    if contributors:
        top = contributors[0]
        lines.append(
            f"The most contributing variable is **{top['label']}** "
            f"({top['value']} vs average {top['mean']}) — value {top['direction']} compared to park."
        )

    if n_models < 5:
        lines.append(
            "⚠️ Dataset encore petit — cet indice est indicatif. "
            "It will gain reliability with more models."
        )

    return " ".join(lines)


def _unavailable(reason: str) -> dict:
    return {
        "available":        False,
        "reason":           reason,
        "anomaly_score":    None,
        "raw_score":        None,
        "label":            None,
        "confidence":       None,
        "top_contributors": [],
        "interpretation":   "",
        "n_models":         0,
    }
