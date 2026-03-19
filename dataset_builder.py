import pandas as pd
import datetime
import os
import hashlib

DATASET_FILE = "bim_dataset.csv"

COLUMNS = [
    "file",
    "file_hash",
    "model_scope",
    "is_benchmark_eligible",
    "discipline_primary",
    "score_global",
    "score_metier",
    "score_data",
    "walls",
    "doors",
    "windows",
    "slabs",
    "railings",
    "total_objects",
    "errors_critical",
    "errors_major",
    "errors_minor",
    "errors_total",
    "date"
]


def compute_hash(ifc_path: str) -> str:
    """Calcule le hash SHA256 du fichier IFC."""
    sha256 = hashlib.sha256()
    with open(ifc_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def check_duplicate(ifc_name: str, file_hash: str, df: pd.DataFrame) -> dict:
    """
    Vérifie si le fichier est un doublon dans le dataset existant.
    Ignore les lignes sans hash valide (anciennes entrées pré-migration).

    Retourne un dict :
    {
        "is_duplicate": bool,
        "type": None | "exact" | "renamed",
        "message": str,
        "existing_file": str | None,
        "existing_date": str | None
    }
    """
    if df.empty or "file_hash" not in df.columns:
        return {"is_duplicate": False, "type": None, "message": "", "existing_file": None, "existing_date": None}

    # Filtre strict : on ignore les hashes vides, NaN, ou "None" texte
    valid_mask = (
        df["file_hash"].notna() &
        (df["file_hash"].astype(str).str.strip() != "") &
        (df["file_hash"].astype(str).str.lower() != "none") &
        (df["file_hash"].astype(str).str.lower() != "nan")
    )
    matches = df[valid_mask & (df["file_hash"] == file_hash)]

    if matches.empty:
        return {"is_duplicate": False, "type": None, "message": "", "existing_file": None, "existing_date": None}

    existing = matches.iloc[-1]  # dernière occurrence connue
    existing_file = existing["file"]
    existing_date = str(existing["date"])

    if existing_file == ifc_name:
        return {
            "is_duplicate": True,
            "type": "exact",
            "message": f"Ce fichier a déjà été analysé le {existing_date[:10]}.",
            "existing_file": existing_file,
            "existing_date": existing_date
        }
    else:
        return {
            "is_duplicate": True,
            "type": "renamed",
            "message": f"Ce fichier est identique à **{existing_file}** (analysé le {existing_date[:10]}), mais a été renommé.",
            "existing_file": existing_file,
            "existing_date": existing_date
        }


def update_dataset(
    ifc_name: str,
    ifc_path: str,
    scores: dict,
    counts: dict,
    df_results,
    scope_info: dict = None,
    discipline_info: dict = None,
    force: bool = False
) -> dict:
    """
    Ajoute une ligne dans le dataset BIM après chaque analyse.

    Paramètres :
    - ifc_name       : nom du fichier IFC (str)
    - ifc_path       : chemin local vers le fichier IFC (pour le hash SHA256)
    - scores         : dict avec clés 'Global', 'Metier', 'Data BIM'
    - counts         : dict avec clés 'walls', 'doors', 'windows', 'slabs', 'railings'
    - df_results     : DataFrame complet des résultats d'analyse
    - scope_info     : dict retourné par model_scope_detector.detect_model_scope()
    - discipline_info: dict retourné par discipline_detector.detect_discipline()
    - force          : si True, remplace l'ancienne entrée en cas de doublon

    Retourne un dict avec :
    - "status"    : "added" | "duplicate_blocked" | "duplicate_replaced"
    - "duplicate" : dict info doublon (ou None)
    - "df"        : DataFrame mis à jour
    """

    # Calcul du hash SHA256
    file_hash = compute_hash(ifc_path)

    # Chargement dataset existant
    if os.path.exists(DATASET_FILE):
        df = pd.read_csv(DATASET_FILE)
    else:
        df = pd.DataFrame(columns=COLUMNS)

    # Vérification doublon
    dup = check_duplicate(ifc_name, file_hash, df)

    if dup["is_duplicate"] and not force:
        print(f"[Dataset] Doublon détecté ({dup['type']}) : {ifc_name} — ajout bloqué.")
        return {
            "status": "duplicate_blocked",
            "duplicate": dup,
            "df": df
        }

    # Comptage erreurs
    errors_critical = errors_major = errors_minor = 0
    if df_results is not None and not df_results.empty:
        fail_df = df_results[df_results["Status"] == "FAIL"]
        errors_critical = int((fail_df["Priority"] == "Critical").sum())
        errors_major    = int((fail_df["Priority"] == "Major").sum())
        errors_minor    = int((fail_df["Priority"] == "Minor").sum())
    errors_total = errors_critical + errors_major + errors_minor

    model_scope           = scope_info.get("scope", "Unknown") if scope_info else "Unknown"
    is_benchmark_eligible = scope_info.get("is_benchmark_eligible", True) if scope_info else True
    discipline_primary    = discipline_info.get("primary", "Unknown") if discipline_info else "Unknown"

    row = {
        "file":                  ifc_name,
        "file_hash":             file_hash,
        "model_scope":           model_scope,
        "is_benchmark_eligible": is_benchmark_eligible,
        "discipline_primary":    discipline_primary,
        "score_global":          scores.get("Global", 0),
        "score_metier":    scores.get("Metier", 0),
        "score_data":      scores.get("Data BIM", 0),
        "walls":           counts.get("walls", 0),
        "doors":           counts.get("doors", 0),
        "windows":         counts.get("windows", 0),
        "slabs":           counts.get("slabs", 0),
        "railings":        counts.get("railings", 0),
        "total_objects":   sum(counts.values()),
        "errors_critical": errors_critical,
        "errors_major":    errors_major,
        "errors_minor":    errors_minor,
        "errors_total":    errors_total,
        "date":            datetime.datetime.now().isoformat()
    }

    df_new = pd.DataFrame([row], columns=COLUMNS)

    if dup["is_duplicate"] and force:
        # Supprime l'ancienne entrée avant d'ajouter la nouvelle
        df = df[df["file_hash"] != file_hash]
        status = "duplicate_replaced"
        print(f"[Dataset] Doublon remplacé : {ifc_name}")
    else:
        status = "added"
        print(f"[Dataset] Entrée ajoutée : {ifc_name} | Score global : {scores.get('Global', 0)}")

    df = pd.concat([df, df_new], ignore_index=True)
    df.to_csv(DATASET_FILE, index=False)

    return {
        "status": status,
        "duplicate": dup if dup["is_duplicate"] else None,
        "df": df
    }


def load_dataset() -> pd.DataFrame:
    """Charge le dataset existant ou retourne un DataFrame vide."""
    if os.path.exists(DATASET_FILE):
        return pd.read_csv(DATASET_FILE)
    return pd.DataFrame(columns=COLUMNS)
