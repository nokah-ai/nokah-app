"""
bim_engine.py
─────────────
Moteur BIM QA partagé — nokah Alpha
Utilisé par dashboard_v3.py (interface développeur) et nokah_app.py (interface produit)

NE PAS importer streamlit dans ce module.
"""

import pandas as pd
import ifcopenshell
import ifcopenshell.geom
import numpy as np
import json
import tempfile
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

from dataset_builder import update_dataset, load_dataset
from discipline_detector import detect_discipline, discipline_badge
from model_scope_detector import detect_model_scope, scope_badge, scope_color
from rules_mep import run_mep_rules
from rules_structure import run_structure_rules
from rules_archi_advanced import run_archi_advanced_rules
from bim_json_builder import build_bim_json, bim_json_to_string
from bim_ai_v1 import run_anomaly_detection
from bim_summary import generate_summary


CONFIG_FILE = "rules_config.json"


# =========================================================
# CONFIG
# =========================================================

default_config = {
    # Architecture
    "R006": {"enabled": True, "priority": "Critical"},
    "R007": {"enabled": True, "min_height": 0.90, "priority": "Critical"},
    "R008": {"enabled": True, "min_z": 0.40, "priority": "Major"},
    "R011": {"enabled": True, "priority": "Minor"},
    "R012": {"enabled": True, "min_width": 0.80, "priority": "Major"},
    "I006": {"enabled": True, "min_height": 1.00, "priority": "Major"},
    # Architecture Avancée
    "A001": {"enabled": True, "priority": "Minor"},
    "A002": {"enabled": True, "priority": "Minor"},   # Check — recommandation thermique
    "A003": {"enabled": True, "priority": "Major", "min_window_height": 0.30, "max_window_height": 4.00},
    "A004": {"enabled": True, "priority": "Minor", "pmr_min_width": 0.83},  # Check — accessibilité PMR
    "A005": {"enabled": True, "priority": "Critical"},
    "A006": {"enabled": True, "priority": "Minor"},
    "A007": {"enabled": True, "priority": "Minor"},
    "A008": {"enabled": True, "priority": "Minor"},
    # MEP
    "M001": {"enabled": True, "priority": "Major"},
    "M002": {"enabled": True, "priority": "Major"},
    "M003": {"enabled": True, "priority": "Major"},
    "M004": {"enabled": True, "priority": "Minor"},
    "M005": {"enabled": True, "priority": "Minor"},
    "M006": {"enabled": True, "priority": "Major", "min_length": 0.05},
    "M007": {"enabled": True, "priority": "Major"},
    # Structure
    "S001": {"enabled": True, "priority": "Major"},
    "S002": {"enabled": True, "priority": "Major"},
    "S003": {"enabled": True, "priority": "Minor"},
    "S004": {"enabled": True, "priority": "Major"},
    "S005": {"enabled": True, "priority": "Minor"},
    "S006": {"enabled": True, "priority": "Minor"},
    "S007": {"enabled": True, "priority": "Major", "min_height": 0.50},
}

config_path = Path(CONFIG_FILE)
if config_path.exists():
    with open(config_path, "r", encoding="utf-8") as f:
        user_config = json.load(f)
else:
    user_config = {}

rule_config = default_config.copy()
for key, value in user_config.items():
    if key in rule_config:
        rule_config[key].update(value)
    else:
        rule_config[key] = value


# =========================================================
# OUTILS
# =========================================================

def score_label(score: float) -> str:
    if score >= 90:
        return "Excellent"
    elif score >= 75:
        return "Good"
    elif score >= 60:
        return "Medium"
    elif score >= 40:
        return "Low"
    return "Critical"


def build_priority_map(df_results: pd.DataFrame) -> dict:
    priority_rank = {
        "Critical": 5,
        "Major": 4,
        "Minor": 3,
        "Accepted": 2,
        "Check": 1,
        "Normal": 0,
    }

    guid_map = {}

    if df_results.empty:
        return guid_map

    for _, row in df_results.iterrows():
        guid = row.get("GUID")
        status = row.get("Status")
        priority = row.get("Priority")
        bucket = row.get("Bucket")

        if pd.isna(guid):
            continue

        if status == "ACCEPTED":
            label = "Accepted"
        elif bucket == "Check" or status == "CHECK":
            label = "Check"
        elif priority in ["Critical", "Major", "Minor"]:
            label = priority
        else:
            label = "Normal"

        if guid not in guid_map or priority_rank[label] > priority_rank[guid_map[guid]]:
            guid_map[guid] = label

    return guid_map


def analyze_ifc(ifc_path: str, rule_config: dict):
    ifc = ifcopenshell.open(ifc_path)
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    results = []
    applicability_rows = []
    viewer_rows = []

    def get_elements(entity_type):
        try:
            return ifc.by_type(entity_type)
        except RuntimeError:
            return []

    def get_element_name(element):
        if hasattr(element, "Name") and element.Name:
            return str(element.Name)
        return "No name"

    def get_element_type_name(element):
        if hasattr(element, "IsDefinedBy") and element.IsDefinedBy:
            for rel in element.IsDefinedBy:
                if rel.is_a("IfcRelDefinesByType"):
                    try:
                        if rel.RelatingType and rel.RelatingType.Name:
                            return str(rel.RelatingType.Name)
                    except Exception:
                        pass
        return "Unknown type"

    def get_pset_property(element, pset_name, prop_name):
        if hasattr(element, "IsDefinedBy") and element.IsDefinedBy:
            for rel in element.IsDefinedBy:
                if rel.is_a("IfcRelDefinesByProperties"):
                    pset = rel.RelatingPropertyDefinition
                    if pset.is_a("IfcPropertySet") and pset.Name == pset_name:
                        for prop in pset.HasProperties:
                            if prop.Name == prop_name and hasattr(prop, "NominalValue"):
                                try:
                                    return prop.NominalValue.wrappedValue
                                except Exception:
                                    return None
        return None

    def get_shape_vertices(element):
        try:
            shape = ifcopenshell.geom.create_shape(settings, element)
        except Exception:
            return None

        verts = shape.geometry.verts
        if not verts:
            return None

        vertices = np.array(verts, dtype=float).reshape((-1, 3))
        return vertices

    def get_shape_z_range(element):
        vertices = get_shape_vertices(element)
        if vertices is None:
            return None, None
        return float(vertices[:, 2].min()), float(vertices[:, 2].max())

    def get_shape_center(element):
        vertices = get_shape_vertices(element)
        if vertices is None:
            return None
        center = vertices.mean(axis=0)
        return center.tolist()

    def has_material(element):
        return bool(getattr(element, "HasAssociations", None))

    def is_probable_skylight(element):
        text = (get_element_name(element) + " " + get_element_type_name(element)).lower()
        keywords = ["sky", "skylight", "roof", "rooflight", "lucarne", "lanterneau", "toit", "velux"]
        return any(keyword in text for keyword in keywords)

    def is_probable_exterior_wall(element):
        text = (get_element_name(element) + " " + get_element_type_name(element)).lower()
        keywords = ["ext", "extérieur", "exterieur", "facade", "façade", "outside", "external"]
        return any(keyword in text for keyword in keywords)

    def get_containing_structure_name(element):
        if hasattr(element, "ContainedInStructure") and element.ContainedInStructure:
            for rel in element.ContainedInStructure:
                try:
                    if rel.RelatingStructure and rel.RelatingStructure.Name:
                        return str(rel.RelatingStructure.Name)
                except Exception:
                    pass
        return None

    def add_result(
        rule_id,
        category,
        severity,
        priority,
        element,
        message,
        applicable_count,
        status="FAIL",
        bucket="Metier",
        suggestion="",
        autofix_possible=False,
        fix_type=""
    ):
        zmin, zmax = get_shape_z_range(element)

        results.append({
            "RuleID": rule_id,
            "Category": category,
            "Severity": severity,
            "Priority": priority,
            "Bucket": bucket,
            "Status": status,
            "IFC_Type": element.is_a(),
            "GUID": element.GlobalId,
            "Name": get_element_name(element),
            "TypeObject": get_element_type_name(element),
            "Storey": get_containing_structure_name(element),
            "Message": message,
            "Suggestion": suggestion,
            "AutoFixPossible": autofix_possible,
            "FixType": fix_type,
            "Zmin": zmin,
            "Zmax": zmax,
            "ApplicableCount": applicable_count
        })

    def add_applicability(rule_id, category, applicable, reason):
        applicability_rows.append({
            "RuleID": rule_id,
            "Category": category,
            "Applicable": "YES" if applicable else "NO",
            "Reason": reason
        })

    def is_rule_enabled(rule_id):
        return rule_config.get(rule_id, {}).get("enabled", True)

    def get_rule_priority(rule_id, fallback):
        return rule_config.get(rule_id, {}).get("priority", fallback)

    def get_rule_value(rule_id, key, fallback):
        return rule_config.get(rule_id, {}).get(key, fallback)

    count_walls = len(get_elements("IfcWall"))
    count_doors = len(get_elements("IfcDoor"))
    count_windows = len(get_elements("IfcWindow"))
    count_railings = len(get_elements("IfcRailing"))
    count_slabs = len(get_elements("IfcSlab"))

    coord_types = ["IfcWall", "IfcDoor", "IfcWindow", "IfcSlab", "IfcRailing"]
    coord_total = sum(len(get_elements(t)) for t in coord_types)

    # ---------------------------
    # Règles
    # ---------------------------

    # Discipline détectée ici pour filtrer les règles de base
    discipline_info = detect_discipline(ifc)
    _primary_early   = discipline_info.get("primary", "Unknown")
    _disc_scores_early = discipline_info.get("scores", {})
    _primary_score_early = _disc_scores_early.get(_primary_early, 1)
    _active_early = {_primary_early}
    for _d, _s in _disc_scores_early.items():
        if _d != _primary_early and _s > 0 and (_s / max(_primary_score_early, 1)) >= 0.10:
            _active_early.add(_d)
    _run_archi_base = "Architecture" in _active_early or _primary_early == "Unknown"
    _run_coord      = True  # R021/R024 toujours actifs (data BIM universel)

    if _run_archi_base:
        rule_id = "R003"
        category = "Architecture"
        applicable = count_windows > 0
        add_applicability(rule_id, category, applicable, "IfcWindow present" if applicable else "No IfcWindow")
        if applicable:
            for window in get_elements("IfcWindow"):
                if not window.FillsVoids:
                    if is_probable_skylight(window):
                        add_result(
                            rule_id, category, "INFO", "Minor", window,
                            "Skylight accepted", count_windows,
                            status="ACCEPTED", bucket="Accepted",
                            suggestion="No correction needed: roof window recognized.",
                            autofix_possible=False, fix_type="None"
                        )
                    else:
                        add_result(
                            rule_id, category, "ERROR", "Major", window,
                            "Window without host wall", count_windows,
                            suggestion="Assign the window to a valid opening or host element.",
                            autofix_possible=False, fix_type="HostRelation"
                        )
    
        rule_id = "R006"
        category = "Architecture"
        applicable = count_walls > 0 and is_rule_enabled(rule_id)
        add_applicability(rule_id, category, applicable, "IfcWall present and rule enabled" if applicable else "Rule disabled or no IfcWall")
        if applicable:
            priority = get_rule_priority(rule_id, "Critical")
            for wall in get_elements("IfcWall"):
                is_external = get_pset_property(wall, "Pset_WallCommon", "IsExternal")
                if is_probable_exterior_wall(wall) and is_external is not True:
                    add_result(
                        rule_id, category, "ERROR", priority, wall,
                        "Exterior wall type but IsExternal property missing or false",
                        count_walls,
                        suggestion="Add or correct Pset_WallCommon.IsExternal = TRUE.",
                        autofix_possible=True, fix_type="Property"
                    )
    
        rule_id = "R007"
        category = "Architecture"
        applicable = count_railings > 0 and is_rule_enabled(rule_id)
        add_applicability(rule_id, category, applicable, "IfcRailing present and rule enabled" if applicable else "Rule disabled or no IfcRailing")
        if applicable:
            min_height = get_rule_value(rule_id, "min_height", 0.90)
            priority = get_rule_priority(rule_id, "Critical")
            for railing in get_elements("IfcRailing"):
                zmin, zmax = get_shape_z_range(railing)
                if zmin is not None and zmax is not None:
                    height = zmax - zmin
                    if height < min_height:
                        add_result(
                            rule_id, category, "WARNING", priority, railing,
                            f"Railing too low ({height:.2f} m < {min_height:.2f} m)",
                            count_railings,
                            suggestion=f"Raise the railing height to at least {min_height:.2f} m.",
                            autofix_possible=False, fix_type="Geometry"
                        )
    
        rule_id = "R008"
        category = "Architecture"
        applicable = count_windows > 0 and is_rule_enabled(rule_id)
        add_applicability(rule_id, category, applicable, "IfcWindow present and rule enabled" if applicable else "Rule disabled or no IfcWindow")
        if applicable:
            min_z = get_rule_value(rule_id, "min_z", 0.40)
            priority = get_rule_priority(rule_id, "Major")
            for window in get_elements("IfcWindow"):
                if is_probable_skylight(window):
                    continue
                zmin, _ = get_shape_z_range(window)
                if zmin is not None and zmin < min_z:
                    delta = round(min_z - zmin, 3)
                    add_result(
                        rule_id, category, "WARNING", priority, window,
                        f"Low window (Zmin={zmin:.2f} m < {min_z:.2f} m)",
                        count_windows,
                        suggestion=f"Raise the window by approximately {delta:.2f} m to reach Zmin >= {min_z:.2f} m.",
                        autofix_possible=False, fix_type="Geometry"
                    )
    
        rule_id = "R011"
        category = "Architecture"
        applicable = count_walls > 0 and is_rule_enabled(rule_id)
        add_applicability(rule_id, category, applicable, "IfcWall present and rule enabled" if applicable else "Rule disabled or no IfcWall")
        if applicable:
            priority = get_rule_priority(rule_id, "Minor")
            for wall in get_elements("IfcWall"):
                if is_probable_exterior_wall(wall):
                    type_text = (get_element_name(wall) + " " + get_element_type_name(wall)).lower()
                    insulation_keywords = ["laine", "insulation", "isol", "rockwool", "glasswool", "eps", "xps", "mineral wool"]
                    if not any(k in type_text for k in insulation_keywords):
                        add_result(
                            rule_id, category, "WARNING", priority, wall,
                            "Exterior wall without detectable insulation in name/type",
                            count_walls,
                            bucket="Check",
                            suggestion="Check the wall composition and add an insulation layer if needed.",
                            autofix_possible=False, fix_type="TypeDefinition"
                        )
    
        rule_id = "R012"
        category = "Architecture"
        applicable = count_doors > 0 and is_rule_enabled(rule_id)
        add_applicability(rule_id, category, applicable, "IfcDoor present and rule enabled" if applicable else "Rule disabled or no IfcDoor")
        if applicable:
            min_width = get_rule_value(rule_id, "min_width", 0.80)
            priority = get_rule_priority(rule_id, "Major")
            for door in get_elements("IfcDoor"):
                overall_width = getattr(door, "OverallWidth", None)
                if overall_width is not None:
                    try:
                        width = float(overall_width)
                        if width < min_width:
                            add_result(
                                rule_id, category, "WARNING", priority, door,
                                f"Door too narrow ({width:.2f} m < {min_width:.2f} m)",
                                count_doors,
                                suggestion=f"Increase the door width to at least {min_width:.2f} m, ideally 0.90 m depending on use.",
                                autofix_possible=False, fix_type="Geometry"
                            )
                    except Exception:
                        pass
    
        rule_id = "I006"
        category = "Interior Design"
        applicable = count_railings > 0 and is_rule_enabled(rule_id)
        add_applicability(rule_id, category, applicable, "IfcRailing present and rule enabled" if applicable else "Rule disabled or no IfcRailing")
        if applicable:
            min_height = get_rule_value(rule_id, "min_height", 1.00)
            priority = get_rule_priority(rule_id, "Major")
            for railing in get_elements("IfcRailing"):
                zmin, zmax = get_shape_z_range(railing)
                if zmin is not None and zmax is not None:
                    height = zmax - zmin
                    if height < min_height:
                        add_result(
                            rule_id, category, "WARNING", priority, railing,
                            f"Interior railing potentially insufficient ({height:.2f} m < {min_height:.2f} m)",
                            count_railings,
                            suggestion=f"Raise the interior railing to {min_height:.2f} m minimum if required by project rules.",
                            autofix_possible=False, fix_type="Geometry"
                        )
    
    
    rule_id = "R021"
    category = "Coordination"
    applicable = coord_total > 0
    add_applicability(rule_id, category, applicable, "Coordination objects present" if applicable else "No target object")
    if applicable:
        for t in coord_types:
            for element in get_elements(t):
                classification = get_pset_property(element, "Classification", "Code")
                ref = get_pset_property(element, "Identity Data", "Assembly Code")
                if classification is None and ref is None:
                    add_result(
                        rule_id, category, "INFO", "Data", element,
                        "Object without detectable classification",
                        coord_total,
                        bucket="Data BIM",
                        suggestion="Add a classification code or Assembly Code.",
                        autofix_possible=False, fix_type="Classification"
                    )

    rule_id = "R024"
    category = "Coordination"
    applicable = coord_total > 0
    add_applicability(rule_id, category, applicable, "Coordination objects present" if applicable else "No target object")
    if applicable:
        for t in coord_types:
            for element in get_elements(t):
                if get_element_type_name(element) == "Unknown type":
                    add_result(
                        rule_id, category, "WARNING", "Data", element,
                        "Object without type",
                        coord_total,
                        bucket="Data BIM",
                        suggestion="Assign the object to a defined BIM type.",
                        autofix_possible=False, fix_type="TypeAssignment"
                    )

    # ---------------------------
    # Scope discipline — détermine quelles règles activer
    # ---------------------------
    # Logique :
    #   Archi pur       → archi de base + archi avancée + intérieur + data
    #   MEP pur         → MEP + data
    #   Structure pure  → structure + data
    #   Mixte           → toutes les disciplines détectées
    #   Unknown         → toutes les règles (safe fallback)

    primary    = discipline_info.get("primary", "Unknown")
    secondary  = discipline_info.get("secondary", [])
    disc_scores = discipline_info.get("scores", {})

    # Une discipline secondaire est "réelle" si elle dépasse 10% du score primaire
    primary_score = disc_scores.get(primary, 1)
    active_disciplines = {primary}
    for disc, score in disc_scores.items():
        if disc != primary and score > 0:
            ratio = score / max(primary_score, 1)
            if ratio >= 0.10:   # au moins 10% du volume primaire
                active_disciplines.add(disc)

    run_archi  = "Architecture" in active_disciplines or primary == "Unknown"
    run_mep    = "MEP" in active_disciplines or primary == "Unknown"
    run_struct = "Structure" in active_disciplines or primary == "Unknown"
    run_interior = run_archi  # intérieur suit toujours l'archi

    # ---------------------------
    # Règles MEP
    # ---------------------------

    if run_mep:
        mep_counts = run_mep_rules(
            ifc, rule_config,
            get_elements, get_element_name, get_element_type_name,
            get_containing_structure_name, get_pset_property,
            add_result, add_applicability
        )
    else:
        mep_counts = {"count_mep": 0, "count_ducts": 0, "count_pipes": 0,
                      "count_terminals": 0, "count_equip": 0}

    # ---------------------------
    # Règles Structure
    # ---------------------------

    if run_struct:
        struct_counts = run_structure_rules(
            ifc, rule_config,
            get_elements, get_element_name, get_element_type_name,
            get_containing_structure_name, get_pset_property, get_shape_z_range,
            add_result, add_applicability
        )
    else:
        struct_counts = {"count_beams": 0, "count_columns": 0, "count_members": 0,
                         "count_footings": 0, "count_piles": 0, "count_structural": 0}

    # ---------------------------
    # Règles Architecture Avancée + Intérieure
    # ---------------------------

    if run_archi:
        archi_adv_counts = run_archi_advanced_rules(
            ifc, rule_config,
            get_elements, get_element_name, get_element_type_name,
            get_containing_structure_name, get_pset_property,
            get_shape_z_range, add_result, add_applicability,
            is_probable_exterior_wall, is_probable_skylight
        )
    else:
        archi_adv_counts = {"count_ext_walls": 0, "count_int_walls": 0,
                            "count_windows_checked": 0, "count_doors_checked": 0}

    # ---------------------------
    # Viewer rows (toutes disciplines)
    # ---------------------------

    viewer_types = [
        "IfcWall", "IfcDoor", "IfcWindow", "IfcSlab", "IfcRailing",
        "IfcBeam", "IfcColumn", "IfcMember",
        "IfcDuctSegment", "IfcPipeSegment", "IfcFlowTerminal",
    ]
    for entity_type in viewer_types:
        for element in get_elements(entity_type):
            center = get_shape_center(element)
            if center is None:
                continue
            viewer_rows.append({
                "GUID": element.GlobalId,
                "IFC_Type": entity_type,
                "Name": get_element_name(element),
                "x": center[0],
                "y": center[1],
                "z": center[2],
            })

    df = pd.DataFrame(results)
    df_app = pd.DataFrame(applicability_rows)
    df_viewer = pd.DataFrame(viewer_rows)

    priority_weights = {
        "Critical": 1.0,
        "Major": 0.6,
        "Minor": 0.2,
        "Data": 0.08
    }

    def compute_normalized_score(df_subset):
        if df_subset.empty:
            return 100.0

        fail_df = df_subset[df_subset["Status"] == "FAIL"].copy()
        if fail_df.empty:
            return 100.0

        grouped = fail_df.groupby(["RuleID", "Priority", "ApplicableCount"]).size().reset_index(name="FailCount")

        penalty = 0.0
        for _, row in grouped.iterrows():
            applicable_count = max(1, int(row["ApplicableCount"]))
            fail_count = int(row["FailCount"])
            ratio = fail_count / applicable_count
            weight = priority_weights.get(row["Priority"], 0.1)
            rule_penalty = min(100.0, ratio * 100.0 * weight)
            penalty += rule_penalty

        penalty = min(100.0, penalty)
        score = max(0.0, 100.0 - penalty)
        return round(score, 2)

    if df.empty:
        summary = pd.DataFrame(columns=["RuleID", "Category", "Severity", "Priority", "Bucket", "Status", "Count"])
        df_metier = pd.DataFrame()
        df_data = pd.DataFrame()
        df_accepted = pd.DataFrame()
        score_global = 100.0
        score_metier = 100.0
        score_data = 100.0
        score_accepted = 100.0
        top_anomalies = pd.DataFrame()
    else:
        df_metier = df[df["Bucket"] == "Metier"].copy()
        df_data = df[df["Bucket"] == "Data BIM"].copy()
        df_accepted = df[df["Bucket"].isin(["Accepted", "Check"])].copy()

        summary = df.groupby(
            ["RuleID", "Category", "Severity", "Priority", "Bucket", "Status"]
        ).size().reset_index(name="Count")

        score_metier = compute_normalized_score(df_metier)
        score_data = compute_normalized_score(df_data)
        score_accepted = compute_normalized_score(df_accepted)
        score_global = round((score_metier * 0.7) + (score_data * 0.3), 2)

        fail_df = df[df["Status"] == "FAIL"].copy()
        priority_order = {"Critical": 0, "Major": 1, "Minor": 2, "Data": 3}
        fail_df["PrioritySort"] = fail_df["Priority"].map(priority_order).fillna(99)
        top_anomalies = fail_df.sort_values(["PrioritySort", "RuleID"]).head(20).drop(columns=["PrioritySort"])

    df_scores = pd.DataFrame([
        {"ScoreName": "Global", "ScoreValue": score_global, "Label": score_label(score_global)},
        {"ScoreName": "Metier", "ScoreValue": score_metier, "Label": score_label(score_metier)},
        {"ScoreName": "Data BIM", "ScoreValue": score_data, "Label": score_label(score_data)},
        {"ScoreName": "Accepted_Check", "ScoreValue": score_accepted, "Label": score_label(score_accepted)},
    ])

    counts = {
        "walls":    count_walls,
        "doors":    count_doors,
        "windows":  count_windows,
        "slabs":    count_slabs,
        "railings": count_railings,
        **{f"mep_{k}": v for k, v in mep_counts.items()},
        **{f"str_{k}": v for k, v in struct_counts.items()},
        **{f"adv_{k}": v for k, v in archi_adv_counts.items()},
    }

    return {
        "scores":       df_scores,
        "summary":      summary,
        "metier":       df_metier,
        "data_bim":     df_data,
        "accepted":     df_accepted,
        "applicability": df_app,
        "top_anomalies": top_anomalies,
        "all_results":  df,
        "viewer":       df_viewer,
        "counts":       counts,
        "discipline":   discipline_info,
        "scope":        detect_model_scope(ifc, discipline_info),
    }


def make_3d_figure(df_viewer: pd.DataFrame, df_results: pd.DataFrame):
    fig = go.Figure()

    if df_viewer.empty:
        return fig

    priority_map = build_priority_map(df_results)

    df_plot = df_viewer.copy()
    df_plot["VisualPriority"] = df_plot["GUID"].map(priority_map).fillna("Normal")

    color_map = {
        "Critical": "red",
        "Major": "orange",
        "Minor": "yellow",
        "Accepted": "cyan",
        "Check": "white",
        "Normal": "lightgray",
    }

    order = ["Critical", "Major", "Minor", "Accepted", "Check", "Normal"]

    for label in order:
        subset = df_plot[df_plot["VisualPriority"] == label]
        if subset.empty:
            continue

        hover_text = (
            "GUID: " + subset["GUID"].astype(str) +
            "<br>Type: " + subset["IFC_Type"].astype(str) +
            "<br>Nom: " + subset["Name"].astype(str) +
            "<br>Priorité: " + subset["VisualPriority"].astype(str)
        )

        fig.add_trace(go.Scatter3d(
            x=subset["x"],
            y=subset["y"],
            z=subset["z"],
            mode="markers",
            name=label,
            text=hover_text,
            hoverinfo="text",
            marker=dict(
                size=3 if label == "Normal" else 5,
                color=color_map[label],
                opacity=0.9
            )
        ))

    fig.update_layout(
        height=700,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(itemsizing="constant")
    )
    return fig


