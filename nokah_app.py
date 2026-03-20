import streamlit as st
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

import base64

from dataset_builder import update_dataset, load_dataset
from discipline_detector import detect_discipline, discipline_badge
from model_scope_detector import detect_model_scope, scope_badge, scope_color
from rules_mep import run_mep_rules
from rules_structure import run_structure_rules
from rules_archi_advanced import run_archi_advanced_rules
from bim_json_builder import build_bim_json, bim_json_to_string
from bim_ai_v1 import run_anomaly_detection
from bim_summary import generate_summary

LOGO_ICON   = Path("path3.png")
LOGO_TEXT   = Path("path43.png")
CONFIG_FILE = "rules_config.json"

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
            "<br>Priority: " + subset["VisualPriority"].astype(str)
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


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="nokah — BIM Quality Intelligence",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Session state init (must be before any st.* call) ────────────────────────
if "nk_done" not in st.session_state:
    st.session_state.nk_done = False
if "nk_lang" not in st.session_state:
    st.session_state.nk_lang = "EN"
if "nk_file" not in st.session_state:
    st.session_state.nk_file = None
if "nk_chat_history" not in st.session_state:
    st.session_state.nk_chat_history = []

def _t(en, fr):
    return fr if st.session_state.nk_lang == "FR" else en

# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #F8FAFC; }
[data-testid="stHeader"] { display: none; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.nk-nav {
    display: flex; align-items: center; gap: 10px;
    padding: 0.75rem 2rem; background: #0F172A;
    border-bottom: 0.5px solid #1E3A8A;
    position: sticky; top: 0; z-index: 100;
}
.nk-baseline {
    font-size: 13px; color: #475569;
    letter-spacing: 0.07em; text-transform: uppercase;
    margin-left: 0;
}
.nk-landing { max-width: 720px; margin: 0 auto; padding: 5rem 2rem 4rem; text-align: center; }
.nk-tag { font-size: 14px; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; color: #22D3EE; margin-bottom: 1.25rem; }
.nk-hero { font-size: 42px; font-weight: 500; color: #F1F5F9; line-height: 1.3; margin-bottom: 1rem; }
.nk-sub { font-size: 18px; color: #94A3B8; line-height: 1.7; margin-bottom: 3rem; }
.nk-hint { font-size: 14px; color: #475569; margin-top: 1rem; text-align: center; }
.nk-upload-pills {
    display: flex; gap: 8px; align-items: center; justify-content: center;
    margin-bottom: 0.5rem;
}
.nk-upload-pill {
    font-size: 12px; padding: 4px 12px; border-radius: 999px;
    border: 0.5px solid #334155; color: #64748B; background: #1E293B;
}
.nk-card { background: #1E293B; border-radius: 12px; border: 0.5px solid #334155; padding: 1.5rem 1.75rem; margin-bottom: 1rem; }
.nk-card-title { font-size: 13px; font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; color: #64748B; margin-bottom: 1rem; }
.score-big { font-size: 48px; font-weight: 500; line-height: 1; }
.score-red { color: #F87171; } .score-orange { color: #FB923C; }
.score-yellow { color: #FCD34D; } .score-green { color: #4ADE80; }
.nk-pill { display: inline-block; padding: 5px 14px; border-radius: 999px; font-size: 13px; font-weight: 500; margin-right: 6px; margin-bottom: 6px; }
.pill-disc { background: #1E3A8A; color: #93C5FD; }
.pill-scope { background: #14532D; color: #86EFAC; }
.pill-scope-warn { background: #78350F; color: #FCD34D; }
.pill-mixed { background: #4C1D95; color: #C4B5FD; }
.issue-row { display: flex; align-items: flex-start; gap: 10px; padding: 12px 14px; border-radius: 8px; border: 0.5px solid #334155; background: #0F172A; margin-bottom: 8px; }
.issue-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; margin-top: 5px; }
.dot-c { background: #EF4444; } .dot-m { background: #F59E0B; } .dot-n { background: #60A5FA; }
.issue-title { font-size: 15px; font-weight: 500; color: #E2E8F0; margin-bottom: 4px; }
.issue-desc { font-size: 13px; color: #64748B; line-height: 1.5; }
.issue-badge { font-size: 12px; padding: 3px 10px; border-radius: 999px; font-weight: 500; white-space: nowrap; flex-shrink: 0; align-self: flex-start; margin-top: 2px; }
.badge-c { background: #FEE2E2; color: #991B1B; }
.badge-m { background: #FEF3C7; color: #92400E; }
.badge-d { background: #DBEAFE; color: #1E40AF; }
.ai-card { border: 0.5px solid #1E3A8A; border-radius: 12px; padding: 1.5rem 1.75rem; background: #1E293B; margin-bottom: 1rem; }
.ai-header { display: flex; align-items: center; gap: 8px; margin-bottom: 0.75rem; }
.ai-icon { width: 28px; height: 28px; border-radius: 50%; background: #1E3A8A; display: flex; align-items: center; justify-content: center; }
.ai-title { font-size: 16px; font-weight: 500; color: #E2E8F0; }
.ai-sub { font-size: 13px; color: #475569; }
.ai-body { font-size: 15px; color: #94A3B8; line-height: 1.7; }
.bench-badge { display: inline-block; padding: 7px 18px; border-radius: 999px; font-size: 15px; font-weight: 500; margin-bottom: 0.75rem; }
.bb-excellent { background: #DCFCE7; color: #14532D; }
.bb-good { background: #D1FAE5; color: #065F46; }
.bb-average { background: #DBEAFE; color: #1E40AF; }
.bb-below { background: #FEF3C7; color: #92400E; }
.atypie-wrap { display: flex; align-items: center; gap: 10px; margin-top: 4px; }
.atypie-track { flex: 1; height: 6px; background: #F1F5F9; border-radius: 999px; overflow: hidden; }
.atypie-fill { height: 100%; border-radius: 999px; }
.atypie-val { font-size: 13px; font-weight: 500; color: #0F172A; min-width: 44px; text-align: right; }
.atypie-lbl { font-size: 12px; color: #64748B; min-width: 72px; }
.nk-div { height: 0.5px; background: #334155; margin: 0.875rem 0; }

/* Style native uploader as a pill chat bar */
[data-testid="stFileUploader"] {
    max-width: 560px;
    margin: 0 auto;
}
[data-testid="stFileUploader"] {
    max-width: 580px !important;
    margin: 0 auto !important;
}
[data-testid="stFileUploaderDropzone"] {
    border: 1.5px solid #378ADD !important;
    border-radius: 999px !important;
    background: #1E293B !important;
    padding: 14px 16px 14px 28px !important;
    min-height: unset !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div > small {
    font-size: 0 !important;
    color: transparent !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div > small::before {
    content: "Upload or drop an IFC file..." !important;
    font-size: 16px !important;
    color: #64748B !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div > span {
    display: none !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] svg {
    display: none !important;
}
[data-testid="stFileUploaderDropzone"] button {
    border-radius: 999px !important;
    background: #1E3A8A !important;
    color: white !important;
    border: none !important;
    padding: 10px 24px !important;
    font-size: 15px !important;
    font-weight: 500 !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background: #1e40af !important;
}
/* Background bleu nuit */
body, [data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="block-container"],
.main, section.main {
    background: #0F172A !important;
}
[data-testid="stHeader"] { background: #0F172A !important; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS
# =========================================================

def _img_b64(path: Path) -> str:
    if path.exists():
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

def _score_color(score: float) -> str:
    if score >= 75: return "score-green"
    if score >= 60: return "score-yellow"
    if score >= 40: return "score-orange"
    return "score-red"

def _score_hex(score: float) -> str:
    if score >= 75: return "#16A34A"
    if score >= 60: return "#CA8A04"
    if score >= 40: return "#D97706"
    return "#DC2626"

def _score_en(score: float) -> str:
    if score >= 90: return "Excellent"
    if score >= 75: return "Good"
    if score >= 60: return "Medium"
    if score >= 40: return "Low"
    return "Critical"

def _disc_en(disc: str) -> str:
    if st.session_state.get("nk_lang") == "FR":
        return {"Architecture": "Architecture", "MEP": "MEP / CVC",
                "Structure": "Structure", "Interior": "Interior",
                "Unknown": "Inconnu"}.get(disc, disc)
    return {"Architecture": "Architecture", "MEP": "MEP / HVAC",
            "Structure": "Structure", "Interior": "Interior",
            "Unknown": "Unknown"}.get(disc, disc)

def _scope_label(scope: str) -> str:
    if st.session_state.get("nk_lang") == "FR":
        return {
            "BuildingModel":       "🏢 Building model",
            "DisciplineSubmodel":  "📐 Discipline submodel",
            "ObjectOnly":          "🔩 Object only",
            "InvalidForBenchmark": "⚠️ Non benchmarkable",
        }.get(scope, scope)
    return {
        "BuildingModel":       "🏢 Building model",
        "DisciplineSubmodel":  "📐 Discipline submodel",
        "ObjectOnly":          "🔩 Object only",
        "InvalidForBenchmark": "⚠️ Not benchmarkable",
    }.get(scope, scope)

def _priority_classes(p: str):
    if p == "Critical": return "dot-c", "badge-c", "Critical"
    if p == "Major":    return "dot-m", "badge-m", "Major"
    return "dot-n", "badge-d", "Data"

def _bench_badge(position: str, score_global: float, score_moyen: float):
    delta = score_global - score_moyen
    if score_global >= 90:   return "bb-excellent", _t("Top range of analyzed BIM models", "Meilleure catégorie du parc")
    elif delta >= 5:          return "bb-good",      _t("Above typical BIM quality level", "Au-dessus du niveau BIM typique")
    elif delta >= -10:        return "bb-average",   _t("Within typical BIM quality range", "Dans la norme BIM typique")
    else:                     return "bb-below",     _t("Below typical BIM quality level", "En-dessous du niveau BIM typique")

def _percentile_msg(score_global: float, df_dataset) -> str:
    if df_dataset is None or df_dataset.empty or len(df_dataset) < 2:
        return ""
    import numpy as np
    scores = df_dataset["score_global"].values
    pct = int(round((scores < score_global).sum() / len(scores) * 100))
    if st.session_state.get("nk_lang") == "FR":
        if pct >= 75:   return f"Your model ranks in the top {100 - pct}% of analyzed BIM models."
        elif pct >= 50: return f"Your model is above {pct}% of analyzed BIM models."
        else:           return f"Your model quality is in the lower {100 - pct}% of analyzed BIM models."
    if pct >= 75:   return f"Your model ranks in the top {100 - pct}% of analyzed BIM models."
    elif pct >= 50: return f"Your model is above {pct}% of analyzed BIM models."
    else:           return f"Your model quality is in the lower {100 - pct}% of analyzed BIM models."

# =========================================================
# NAV
# =========================================================

icon_b64 = _img_b64(LOGO_ICON)
text_b64 = _img_b64(LOGO_TEXT)
icon_html = f'<img src="data:image/png;base64,{icon_b64}" style="height:144px;width:144px;object-fit:contain"/>' if icon_b64 else '<div style="width:72px;height:72px;border-radius:50%;background:#1E3A8A"></div>'
text_html = f'<img src="data:image/png;base64,{text_b64}" style="height:32px;object-fit:contain;margin-left:4px"/>' if text_b64 else '<span style="font-size:28px;font-weight:500;color:#1E3A8A">nokah</span>'

st.markdown(f'''
<div class="nk-nav">
    {icon_html}
    <div style="display:flex;flex-direction:column;justify-content:center;gap:2px">
        {text_html}
        <span class="nk-baseline">BIM Quality Intelligence</span>
    </div>
</div>
''', unsafe_allow_html=True)

# =========================================================
# LANDING
# =========================================================

if "nk_done" not in st.session_state:
    st.session_state.nk_done = False

if not st.session_state.nk_done:
    st.markdown('''
    <div class="nk-landing">
        <div class="nk-tag">Powered by AI</div>
        <div class="nk-hero">Your BIM model,<br>analyzed in seconds</div>
        <div class="nk-sub">Drop an IFC file — nokah detects the discipline,
        scores quality, identifies issues, and benchmarks your model
        against similar projects.</div>
    </div>
    ''', unsafe_allow_html=True)

    _, col_c, _ = st.columns([1, 2, 1])
    with col_c:
        st.markdown('''
        <div class="nk-upload-pills">
            <span class="nk-upload-pill">Architecture</span>
            <span class="nk-upload-pill">MEP / HVAC</span>
            <span class="nk-upload-pill">Structure</span>
            <span class="nk-upload-pill">Mixed</span>
        </div>
        ''', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload IFC", type=["ifc"], label_visibility="collapsed",
            key="nk_uploader")
        st.markdown('<div class="nk-hint">Supports IFC 2x3 and IFC 4</div>', unsafe_allow_html=True)

    if uploaded is not None:
        st.session_state.nk_file = uploaded
        st.session_state.nk_done = True
        st.rerun()
    st.stop()

# =========================================================
# ANALYSIS
# =========================================================

uploaded = st.session_state.get("nk_file")
if uploaded is None:
    st.session_state.nk_done = False
    st.rerun()

# Cache analysis in session state to avoid re-reading file on rerun
_cache_key = "nk_analysis_" + (uploaded.name if uploaded else "")
if _cache_key not in st.session_state or st.session_state.get("nk_last_file") != uploaded.name:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ifc") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name
    with st.spinner(_t("Analyzing your BIM model...", "Analyse de votre maquette BIM...")):
        _analysis = analyze_ifc(tmp_path, rule_config)
        st.session_state[_cache_key] = _analysis
        st.session_state["nk_tmp_path"] = tmp_path
        st.session_state["nk_last_file"] = uploaded.name

analysis = st.session_state[_cache_key]
tmp_path  = st.session_state.get("nk_tmp_path", "")

df_scores  = analysis["scores"]
df_all     = analysis["all_results"]
df_top     = analysis["top_anomalies"]
df_viewer  = analysis["viewer"]
counts     = analysis["counts"]
discipline = analysis["discipline"]
scope_info = analysis["scope"]

score_map = {row["ScoreName"]: row["ScoreValue"] for _, row in df_scores.iterrows()}

# Dataset update
ds_result = update_dataset(
    ifc_name=uploaded.name, ifc_path=tmp_path,
    scores=score_map, counts=counts, df_results=df_all,
    scope_info=scope_info, discipline_info=discipline, force=False
)

# Load benchmark-eligible dataset
df_full = load_dataset()
if not df_full.empty and "is_benchmark_eligible" in df_full.columns:
    df_bench = df_full[df_full["is_benchmark_eligible"] == True].copy()
else:
    df_bench = df_full.copy()

# AI outputs
bim_json   = build_bim_json(ifc_name=uploaded.name, discipline=discipline,
                             score_map=score_map, counts=counts,
                             df_all=df_all, df_top=df_top, df_dataset=df_bench)
ai_result  = run_anomaly_detection(df_bench, bim_json)

# Values
score_global = score_map.get("Global", 0)
score_metier = score_map.get("Metier", 0)
score_data   = score_map.get("Data BIM", 0)
primary      = discipline.get("primary", "Unknown")
secondary    = discipline.get("secondary", [])
is_mixed     = discipline.get("is_mixed", False)
scope        = scope_info.get("scope", "Unknown")
is_eligible  = scope_info.get("is_benchmark_eligible", True)
benchmark    = bim_json.get("benchmark")
errors       = bim_json.get("errors", {})

# =========================================================
# =========================================================
# RESULTS — Chat interface
# =========================================================

# ── CSS chat bubbles ──────────────────────────────────────────────────────────
st.markdown('''
<style>
.chat-wrap { max-width: 760px; margin: 0 auto; padding: 2rem 1.5rem; display: flex; flex-direction: column; gap: 20px; }
.chat-row { display: flex; width: 100%; }
.chat-row-right { justify-content: flex-end; }
.chat-row-left  { justify-content: flex-start; }

/* Client bubble — right */
.bubble-client {
    background: #1E3A8A;
    border: 1.5px solid #378ADD;
    border-radius: 18px 18px 4px 18px;
    padding: 14px 20px;
    max-width: 60%;
}
.bubble-client .bc-filename { font-size: 16px; font-weight: 500; color: #E2E8F0; margin-bottom: 8px; }
.bubble-client .bc-pills { display: flex; gap: 6px; flex-wrap: wrap; }
.bubble-client .bc-pill { font-size: 12px; padding: 3px 10px; border-radius: 999px; background: #0F172A; color: #93C5FD; border: 0.5px solid #378ADD; }

/* nokah bubble — left */
.bubble-nokah {
    background: #1E293B;
    border: 0.5px solid #334155;
    border-radius: 18px 18px 18px 4px;
    padding: 18px 22px;
    max-width: 85%;
}
.bk-label { font-size: 12px; font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; color: #475569; margin-bottom: 12px; display: flex; align-items: center; gap: 6px; }
.bk-icon { width: 20px; height: 20px; border-radius: 50%; background: #1E3A8A; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }

/* Score */
.nk-score-big { font-size: 48px; font-weight: 500; line-height: 1; margin-bottom: 6px; }
.nk-score-sub { font-size: 17px; color: #475569; margin-bottom: 14px; }
.nk-score-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.nk-score-item { background: #0F172A; border-radius: 8px; padding: 12px 14px; border: 0.5px solid #334155; }
.nk-score-item-lbl { font-size: 12px; color: #475569; margin-bottom: 4px; }
.nk-score-item-val { font-size: 22px; font-weight: 500; }
.nk-score-item-sub { font-size: 12px; color: #475569; margin-top: 3px; }

/* AI summary */
.nk-ai-body { font-size: 15px; color: #94A3B8; line-height: 1.75; margin-bottom: 14px; }
.nk-bench-badge { display: inline-block; padding: 5px 16px; border-radius: 999px; font-size: 13px; font-weight: 500; margin-bottom: 8px; }
.nk-bench-msg { font-size: 14px; color: #475569; line-height: 1.6; }

/* Issues */
.nk-issue-row { display: flex; align-items: flex-start; gap: 10px; padding: 10px 12px; border-radius: 8px; border: 0.5px solid #334155; background: #0F172A; margin-bottom: 8px; }
.nk-issue-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; margin-top: 5px; }
.nd-c { background: #F87171; } .nd-m { background: #FB923C; } .nd-d { background: #60A5FA; }
.nk-issue-title { font-size: 14px; font-weight: 500; color: #E2E8F0; margin-bottom: 3px; }
.nk-issue-desc  { font-size: 12px; color: #64748B; line-height: 1.4; }
.nk-issue-badge { font-size: 11px; padding: 2px 9px; border-radius: 999px; font-weight: 500; white-space: nowrap; flex-shrink: 0; align-self: flex-start; margin-top: 2px; }
.nib-c { background: #450A0A; color: #F87171; }
.nib-m { background: #431407; color: #FB923C; }
.nib-d { background: #172554; color: #93C5FD; }

/* Atypie */
.nk-atypie-row { display: flex; align-items: center; gap: 12px; margin-top: 8px; width: 100%; }
.nk-atypie-lbl { font-size: 14px; font-weight: 500; min-width: 80px; flex-shrink: 0; }
.nk-atypie-lbl.nk-atypie-normal { color: #4ADE80; }
.nk-atypie-lbl.nk-atypie-atypical { color: #FB923C; }
.nk-atypie-lbl.nk-atypie-very-atypical { color: #F87171; }
.nk-atypie-track { flex: 1; min-width: 0; height: 8px; background: #0F172A; border-radius: 999px; overflow: hidden; border: 0.5px solid #334155; }
.nk-atypie-fill { height: 100%; border-radius: 999px; background: #4ADE80; }
.nk-atypie-fill.nk-atypie-bar-normal { background: #4ADE80; }
.nk-atypie-fill.nk-atypie-bar-atypical { background: #FB923C; }
.nk-atypie-fill.nk-atypie-bar-very-atypical { background: #F87171; }
.nk-atypie-val { font-size: 14px; font-weight: 500; color: #E2E8F0; min-width: 36px; flex-shrink: 0; text-align: right; }
.nk-atypie-note { font-size: 12px; color: #475569; margin-top: 8px; }

/* Timestamp */
.nk-ts { font-size: 11px; color: #334155; margin-top: 5px; }
.nk-ts-right { text-align: right; }

/* Upload another — right bubble */
.bubble-upload-another {
    background: #1E3A8A;
    border: 1.5px solid #378ADD;
    border-radius: 18px 18px 4px 18px;
    padding: 14px 22px;
    display: inline-flex; align-items: center; gap: 10px;
    cursor: pointer;
}
.bua-text { font-size: 15px; font-weight: 500; color: #E2E8F0; }
.bua-icon { width: 32px; height: 32px; border-radius: 50%; background: #0F172A; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
</style>
''', unsafe_allow_html=True)

# ── Build bubble data ─────────────────────────────────────────────────────────
disc_pills_html = f'<span class="bc-pill">{_disc_en(primary)}</span>'
if is_mixed:
    for s in secondary[:2]:
        disc_pills_html += f'<span class="bc-pill">+ {_disc_en(s)}</span>'
scope_cls_pill = "pill-scope" if is_eligible else "pill-scope-warn"
disc_pills_html += '<span class="bc-pill">' + _scope_label(scope) + '</span>'

# Duplicate / not eligible notices (subtle)
notices = ""
if ds_result["status"] == "duplicate_blocked":
    notices = '<div style="text-align:center;font-size:13px;color:#378ADD;margin-bottom:0.5rem">Previously analyzed — showing cached results</div>'
if not is_eligible:
    notices += '<div style="text-align:center;font-size:13px;color:#FCD34D;margin-bottom:0.5rem">Model not included in benchmark comparisons</div>'

# AI summary text
errors_n = bim_json.get("errors", {})
crit_n   = errors_n.get("critical", 0)
total_err = errors_n.get("total", 0)
if total_err == 0:
    ai_msg_html = f'This <strong style="color:#E2E8F0">{_disc_en(primary).lower()}</strong> model has no detected issues — fully compliant with all configured rules.'
else:
    if score_metier < score_data - 15:
        dom = _t("Issues are primarily <strong style=\"color:#E2E8F0\">conformity-related</strong> rather than data quality.", "Les problèmes sont principalement liés à la <strong style=\"color:#E2E8F0\">conformité</strong> plutôt qu'à la qualité des données.")
    elif score_data < score_metier - 15:
        dom = "Issues are primarily <strong style=\"color:#E2E8F0\">data quality related</strong> rather than conformity."
    else:
        dom = "Conformity and data quality issues are <strong style=\"color:#E2E8F0\">relatively balanced</strong>."
    crit_html = f' <strong style="color:#F87171">{crit_n} critical anomaly(-ies)</strong> require immediate attention.' if crit_n > 0 else ""
    ai_msg_html = f'This <strong style="color:#E2E8F0">{_disc_en(primary).lower()}</strong> model scores <strong style="color:#E2E8F0">{score_global:.1f}/100</strong>. {dom}{crit_html}'

# Benchmark bubble content
if benchmark and is_eligible:
    bb_cls, bb_msg = _bench_badge(benchmark.get("position",""), score_global, benchmark.get("score_moyen", 50))
    pct_txt = _percentile_msg(score_global, df_bench)
    bench_html = f'''
    <span class="nk-bench-badge {bb_cls}">{bb_msg}</span>
    <div class="nk-bench-msg">{pct_txt}</div>'''
elif not is_eligible:
    bench_html = '<div style="font-size:14px;color:#475569">Not included in benchmark comparisons.</div>'
else:
    bench_html = '<div style="font-size:14px;color:#475569">Not enough data for benchmark yet.</div>'

# Score color
sc_color = _score_hex(score_global)
sm_color = _score_hex(score_metier)
sd_color = _score_hex(score_data)

# Issues HTML
issues_html = ""
if df_top.empty:
    issues_html = '<div style="font-size:14px;color:#4ADE80">No issues detected — model is fully compliant.</div>'
else:
    shown = {}
    for _, row in df_top.iterrows():
        msg = row.get("Message","")
        prio = row.get("Priority","Minor")
        if msg not in shown:
            shown[msg] = {"priority": prio, "suggestion": row.get("Suggestion",""), "count": 1}
        else:
            shown[msg]["count"] += 1
        if len(shown) >= 6: break
    for msg, d in shown.items():
        if d["priority"] == "Critical":    dc, bc, bl = "nd-c","nib-c","Critical"
        elif d["priority"] == "Major":     dc, bc, bl = "nd-m","nib-m","Major"
        else:                               dc, bc, bl = "nd-d","nib-d","Data"
        cnt = f" · {d['count']}x" if d["count"] > 1 else ""
        sug = d["suggestion"][:120] + ("..." if len(d["suggestion"]) > 120 else "")
        issues_html += f'''<div class="nk-issue-row">
            <div class="nk-issue-dot {dc}"></div>
            <div style="flex:1">
                <div class="nk-issue-title">{msg}</div>
                <div class="nk-issue-desc">{sug}{cnt}</div>
            </div>
            <span class="nk-issue-badge {bc}">{bl}</span>
        </div>'''

# ── RENDER CHAT — each bubble in its own st.markdown ─────────────────────────

# Shared wrapper CSS reset
_W = '<div class="chat-wrap" style="padding-bottom:0">'
_WE = '</div>'

# Notices
if notices:
    st.markdown(_W + notices + _WE, unsafe_allow_html=True)

# ── Bubble 1: CLIENT — IFC file (right) ──────────────────────────────────────
_b1 = (
    _W
    + '<div class="chat-row chat-row-right"><div>'
    + '<div class="bubble-client">'
    + '<div class="bc-filename">' + uploaded.name + '</div>'
    + '<div class="bc-pills">' + disc_pills_html + '</div>'
    + '</div>'
    + '<div class="nk-ts nk-ts-right">Just now</div>'
    + '</div></div>'
    + _WE
)
st.markdown(_b1, unsafe_allow_html=True)

# ── Bubble 2: nokah — AI summary + benchmark (left) ──────────────────────────
_svg_icon = (
    '<div class="bk-icon">'
    '<svg width="11" height="11" viewBox="0 0 11 11" fill="none">'
    '<circle cx="5.5" cy="5.5" r="4" stroke="#22D3EE" stroke-width="1.1"/>'
    '<path d="M3.5 5.5h4M5.5 3.5v4" stroke="#22D3EE" stroke-width="1.1" stroke-linecap="round"/>'
    '</svg></div>'
)
_b2 = (
    _W
    + '<div class="chat-row chat-row-left"><div style="max-width:85%">'
    + '<div class="bubble-nokah">'
    + '<div class="bk-label">' + _svg_icon + 'nokah intelligence</div>'
    + '<div class="nk-ai-body">' + ai_msg_html + '</div>'
    + '<div class="nk-div"></div>'
    + bench_html
    + '</div>'
    + '<div class="nk-ts">nokah &middot; just now</div>'
    + '</div></div>'
    + _WE
)
st.markdown(_b2, unsafe_allow_html=True)

# ── Bubble 3: nokah — Scores (left) ──────────────────────────────────────────
_b3 = (
    _W
    + '<div class="chat-row chat-row-left"><div style="max-width:75%">'
    + '<div class="bubble-nokah">'
    + '<div class="bk-label">Quality scores</div>'
    + '<div class="nk-score-big" style="color:' + sc_color + '">' + f"{score_global:.1f}" + '</div>'
    + '<div class="nk-score-sub">/ 100 &nbsp;&middot;&nbsp; ' + _score_en(score_global) + '</div>'
    + '<div class="nk-score-grid">'
    + '<div class="nk-score-item">'
    + '<div class="nk-score-item-lbl">' + _disc_en(primary) + '</div>'
    + '<div class="nk-score-item-val" style="color:' + sm_color + '">' + f"{score_metier:.1f}" + '</div>'
    + '<div class="nk-score-item-sub">' + _score_en(score_metier) + '</div>'
    + '</div>'
    + '<div class="nk-score-item">'
    + '<div class="nk-score-item-lbl">Data BIM</div>'
    + '<div class="nk-score-item-val" style="color:' + sd_color + '">' + f"{score_data:.1f}" + '</div>'
    + '<div class="nk-score-item-sub">' + _score_en(score_data) + '</div>'
    + '</div>'
    + '</div>'
    + '</div></div></div>'
    + _WE
)
st.markdown(_b3, unsafe_allow_html=True)

# ── Bubble 4: nokah — Key issues (left) ──────────────────────────────────────
_b4 = (
    _W
    + '<div class="chat-row chat-row-left"><div style="max-width:90%">'
    + '<div class="bubble-nokah">'
    + '<div class="bk-label">Key issues</div>'
    + issues_html
    + '</div></div></div>'
    + _WE
)
st.markdown(_b4, unsafe_allow_html=True)

# ── Bubble 5: nokah — Atypicality ───────────────────────────────────────────
if ai_result and ai_result.get("available"):
    ai_score = ai_result.get("anomaly_score", 0)
    lbl      = ai_result.get("label", "")
    conf     = ai_result.get("confidence","").capitalize()
    n_mod    = ai_result.get("n_models", 0)
    lbl_map  = {"Normal": ("Normal", "#4ADE80"), "Atypique": ("Atypical", "#FB923C"), "Très atypique": ("Very atypical", "#F87171")}
    lbl_en, lbl_hex = lbl_map.get(lbl, (lbl, "#64748B"))
    bar_pct = int(ai_score * 100)
    # Atypie — SVG with no inline styles (all attributes native)
    _bar = int(ai_score * 300)
    _b5 = (
        '<div class="chat-wrap" style="padding-top:0;padding-bottom:0">'
        '<div class="chat-row chat-row-left"><div style="max-width:85%">'
        '<div class="bubble-nokah">'
        '<div class="bk-label">Atypicality index</div>'
        '<svg width="100%" height="28" viewBox="0 0 460 28" xmlns="http://www.w3.org/2000/svg">'
        '<text x="0" y="20" font-size="15" font-weight="bold" fill="' + lbl_hex + '">' + lbl_en + '</text>'
        '<rect x="115" y="10" width="270" height="8" rx="4" fill="#1E3A8A"/>'
        '<rect x="115" y="10" width="' + str(int(ai_score * 270)) + '" height="8" rx="4" fill="' + lbl_hex + '"/>'
        '<text x="460" y="20" font-size="14" font-weight="bold" fill="#E2E8F0" text-anchor="end">' + f"{ai_score:.2f}" + '</text>'
        '</svg>'
        '<p style="font-size:12px;color:#475569;margin:4px 0 0 0">Confidence: ' + conf + '</p>'
        '</div></div></div></div>'
    )
    st.markdown(_b5, unsafe_allow_html=True)

# ── Bubble 6: nokah — 3D Viewer ─────────────────────────────────────────────
st.markdown(
    '<div class="chat-wrap" style="padding-top:0;padding-bottom:0">'
    '<div class="chat-row chat-row-left"><div style="max-width:95%;width:95%">'
    '<div class="bubble-nokah">'
    '<div class="bk-label">3D viewer &mdash; anomalies highlighted</div>',
    unsafe_allow_html=True
)
st.caption("Red = Critical · Orange = Major · Yellow = Minor · Cyan = Accepted · Gray = Normal")
fig = make_3d_figure(df_viewer, df_all)
fig.update_layout(height=480, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0))
st.plotly_chart(fig, width="stretch")
st.markdown(
    '</div></div></div></div>',
    unsafe_allow_html=True
)

# ── Bubble 7 + 8: CSS-only approach — Streamlit widgets can't be inside HTML ──
# We constrain width and alignment via CSS targeting Streamlit's own containers

st.markdown('''<style>
/* Expert mode expander — constrain to chat width, left-aligned */
[data-testid="stExpander"] {
    max-width: 650px !important;
    margin-left: calc((100% - 760px) / 2) !important;
    background: #1E293B !important;
    border: 0.5px solid #334155 !important;
    border-radius: 18px 18px 18px 4px !important;
    overflow: hidden !important;
    margin-bottom: 1rem !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] > details > summary {
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: #475569 !important;
    padding: 14px 18px !important;
    background: #1E293B !important;
}
[data-testid="stExpanderDetails"] {
    padding: 0 18px 16px !important;
    background: #1E293B !important;
}

/* Upload button — right aligned, pill style, no text wrapping */
div[data-testid="stButton"] > button {
    border-radius: 18px 18px 4px 18px !important;
    border: 1.5px solid #378ADD !important;
    color: #E2E8F0 !important;
    background: #1E3A8A !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    padding: 12px 24px !important;
    white-space: nowrap !important;
    display: block !important;
    width: auto !important;
    text-align: center !important;
}
div[data-testid="stButton"] > button:hover {
    background: #1e40af !important;
}
</style>''', unsafe_allow_html=True)

with st.expander(_t("Expert mode — full results, JSON export, dataset stats",
                     "Expert mode — full results, JSON export, dataset stats")):
    st.subheader(_t("Scores", "Scores"))
    ec1, ec2, ec3, ec4 = st.columns(4)
    ec1.metric("Global",           f"{score_global:.2f}")
    ec2.metric(_disc_en(primary),  f"{score_metier:.2f}")
    ec3.metric("Data BIM",         f"{score_data:.2f}")
    ec4.metric("Accepted/Check",   f"{score_map.get('Accepted_Check', 0):.2f}")
    st.divider()
    st.subheader(_t("All anomalies", "Toutes les anomalies"))
    if not df_all.empty:
        cols = [c for c in ["RuleID","Category","Priority","Bucket","Status","IFC_Type","GUID","Name","Storey","Message","Suggestion"] if c in df_all.columns]
        ef1, ef2 = st.columns(2)
        sel_b = ef1.selectbox(_t("Bucket","Bucket"),   ["All"] + sorted(df_all["Bucket"].dropna().unique().tolist()),   key="xb")
        sel_p = ef2.selectbox(_t("Priority","Priorité"), ["All"] + sorted(df_all["Priority"].dropna().unique().tolist()), key="xp")
        filt = df_all.copy()
        if sel_b != "All": filt = filt[filt["Bucket"] == sel_b]
        if sel_p != "All": filt = filt[filt["Priority"] == sel_p]
        st.write(f"{len(filt)} " + _t("results","résultats"))
        st.dataframe(filt[cols], width="stretch")
    st.divider()
    if benchmark:
        st.subheader(_t("Benchmark detail","Détail benchmark"))
        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric(_t("Park average","Moyenne parc"), f"{benchmark.get('score_moyen',0):.1f}")
        bc2.metric(_t("Park min","Min parc"),         f"{benchmark.get('score_min',0):.1f}")
        bc3.metric(_t("Park max","Max parc"),         f"{benchmark.get('score_max',0):.1f}")
        bc4.metric(_t("Models","Modèles"),            benchmark.get('nb_models', 0))
        # JSON export removed from UI (internal use only)

# Upload another IFC → use + button in sticky chat bar
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ── CHAT BAR ──────────────────────────────────────────────────────────────────
import os as _os
try:
    from nokah_chat import get_chat_response
    _chat_ok = True
except ImportError:
    _chat_ok = False

# Pass Groq key via env so nokah_chat.py can access it
try:
    _gk = st.secrets.get("GROQ_API_KEY", "")
    if _gk: _os.environ["GROQ_API_KEY"] = _gk
except Exception:
    pass

# Build real context from bim_json
_errors_ctx = bim_json.get("errors", {})
_top_issues_ctx = []
_bj_top = bim_json.get("top_issues", [])
for _r in _bj_top[:5]:
    if isinstance(_r, dict):
        _m = _r.get("message","") or _r.get("Message","") or _r.get("title","") or str(_r)
        if _m: _top_issues_ctx.append(str(_m)[:100])
    elif isinstance(_r, str):
        _top_issues_ctx.append(_r[:100])

_bench_ctx = ""
if benchmark: _bench_ctx = benchmark.get("position","")

_atypie_ctx = ""
if ai_result and ai_result.get("available"):
    _lmap = {"Normal":"Normal","Atypique":"Atypical","Très atypique":"Very atypical"}
    _atypie_ctx = _lmap.get(ai_result.get("label",""), "")

_chat_ctx = {
    "filename": uploaded.name,
    "discipline": _disc_en(primary),
    "score_global": round(score_global, 1),
    "score_metier": round(score_metier, 1),
    "score_data_bim": round(score_data, 1),
    "n_critical": _errors_ctx.get("critical", 0),
    "n_major": _errors_ctx.get("major", 0),
    "n_minor": _errors_ctx.get("minor", 0),
    "top_issues": _top_issues_ctx,
    "benchmark_position": _bench_ctx,
    "atypie_label": _atypie_ctx,
    "objects": bim_json.get("objects", {}),
}

st.markdown(
    '<div class="chat-wrap" style="padding-top:0;padding-bottom:0">' +
    '<div class="chat-row chat-row-left"><div style="max-width:85%">' +
    '<div class="bubble-nokah">' +
    '<div class="bk-label">nokah — ' + _t("Ask anything about your model","Posez vos questions sur votre maquette") + '</div>' +
    '<div style="font-size:13px;color:#94A3B8">' +
    _t("Issues · Score · What to fix · Norms · Compare to other models",
       "Anomalies · Score · Quoi corriger · Normes · Comparer aux autres maquettes") +
    '</div></div></div></div></div>',
    unsafe_allow_html=True
)


for _msg in st.session_state.nk_chat_history:
    if _msg["role"] == "user":
        st.markdown(
            '<div class="chat-wrap" style="padding-top:0;padding-bottom:4px">' +
            '<div class="chat-row chat-row-right">' +
            '<div class="bubble-client" style="max-width:70%">' +
            f'<p style="margin:0;font-size:14px">{_msg["content"]}</p>' +
            '</div></div></div>', unsafe_allow_html=True)
    else:
        import re as _re
        _fmt = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', _msg["content"])
        _fmt = _fmt.replace('\n', '<br>')
        _badge = '<span style="font-size:10px;color:#22D3EE;float:right">⚡ AI</span>' if _msg.get("source") == "groq" else ""
        st.markdown(
            '<div class="chat-wrap" style="padding-top:0;padding-bottom:4px">' +
            '<div class="chat-row chat-row-left">' +
            '<div class="bubble-nokah" style="max-width:85%">' +
            _badge + f'<p style="margin:0;font-size:14px;line-height:1.7">{_fmt}</p>' +
            '</div></div></div>', unsafe_allow_html=True)

# ── CHAT SECTION ─────────────────────────────────────────────────────────────
st.markdown("""<style>
div[data-testid="stChatInputSubmitButton"] button {
    background: #378ADD !important; border-radius: 50% !important;
    border: none !important; width: 34px !important; height: 34px !important;
}
div[data-testid="stChatInputSubmitButton"] button svg { display:none !important; }
div[data-testid="stChatInputSubmitButton"] button::after {
    content: "\u2191"; color: white; font-size: 18px; font-weight: bold;
}
div[data-testid="stChatInput"] { max-width:640px !important; margin:0 auto !important; }
div[data-testid="stChatInput"] > div {
    border:1.5px solid #378ADD !important;
    border-radius:24px !important; background:#1E293B !important;
}
section[data-testid="stMain"] .block-container { padding-bottom:90px !important; }
/* Disclaimer below sticky bar */
div[data-testid="stBottom"]::after {
    content: "nokah est une IA et peut faire des erreurs. Veuillez vérifier les réponses.";
    display: block; text-align: center;
    font-size: 11px; color: #475569;
    padding: 2px 0 3px 0;
    background: inherit;
}
</style>""", unsafe_allow_html=True)

# (disclaimer shown below chat bar via CSS)

# Plus button
_, _plus_col = st.columns([12, 1])
with _plus_col:
    if st.button("＋", key="nk_plus", help=_t("Upload a new IFC", "Analyser un nouveau IFC")):
        st.session_state.nk_done = False
        st.session_state.nk_file = None
        st.session_state.nk_chat_history = []
        st.rerun()

# Native sticky chat input
_q = st.chat_input(
    _t("Ask about issues, score, corrections, norms...",
       "Anomalies, score, corrections, normes..."),
    key="nk_chat_input"
)

if _q and _q.strip() and _chat_ok:
    with st.spinner(_t("nokah is thinking...", "nokah réfléchit...")):
        _resp, _src = get_chat_response(_q, _chat_ctx,
            st.session_state.nk_chat_history,
            st.session_state.get("nk_lang", "EN"))
    st.session_state.nk_chat_history.append({"role": "user", "content": _q})
    st.session_state.nk_chat_history.append({"role": "assistant", "content": _resp, "source": _src})
    st.rerun()
elif _q and _q.strip() and not _chat_ok:
    st.error("nokah_chat.py not found.")