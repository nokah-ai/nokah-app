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
try:
    from nokah_chat import get_chat_response
    _chat_ok = True
except ImportError:
    _chat_ok = False

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

# ── Sticky chat CSS ──────────────────────────────────────────────────────────
st.markdown('''<style>
.nk-sticky-spacer { height: 90px; }
section[data-testid="stMain"] > div:last-child { padding-bottom: 90px; }
div[data-testid="stFixedWidthColumn"] { z-index: 999; }
</style>''', unsafe_allow_html=True)

st.markdown('<div class="nk-sticky-spacer"></div>', unsafe_allow_html=True)

# ── Sticky bar using st.container with custom CSS ────────────────────────────
_path2_b64 = """iVBORw0KGgoAAAANSUhEUgAABAAAAAQACAYAAAB/HSuDAAEAAElEQVR4nOzdd3MbSbIu/KctvPeGBEErMzO759443/8r3HjP2d3ZnZE0cqToHTzQXe8fYBULzQY9Rff8IhSiSJgG0KCQWZlZABERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERvQbGYx8AERERvUyGMf2YIYS49HKWZcE0TQgh4HnelZcnIiKi22ECgIiIiH4KwzAuJAVM04TjOLBtWyUAPM/DeDx+zEMlIiJ6kezHPgAiIiJ6HYQQKvA3DAO2bSMSicB1Xdi2Dc/zMBwO4XneIx8pERHRy8QEABEREd070zTh+z6A8FYA27aRTCaRTqcRi8UEAPR6PWMymajrERER0f1iAoCIiIjunR7Em6YJYJoAkCv/8XgcpVIJlUpFRKNRDAYD7O/vi16vZ8iEAREREd0vJgCIiIjo3hmGoVb8fd9XX9u2Ddd1UalU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AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAuEv/P5XhpwJysCK5AAAAAElFTkSuQmCC"""
_send_icon_html = f'<img src="data:image/png;base64,{_path2_b64}" style="width:26px;height:26px;filter:brightness(1.2)"/>'

# Disclaimer
_disclaimer = _t(
    "nokah is an AI and may make mistakes. Please verify responses.",
    "nokah est une IA et peut faire des erreurs. Veuillez vérifier les réponses."
)

st.markdown(f'''<style>
.nk-chat-fixed {{
    position: fixed;
    bottom: 0; left: 0; right: 0;
    z-index: 9999;
    background: #0F172A;
    border-top: 1px solid #1E3A8A;
    padding: 10px 0 6px 0;
}}
.nk-chat-fixed-inner {{
    max-width: 620px;
    margin: 0 auto;
    padding: 0 16px;
}}
.nk-chat-row {{
    display: flex;
    align-items: center;
    gap: 8px;
    background: #1E293B;
    border: 1.5px solid #378ADD;
    border-radius: 24px;
    padding: 4px 8px;
}}
.nk-disclaimer {{
    text-align: center;
    font-size: 11px;
    color: #334155;
    margin-top: 4px;
}}
</style>
<div class="nk-chat-fixed">
  <div class="nk-chat-fixed-inner">
    <div class="nk-chat-row" id="nk-chat-row"></div>
    <div class="nk-disclaimer">{_disclaimer}</div>
  </div>
</div>
''', unsafe_allow_html=True)

# Streamlit widgets positioned inside the bar via columns
_c_plus, _c_input, _c_send = st.columns([1, 10, 1])

with _c_plus:
    if st.button("＋", key="nk_plus",
                 help=_t("Upload a new IFC", "Analyser un nouveau IFC")):
        st.session_state.nk_done = False
        st.session_state.nk_file = None
        st.session_state.nk_chat_history = []
        st.rerun()

with _c_input:
    _q = st.text_input("chat",
        placeholder=_t("Ask about issues, score, corrections, norms...",
                       "Anomalies, score, corrections, normes..."),
        label_visibility="collapsed", key="nk_chat_input")

with _c_send:
    _send = st.button(_send_icon_html, key="nk_send")

if _send and _q.strip() and _chat_ok:
    with st.spinner(_t("nokah is thinking...", "nokah réfléchit...")):
        _resp, _src = get_chat_response(_q, _chat_ctx,
            st.session_state.nk_chat_history,
            st.session_state.get("nk_lang", "EN"))
    st.session_state.nk_chat_history.append({"role": "user", "content": _q})
    st.session_state.nk_chat_history.append({"role": "assistant", "content": _resp, "source": _src})
    st.rerun()
elif _send and _q.strip() and not _chat_ok:
    st.error("nokah_chat.py not found.")

st.markdown("<div style='height:3rem'></div>", unsafe_allow_html=True)
