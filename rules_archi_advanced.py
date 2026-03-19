"""
rules_archi_advanced.py
───────────────────────
Règles QA avancées pour les modèles Architecture et Architecture Intérieure.
Ces règles détectent des incohérences de conception, pas seulement des propriétés manquantes.

Règles intelligentes selon la vision NOKAH :
  A001 — Ratio façade/fenêtres atypique
  A002 — Mur extérieur sans isolation thermique détectable
  A003 — Fenêtre hors plage de hauteur standard
  A004 — Porte non accessible (largeur < 0.83 m PMR)
  A005 — Garde-corps avec hauteur incohérente (quasi nulle)
  A006 — Élément sans niveau (storey) — archi
  A007 — Mur intérieur sans type défini
  A008 — Incohérence ratio murs/dalles (modèle fragmenté)
"""

import numpy as np


def run_archi_advanced_rules(
    ifc, rule_config: dict,
    get_elements, get_element_name, get_element_type_name,
    get_containing_structure_name, get_pset_property,
    get_shape_z_range, add_result, add_applicability,
    is_probable_exterior_wall, is_probable_skylight
):
    """
    Exécute les règles avancées Architecture sur le modèle IFC ouvert.
    """

    def is_rule_enabled(rule_id):
        return rule_config.get(rule_id, {}).get("enabled", True)

    def get_rule_priority(rule_id, fallback):
        return rule_config.get(rule_id, {}).get("priority", fallback)

    def get_rule_value(rule_id, key, fallback):
        return rule_config.get(rule_id, {}).get(key, fallback)

    # Comptages de base
    all_walls    = get_elements("IfcWall") + get_elements("IfcWallStandardCase")
    all_windows  = get_elements("IfcWindow")
    all_doors    = get_elements("IfcDoor")
    all_slabs    = get_elements("IfcSlab")
    all_railings = get_elements("IfcRailing")

    exterior_walls = [w for w in all_walls if is_probable_exterior_wall(w)]
    non_skylight_windows = [w for w in all_windows if not is_probable_skylight(w)]

    count_walls    = len(all_walls)
    count_ext_walls = len(exterior_walls)
    count_windows  = len(non_skylight_windows)
    count_doors    = len(all_doors)
    count_slabs    = len(all_slabs)
    count_railings = len(all_railings)

    # ─────────────────────────────────────────────────────
    # A001 — Ratio façade / fenêtres atypique
    # ─────────────────────────────────────────────────────
    rule_id  = "A001"
    category = "Advanced Architecture"
    applicable = count_ext_walls > 0 and count_windows > 0 and is_rule_enabled(rule_id)
    add_applicability(
        rule_id, category, applicable,
        f"{count_ext_walls} exterior wall(s), {count_windows} window(s)" if applicable
        else "Insufficient exterior walls or windows"
    )
    if applicable:
        priority = get_rule_priority(rule_id, "Minor")
        ratio = count_windows / max(count_ext_walls, 1)

        # Seuils typiques : 0.1 à 1.5 fenêtres par mur extérieur
        if ratio < 0.05:
            add_result(
                rule_id, category, "WARNING", priority,
                exterior_walls[0],
                f"Very low window/exterior wall ratio ({ratio:.2f}) — possible missing windows",
                count_ext_walls,
                bucket="Check",
                suggestion=(
                    f"Ratio actuel : {count_windows} fenêtres pour {count_ext_walls} murs extérieurs. "
                    "Vérifier si des fenêtres sont manquantes ou mal classées."
                ),
                autofix_possible=False, fix_type="ModelCoherence"
            )
        elif ratio > 3.0:
            add_result(
                rule_id, category, "WARNING", priority,
                exterior_walls[0],
                f"Very high window/exterior wall ratio ({ratio:.2f}) — possible modeling error",
                count_ext_walls,
                bucket="Check",
                suggestion=(
                    f"Ratio actuel : {count_windows} fenêtres pour {count_ext_walls} murs extérieurs. "
                    "Vérifier si des murs extérieurs sont manquants ou mal classés."
                ),
                autofix_possible=False, fix_type="ModelCoherence"
            )

    # ─────────────────────────────────────────────────────
    # A002 — Mur extérieur sans isolation thermique détectable
    # ─────────────────────────────────────────────────────
    rule_id  = "A002"
    category = "Advanced Architecture"
    applicable = count_ext_walls > 0 and is_rule_enabled(rule_id)
    add_applicability(
        rule_id, category, applicable,
        f"{count_ext_walls} exterior wall(s) to check" if applicable
        else "No exterior wall detected"
    )
    if applicable:
        priority = get_rule_priority(rule_id, "Major")
        insulation_keywords = [
            "laine", "insulation", "isol", "rockwool", "glasswool",
            "eps", "xps", "mineral wool", "ite ", "etics", "sarking",
            "isolant", "isolation"
        ]
        for wall in exterior_walls:
            type_text = (
                get_element_name(wall) + " " + get_element_type_name(wall)
            ).lower()
            # Vérifie aussi la propriété R (résistance thermique)
            r_value = get_pset_property(wall, "Pset_WallCommon", "ThermalTransmittance")
            has_insulation_name = any(k in type_text for k in insulation_keywords)
            has_r_value = r_value is not None

            if not has_insulation_name and not has_r_value:
                add_result(
                    rule_id, category, "WARNING", priority, wall,
                    "Exterior wall without detectable thermal insulation",
                    count_ext_walls,
                    bucket="Check",
                    suggestion=(
                        "Aucune couche d'isolation ni propriété ThermalTransmittance trouvée. "
                        "Check wall composition and add insulation layer if needed."
                    ),
                    autofix_possible=False, fix_type="ThermalEnvelope"
                )

    # ─────────────────────────────────────────────────────
    # A003 — Fenêtre avec hauteur incohérente (trop haute ou quasi nulle)
    # ─────────────────────────────────────────────────────
    rule_id  = "A003"
    category = "Advanced Architecture"
    applicable = count_windows > 0 and is_rule_enabled(rule_id)
    add_applicability(
        rule_id, category, applicable,
        f"{count_windows} window(s) to check" if applicable else "No window detected"
    )
    if applicable:
        priority = get_rule_priority(rule_id, "Major")
        min_height = get_rule_value(rule_id, "min_window_height", 0.30)
        max_height = get_rule_value(rule_id, "max_window_height", 4.00)

        for window in non_skylight_windows:
            zmin, zmax = get_shape_z_range(window)
            if zmin is not None and zmax is not None:
                height = zmax - zmin
                if height < min_height:
                    add_result(
                        rule_id, category, "WARNING", priority, window,
                        f"Window with near-zero height ({height:.2f} m) — possible modeling error",
                        count_windows,
                        suggestion=(
                            f"Detected height: {height:.2f} m. "
                            "A standard window measures between 0.60 m and 2.50 m. Check the geometry."
                        ),
                        autofix_possible=False, fix_type="GeometryCoherence"
                    )
                elif height > max_height:
                    add_result(
                        rule_id, category, "WARNING", "Minor", window,
                        f"Abnormally tall window ({height:.2f} m > {max_height:.2f} m)",
                        count_windows,
                        bucket="Check",
                        suggestion=(
                            f"Detected height: {height:.2f} m. "
                            "Check if this is full-height glazing or a modeling error."
                        ),
                        autofix_possible=False, fix_type="GeometryCoherence"
                    )

    # ─────────────────────────────────────────────────────
    # A004 — Porte non accessible PMR (largeur < 0.83 m)
    # ─────────────────────────────────────────────────────
    rule_id  = "A004"
    category = "Interior Architecture"
    applicable = count_doors > 0 and is_rule_enabled(rule_id)
    add_applicability(
        rule_id, category, applicable,
        f"{count_doors} door(s) to check for accessibility" if applicable
        else "No door detected"
    )
    if applicable:
        priority = get_rule_priority(rule_id, "Major")
        pmr_min = get_rule_value(rule_id, "pmr_min_width", 0.83)

        for door in all_doors:
            overall_width = getattr(door, "OverallWidth", None)
            if overall_width is not None:
                try:
                    width = float(overall_width)
                    if 0.01 < width < pmr_min:
                        add_result(
                            rule_id, category, "WARNING", priority, door,
                            f"Porte potentiellement non conforme PMR ({width:.2f} m < {pmr_min:.2f} m)",
                            count_doors,
                            bucket="Check",
                            suggestion=(
                                f"Current width: {width:.2f} m. "
                                f"La réglementation PMR impose une largeur minimale de {pmr_min:.2f} m "
                                "(passage utile ≥ 0.77 m). Vérifier si cette porte dessert un espace accessible."
                            ),
                            autofix_possible=False, fix_type="Accessibility"
                        )
                except Exception:
                    pass

    # ─────────────────────────────────────────────────────
    # A005 — Garde-corps avec hauteur quasi nulle (erreur modélisation)
    # ─────────────────────────────────────────────────────
    rule_id  = "A005"
    category = "Interior Architecture"
    applicable = count_railings > 0 and is_rule_enabled(rule_id)
    add_applicability(
        rule_id, category, applicable,
        f"{count_railings} railing(s) to check" if applicable
        else "No railing detected"
    )
    if applicable:
        priority = get_rule_priority(rule_id, "Critical")
        for railing in all_railings:
            zmin, zmax = get_shape_z_range(railing)
            if zmin is not None and zmax is not None:
                height = zmax - zmin
                if 0 < height < 0.10:
                    add_result(
                        rule_id, category, "ERROR", priority, railing,
                        f"Railing with near-zero height ({height:.2f} m) — likely modeling error",
                        count_railings,
                        suggestion=(
                            f"Detected height: {height:.2f} m. "
                            "A railing must be at least 0.90 m (exterior) or 1.00 m (interior). "
                            "Check the geometry in the authoring tool."
                        ),
                        autofix_possible=False, fix_type="SafetyGeometry"
                    )

    # ─────────────────────────────────────────────────────
    # A006 — Élément architectural sans niveau (storey)
    # ─────────────────────────────────────────────────────
    rule_id  = "A006"
    category = "Advanced Architecture"
    all_archi = all_walls + all_doors + all_windows
    applicable = len(all_archi) > 0 and is_rule_enabled(rule_id)
    add_applicability(
        rule_id, category, applicable,
        f"{len(all_archi)} architectural element(s) to check" if applicable
        else "No architectural element detected"
    )
    if applicable:
        priority = get_rule_priority(rule_id, "Minor")
        count_no_storey = 0
        first_el_no_storey = None
        for el in all_archi:
            storey = get_containing_structure_name(el)
            if storey is None:
                count_no_storey += 1
                if first_el_no_storey is None:
                    first_el_no_storey = el

        if count_no_storey > 0 and first_el_no_storey is not None:
            add_result(
                rule_id, category, "WARNING", priority, first_el_no_storey,
                f"{count_no_storey} architectural element(s) without floor level (IfcBuildingStorey)",
                len(all_archi),
                bucket="Data BIM",
                suggestion=(
                    "Rattacher ces éléments à un niveau dans la structure spatiale IFC "
                    "pour faciliter la coordination et l'extraction de données."
                ),
                autofix_possible=False, fix_type="SpatialStructure"
            )

    # ─────────────────────────────────────────────────────
    # A007 — Mur intérieur sans type défini
    # ─────────────────────────────────────────────────────
    rule_id  = "A007"
    category = "Interior Architecture"
    interior_walls = [w for w in all_walls if not is_probable_exterior_wall(w)]
    applicable = len(interior_walls) > 0 and is_rule_enabled(rule_id)
    add_applicability(
        rule_id, category, applicable,
        f"{len(interior_walls)} interior wall(s) to check" if applicable
        else "No interior wall detected"
    )
    if applicable:
        priority = get_rule_priority(rule_id, "Minor")
        for wall in interior_walls:
            if get_element_type_name(wall) == "Type inconnu":
                add_result(
                    rule_id, category, "WARNING", priority, wall,
                    "Interior wall without BIM type defined",
                    len(interior_walls),
                    bucket="Data BIM",
                    suggestion=(
                        "Associer ce mur à un IfcWallType pour permettre l'extraction "
                        "des quantités, des matériaux et faciliter la coordination."
                    ),
                    autofix_possible=False, fix_type="TypeAssignment"
                )

    # ─────────────────────────────────────────────────────
    # A008 — Ratio murs/dalles incohérent (modèle fragmenté)
    # ─────────────────────────────────────────────────────
    rule_id  = "A008"
    category = "Advanced Architecture"
    applicable = count_walls > 0 and count_slabs > 0 and is_rule_enabled(rule_id)
    add_applicability(
        rule_id, category, applicable,
        f"{count_walls} mur(s) / {count_slabs} dalle(s)" if applicable
        else "Insufficient walls or slabs for this rule"
    )
    if applicable:
        priority = get_rule_priority(rule_id, "Minor")
        ratio_mur_dalle = count_walls / max(count_slabs, 1)

        # Seuils : un bâtiment typique a entre 3 et 40 murs par dalle
        if ratio_mur_dalle > 60:
            add_result(
                rule_id, category, "WARNING", priority,
                all_slabs[0] if all_slabs else all_walls[0],
                f"Very high wall/slab ratio ({ratio_mur_dalle:.0f}) — model potentially fragmented",
                count_walls,
                bucket="Check",
                suggestion=(
                    f"{count_walls} murs pour seulement {count_slabs} dalle(s). "
                    "A high ratio may indicate missing slabs or fragmented modeling per level."
                ),
                autofix_possible=False, fix_type="ModelCoherence"
            )
        elif ratio_mur_dalle < 0.5:
            add_result(
                rule_id, category, "WARNING", priority,
                all_walls[0] if all_walls else all_slabs[0],
                f"Very low wall/slab ratio ({ratio_mur_dalle:.2f}) — walls potentially missing",
                count_slabs,
                bucket="Check",
                suggestion=(
                    f"Seulement {count_walls} mur(s) pour {count_slabs} dalle(s). "
                    "Check if walls are missing or exported in another IFC file."
                ),
                autofix_possible=False, fix_type="ModelCoherence"
            )

    return {
        "count_ext_walls":    count_ext_walls,
        "count_int_walls":    len(interior_walls),
        "count_windows_checked": count_windows,
        "count_doors_checked":   count_doors,
    }
