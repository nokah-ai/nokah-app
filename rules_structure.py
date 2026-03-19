"""
rules_structure.py
──────────────────
Pack de règles QA pour les modèles Structure (IFC).
Conçu pour être appelé depuis analyze_ifc() dans dashboard_v3.py.
"""


def run_structure_rules(ifc, rule_config: dict, get_elements, get_element_name,
                        get_element_type_name, get_containing_structure_name,
                        get_pset_property, get_shape_z_range,
                        add_result, add_applicability):
    """
    Exécute toutes les règles Structure sur le modèle IFC ouvert.
    """

    def is_rule_enabled(rule_id):
        return rule_config.get(rule_id, {}).get("enabled", True)

    def get_rule_priority(rule_id, fallback):
        return rule_config.get(rule_id, {}).get("priority", fallback)

    def get_rule_value(rule_id, key, fallback):
        return rule_config.get(rule_id, {}).get(key, fallback)

    count_beams   = len(get_elements("IfcBeam"))
    count_columns = len(get_elements("IfcColumn"))
    count_members = len(get_elements("IfcMember"))
    count_footings = len(get_elements("IfcFooting"))
    count_piles   = len(get_elements("IfcPile"))

    all_structural = (
        get_elements("IfcBeam") +
        get_elements("IfcColumn") +
        get_elements("IfcMember") +
        get_elements("IfcFooting") +
        get_elements("IfcPile")
    )
    count_total = len(all_structural)

    # ─────────────────────────────────────────────────────
    # S001 — Élément structurel sans type défini
    # ─────────────────────────────────────────────────────
    rule_id  = "S001"
    category = "Structure"
    applicable = count_total > 0 and is_rule_enabled(rule_id)
    add_applicability(
        rule_id, category, applicable,
        f"{count_total} structural element(s) present" if applicable else "No structural element or rule disabled"
    )
    if applicable:
        priority = get_rule_priority(rule_id, "Major")
        for el in all_structural:
            if get_element_type_name(el) == "Type inconnu":
                add_result(
                    rule_id, category, "WARNING", priority, el,
                    "Structural element without BIM type defined",
                    count_total,
                    suggestion="Assign the element to a structural type (IfcBeamType, IfcColumnType, etc.).",
                    autofix_possible=False, fix_type="TypeAssignment"
                )

    # ─────────────────────────────────────────────────────
    # S002 — Élément structurel sans niveau (storey)
    # ─────────────────────────────────────────────────────
    rule_id  = "S002"
    category = "Structure"
    applicable = count_total > 0 and is_rule_enabled(rule_id)
    add_applicability(
        rule_id, category, applicable,
        f"{count_total} structural element(s) present" if applicable else "No structural element or rule disabled"
    )
    if applicable:
        priority = get_rule_priority(rule_id, "Major")
        for el in all_structural:
            storey = get_containing_structure_name(el)
            if storey is None:
                add_result(
                    rule_id, category, "WARNING", priority, el,
                    "Structural element not assigned to any floor level (IfcBuildingStorey)",
                    count_total,
                    suggestion="Assign the element to a level in the IFC spatial structure.",
                    autofix_possible=False, fix_type="SpatialAssignment"
                )

    # ─────────────────────────────────────────────────────
    # S003 — Poteau sans fondation détectable
    # ─────────────────────────────────────────────────────
    rule_id  = "S003"
    category = "Structure"
    applicable = count_columns > 0 and is_rule_enabled(rule_id)
    add_applicability(
        rule_id, category, applicable,
        f"{count_columns} column(s) present" if applicable else "No column or rule disabled"
    )
    if applicable:
        priority = get_rule_priority(rule_id, "Minor")
        # Vérification simple : si poteaux présents mais 0 fondations
        if count_footings == 0 and count_piles == 0:
            for el in get_elements("IfcColumn"):
                add_result(
                    rule_id, category, "WARNING", priority, el,
                    "Column present but no foundation (IfcFooting/IfcPile) detected in the model",
                    count_columns,
                    bucket="Check",
                    suggestion="Check if foundations are in a separate model or add them.",
                    autofix_possible=False, fix_type="ModelCompleteness"
                )

    # ─────────────────────────────────────────────────────
    # S004 — Élément structurel sans matériau
    # ─────────────────────────────────────────────────────
    rule_id  = "S004"
    category = "Structure"
    applicable = count_total > 0 and is_rule_enabled(rule_id)
    add_applicability(
        rule_id, category, applicable,
        f"{count_total} structural element(s) present" if applicable else "No structural element or rule disabled"
    )
    if applicable:
        priority = get_rule_priority(rule_id, "Major")
        for el in all_structural:
            has_mat = False
            if hasattr(el, "HasAssociations") and el.HasAssociations:
                for assoc in el.HasAssociations:
                    if assoc.is_a("IfcRelAssociatesMaterial"):
                        has_mat = True
                        break
            if not has_mat:
                add_result(
                    rule_id, category, "WARNING", priority, el,
                    "Structural element without associated material",
                    count_total,
                    suggestion="Assign a material (concrete, steel, wood) to the structural element.",
                    autofix_possible=False, fix_type="Material"
                )

    # ─────────────────────────────────────────────────────
    # S005 — Élément structurel sans classification
    # ─────────────────────────────────────────────────────
    rule_id  = "S005"
    category = "Structure"
    applicable = count_total > 0 and is_rule_enabled(rule_id)
    add_applicability(
        rule_id, category, applicable,
        f"{count_total} structural element(s) present" if applicable else "No structural element or rule disabled"
    )
    if applicable:
        priority = get_rule_priority(rule_id, "Minor")
        for el in all_structural:
            classification = get_pset_property(el, "Classification", "Code")
            ref = get_pset_property(el, "Identity Data", "Assembly Code")
            if classification is None and ref is None:
                add_result(
                    rule_id, category, "INFO", priority, el,
                    "Structural element without classification code",
                    count_total,
                    bucket="Data BIM",
                    suggestion="Add a classification code (Uniclass, etc.).",
                    autofix_possible=False, fix_type="Classification"
                )


    # ─────────────────────────────────────────────────────
    # S006 — Ratio poutres/poteaux atypique
    # ─────────────────────────────────────────────────────
    rule_id  = "S006"
    category = "Structure"
    applicable = count_beams > 0 and count_columns > 0 and is_rule_enabled(rule_id)
    add_applicability(
        rule_id, category, applicable,
        f"{count_beams} poutre(s) / {count_columns} poteau(x)" if applicable
        else "Poutres ou poteaux insuffisants pour la règle"
    )
    if applicable:
        priority = get_rule_priority(rule_id, "Minor")
        ratio = count_beams / max(count_columns, 1)
        # Ratio typique : 0.5 à 5 poutres par poteau
        if ratio > 10:
            add_result(
                rule_id, category, "WARNING", priority,
                get_elements("IfcBeam")[0],
                f"Very high beam/column ratio ({ratio:.1f}) — potentially atypical structure",
                count_beams,
                bucket="Check",
                suggestion=(
                    f"{count_beams} poutres pour {count_columns} poteau(x). "
                    "Un ratio très élevé peut indiquer des poteaux manquants ou une structure particulière. "
                    "Vérifier la cohérence structurelle."
                ),
                autofix_possible=False, fix_type="StructuralCoherence"
            )
        elif ratio < 0.2:
            add_result(
                rule_id, category, "WARNING", priority,
                get_elements("IfcColumn")[0],
                f"Very low beam/column ratio ({ratio:.2f}) — beams potentially missing",
                count_columns,
                bucket="Check",
                suggestion=(
                    f"Seulement {count_beams} poutre(s) pour {count_columns} poteau(x). "
                    "Vérifier si les poutres sont dans un modèle séparé ou si elles sont manquantes."
                ),
                autofix_possible=False, fix_type="StructuralCoherence"
            )

    # ─────────────────────────────────────────────────────
    # S007 — Poteau de hauteur incohérente (trop court)
    # ─────────────────────────────────────────────────────
    rule_id  = "S007"
    category = "Structure"
    applicable = count_columns > 0 and is_rule_enabled(rule_id)
    add_applicability(
        rule_id, category, applicable,
        f"{count_columns} poteau(x) à contrôler" if applicable
        else "Aucun poteau ou règle désactivée"
    )
    if applicable:
        priority = get_rule_priority(rule_id, "Major")
        min_col_height = rule_config.get(rule_id, {}).get("min_height", 0.50)
        for col in get_elements("IfcColumn"):
            zmin, zmax = get_shape_z_range(col)
            if zmin is not None and zmax is not None:
                height = zmax - zmin
                if 0 < height < min_col_height:
                    add_result(
                        rule_id, category, "WARNING", priority, col,
                        f"Column with abnormally low height ({height:.2f} m < {min_col_height:.2f} m)",
                        count_columns,
                        suggestion=(
                            f"Detected height: {height:.2f} m. "
                            "Un poteau structurel doit mesurer au moins 0.50 m. "
                            "Vérifier si c'est un artefact de modélisation ou un poteau de liaison court."
                        ),
                        autofix_possible=False, fix_type="GeometryCoherence"
                    )

    return {
        "count_beams":      count_beams,
        "count_columns":    count_columns,
        "count_members":    count_members,
        "count_footings":   count_footings,
        "count_piles":      count_piles,
        "count_structural": count_total,
    }
