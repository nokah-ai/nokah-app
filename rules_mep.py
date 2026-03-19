
def _get_segment_length(element):
    """Tente de récupérer la longueur d'un segment MEP via ses propriétés."""
    try:
        length = getattr(element, "Length", None)
        if length is not None:
            return float(length)
        # Tentative via Pset
        if hasattr(element, "IsDefinedBy") and element.IsDefinedBy:
            for rel in element.IsDefinedBy:
                if rel.is_a("IfcRelDefinesByProperties"):
                    pset = rel.RelatingPropertyDefinition
                    if hasattr(pset, "HasProperties"):
                        for prop in pset.HasProperties:
                            if prop.Name in ("Length", "Longueur") and hasattr(prop, "NominalValue"):
                                return float(prop.NominalValue.wrappedValue)
    except Exception:
        pass
    return None

"""
rules_mep.py
────────────
Pack de règles QA pour les modèles MEP / CVC (IFC).
Conçu pour être appelé depuis analyze_ifc() dans dashboard_v3.py.
"""


def run_mep_rules(ifc, rule_config: dict, get_elements, get_element_name,
                  get_element_type_name, get_containing_structure_name,
                  get_pset_property, add_result, add_applicability):
    """
    Exécute toutes les règles MEP sur le modèle IFC ouvert.

    Les fonctions utilitaires (get_elements, add_result, etc.)
    sont passées depuis analyze_ifc() pour éviter la duplication de code.
    """

    def is_rule_enabled(rule_id):
        return rule_config.get(rule_id, {}).get("enabled", True)

    def get_rule_priority(rule_id, fallback):
        return rule_config.get(rule_id, {}).get("priority", fallback)

    # Types MEP principaux
    duct_types   = ["IfcDuctSegment", "IfcDuctFitting", "IfcDuctSilencer"]
    pipe_types   = ["IfcPipeSegment", "IfcPipeFitting"]
    flow_types   = ["IfcFlowSegment", "IfcFlowFitting", "IfcFlowTerminal",
                    "IfcFlowController", "IfcFlowMovingDevice"]
    terminal_types = ["IfcAirTerminal", "IfcAirTerminalBox",
                      "IfcFlowTerminal", "IfcUnitaryEquipment",
                      "IfcEnergyConversionDevice"]
    all_mep_types = duct_types + pipe_types + flow_types + ["IfcUnitaryEquipment", "IfcEnergyConversionDevice"]

    def get_mep_elements(type_list):
        result = []
        for t in type_list:
            result.extend(get_elements(t))
        return result

    all_mep = get_mep_elements(all_mep_types)
    all_ducts = get_mep_elements(duct_types)
    all_pipes = get_mep_elements(pipe_types)
    all_flow = get_mep_elements(flow_types)
    all_terminals = get_mep_elements(terminal_types)
    all_equip = get_elements("IfcUnitaryEquipment") + get_elements("IfcEnergyConversionDevice")

    count_mep       = len(all_mep)
    count_ducts     = len(all_ducts)
    count_pipes     = len(all_pipes)
    count_terminals = len(all_terminals)
    count_equip     = len(all_equip)

    # ─────────────────────────────────────────────────────
    # M001 — Segment MEP sans système de référence
    # ─────────────────────────────────────────────────────
    rule_id  = "M001"
    category = "MEP"
    segments = all_ducts + all_pipes + get_mep_elements(flow_types)
    applicable = len(segments) > 0 and is_rule_enabled(rule_id)
    add_applicability(
        rule_id, category, applicable,
        f"{len(segments)} MEP segment(s) present" if applicable else "No MEP segment or rule disabled"
    )
    if applicable:
        priority = get_rule_priority(rule_id, "Major")
        for el in segments:
            # Vérifie si l'élément appartient à un système MEP
            in_system = False
            if hasattr(el, "HasAssignments") and el.HasAssignments:
                for assign in el.HasAssignments:
                    if assign.is_a("IfcRelAssignsToGroup"):
                        try:
                            if assign.RelatingGroup and assign.RelatingGroup.is_a("IfcSystem"):
                                in_system = True
                                break
                        except Exception:
                            pass
            if not in_system:
                add_result(
                    rule_id, category, "WARNING", priority, el,
                    "MEP segment without system reference (IfcSystem)",
                    len(segments),
                    suggestion="Assign this segment to a MEP system (IfcSystem) in Revit or the authoring tool.",
                    autofix_possible=False, fix_type="SystemAssignment"
                )

    # ─────────────────────────────────────────────────────
    # M002 — Terminal MEP sans rattachement spatial
    # ─────────────────────────────────────────────────────
    rule_id  = "M002"
    category = "MEP"
    applicable = count_terminals > 0 and is_rule_enabled(rule_id)
    add_applicability(
        rule_id, category, applicable,
        f"{count_terminals} MEP terminal(s) present" if applicable else "No MEP terminal or rule disabled"
    )
    if applicable:
        priority = get_rule_priority(rule_id, "Major")
        for el in all_terminals:
            storey = get_containing_structure_name(el)
            if storey is None:
                add_result(
                    rule_id, category, "WARNING", priority, el,
                    "MEP terminal not assigned to any floor level (IfcBuildingStorey)",
                    count_terminals,
                    suggestion="Assign the terminal to a level in the IFC spatial structure.",
                    autofix_possible=False, fix_type="SpatialAssignment"
                )

    # ─────────────────────────────────────────────────────
    # M003 — Équipement MEP sans type défini
    # ─────────────────────────────────────────────────────
    rule_id  = "M003"
    category = "MEP"
    applicable = count_equip > 0 and is_rule_enabled(rule_id)
    add_applicability(
        rule_id, category, applicable,
        f"{count_equip} MEP equipment present" if applicable else "No MEP equipment or rule disabled"
    )
    if applicable:
        priority = get_rule_priority(rule_id, "Major")
        for el in all_equip:
            if get_element_type_name(el) == "Type inconnu":
                add_result(
                    rule_id, category, "WARNING", priority, el,
                    "MEP equipment without BIM type defined",
                    count_equip,
                    suggestion="Assign the equipment to an IfcUnitaryEquipmentType or IfcEnergyConversionDeviceType.",
                    autofix_possible=False, fix_type="TypeAssignment"
                )

    # ─────────────────────────────────────────────────────
    # M004 — Objet MEP sans nom renseigné
    # ─────────────────────────────────────────────────────
    rule_id  = "M004"
    category = "MEP"
    applicable = count_mep > 0 and is_rule_enabled(rule_id)
    add_applicability(
        rule_id, category, applicable,
        f"{count_mep} MEP object(s) present" if applicable else "No MEP object or rule disabled"
    )
    if applicable:
        priority = get_rule_priority(rule_id, "Minor")
        for el in all_mep:
            name = get_element_name(el)
            if name == "No name" or name.strip() == "":
                add_result(
                    rule_id, category, "INFO", priority, el,
                    "MEP object without name",
                    count_mep,
                    bucket="Data BIM",
                    suggestion="Fill in the MEP object name to facilitate coordination.",
                    autofix_possible=False, fix_type="Naming"
                )

    # ─────────────────────────────────────────────────────
    # M005 — Équipement MEP sans classification
    # ─────────────────────────────────────────────────────
    rule_id  = "M005"
    category = "MEP"
    applicable = count_equip > 0 and is_rule_enabled(rule_id)
    add_applicability(
        rule_id, category, applicable,
        f"{count_equip} MEP equipment present" if applicable else "No MEP equipment or rule disabled"
    )
    if applicable:
        priority = get_rule_priority(rule_id, "Minor")
        for el in all_equip:
            classification = get_pset_property(el, "Classification", "Code")
            ref = get_pset_property(el, "Identity Data", "Assembly Code")
            if classification is None and ref is None:
                add_result(
                    rule_id, category, "INFO", priority, el,
                    "MEP equipment without classification code",
                    count_equip,
                    bucket="Check",
                    suggestion="Add a classification code (Uniclass, OmniClass, etc.) to the equipment.",
                    autofix_possible=False, fix_type="Classification"
                )


    # ─────────────────────────────────────────────────────
    # M006 — Conduit MEP de longueur quasi nulle (erreur export)
    # ─────────────────────────────────────────────────────
    rule_id  = "M006"
    category = "MEP"
    duct_pipe_segments = all_ducts + all_pipes
    applicable = len(duct_pipe_segments) > 0 and is_rule_enabled(rule_id)
    add_applicability(
        rule_id, category, applicable,
        f"{len(duct_pipe_segments)} duct/pipe segment(s) to check" if applicable
        else "No duct/pipe segment or rule disabled"
    )
    if applicable:
        priority = get_rule_priority(rule_id, "Major")
        min_length = rule_config.get(rule_id, {}).get("min_length", 0.05)
        for el in duct_pipe_segments:
            length = _get_segment_length(el)
            if length is not None and 0 < length < min_length:
                add_result(
                    rule_id, category, "WARNING", priority, el,
                    f"MEP segment with near-zero length ({length:.3f} m) — possible export error",
                    len(duct_pipe_segments),
                    suggestion=(
                        f"Detected length: {length:.3f} m. "
                        "Very short segments are often Revit export artifacts. "
                        "Check and remove residual segments."
                    ),
                    autofix_possible=False, fix_type="GeometryCleanup"
                )

    # ─────────────────────────────────────────────────────
    # M007 — Terminal MEP sans connexion réseau détectable
    # ─────────────────────────────────────────────────────
    rule_id  = "M007"
    category = "MEP"
    applicable = count_terminals > 0 and is_rule_enabled(rule_id)
    add_applicability(
        rule_id, category, applicable,
        f"{count_terminals} MEP terminal(s) to check" if applicable
        else "No MEP terminal or rule disabled"
    )
    if applicable:
        priority = get_rule_priority(rule_id, "Major")
        for el in all_terminals:
            connected = False
            try:
                if hasattr(el, "IsConnectedFrom") and el.IsConnectedFrom:
                    connected = True
                if hasattr(el, "ConnectedFrom") and el.ConnectedFrom:
                    connected = True
                # Vérification via HasPorts
                if hasattr(el, "HasPorts") and el.HasPorts:
                    for port_rel in el.HasPorts:
                        try:
                            port = port_rel.RelatingPort if hasattr(port_rel, "RelatingPort") else port_rel
                            if hasattr(port, "ConnectedTo") and port.ConnectedTo:
                                connected = True
                                break
                            if hasattr(port, "ConnectedFrom") and port.ConnectedFrom:
                                connected = True
                                break
                        except Exception:
                            pass
            except Exception:
                pass

            if not connected:
                add_result(
                    rule_id, category, "WARNING", priority, el,
                    "MEP terminal with no detectable network connection — network may be incomplete",
                    count_terminals,
                    bucket="Check",
                    suggestion=(
                        "Ce terminal n'est pas connecté à un réseau de distribution. "
                        "Check MEP connectivity in the authoring tool (Revit, MagiCAD, etc.)."
                    ),
                    autofix_possible=False, fix_type="NetworkConnectivity"
                )

    return {
        "count_mep":       count_mep,
        "count_ducts":     count_ducts,
        "count_pipes":     count_pipes,
        "count_terminals": count_terminals,
        "count_equip":     count_equip,
    }
