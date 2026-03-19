"""
discipline_detector.py
──────────────────────
Détecte automatiquement la discipline dominante d'un fichier IFC
en comptant les objets caractéristiques de chaque discipline.
"""

# Objets IFC caractéristiques par discipline
DISCIPLINE_SIGNATURES = {
    "Architecture": {
        "types": ["IfcWall", "IfcDoor", "IfcWindow", "IfcSlab", "IfcRoof", "IfcStair", "IfcRailing"],
        "weight": 1.0
    },
    "MEP": {
        "types": [
            "IfcDuctSegment", "IfcDuctFitting", "IfcDuctSilencer",
            "IfcPipeSegment", "IfcPipeFitting",
            "IfcFlowSegment", "IfcFlowFitting", "IfcFlowTerminal",
            "IfcFlowController", "IfcFlowMovingDevice", "IfcFlowStorageDevice",
            "IfcUnitaryEquipment", "IfcEnergyConversionDevice",
            "IfcAirTerminal", "IfcAirTerminalBox",
            "IfcCableSegment", "IfcCableFitting",
            "IfcElectricAppliance", "IfcLightFixture",
        ],
        "weight": 1.0
    },
    "Structure": {
        "types": [
            "IfcBeam", "IfcColumn", "IfcMember",
            "IfcFooting", "IfcPile", "IfcReinforcingBar",
            "IfcReinforcingMesh", "IfcPlate", "IfcTendon",
        ],
        "weight": 1.0
    },
    "Interior": {
        "types": [
            "IfcFurnishingElement", "IfcCovering",
            "IfcSanitaryTerminal", "IfcSystemFurnitureElement",
        ],
        "weight": 0.8
    },
}

# Seuil minimal d'objets pour qu'une discipline soit "présente"
MIN_OBJECTS_THRESHOLD = 3


def detect_discipline(ifc) -> dict:
    """
    Analyse un fichier IFC ouvert et retourne la discipline détectée.

    Paramètres :
    - ifc : objet ifcopenshell ouvert

    Retourne un dict :
    {
        "primary":   str,          # discipline dominante
        "secondary": list[str],    # disciplines secondaires présentes
        "scores":    dict,         # nb objets par discipline
        "all_counts": dict,        # comptage brut par type IFC
        "is_mixed":  bool,         # modèle multi-discipline ?
    }
    """

    def count_type(entity_type):
        try:
            return len(ifc.by_type(entity_type))
        except Exception:
            return 0

    # Comptage brut
    all_counts = {}
    discipline_scores = {}

    for discipline, config in DISCIPLINE_SIGNATURES.items():
        total = 0
        for ifc_type in config["types"]:
            n = count_type(ifc_type)
            all_counts[ifc_type] = n
            total += n
        discipline_scores[discipline] = total

    # Disciplines présentes (au-dessus du seuil)
    present = {
        disc: score
        for disc, score in discipline_scores.items()
        if score >= MIN_OBJECTS_THRESHOLD
    }

    if not present:
        # Aucune discipline détectée clairement
        return {
            "primary": "Unknown",
            "secondary": [],
            "scores": discipline_scores,
            "all_counts": all_counts,
            "is_mixed": False,
        }

    # Discipline dominante = celle avec le plus d'objets
    primary = max(present, key=lambda d: present[d])
    secondary = [d for d in present if d != primary]
    is_mixed = len(present) > 1

    return {
        "primary": primary,
        "secondary": secondary,
        "scores": discipline_scores,
        "all_counts": all_counts,
        "is_mixed": is_mixed,
    }


def discipline_badge(discipline: str) -> str:
    """Retourne un emoji badge pour l'affichage Streamlit."""
    badges = {
        "Architecture": "🏛️ Architecture",
        "MEP":          "🌬️ MEP / CVC",
        "Structure":    "🏗️ Structure",
        "Interior":     "🛋️ Intérieur",
        "Unknown":      "❓ Indéterminé",
    }
    return badges.get(discipline, f"📁 {discipline}")
