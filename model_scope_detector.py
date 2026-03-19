"""
model_scope_detector.py
───────────────────────
Qualifie chaque modèle IFC avant intégration au benchmark.
Protège le dataset contre les objets isolés, familles paramétriques,
ou IFC trop pauvres pour être comparés à de vrais modèles de bâtiment.

Statuts possibles :
  BuildingModel       — vrai modèle de bâtiment complet
  DisciplineSubmodel  — sous-modèle métier cohérent (archi, MEP, structure)
  ObjectOnly          — objet isolé ou contenu trop limité
  InvalidForBenchmark — fichier trop pauvre ou non comparable
"""

# ── Seuils de qualification ───────────────────────────────────────────────────

# Nombre minimum d'objets pour être un sous-modèle valide
MIN_OBJECTS_SUBMODEL = 10

# Nombre minimum de types IFC différents pour être considéré varié
MIN_IFC_TYPES_VARIETY = 2

# Nombre minimum d'objets pour être un bâtiment complet
MIN_OBJECTS_BUILDING = 50

# Nombre minimum de niveaux (IfcBuildingStorey) pour un bâtiment
MIN_STOREYS_BUILDING = 1

# Types IFC "riches" qui indiquent un vrai projet
RICH_TYPES = {
    "IfcWall", "IfcWallStandardCase", "IfcDoor", "IfcWindow", "IfcSlab",
    "IfcBeam", "IfcColumn", "IfcRoof", "IfcStair", "IfcRailing",
    "IfcDuctSegment", "IfcPipeSegment", "IfcFlowTerminal",
    "IfcUnitaryEquipment", "IfcEnergyConversionDevice",
    "IfcSpace", "IfcZone",
}


def detect_model_scope(ifc, discipline_info: dict) -> dict:
    """
    Analyse un fichier IFC ouvert et détermine son scope.

    Paramètres :
    - ifc            : objet ifcopenshell ouvert
    - discipline_info: dict retourné par detect_discipline()

    Retourne un dict :
    {
        "scope":                 str,   # BuildingModel | DisciplineSubmodel | ObjectOnly | InvalidForBenchmark
        "is_benchmark_eligible": bool,
        "reason":                str,
        "details": {
            "total_objects":    int,
            "ifc_types_count":  int,
            "has_storeys":      bool,
            "has_spatial":      bool,
            "rich_types_found": list,
        }
    }
    """

    def count_type(entity_type):
        try:
            return len(ifc.by_type(entity_type))
        except Exception:
            return 0

    # ── Comptages de base ─────────────────────────────────
    # Nombre de niveaux
    n_storeys = count_type("IfcBuildingStorey")
    n_buildings = count_type("IfcBuilding")
    n_sites = count_type("IfcSite")
    n_spaces = count_type("IfcSpace")

    has_storeys  = n_storeys >= MIN_STOREYS_BUILDING
    has_spatial  = n_buildings > 0 or n_sites > 0 or n_storeys > 0

    # Types IFC présents (parmi les types "riches")
    rich_types_found = [t for t in RICH_TYPES if count_type(t) > 0]
    n_rich_types = len(rich_types_found)

    # Total objets tous types confondus
    discipline_scores = discipline_info.get("scores", {})
    total_objects = sum(discipline_scores.values())

    # ── Classification ────────────────────────────────────

    # Cas 1 — trop pauvre pour être analysé
    if total_objects == 0:
        return _result(
            scope="InvalidForBenchmark",
            eligible=False,
            reason="Aucun objet IFC reconnu dans le fichier.",
            total_objects=total_objects,
            ifc_types_count=n_rich_types,
            has_storeys=has_storeys,
            has_spatial=has_spatial,
            rich_types=rich_types_found
        )

    # Cas 2 — objet isolé
    if total_objects < MIN_OBJECTS_SUBMODEL or n_rich_types < MIN_IFC_TYPES_VARIETY:
        return _result(
            scope="ObjectOnly",
            eligible=False,
            reason=(
                f"Modèle trop limité : {total_objects} objet(s) de {n_rich_types} type(s) différent(s). "
                "Semble être un objet ou une famille isolée."
            ),
            total_objects=total_objects,
            ifc_types_count=n_rich_types,
            has_storeys=has_storeys,
            has_spatial=has_spatial,
            rich_types=rich_types_found
        )

    # Cas 3 — bâtiment complet
    if (total_objects >= MIN_OBJECTS_BUILDING and
            has_spatial and
            n_rich_types >= 3):
        return _result(
            scope="BuildingModel",
            eligible=True,
            reason=(
                f"Modèle de bâtiment complet : {total_objects} objets, "
                f"{n_rich_types} types IFC, structure spatiale présente."
            ),
            total_objects=total_objects,
            ifc_types_count=n_rich_types,
            has_storeys=has_storeys,
            has_spatial=has_spatial,
            rich_types=rich_types_found
        )

    # Cas 4 — sous-modèle discipline valide
    if total_objects >= MIN_OBJECTS_SUBMODEL:
        return _result(
            scope="DisciplineSubmodel",
            eligible=True,
            reason=(
                f"Sous-modèle métier valide : {total_objects} objets, "
                f"discipline {discipline_info.get('primary', 'Inconnue')}."
            ),
            total_objects=total_objects,
            ifc_types_count=n_rich_types,
            has_storeys=has_storeys,
            has_spatial=has_spatial,
            rich_types=rich_types_found
        )

    # Cas 5 — fallback
    return _result(
        scope="InvalidForBenchmark",
        eligible=False,
        reason=f"Modèle non classifiable : {total_objects} objets, {n_rich_types} types.",
        total_objects=total_objects,
        ifc_types_count=n_rich_types,
        has_storeys=has_storeys,
        has_spatial=has_spatial,
        rich_types=rich_types_found
    )


def _result(scope, eligible, reason, total_objects,
            ifc_types_count, has_storeys, has_spatial, rich_types):
    return {
        "scope":                 scope,
        "is_benchmark_eligible": eligible,
        "reason":                reason,
        "details": {
            "total_objects":    total_objects,
            "ifc_types_count":  ifc_types_count,
            "has_storeys":      has_storeys,
            "has_spatial":      has_spatial,
            "rich_types_found": rich_types,
        }
    }


def scope_badge(scope: str) -> str:
    """Retourne un badge lisible pour l'affichage Streamlit."""
    badges = {
        "BuildingModel":       "🏢 Bâtiment complet",
        "DisciplineSubmodel":  "📐 Sous-modèle métier",
        "ObjectOnly":          "🔩 Objet isolé",
        "InvalidForBenchmark": "⚠️ Non benchmarkable",
    }
    return badges.get(scope, f"❓ {scope}")


def scope_color(scope: str) -> str:
    """Retourne une couleur Streamlit selon le scope."""
    colors = {
        "BuildingModel":       "success",
        "DisciplineSubmodel":  "info",
        "ObjectOnly":          "warning",
        "InvalidForBenchmark": "error",
    }
    return colors.get(scope, "info")
