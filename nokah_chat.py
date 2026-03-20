"""
nokah_chat.py
─────────────
Moteur de chat conversationnel nokah
Couche 1 : réponses locales basées sur les données réelles de la maquette
Couche 2 : LLM Groq optionnel (si clé API configurée dans Streamlit secrets)
"""

import random
import re
import os

# ── Détection d'intention ─────────────────────────────────────────────────────

INTENT_PATTERNS = {
    "problems": [
        r"probl[eè]m", r"anomali", r"erreur", r"issue", r"soucis", r"défaut",
        r"problem", r"what.*(wrong|bad|issue)", r"qu['\s]est.ce qui", r"quoi.*tort"
    ],
    "score": [
        r"score", r"note", r"qualit", r"évaluat", r"résultat",
        r"how.*good", r"quality", r"rating", r"combien"
    ],
    "fix": [
        r"corrig", r"fix", r"réparer", r"améliorer", r"priorit", r"d'abord",
        r"first", r"correct", r"improve", r"que faire", r"comment"
    ],
    "compare": [
        r"compar", r"autres", r"benchmark", r"moyenne", r"classement",
        r"compare", r"others", r"average", r"ranking", r"mieux", r"pire"
    ],
    "explain": [
        r"pourquoi", r"expliqu", r"why", r"explain", r"reason", r"cause",
        r"qu['\s]est.ce que", r"signifi", r"meaning", r"comprend"
    ],
    "discipline": [
        r"architectur", r"mep", r"hvac", r"structure", r"disciplin",
        r"type.*maquette", r"quel.*modèle"
    ],
    "norms": [
        r"norme", r"réglementation", r"code.*construction", r"dtu", r"eurocode",
        r"standard", r"regulation", r"legal", r"obligatoire", r"required",
        r"pays", r"france", r"suisse", r"belgique", r"uk", r"europe"
    ],
    "greeting": [
        r"^(bonjour|hello|salut|hi|hey|bonsoir|coucou)[\s!?.]*$"
    ],
    "thanks": [
        r"^(merci|thank|thanks|bravo|super|parfait|génial|cool|ok|bien)[\s!.]*$"
    ],
    "offtrack": [
        r"météo", r"weather", r"recette", r"recipe", r"sport", r"foot",
        r"politique", r"politic", r"actualit", r"news", r"film", r"musique",
        r"blague", r"joke", r"histoire", r"story(?!.*bim)", r"code.*python(?!.*bim)",
    ],
    "identity": [
        r"qui es.tu", r"tu es.*ia", r"c['\s]est quoi ton nom", r"ton nom",
        r"what.*your name", r"who are you", r"what are you", r"quel.*ia",
        r"nom.*ia", r"ia.*nom",
    ],
    "element_count": [
        r"combien.*(porte|fenêtre|mur|poteau|dalle|escalier|pièce|zone|espace)",
        r"how many.*(door|window|wall|column|slab|stair|room|space|element)",
        r"nombre.*(porte|fenêtre|mur)", r"count.*element",
    ],
}

def detect_intent(question: str) -> str:
    """Détecte l'intention principale de la question."""
    q = question.lower().strip()
    for intent, patterns in INTENT_PATTERNS.items():
        for p in patterns:
            if re.search(p, q):
                return intent
    return "general"


# ── Générateur de réponses locales ────────────────────────────────────────────

def _score_label(score: float) -> str:
    if score >= 85: return "excellent"
    if score >= 70: return "good"
    if score >= 55: return "average"
    if score >= 40: return "below average"
    return "weak"

def _score_label_fr(score: float) -> str:
    if score >= 85: return "excellent"
    if score >= 70: return "bon"
    if score >= 55: return "moyen"
    if score >= 40: return "en dessous de la moyenne"
    return "faible"

def _pick(lst): return random.choice(lst)

def generate_local_response(question: str, analysis: dict, lang: str = "EN") -> str:
    """
    Génère une réponse locale basée sur les données réelles de la maquette.
    analysis = dict avec scores, anomalies, discipline, benchmark, etc.
    """
    intent = detect_intent(question)
    fr = lang == "FR"

    # Extraire les données clés
    score_global = analysis.get("score_global", 0)
    score_metier = analysis.get("score_metier", 0)
    score_data = analysis.get("score_data_bim", 0)
    discipline = analysis.get("discipline", "Unknown")
    n_critical = analysis.get("n_critical", 0)
    n_major = analysis.get("n_major", 0)
    n_minor = analysis.get("n_minor", 0)
    top_issues = analysis.get("top_issues", [])
    benchmark = analysis.get("benchmark_position", "")
    filename = analysis.get("filename", "this model")
    atypie = analysis.get("atypie_label", "")

    # ── HORS SUJET ────────────────────────────────────────────────────────────
    if intent == "offtrack":
        return _pick([
            _("I'm specialized in BIM quality analysis — that topic is outside my scope. Want to explore the results of your IFC model instead?",
              "Je suis spécialisé dans l'analyse qualité BIM — ce sujet dépasse mon périmètre. Je peux vous aider à analyser les résultats de votre maquette IFC si vous le souhaitez.",
              fr),
            _("That's outside my expertise. I'm here to help you understand and improve your BIM model. What would you like to know about it?",
              "Ce n'est pas mon domaine. Je suis là pour vous aider à comprendre et améliorer votre maquette BIM. Que voulez-vous savoir à son sujet ?",
              fr),
        ])

    # ── IDENTITÉ ─────────────────────────────────────────────────────────────
    if intent == "identity":
        return _pick([
            _("I'm **nokah** — an AI assistant specialized in BIM quality analysis. "
              "I analyze IFC models, detect anomalies, score quality, and help you understand your BIM.",
              "Je suis **nokah** — un assistant IA spécialisé dans l'analyse qualité BIM. "
              "J'analyse les maquettes IFC, détecte les anomalies, évalue la qualité et vous aide à comprendre votre BIM.",
              fr),
            _("I'm **nokah**, your BIM quality intelligence assistant. "
              "Ask me about your model's results, scores, or BIM best practices.",
              "Je suis **nokah**, votre assistant d'intelligence qualité BIM. "
              "Posez-moi des questions sur les résultats de votre maquette, les scores ou les bonnes pratiques BIM.",
              fr),
        ])

    # ── COMPTAGE ÉLÉMENTS ────────────────────────────────────────────────────
    if intent == "element_count":
        # Get counts from bim_json objects if available
        objects = analysis.get("objects", {})
        if objects:
            counts = "\n".join([f"• {k.capitalize()}: {v}" for k, v in list(objects.items())[:8]])
            return _(f"Here are the element counts for this model:\n{counts}",
                     f"Voici les comptages d'éléments de cette maquette :\n{counts}", fr)
        return _(
            "Element counts are available in the Expert mode section of the analysis. "
            "I can tell you about quality issues, scores, and recommendations based on those elements.",
            "Les comptages d'éléments sont disponibles dans la section Expert mode de l'analyse. "
            "Je peux vous informer sur les anomalies, scores et recommandations basés sur ces éléments.",
            fr)

    # ── SALUTATION ────────────────────────────────────────────────────────────
    if intent == "greeting":
        return _pick([
            _("Hello! I've analyzed your IFC model. Ask me anything about the results — scores, issues, recommendations.",
              "Bonjour ! J'ai analysé votre maquette IFC. Posez-moi vos questions sur les résultats — scores, anomalies, recommandations.",
              fr),
            _("Hi! Your model has been analyzed. I'm ready to answer your questions.",
              "Bonjour ! Votre maquette a été analysée. Je suis prêt à répondre à vos questions.",
              fr),
        ])

    # ── REMERCIEMENTS ─────────────────────────────────────────────────────────
    if intent == "thanks":
        return _pick([
            _("Glad to help! Any other questions about the model?",
              "Avec plaisir ! D'autres questions sur la maquette ?",
              fr),
            _("You're welcome. Feel free to ask anything else about your BIM.",
              "De rien. N'hésitez pas si vous avez d'autres questions sur votre BIM.",
              fr),
        ])

    # ── PROBLÈMES ─────────────────────────────────────────────────────────────
    if intent == "problems":
        if not top_issues:
            return _("No significant issues were detected in this model.",
                     "Aucune anomalie significative n'a été détectée dans cette maquette.", fr)

        issues_text = "\n".join([f"• {issue}" for issue in top_issues[:5]])
        severity = ""
        if n_critical > 0:
            severity = _pick([
                _(f"⚠️ {n_critical} critical issue{'s' if n_critical > 1 else ''} require immediate attention.",
                  f"⚠️ {n_critical} anomalie{'s' if n_critical > 1 else ''} critique{'s' if n_critical > 1 else ''} nécessite{'nt' if n_critical > 1 else ''} une attention immédiate.",
                  fr),
                _(f"The most urgent point: {n_critical} critical anomal{'ies' if n_critical > 1 else 'y'} detected.",
                  f"Le point le plus urgent : {n_critical} anomalie{'s' if n_critical > 1 else ''} critique{'s' if n_critical > 1 else ''} détectée{'s' if n_critical > 1 else ''}.",
                  fr),
            ])

        intro = _pick([
            _("Here are the main issues identified in your model:",
              "Voici les principaux problèmes identifiés dans votre maquette :", fr),
            _("The analysis detected the following issues:",
              "L'analyse a détecté les anomalies suivantes :", fr),
        ])
        total = _( f"In total: {n_critical} critical, {n_major} major, {n_minor} minor.",
                   f"Au total : {n_critical} critique{'s' if n_critical!=1 else ''}, {n_major} majeur{'s' if n_major!=1 else ''}, {n_minor} mineur{'s' if n_minor!=1 else ''}.", fr)

        return f"{severity}\n\n{intro}\n{issues_text}\n\n{total}"

    # ── SCORE ─────────────────────────────────────────────────────────────────
    if intent == "score":
        lbl = _score_label_fr(score_global) if fr else _score_label(score_global)
        return _pick([
            _(f"Your model scores **{score_global:.1f}/100** — {lbl}.\n\n"
              f"• Technical compliance: {score_metier:.1f}/100\n"
              f"• BIM data quality: {score_data:.1f}/100\n\n"
              f"{'The technical compliance score is pulling the overall score down.' if score_metier < score_data else 'BIM data quality is the main area to improve.' if score_data < score_metier else 'Both scores are balanced.'}",
              f"Votre maquette obtient **{score_global:.1f}/100** — niveau {lbl}.\n\n"
              f"• Conformité métier : {score_metier:.1f}/100\n"
              f"• Qualité données BIM : {score_data:.1f}/100\n\n"
              f"{'La conformité métier tire le score global vers le bas.' if score_metier < score_data else 'La qualité des données BIM est le principal axe d amélioration.' if score_data < score_metier else 'Les deux scores sont équilibrés.'}",
              fr),
        ])

    # ── CORRIGER / PRIORITÉS ──────────────────────────────────────────────────
    if intent == "fix":
        recs = []
        if n_critical > 0 and top_issues:
            recs.append(_(f"1. **Priority 1 — Critical issues first** ({n_critical} to fix): {top_issues[0] if top_issues else 'see list above'}",
                          f"1. **Priorité 1 — Anomalies critiques** ({n_critical} à corriger) : {top_issues[0] if top_issues else 'voir liste ci-dessus'}",
                          fr))
        if n_major > 0:
            recs.append(_(f"2. **Priority 2 — Major issues** ({n_major}): these affect functionality",
                          f"2. **Priorité 2 — Anomalies majeures** ({n_major}) : elles affectent la fonctionnalité",
                          fr))
        if score_data < 70:
            recs.append(_(f"3. **BIM data quality** (score: {score_data:.0f}/100): complete missing properties",
                          f"3. **Qualité données BIM** (score : {score_data:.0f}/100) : complétez les propriétés manquantes",
                          fr))
        if not recs:
            return _("Your model has no major issues to fix. Focus on the minor points to reach excellence.",
                     "Votre maquette n'a pas d'anomalies majeures à corriger. Concentrez-vous sur les points mineurs pour atteindre l'excellence.",
                     fr)
        intro = _pick([
            _("Here's the recommended correction order:", "Voici l'ordre de correction recommandé :", fr),
            _("To improve this model efficiently:", "Pour améliorer cette maquette efficacement :", fr),
        ])
        return intro + "\n\n" + "\n".join(recs)

    # ── COMPARAISON / BENCHMARK ───────────────────────────────────────────────
    if intent == "compare":
        if not benchmark:
            return _("Benchmark data is not available for this analysis.",
                     "Les données de benchmark ne sont pas disponibles pour cette analyse.", fr)
        bench_map = {
            "Top range": _("top range", "dans le top", fr),
            "Above typical": _("above average", "au-dessus de la moyenne", fr),
            "Within typical": _("within typical range", "dans la moyenne", fr),
            "Below typical": _("below average", "en dessous de la moyenne", fr),
        }
        bench_label = bench_map.get(benchmark, benchmark)
        return _pick([
            _(f"Compared to similar analyzed BIM models, yours is **{bench_label}** with a score of {score_global:.1f}/100.",
              f"Par rapport aux maquettes BIM similaires analysées, la vôtre se situe **{bench_label}** avec un score de {score_global:.1f}/100.",
              fr),
            _(f"This {discipline} model scores {score_global:.1f}/100, placing it **{bench_label}** in the nokah dataset.",
              f"Cette maquette {discipline} obtient {score_global:.1f}/100, la plaçant **{bench_label}** dans le dataset nokah.",
              fr),
        ])

    # ── EXPLIQUER ─────────────────────────────────────────────────────────────
    if intent == "explain":
        lbl = _score_label_fr(score_global) if fr else _score_label(score_global)
        reasons = []
        if n_critical > 0:
            reasons.append(_(f"{n_critical} critical issue{'s' if n_critical!=1 else ''}",
                             f"{n_critical} anomalie{'s' if n_critical!=1 else ''} critique{'s' if n_critical!=1 else ''}",
                             fr))
        if score_metier < 60:
            reasons.append(_("low technical compliance score",
                             "score de conformité métier faible", fr))
        if score_data < 60:
            reasons.append(_("incomplete BIM data",
                             "données BIM incomplètes", fr))

        if reasons:
            reason_str = ", ".join(reasons)
            return _(f"The score of {score_global:.1f}/100 ({lbl}) is mainly explained by: {reason_str}.\n\n"
                     f"The nokah score combines technical compliance (70%) and BIM data quality (30%).",
                     f"Le score de {score_global:.1f}/100 ({lbl}) s'explique principalement par : {reason_str}.\n\n"
                     f"Le score nokah combine conformité métier (70%) et qualité des données BIM (30%).",
                     fr)
        return _(f"The score of {score_global:.1f}/100 is {lbl}. No major issues were identified.",
                 f"Le score de {score_global:.1f}/100 est {lbl}. Aucune anomalie majeure n'a été identifiée.",
                 fr)

    # ── NORMES ────────────────────────────────────────────────────────────────
    if intent == "norms":
        return _(
            "nokah applies rules based on common BIM standards:\n\n"
            "• **IFC schema** (IFC 2x3 / IFC 4) — geometry and property validation\n"
            "• **Architecture**: wall types, door widths (≥0.80m), railing heights (≥0.90m exterior, ≥1.00m interior)\n"
            "• **Structure**: load-bearing elements, column/foundation matching, section classification\n"
            "• **MEP/HVAC**: duct coverage, equipment zoning, CVC system classification\n\n"
            "For country-specific regulations (DTU, Eurocodes, UK Building Regs), always refer to the official standards for your project location.",
            "nokah applique des règles basées sur les standards BIM courants :\n\n"
            "• **Schéma IFC** (IFC 2x3 / IFC 4) — validation géométrie et propriétés\n"
            "• **Architecture** : types de murs, largeurs de portes (≥0,80m), hauteurs de garde-corps (≥0,90m extérieur, ≥1,00m intérieur)\n"
            "• **Structure** : éléments porteurs, correspondance poteaux/fondations, classification des sections\n"
            "• **MEP/CVC** : couverture réseau, zonage équipements, classification systèmes\n\n"
            "Pour les réglementations locales spécifiques (DTU, Eurocodes, normes suisses), référez-vous toujours aux textes officiels applicables à votre projet.",
            fr)

    # ── DISCIPLINE ────────────────────────────────────────────────────────────
    if intent == "discipline":
        return _(f"This model was identified as **{discipline}**.\n"
                 f"nokah automatically detected the discipline based on the IFC element types and applied the corresponding rules.",
                 f"Cette maquette a été identifiée comme **{discipline}**.\n"
                 f"nokah a automatiquement détecté la discipline à partir des types d'éléments IFC et appliqué les règles correspondantes.",
                 fr)

    # ── RÉPONSE GÉNÉRALE ──────────────────────────────────────────────────────
    return _pick([
        _(f"Your {discipline} model scores {score_global:.1f}/100 with {n_critical} critical and {n_major} major issues. "
          f"Ask me about the problems, the score, how to fix it, or how it compares to other models.",
          f"Votre maquette {discipline} obtient {score_global:.1f}/100 avec {n_critical} anomalie{'s' if n_critical!=1 else ''} critique{'s' if n_critical!=1 else ''} et {n_major} majeure{'s' if n_major!=1 else ''}. "
          f"Posez-moi des questions sur les anomalies, le score, les corrections ou la comparaison avec d'autres maquettes.",
          fr),
        _(f"I can help you understand this model's analysis. "
          f"You can ask me: what are the issues? why this score? what to fix first? how does it compare?",
          f"Je peux vous aider à comprendre l'analyse de cette maquette. "
          f"Vous pouvez me demander : quelles sont les anomalies ? pourquoi ce score ? que corriger en priorité ? comment se compare-t-elle ?",
          fr),
    ])


def _(en: str, fr_text: str, use_fr: bool) -> str:
    """Helper bilingue."""
    return fr_text if use_fr else en


# ── Couche Groq (optionnelle) ─────────────────────────────────────────────────

def build_groq_prompt(question: str, analysis: dict, history: list, lang: str) -> list:
    """Construit le prompt pour Groq avec contexte BIM complet."""
    fr = lang == "FR"

    system_prompt = f"""You are nokah, an AI assistant specialized exclusively in BIM (Building Information Modeling) quality analysis.

You have analyzed an IFC model with the following results:
- Filename: {analysis.get('filename', 'unknown')}
- Discipline: {analysis.get('discipline', 'unknown')}
- Global quality score: {analysis.get('score_global', 0):.1f}/100
- Technical compliance score: {analysis.get('score_metier', 0):.1f}/100
- BIM data quality score: {analysis.get('score_data_bim', 0):.1f}/100
- Critical issues: {analysis.get('n_critical', 0)}
- Major issues: {analysis.get('n_major', 0)}
- Minor issues: {analysis.get('n_minor', 0)}
- Top issues: {', '.join(analysis.get('top_issues', [])[:5])}
- Benchmark position: {analysis.get('benchmark_position', 'N/A')}
- Atypicality: {analysis.get('atypie_label', 'N/A')}

IMPORTANT RULES:
1. Answer ONLY questions related to BIM, construction, architecture, MEP, structure, IFC, building standards, or this specific model's analysis.
2. If the question is off-topic (weather, politics, recipes, sports, general coding, etc.), politely redirect to BIM topics.
3. Base your answers on the REAL data above — never invent scores or issues.
4. Be precise, helpful, and professional.
5. Keep answers concise but complete.
6. Respond in {"French" if fr else "English"}.
7. You are nokah — not ChatGPT, not Claude. Don't mention other AI systems."""

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-6:]:  # Keep last 6 messages for context
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})
    return messages


def call_groq(question: str, analysis: dict, history: list, lang: str) -> str | None:
    """
    Appelle l'API Groq si la clé est disponible.
    Retourne None si pas de clé ou erreur.
    """
    try:
        import streamlit as st
        api_key = st.secrets.get("GROQ_API_KEY", "") or os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            return None

        import requests
        messages = build_groq_prompt(question, analysis, history, lang)

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.1-70b-versatile",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 600,
            },
            timeout=15
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        return None

    except Exception:
        return None


def get_chat_response(question: str, analysis: dict, history: list, lang: str = "EN") -> tuple[str, str]:
    """
    Point d'entrée principal du chat.
    Retourne (réponse, source) où source = "groq" ou "local"
    """
    # Essayer Groq d'abord
    groq_response = call_groq(question, analysis, history, lang)
    if groq_response:
        return groq_response, "groq"

    # Fallback local
    local_response = generate_local_response(question, analysis, lang)
    return local_response, "local"
