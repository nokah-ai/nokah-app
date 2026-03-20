"""
nokah_chat.py — Moteur de chat nokah
Couche 1 : réponses locales basées sur les vraies données
Couche 2 : Groq LLM (si GROQ_API_KEY dans os.environ)
"""
import random, re, os

INTENT_PATTERNS = {
    "problems":   [r"probl", r"anomali", r"erreur", r"issue", r"soucis", r"problem", r"wrong", r"bad"],
    "score":      [r"score", r"note", r"qualit", r"rating", r"combien", r"how good"],
    "fix":        [r"corrig", r"fix", r"priorit", r"first", r"correct", r"improve", r"que faire", r"what to"],
    "compare":    [r"compar", r"benchmark", r"moyenne", r"average", r"others", r"mieux", r"pire"],
    "explain":    [r"pourquoi", r"expliqu", r"why", r"explain", r"reason", r"cause", r"signifi"],
    "norms":      [r"norme", r"regl", r"dtu", r"eurocode", r"standard", r"legal", r"pays", r"france"],
    "greeting":   [r"^(bonjour|hello|salut|hi|hey|bonsoir)[\s!?.]*$"],
    "thanks":     [r"^(merci|thank|thanks|ok|bien|super|parfait)[\s!.]*$"],
    "identity":   [r"qui es.tu", r"ton nom", r"your name", r"who are you", r"what are you", r"nom.*ia"],
    "offtrack":   [r"meteo", r"weather", r"recette", r"recipe", r"sport", r"foot", r"politique", r"film", r"blague"],
}

def detect_intent(q):
    q = q.lower().strip()
    for intent, patterns in INTENT_PATTERNS.items():
        for p in patterns:
            if re.search(p, q):
                return intent
    return "general"

def _(en, fr, use_fr):
    return fr if use_fr else en

def _pick(lst):
    return random.choice(lst)

def generate_local_response(question, analysis, lang="EN"):
    intent = detect_intent(question)
    fr = lang == "FR"
    sg = analysis.get("score_global", 0)
    sm = analysis.get("score_metier", 0)
    sd = analysis.get("score_data_bim", 0)
    disc = analysis.get("discipline", "Unknown")
    nc = analysis.get("n_critical", 0)
    nma = analysis.get("n_major", 0)
    nmi = analysis.get("n_minor", 0)
    issues = analysis.get("top_issues", [])
    bench = analysis.get("benchmark_position", "")

    if intent == "offtrack":
        return _pick([
            _("I'm specialized in BIM quality — that's outside my scope. Ask me about your model.",
              "Je suis spécialisé en qualité BIM — ce sujet dépasse mon périmètre. Posez-moi des questions sur votre maquette.", fr),
        ])
    if intent == "identity":
        return _pick([
            _("I'm **nokah** — an AI specialized in BIM quality analysis. I analyze IFC files, score quality, and detect anomalies.",
              "Je suis **nokah** — une IA spécialisée dans l'analyse qualité BIM. J'analyse les fichiers IFC, évalue la qualité et détecte les anomalies.", fr),
        ])
    if intent == "greeting":
        return _("Hello! Your model has been analyzed. Ask me anything about the results.",
                 "Bonjour ! Votre maquette a été analysée. Posez-moi vos questions.", fr)
    if intent == "thanks":
        return _("Glad to help! Any other questions?", "Avec plaisir ! D'autres questions ?", fr)

    if intent == "problems":
        if not issues and nc == 0 and nma == 0:
            return _("No significant issues detected.", "Aucune anomalie significative détectée.", fr)
        lines = "\n".join([f"• {i}" for i in issues[:5]])
        total = _(f"Total: {nc} critical, {nma} major, {nmi} minor.",
                  f"Total : {nc} critique(s), {nma} majeure(s), {nmi} mineure(s).", fr)
        intro = _("Main issues detected:", "Principales anomalies détectées :", fr)
        warn = _(f"⚠️ {nc} critical issue(s) require immediate attention.\n\n",
                 f"⚠️ {nc} anomalie(s) critique(s) nécessite(nt) une attention immédiate.\n\n", fr) if nc > 0 else ""
        return f"{warn}{intro}\n{lines}\n\n{total}"

    if intent == "score":
        lbl = ("excellent" if sg>=85 else "bon" if sg>=70 else "moyen" if sg>=55 else "faible") if fr else ("excellent" if sg>=85 else "good" if sg>=70 else "average" if sg>=55 else "weak")
        dom = _("Technical compliance is lower." if sm < sd-10 else "BIM data quality is lower." if sd < sm-10 else "Both scores are balanced.",
                "La conformité métier est plus faible." if sm < sd-10 else "La qualité des données est plus faible." if sd < sm-10 else "Les deux scores sont équilibrés.", fr)
        return _(f"Score: **{sg:.1f}/100** ({lbl})\n• Technical: {sm:.1f}/100\n• BIM data: {sd:.1f}/100\n\n{dom}",
                 f"Score : **{sg:.1f}/100** ({lbl})\n• Conformité : {sm:.1f}/100\n• Données BIM : {sd:.1f}/100\n\n{dom}", fr)

    if intent == "fix":
        recs = []
        if nc > 0:
            recs.append(_(f"1. **Critical first** ({nc}): {issues[0] if issues else 'see list'}",
                          f"1. **Critiques en priorité** ({nc}) : {issues[0] if issues else 'voir liste'}", fr))
        if nma > 0:
            recs.append(_(f"2. **Major issues** ({nma}): affect functionality",
                          f"2. **Anomalies majeures** ({nma}) : affectent la fonctionnalité", fr))
        if sd < 70:
            recs.append(_(f"3. **BIM data quality** ({sd:.0f}/100): complete missing properties",
                          f"3. **Qualité données BIM** ({sd:.0f}/100) : compléter les propriétés manquantes", fr))
        if not recs:
            return _("No major issues. Focus on minor points for excellence.",
                     "Pas d'anomalies majeures. Concentrez-vous sur les points mineurs.", fr)
        intro = _("Recommended correction order:", "Ordre de correction recommandé :", fr)
        return intro + "\n\n" + "\n".join(recs)

    if intent == "compare":
        if not bench:
            return _("No benchmark data available yet.", "Pas encore de données benchmark disponibles.", fr)
        return _(f"This {disc} model ({sg:.1f}/100) is positioned: **{bench}** vs similar models.",
                 f"Cette maquette {disc} ({sg:.1f}/100) se positionne : **{bench}** par rapport aux maquettes similaires.", fr)

    if intent == "explain":
        reasons = []
        if nc > 0: reasons.append(_(f"{nc} critical issue(s)", f"{nc} anomalie(s) critique(s)", fr))
        if sm < 60: reasons.append(_("low technical compliance", "conformité métier faible", fr))
        if sd < 60: reasons.append(_("incomplete BIM data", "données BIM incomplètes", fr))
        lbl = ("excellent" if sg>=85 else "bon" if sg>=70 else "moyen" if sg>=55 else "faible") if fr else ("excellent" if sg>=85 else "good" if sg>=70 else "average" if sg>=55 else "weak")
        if reasons:
            return _(f"Score {sg:.1f}/100 ({lbl}) explained by: {', '.join(reasons)}.\nFormula: 70% technical + 30% BIM data quality.",
                     f"Score {sg:.1f}/100 ({lbl}) expliqué par : {', '.join(reasons)}.\nFormule : 70% conformité + 30% qualité données.", fr)
        return _(f"Score {sg:.1f}/100 ({lbl}). No major issues identified.",
                 f"Score {sg:.1f}/100 ({lbl}). Aucune anomalie majeure identifiée.", fr)

    if intent == "norms":
        return _("nokah rules are based on IFC schema (2x3/4), architecture (door widths ≥0.80m, railing heights), structure (load-bearing), and MEP/HVAC standards. For country-specific rules (DTU, Eurocodes), refer to official documentation.",
                 "Les règles nokah sont basées sur le schéma IFC (2x3/4), l'architecture (largeurs portes ≥0,80m, garde-corps), la structure (porteurs) et les standards MEP/CVC. Pour les normes locales (DTU, Eurocodes), consultez les textes officiels.", fr)

    return _(f"Your {disc} model: {sg:.1f}/100 — {nc} critical, {nma} major issues. Ask about problems, score, fixes, or benchmark.",
             f"Maquette {disc} : {sg:.1f}/100 — {nc} critique(s), {nma} majeure(s). Posez des questions sur les anomalies, le score, les corrections ou le benchmark.", fr)


def call_groq(question, analysis, history, lang):
    """Appelle Groq si GROQ_API_KEY est dans os.environ."""
    try:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            return None
        import requests
        fr = lang == "FR"
        system = f"""You are nokah, an AI specialized in BIM quality analysis.
Model analyzed: {analysis.get('filename','unknown')} | Discipline: {analysis.get('discipline','unknown')}
Score: {analysis.get('score_global',0):.1f}/100 | Technical: {analysis.get('score_metier',0):.1f}/100 | BIM data: {analysis.get('score_data_bim',0):.1f}/100
Critical issues: {analysis.get('n_critical',0)} | Major: {analysis.get('n_major',0)} | Minor: {analysis.get('n_minor',0)}
Top issues: {", ".join(analysis.get('top_issues',[])[:5])}
Benchmark: {analysis.get('benchmark_position','N/A')}
RULES: Only answer BIM/construction/IFC questions. If off-topic, redirect politely. Never invent data. Respond in {"French" if fr else "English"}."""
        msgs = [{"role":"system","content":system}]
        for m in history[-6:]:
            msgs.append({"role":m["role"],"content":m["content"]})
        msgs.append({"role":"user","content":question})
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"},
            json={"model":"llama-3.3-70b-versatile","messages":msgs,"temperature":0.7,"max_tokens":600},
            timeout=15
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        return None
    except Exception:
        return None


def get_chat_response(question, analysis, history, lang="EN"):
    """Point d'entrée — essaie Groq d'abord, fallback local."""
    resp = call_groq(question, analysis, history, lang)
    if resp:
        return resp, "groq"
    return generate_local_response(question, analysis, lang), "local"
