"""
nokah — Freemium Module
Compteur persistant via cookies navigateur + page de blocage
"""
import streamlit as st
import json

# ── Configuration ─────────────────────────────────────────────────────────────
FREE_ANALYSES = 3
STARTER_BONUS = 3
APP_URL        = "https://nokah-app-8ecvxvmgro2m7jv6aszjjs.streamlit.app"
STRIPE_STARTER = "https://buy.stripe.com/14A3cp4pn7fggHtdeyb7y01"
STRIPE_PRO     = "https://buy.stripe.com/3cIeV79JHbvwaj56Qab7y00"

ACCESS_CODES = {
    "NK-STARTER": {"plan": "starter", "extra_analyses": STARTER_BONUS},
    "NK-PRO":     {"plan": "pro",     "extra_analyses": 999999},
    "NK-OLIVIA":  {"plan": "starter", "extra_analyses": STARTER_BONUS},
    "NK-ELIOTT":  {"plan": "starter", "extra_analyses": STARTER_BONUS},
}

# ── Cookie JS helpers ──────────────────────────────────────────────────────────
_COOKIE_READER = """
<script>
(function() {
    function getCookie(name) {
        const v = document.cookie.split('; ').find(r => r.startsWith(name + '='));
        return v ? decodeURIComponent(v.split('=')[1]) : null;
    }
    function setCookie(name, value, days) {
        const d = new Date();
        d.setTime(d.getTime() + days*24*60*60*1000);
        document.cookie = name + '=' + encodeURIComponent(value) + ';expires=' + d.toUTCString() + ';path=/';
    }

    // Lire le cookie existant
    const raw = getCookie('nokah_freemium');
    const data = raw ? JSON.parse(raw) : {count: 0, plan: 'free', extra: 0};

    // Injecter dans un input caché pour que Streamlit puisse le lire
    let inp = document.getElementById('nokah_cookie_bridge');
    if (!inp) {
        inp = document.createElement('input');
        inp.type = 'hidden';
        inp.id = 'nokah_cookie_bridge';
        document.body.appendChild(inp);
    }
    inp.value = JSON.stringify(data);

    // Exposer globalement pour les mises à jour
    window._nokahData = data;
    window._nokahSetCookie = function(data) {
        setCookie('nokah_freemium', JSON.stringify(data), 365);
        window._nokahData = data;
    };
})();
</script>
"""

_COOKIE_INCREMENT = """
<script>
(function() {
    function getCookie(name) {
        const v = document.cookie.split('; ').find(r => r.startsWith(name + '='));
        return v ? decodeURIComponent(v.split('=')[1]) : null;
    }
    function setCookie(name, value, days) {
        const d = new Date();
        d.setTime(d.getTime() + days*24*60*60*1000);
        document.cookie = name + '=' + encodeURIComponent(value) + ';expires=' + d.toUTCString() + ';path=/';
    }
    const raw = getCookie('nokah_freemium');
    const data = raw ? JSON.parse(raw) : {count: 0, plan: 'free', extra: 0};
    data.count = (data.count || 0) + 1;
    setCookie('nokah_freemium', JSON.stringify(data), 365);
})();
</script>
"""

def _make_apply_code_js(plan, extra):
    return f"""
<script>
(function() {{
    function getCookie(name) {{
        const v = document.cookie.split('; ').find(r => r.startsWith(name + '='));
        return v ? decodeURIComponent(v.split('=')[1]) : null;
    }}
    function setCookie(name, value, days) {{
        const d = new Date();
        d.setTime(d.getTime() + days*24*60*60*1000);
        document.cookie = name + '=' + encodeURIComponent(value) + ';expires=' + d.toUTCString() + ';path=/';
    }}
    const raw = getCookie('nokah_freemium');
    const data = raw ? JSON.parse(raw) : {{count: 0, plan: 'free', extra: 0}};
    data.plan = '{plan}';
    data.extra = {extra};
    setCookie('nokah_freemium', JSON.stringify(data), 365);
}})();
</script>
"""

# ── Session init ───────────────────────────────────────────────────────────────
def init_session():
    """Initialise la session depuis le cookie navigateur."""
    if "nk_initialized" not in st.session_state:
        st.session_state.nk_initialized   = False
    if "analyses_count" not in st.session_state:
        st.session_state.analyses_count    = 0
    if "plan" not in st.session_state:
        st.session_state.plan              = "free"
    if "extra_analyses" not in st.session_state:
        st.session_state.extra_analyses    = 0
    if "access_code_applied" not in st.session_state:
        st.session_state.access_code_applied = False
    if "pending_increment" not in st.session_state:
        st.session_state.pending_increment = False

    # Injecter le lecteur de cookie — Streamlit va le rendre à chaque run
    # On lit le cookie via query_params comme bridge (méthode fiable)
    st.markdown(_COOKIE_READER, unsafe_allow_html=True)

    # Bridge via query params : si ?nk_count=X est dans l'URL, on l'utilise
    params = st.query_params
    if "nk_count" in params:
        try:
            st.session_state.analyses_count = int(params["nk_count"])
        except Exception:
            pass
    if "nk_plan" in params:
        st.session_state.plan = params["nk_plan"]
    if "nk_extra" in params:
        try:
            st.session_state.extra_analyses = int(params["nk_extra"])
        except Exception:
            pass

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_remaining():
    total = FREE_ANALYSES + st.session_state.extra_analyses
    if st.session_state.plan == "pro":
        return 999999
    return max(0, total - st.session_state.analyses_count)

def can_analyze():
    return get_remaining() > 0 or st.session_state.plan == "pro"

def increment_counter():
    """Incrémente le compteur en session ET dans le cookie."""
    st.session_state.analyses_count += 1
    # Persister dans le cookie via JS
    st.markdown(_COOKIE_INCREMENT, unsafe_allow_html=True)
    # Mettre à jour les query params pour la prochaine session
    st.query_params["nk_count"] = str(st.session_state.analyses_count)
    st.query_params["nk_plan"]  = st.session_state.plan
    st.query_params["nk_extra"] = str(st.session_state.extra_analyses)

def apply_access_code(code: str) -> bool:
    code = code.strip().upper()
    if code in ACCESS_CODES:
        info = ACCESS_CODES[code]
        st.session_state.plan           = info["plan"]
        st.session_state.extra_analyses = info["extra_analyses"]
        st.session_state.access_code_applied = True
        # Persister dans cookie + query params
        st.markdown(_make_apply_code_js(info["plan"], info["extra_analyses"]), unsafe_allow_html=True)
        st.query_params["nk_plan"]  = info["plan"]
        st.query_params["nk_extra"] = str(info["extra_analyses"])
        return True
    return False

# ── UI Components ──────────────────────────────────────────────────────────────
def render_analysis_counter():
    remaining = get_remaining()
    plan = st.session_state.plan
    if plan == "pro":
        st.markdown("""
        <div style="background:linear-gradient(135deg,#0ea5e9,#2563eb);border-radius:8px;
        padding:8px 16px;margin-bottom:16px;display:inline-block;font-size:13px;
        color:white;font-weight:500;">✓ Pro — Unlimited analyses</div>
        """, unsafe_allow_html=True)
    elif plan == "starter":
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#7c3aed,#4f46e5);border-radius:8px;
        padding:8px 16px;margin-bottom:16px;display:inline-block;font-size:13px;
        color:white;font-weight:500;">⚡ Starter — {remaining} {'analysis' if remaining==1 else 'analyses'} remaining</div>
        """, unsafe_allow_html=True)
    else:
        color = "#22c55e" if remaining > 1 else "#f59e0b" if remaining == 1 else "#ef4444"
        st.markdown(f"""
        <div style="background:{color}20;border:1px solid {color}60;border-radius:8px;
        padding:8px 16px;margin-bottom:16px;display:inline-block;font-size:13px;
        color:{color};font-weight:500;">{'✓' if remaining>0 else '✗'} Free — {remaining} {'analysis' if remaining==1 else 'analyses'} remaining</div>
        """, unsafe_allow_html=True)

def render_paywall():
    """Affiche la page de blocage avec auto-scroll."""

    # Auto-scroll
    st.markdown("""
    <div id="nokah-paywall-anchor"></div>
    <script>
        setTimeout(function() {
            var el = document.getElementById('nokah-paywall-anchor');
            if (el) { el.scrollIntoView({behavior: 'smooth', block: 'start'}); }
        }, 400);
    </script>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    .paywall-container{text-align:center;padding:40px 20px;max-width:700px;margin:0 auto;}
    .paywall-title{font-size:28px;font-weight:700;color:#f1f5f9;margin-bottom:8px;}
    .paywall-subtitle{font-size:16px;color:#94a3b8;margin-bottom:40px;}
    .plan-card{background:#1e293b;border:1px solid #334155;border-radius:16px;
               padding:28px 24px;margin:12px;transition:transform 0.2s;}
    .plan-card:hover{transform:translateY(-2px);}
    .plan-card.featured{border:2px solid #3b82f6;
                        background:linear-gradient(135deg,#1e293b,#1e3a5f);}
    .plan-name{font-size:18px;font-weight:700;color:#f1f5f9;margin-bottom:4px;}
    .plan-price{font-size:32px;font-weight:800;color:#3b82f6;margin:8px 0;}
    .plan-price span{font-size:14px;color:#94a3b8;font-weight:400;}
    .plan-feature{font-size:14px;color:#94a3b8;margin:6px 0;}
    .plan-feature::before{content:"✓ ";color:#22c55e;font-weight:700;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="paywall-container">
        <div style="font-size:48px;margin-bottom:16px;">🔒</div>
        <div class="paywall-title">You've used your 3 free analyses</div>
        <div class="paywall-subtitle">Choose a plan to continue or enter an access code</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="plan-card">
            <div class="plan-name">⚡ Starter</div>
            <div class="plan-price">€29<span>/month</span></div>
            <div class="plan-feature">6 IFC analyses per month</div>
            <div class="plan-feature">Quality score & anomaly detection</div>
            <div class="plan-feature">3D anomaly viewer</div>
            <div class="plan-feature">PDF export</div>
        </div>
        """, unsafe_allow_html=True)
        st.link_button("Start Starter →", STRIPE_STARTER, use_container_width=True)

    with col2:
        st.markdown("""
        <div class="plan-card featured">
            <div style="font-size:11px;color:#3b82f6;font-weight:700;margin-bottom:8px;">MOST POPULAR</div>
            <div class="plan-name">🚀 Pro</div>
            <div class="plan-price">€49<span>/month</span></div>
            <div class="plan-feature">Unlimited IFC analyses</div>
            <div class="plan-feature">Custom BIM conventions</div>
            <div class="plan-feature">3D anomaly viewer</div>
            <div class="plan-feature">Priority support</div>
        </div>
        """, unsafe_allow_html=True)
        st.link_button("Start Pro →", STRIPE_PRO, use_container_width=True, type="primary")

    st.markdown(
        "<div style='text-align:center;color:#475569;margin:24px 0;font-size:14px;'>"
        "— or enter an access code —</div>",
        unsafe_allow_html=True
    )

    col_code, col_btn = st.columns([3, 1])
    with col_code:
        code_input = st.text_input(
            "", placeholder="Enter your access code (e.g. NK-STARTER)",
            label_visibility="collapsed", key="paywall_code_input"
        )
    with col_btn:
        if st.button("Apply", use_container_width=True, type="secondary"):
            if code_input and apply_access_code(code_input):
                st.success(f"✓ Code applied! You have {get_remaining()} additional analyses.")
                st.rerun()
            else:
                st.error("Invalid code.")

    st.markdown(
        "<div style='text-align:center;margin-top:32px;'>"
        "<a href='mailto:contact@nokah.ai' style='color:#475569;font-size:13px;'>"
        "Need a custom plan? Contact us →</a></div>",
        unsafe_allow_html=True
    )
