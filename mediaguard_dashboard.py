import streamlit as st
import pandas as pd
import json
import re

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediGuard AI · FWA Dashboard",
    page_icon="🛡️",
    layout="wide",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Hide Streamlit chrome ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid="stHeader"] {display: none !important;}

    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    .block-container { padding: 0 0 2rem; background: #F7F9FC; }

    /* ── Top header bar ── */
    .header-bar {
        background: #1E3A5F;
        padding: 1.1rem 2.5rem .9rem;
        margin-bottom: 1.5rem;
    }
    .header-bar h1 {
        color: #FFFFFF;
        font-size: 1.55rem;
        font-weight: 800;
        margin: 0 0 .15rem;
        line-height: 1.2;
    }
    .header-bar p {
        color: #BFDBFE;
        font-size: .82rem;
        margin: 0;
    }

    /* ── KPI cards ── */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #DBEAFE;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,.05);
    }
    .kpi-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: .35rem;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        line-height: 1;
        color: #2563EB;
    }
    .kpi-value.danger { color: #DC2626; }
    .kpi-value.navy   { color: #1E3A5F; }

    /* ── Risk badges (narrative panel) ── */
    .badge {
        display: inline-block;
        padding: .2rem .65rem;
        border-radius: 9999px;
        font-size: .75rem;
        font-weight: 700;
        letter-spacing: .04em;
    }
    .badge-high   { background: #DC2626; color: #FFFFFF; }
    .badge-medium { background: #D97706; color: #FFFFFF; }
    .badge-low    { background: #16A34A; color: #FFFFFF; }
    .badge-siu    { background: #EDE9FE; color: #5B21B6; }

    /* ── Narrative panel ── */
    .narrative-box {
        background: #FFFFFF;
        border-left: 4px solid #2563EB;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,.06);
        margin-top: .5rem;
    }
    .narrative-box h4 { color: #1E3A5F; margin-top: 0; font-size: 1rem; }
    .narrative-body   { color: #334155; font-size: .9rem; line-height: 1.65; }
    .red-flag-chip {
        display: inline-block;
        background: #FEE2E2;
        color: #991B1B;
        border-radius: 6px;
        padding: .15rem .55rem;
        font-size: .72rem;
        font-weight: 600;
        margin: .15rem .2rem .15rem 0;
    }

    /* ── Right panel card ── */
    .right-panel-card {
        background: #FFFFFF;
        border: 1px solid #DBEAFE;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: .65rem;
        box-shadow: 0 1px 4px rgba(0,0,0,.04);
    }

    /* ── Table container ── */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #DBEAFE;
    }
    .section-header {
        font-size: 1rem;
        font-weight: 700;
        color: #1E3A5F;
        margin: 1.4rem 0 .6rem;
        letter-spacing: .02em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str = "output/fwa_ai_report.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ai_risk_score"] = pd.to_numeric(df["ai_risk_score"], errors="coerce")
    df["composite_risk_score"] = pd.to_numeric(df["composite_risk_score"], errors="coerce")
    df["billed_amount"] = pd.to_numeric(df["billed_amount"], errors="coerce")
    df["date_of_service"] = pd.to_datetime(df["date_of_service"], errors="coerce")

    def risk_tier(score):
        if pd.isna(score):
            return "Unknown"
        if score >= 80:
            return "HIGH"
        if score >= 50:
            return "MEDIUM"
        return "LOW"

    df["risk_tier"] = df["ai_risk_score"].apply(risk_tier)
    return df


df = load_data()

# ── Header bar ────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='header-bar'>"
    "<h1>🛡️ MediGuard AI — FWA Claims Dashboard</h1>"
    "<p>Fraud, Waste &amp; Abuse detection powered by Groq LLM reasoning</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ── KPI area — placeholder rendered here (above filters) ─────────────────────
# Filled after filters are defined and `filtered` is computed.
kpi_area = st.container()

# ── Filter row ────────────────────────────────────────────────────────────────
st.markdown(
    '<hr style="border:none;border-top:1px solid #DBEAFE;margin:.25rem 0 .75rem;">',
    unsafe_allow_html=True,
)

f1, f2, f3, f4, f5 = st.columns(5)

with f1:
    risk_options = ["All"] + sorted(df["risk_tier"].dropna().unique().tolist())
    selected_risk = st.selectbox("Risk Tier", risk_options)

with f2:
    rec_options = ["All"] + sorted(df["ai_recommendation"].dropna().unique().tolist())
    selected_rec = st.selectbox("Recommendation", rec_options)

with f3:
    specialty_options = ["All"] + sorted(df["provider_specialty"].dropna().unique().tolist())
    selected_specialty = st.selectbox("Provider Specialty", specialty_options)

with f4:
    state_options = ["All"] + sorted(df["provider_state"].dropna().unique().tolist())
    selected_state = st.selectbox("Provider State", state_options)

with f5:
    min_score, max_score = st.slider(
        "AI Risk Score Range",
        min_value=0, max_value=100,
        value=(0, 100), step=1,
    )

st.markdown(
    '<hr style="border:none;border-top:1px solid #DBEAFE;margin:.75rem 0 1rem;">',
    unsafe_allow_html=True,
)

# ── Apply filters ─────────────────────────────────────────────────────────────
filtered = df.copy()
if selected_risk != "All":
    filtered = filtered[filtered["risk_tier"] == selected_risk]
if selected_rec != "All":
    filtered = filtered[filtered["ai_recommendation"] == selected_rec]
if selected_specialty != "All":
    filtered = filtered[filtered["provider_specialty"] == selected_specialty]
if selected_state != "All":
    filtered = filtered[filtered["provider_state"] == selected_state]
filtered = filtered[
    (filtered["ai_risk_score"] >= min_score) &
    (filtered["ai_risk_score"] <= max_score)
]

# ── KPI cards — fill the placeholder reserved above the filter row ────────────
with kpi_area:
    total_claims = len(filtered)
    avg_risk = filtered["ai_risk_score"].mean()
    siu_count = (filtered["ai_recommendation"] == "REFER_TO_SIU").sum()
    high_count = (filtered["risk_tier"] == "HIGH").sum()
    total_billed = filtered["billed_amount"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Total Flagged Claims</div>'
            f'<div class="kpi-value">{total_claims:,}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        score_class = "danger" if (avg_risk or 0) >= 50 else ""
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Avg AI Risk Score</div>'
            f'<div class="kpi-value {score_class}">{avg_risk:.1f}</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Refer to SIU</div>'
            f'<div class="kpi-value danger">{siu_count:,}</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">High Risk Claims</div>'
            f'<div class="kpi-value danger">{high_count:,}</div></div>',
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Total Billed Amount</div>'
            f'<div class="kpi-value navy">${total_billed:,.0f}</div></div>',
            unsafe_allow_html=True,
        )

# ── Build display dataframe & styler ─────────────────────────────────────────
TABLE_COLS = [
    "claim_id", "patient_id", "provider_name", "provider_specialty",
    "provider_state", "date_of_service", "cpt_code", "cpt_description",
    "billed_amount", "ai_risk_score", "risk_tier",
    "ai_fraud_category", "ai_recommendation",
]

display_df = filtered[TABLE_COLS].copy()
display_df["date_of_service"] = display_df["date_of_service"].dt.strftime("%Y-%m-%d")
display_df["cpt_code"] = display_df["cpt_code"].apply(
    lambda x: str(int(x)).zfill(5) if pd.notna(x) else ""
)
display_df["billed_amount"] = display_df["billed_amount"].apply(
    lambda x: f"${x:,.2f}" if pd.notna(x) else ""
)

def risk_style(val):
    styles = {
        "HIGH":   "background-color: #DC2626; color: #FFFFFF; font-weight: 700;",
        "MEDIUM": "background-color: #D97706; color: #FFFFFF; font-weight: 700;",
        "LOW":    "background-color: #16A34A; color: #FFFFFF; font-weight: 700;",
    }
    return styles.get(val, "background-color: #FFFFFF; color: #1a1a1a;")

def alternating_rows(row):
    bg = "#F0F7FF" if row.name % 2 == 0 else "#FFFFFF"
    return [f"background-color: {bg}; color: #1a1a1a;" for _ in row]

styled = (
    display_df.style
    .apply(alternating_rows, axis=1)
    .map(risk_style, subset=["risk_tier"])
    .set_table_styles([
        {"selector": "thead th", "props": [
            ("background-color", "#1E3A5F"),
            ("color", "#FFFFFF"),
            ("font-weight", "700"),
            ("font-size", ".8rem"),
            ("letter-spacing", ".04em"),
            ("border-bottom", "2px solid #1E3A5F"),
        ]},
        {"selector": "td", "props": [
            ("font-size", ".82rem"),
            ("border-color", "#DBEAFE"),
        ]},
        {"selector": "table", "props": [("border-collapse", "collapse")]},
    ])
)

# ── Claim detail dialog ───────────────────────────────────────────────────────
@st.dialog("🔍 Claim Investigation", width="large")
def show_claim_dialog(row):
    claim_id = row["claim_id"]
    tier = row.get("risk_tier", "")
    badge_class = {
        "HIGH": "badge-high", "MEDIUM": "badge-medium", "LOW": "badge-low"
    }.get(tier, "")
    rec = str(row.get("ai_recommendation", ""))
    rec_badge = "badge-siu" if rec == "REFER_TO_SIU" else "badge-low"
    risk_val_color = {
        "HIGH": "#DC2626", "MEDIUM": "#D97706", "LOW": "#16A34A"
    }.get(tier, "#2563EB")

    st.markdown(
        f'<div style="font-size:.8rem;color:#64748B;margin-bottom:1rem;">'
        f'Claim ID: <strong style="color:#1E3A5F;">{claim_id}</strong></div>',
        unsafe_allow_html=True,
    )

    # ── Top 3 cards: Risk Score, Recommendation, Fraud Category ──
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f'<div class="right-panel-card"><div class="kpi-label">Risk Score</div>'
            f'<div style="font-size:1.5rem;font-weight:800;color:{risk_val_color};line-height:1;">'
            f'{int(row["ai_risk_score"]) if pd.notna(row["ai_risk_score"]) else "—"}</div>'
            f'<span class="badge {badge_class}">{tier}</span></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="right-panel-card">'
            f'<div class="kpi-label" style="font-size:.6rem;letter-spacing:.05em;">Recommendation</div>'
            f'<div style="margin-top:.3rem;"><span class="badge {rec_badge}">{rec}</span></div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="right-panel-card"><div class="kpi-label">Fraud Category</div>'
            f'<div style="font-weight:700;color:#1E3A5F;font-size:.85rem;margin-top:.25rem;">'
            f'{row.get("ai_fraud_category","—")}</div></div>',
            unsafe_allow_html=True,
        )

    # ── Provider info ──
    st.markdown(
        f'<div class="right-panel-card">'
        f'<div class="kpi-label">Provider</div>'
        f'<div style="font-weight:600;font-size:.88rem;color:#1E293B;">{row.get("provider_name","—")}</div>'
        f'<div style="font-size:.75rem;color:#64748B;">'
        f'{row.get("provider_specialty","—")} &middot; {row.get("provider_state","—")}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Red flags ──
    raw_flags = str(row.get("ai_red_flags", ""))
    flags = [f.strip() for f in raw_flags.split("|") if f.strip() and f.strip() != "nan"]
    flags_html = "".join(f'<span class="red-flag-chip">{f}</span>' for f in flags)
    if flags_html:
        st.markdown(
            f'<div class="right-panel-card">'
            f'<div class="kpi-label" style="margin-bottom:.4rem;">Red Flags</div>'
            f'{flags_html}</div>',
            unsafe_allow_html=True,
        )

    # ── Groq AI narrative ──
    narrative = str(row.get("ai_narrative", "No narrative available."))
    if narrative == "nan":
        narrative = "No narrative available."
    st.markdown(
        f'<div class="narrative-box">'
        f'<h4>🤖 Groq AI Analysis</h4>'
        f'<p class="narrative-body">{narrative}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Rule engine flags (expandable) ──
    raw_rule_flags = row.get("rule_flags", "[]")
    try:
        rule_flags = json.loads(str(raw_rule_flags)) if raw_rule_flags and str(raw_rule_flags) != "nan" else []
    except (json.JSONDecodeError, ValueError):
        rule_flags = []

    if rule_flags:
        with st.expander("Rule Engine Flags Detail"):
            for flag in rule_flags:
                sev = flag.get("severity", "LOW")
                sev_color = {"HIGH": "#DC2626", "MEDIUM": "#D97706", "LOW": "#16A34A"}.get(sev, "#64748B")
                st.markdown(
                    f'<div style="border:1px solid #DBEAFE;border-left:4px solid {sev_color};'
                    f'border-radius:6px;padding:.75rem 1rem;margin-bottom:.5rem;background:#fff;">'
                    f'<strong style="color:{sev_color};">[{sev}]</strong>'
                    f'<strong style="color:#1E293B;"> {flag.get("rule","")}</strong><br>'
                    f'<span style="color:#475569;font-size:.85rem;">{flag.get("detail","")}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Close", type="secondary", use_container_width=True):
        st.rerun()


# ── Full-width claims table ───────────────────────────────────────────────────
st.markdown('<div class="section-header">Flagged Claims</div>', unsafe_allow_html=True)
event = st.dataframe(
    styled,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    height=560,
    column_config={
        "claim_id":           st.column_config.TextColumn("Claim ID"),
        "patient_id":         st.column_config.TextColumn("Patient ID"),
        "provider_name":      st.column_config.TextColumn("Provider"),
        "provider_specialty": st.column_config.TextColumn("Specialty"),
        "provider_state":     st.column_config.TextColumn("State"),
        "date_of_service":    st.column_config.TextColumn("Date of Service"),
        "cpt_code":           st.column_config.TextColumn("CPT"),
        "cpt_description":    st.column_config.TextColumn("CPT Description"),
        "billed_amount":      st.column_config.TextColumn("Billed"),
        "ai_risk_score":      st.column_config.ProgressColumn(
            "Risk Score", min_value=0, max_value=100, format="%d",
        ),
        "risk_tier":          st.column_config.TextColumn("Risk Tier"),
        "ai_fraud_category":  st.column_config.TextColumn("Fraud Category"),
        "ai_recommendation":  st.column_config.TextColumn("Recommendation"),
    },
)

# ── Open dialog when a row is selected ───────────────────────────────────────
selected_rows = event.selection.rows if event.selection else []
if selected_rows:
    row = filtered.iloc[selected_rows[0]]
    show_claim_dialog(row)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:#94A3B8;font-size:.75rem;'>"
    "MediGuard AI · FWA Intelligence Platform · Powered by Groq LLM &amp; LangChain</p>",
    unsafe_allow_html=True,
)
