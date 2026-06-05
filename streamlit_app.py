import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Offer Acceptance Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main { background-color: #F5F3EE; }
.block-container { padding: 2rem 3rem; max-width: 1100px; }

h1, h2, h3 { font-family: 'DM Serif Display', serif; }

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #1A1A1A;
    line-height: 1.2;
    margin-bottom: 0.5rem;
}
.hero-sub {
    font-size: 1rem;
    color: #666;
    font-weight: 300;
    margin-bottom: 2rem;
    line-height: 1.6;
}

.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    border: 0.5px solid #E0DDD6;
    margin-bottom: 1rem;
}
.metric-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #999;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 2rem;
    font-weight: 500;
    color: #1A1A1A;
    line-height: 1;
}
.metric-sub {
    font-size: 12px;
    color: #888;
    margin-top: 4px;
}

.risk-low    { background:#E8F5F0; border:0.5px solid #A5D6C2; border-radius:8px; padding:1rem 1.25rem; }
.risk-medium { background:#FEF6E7; border:0.5px solid #F0CC7A; border-radius:8px; padding:1rem 1.25rem; }
.risk-high   { background:#FDECEA; border:0.5px solid #F5A8A6; border-radius:8px; padding:1rem 1.25rem; }

.risk-low    h3 { color:#1B6B4A; font-family:'DM Serif Display',serif; font-size:1.2rem; }
.risk-medium h3 { color:#7A5200; font-family:'DM Serif Display',serif; font-size:1.2rem; }
.risk-high   h3 { color:#8B1C1A; font-family:'DM Serif Display',serif; font-size:1.2rem; }

.risk-low    p, .risk-low    li { color:#1B6B4A; font-size:13px; line-height:1.7; }
.risk-medium p, .risk-medium li { color:#7A5200; font-size:13px; line-height:1.7; }
.risk-high   p, .risk-high   li { color:#8B1C1A; font-size:13px; line-height:1.7; }

.feature-bar-pos { background:#1D9E75; height:8px; border-radius:4px; }
.feature-bar-neg { background:#E24B4A; height:8px; border-radius:4px; }

.section-label {
    font-size:11px;
    text-transform:uppercase;
    letter-spacing:0.1em;
    color:#999;
    margin-bottom:0.5rem;
    margin-top:1.5rem;
}

.insight-box {
    background:white;
    border-radius:10px;
    border:0.5px solid #E0DDD6;
    padding:1rem 1.25rem;
    margin-bottom:0.75rem;
    font-size:13px;
    color:#444;
    line-height:1.7;
}
.insight-box strong { color:#1A1A1A; }

.divider { border:none; border-top:0.5px solid #E0DDD6; margin:1.5rem 0; }

.author-strip {
    background:#1A1A1A;
    border-radius:10px;
    padding:1rem 1.5rem;
    color:#E8E4DC;
    font-size:12px;
    margin-top:2rem;
    display:flex;
    justify-content:space-between;
    align-items:center;
}

.stButton > button {
    background:#1A1A1A !important;
    color:white !important;
    border:none !important;
    border-radius:8px !important;
    font-family:'DM Sans',sans-serif !important;
    font-size:14px !important;
    padding:0.65rem 2rem !important;
    width:100% !important;
    transition:opacity 0.2s !important;
}
.stButton > button:hover { opacity:0.85 !important; }

.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider > div { font-family:'DM Sans',sans-serif !important; }

label { font-size:13px !important; color:#555 !important; font-weight:400 !important; }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path   = "offer_acceptance_model.pkl"
    features_path = "model_features.pkl"

    if not os.path.exists(model_path):
        st.error(
            "Model file not found. Make sure offer_acceptance_model.pkl "
            "is in the same directory as this app."
        )
        st.stop()

    model    = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features

model, feature_names = load_model()


# ── Helper functions ──────────────────────────────────────────
def compute_stage_drop_risk(response_hours, recruiter_sent, process_days, competing):
    return round(
        (response_hours / 120) * 0.30 +
        (1 - recruiter_sent / 5) * 0.30 +
        (process_days / 90) * 0.20 +
        (competing * 0.20),
        3
    )

def build_input(vals):
    level_map = {"Junior":1, "Mid":2, "Senior":3, "Lead":4, "Director":5}
    cur_num = level_map[vals["current_level"]]
    off_num = level_map[vals["offered_level"]]
    jump    = off_num - cur_num

    stage_drop = compute_stage_drop_risk(
        vals["response_time_hours"],
        vals["recruiter_sentiment"],
        vals["process_length_days"],
        vals["competing_offers"]
    )

    salary_pct = ((vals["offered_salary"] - vals["current_salary"])
                  / vals["current_salary"] * 100) if vals["current_salary"] > 0 else 0

    row = {
        "seniority_jump":            jump,
        "years_experience":          vals["years_experience"],
        "months_in_current_role":    vals["months_in_current_role"],
        "notice_period_days":        vals["notice_period_days"],
        "current_salary":            vals["current_salary"],
        "offered_salary":            vals["offered_salary"],
        "salary_increase_pct":       round(salary_pct, 1),
        "remote_policy_match":       int(vals["remote_policy_match"]),
        "interview_score":           vals["interview_score"],
        "recruiter_sentiment":       vals["recruiter_sentiment"],
        "hiring_manager_sentiment":  vals["hiring_manager_sentiment"],
        "process_length_days":       vals["process_length_days"],
        "response_time_hours":       vals["response_time_hours"],
        "competing_offers":          int(vals["competing_offers"]),
        "employer_brand_score":      vals["employer_brand_score"],
        "technical_test_score":      vals["technical_test_score"] if vals["has_tech_test"] else -1,
        "candidate_nps":             vals["candidate_nps"],
        "relocation_required":       int(vals["relocation_required"]),
        "visa_required":             int(vals["visa_required"]),
        "stage_drop_risk":           stage_drop,
    }

    df = pd.DataFrame([row])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names], salary_pct, jump, stage_drop


def get_feature_contributions(row_df, prob):
    """Approximate contribution of each feature to the prediction."""
    importances = dict(zip(feature_names, model.feature_importances_))
    row = row_df.iloc[0]

    contributions = []

    # Salary
    sal = row.get("salary_increase_pct", 0)
    sal_dir = 1 if sal > 10 else (-1 if sal < 0 else 0)
    sal_mag = min(abs(sal) / 30, 1.0) * importances.get("salary_increase_pct", 0)
    if sal_dir != 0:
        contributions.append({
            "feature": "Salary increase",
            "direction": sal_dir,
            "magnitude": sal_mag,
            "detail": f"{sal:.1f}% lift — {'strong pull' if sal > 20 else 'modest lift' if sal > 10 else 'at-risk territory'}"
        })

    # Seniority jump
    jump = row.get("seniority_jump", 0)
    if jump != 0:
        contributions.append({
            "feature": "Career trajectory",
            "direction": 1 if jump > 0 else -1,
            "magnitude": min(abs(jump) / 2, 1.0) * importances.get("seniority_jump", 0),
            "detail": f"{'Promotion offer' if jump > 0 else 'Step down'} ({'+' if jump > 0 else ''}{jump} level{'s' if abs(jump) > 1 else ''})"
        })

    # Stage drop risk
    sdr = row.get("stage_drop_risk", 0)
    sdr_dir = -1 if sdr > 0.4 else (1 if sdr < 0.2 else 0)
    if sdr_dir != 0:
        contributions.append({
            "feature": "Engagement signal",
            "direction": sdr_dir,
            "magnitude": min(sdr, 1.0) * importances.get("stage_drop_risk", 0),
            "detail": f"Stage drop risk: {sdr:.3f} — {'high concern' if sdr > 0.5 else 'moderate concern' if sdr > 0.3 else 'low concern'}"
        })

    # Process speed
    proc = row.get("process_length_days", 30)
    proc_dir = 1 if proc < 21 else (-1 if proc > 55 else 0)
    if proc_dir != 0:
        contributions.append({
            "feature": "Process speed",
            "direction": proc_dir,
            "magnitude": min(proc / 90, 1.0) * importances.get("process_length_days", 0),
            "detail": f"{proc} days — {'fast process, positive signal' if proc < 21 else 'slow process, reducing pull'}"
        })

    # Competing offer
    if row.get("competing_offers", 0) == 1:
        contributions.append({
            "feature": "Competing offer",
            "direction": -1,
            "magnitude": importances.get("competing_offers", 0) * 1.5,
            "detail": "BATNA active — candidate is comparing, not deciding"
        })

    # HM sentiment
    hm = row.get("hiring_manager_sentiment", 3)
    hm_dir = 1 if hm >= 4 else (-1 if hm <= 2.5 else 0)
    if hm_dir != 0:
        contributions.append({
            "feature": "HM impression",
            "direction": hm_dir,
            "magnitude": ((hm - 3) / 2) * importances.get("hiring_manager_sentiment", 0),
            "detail": f"HM sentiment {hm}/5 — {'strong human connection' if hm >= 4 else 'weak connection, risk factor'}"
        })

    # Months in role
    months = row.get("months_in_current_role", 24)
    months_dir = 1 if months > 36 else (-1 if months < 12 else 0)
    if months_dir != 0:
        contributions.append({
            "feature": "Tenure readiness",
            "direction": months_dir,
            "magnitude": min(months / 84, 1.0) * importances.get("months_in_current_role", 0),
            "detail": f"{months} months in role — {'ready to move' if months > 36 else 'reluctant mover, under 12 months'}"
        })

    # NPS
    nps = row.get("candidate_nps", 0)
    nps_dir = 1 if nps > 50 else (-1 if nps < 0 else 0)
    if nps_dir != 0:
        contributions.append({
            "feature": "Process NPS",
            "direction": nps_dir,
            "magnitude": abs(nps / 100) * importances.get("candidate_nps", 0),
            "detail": f"NPS {nps} — {'process promoter' if nps > 50 else 'process detractor, flag for follow-up'}"
        })

    # Sort by magnitude
    contributions.sort(key=lambda x: abs(x["magnitude"]), reverse=True)
    return contributions[:6]


def get_recommendations(prob, row_df, jump, salary_pct):
    """Return risk level and tailored recommendations."""
    row = row_df.iloc[0]

    if prob >= 0.70:
        risk = "low"
        title = "Low risk — proceed with confidence"
        actions = [
            "Confirm verbal acceptance before issuing the written offer letter.",
            "Brief the hiring manager on the candidate's stated motivators and career goals.",
            "Set a clear offer deadline (5–7 business days) to maintain momentum.",
            "Prepare a concise offer narrative — lead with career trajectory, not just compensation.",
        ]
        if row.get("competing_offers", 0) == 1:
            actions.append(
                "Competing offer is in play even at low risk — accelerate the timeline and "
                "get the HM on a personal call before the written offer arrives."
            )

    elif prob >= 0.50:
        risk = "medium"
        title = "Medium risk — intervene before extending"
        actions = [
            "Do NOT extend the written offer without a pre-offer conversation.",
            "Use the pre-offer call to surface unstated concerns — ask directly: "
            "'Is there anything that would make this decision difficult for you?'",
            "Involve the hiring manager in the closing conversation — HM rapport is your "
            "strongest lever at medium risk.",
        ]
        if salary_pct < 12:
            actions.append(
                f"Salary lift is {salary_pct:.1f}% — below the 12% threshold where "
                "acceptance risk increases sharply. Explore whether the band can flex."
            )
        if row.get("competing_offers", 0) == 1:
            actions.append(
                "Competing offer is active. Ask the candidate what the competing offer is "
                "doing that yours is not — then decide whether to differentiate on comp, "
                "career scope, flexibility, or speed."
            )
        if jump <= 0:
            actions.append(
                "This is a lateral or step-down offer. Reframe the conversation around "
                "non-compensation value: stability, brand, development, long-term trajectory."
            )

    else:
        risk = "high"
        title = "High risk — pause and reassess"
        actions = [
            "Do not extend the offer without a full pre-offer debrief with the hiring manager.",
            "Identify the primary risk driver from the feature breakdown below and address "
            "it directly before the offer call.",
            "Explore whether title, scope, or start date flexibility can compensate for "
            "a weaker compensation package.",
            "If the process has taken more than 55 days, acknowledge it directly with the "
            "candidate — transparency rebuilds trust and reduces attrition risk.",
        ]
        if row.get("stage_drop_risk", 0) > 0.5:
            actions.append(
                "Stage drop risk is critically high. Review the candidate's response times "
                "and engagement pattern — there may already be a silent withdrawal in progress."
            )
        if row.get("months_in_current_role", 24) < 12:
            actions.append(
                "Candidate has been in their current role under 12 months. This is the "
                "single strongest structural indicator of a reluctant mover — "
                "manage expectations on close probability regardless of offer quality."
            )
        actions.append(
            "Document the predicted decline risk in your ATS. If the offer is declined, "
            "use it as a data point to refine pipeline confidence scoring going forward."
        )

    return risk, title, actions


# ══════════════════════════════════════════════════════════════
#  UI — HEADER
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div style="margin-bottom:0.25rem;">
  <span style="font-size:11px;text-transform:uppercase;letter-spacing:0.1em;color:#999;">
    Solomon Obiechina · Talent Science
  </span>
</div>
""", unsafe_allow_html=True)

st.markdown('<h1 class="hero-title">Offer Acceptance Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
<p class="hero-sub">
  Enter a candidate's profile to receive an instant acceptance probability,
  risk classification, feature-level contribution breakdown, and
  evidence-based recruiter recommendations.<br>
  <span style="font-size:12px;color:#aaa;">
    Built on a Random Forest model trained across 2,000 candidates ·
    75% accuracy · Grounded in behavioural economics
  </span>
</p>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  UI — INPUT FORM
# ══════════════════════════════════════════════════════════════

col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:

    st.markdown('<p class="section-label">Candidate profile</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        current_level = st.selectbox(
            "Current seniority level",
            ["Junior", "Mid", "Senior", "Lead", "Director"]
        )
    with c2:
        offered_level = st.selectbox(
            "Offered seniority level",
            ["Junior", "Mid", "Senior", "Lead", "Director"],
            index=2
        )

    c3, c4 = st.columns(2)
    with c3:
        years_exp = st.number_input("Years of experience", 1, 35, 6)
    with c4:
        months_role = st.number_input("Months in current role", 1, 84, 24)

    st.markdown('<p class="section-label">Compensation</p>', unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    with c5:
        current_salary = st.number_input(
            "Current salary (SEK)", 200000, 2000000, 580000, step=10000
        )
    with c6:
        offered_salary = st.number_input(
            "Offered salary (SEK)", 200000, 2000000, 670000, step=10000
        )

    if current_salary > 0:
        pct = (offered_salary - current_salary) / current_salary * 100
        col = "#1D9E75" if pct > 10 else ("#EF9F27" if pct > 0 else "#E24B4A")
        st.markdown(
            f'<p style="font-size:12px;color:{col};margin-top:-0.5rem;">'
            f'Salary lift: {pct:+.1f}%</p>',
            unsafe_allow_html=True
        )

    st.markdown('<p class="section-label">Process signals</p>', unsafe_allow_html=True)

    c7, c8 = st.columns(2)
    with c7:
        process_days = st.number_input("Process length (days)", 7, 120, 32)
    with c8:
        notice_period = st.selectbox(
            "Notice period (days)", [0, 14, 30, 60, 90], index=2
        )

    c9, c10 = st.columns(2)
    with c9:
        response_hours = st.number_input(
            "Avg candidate response time (hrs)", 0.5, 120.0, 8.0, step=0.5
        )
    with c10:
        candidate_nps = st.slider("Candidate NPS", -100, 100, 30)

    st.markdown('<p class="section-label">Assessment scores</p>', unsafe_allow_html=True)

    c11, c12 = st.columns(2)
    with c11:
        recruiter_sent = st.slider("Recruiter sentiment (1–5)", 1.0, 5.0, 3.8, 0.1)
    with c12:
        hm_sent = st.slider("HM sentiment (1–5)", 1.0, 5.0, 3.6, 0.1)

    interview_score = st.slider("Interview score (1–5)", 1.0, 5.0, 3.5, 0.1)
    employer_brand  = st.slider("Employer brand score (1–5)", 1.0, 5.0, 3.7, 0.1)

    st.markdown('<p class="section-label">Contextual factors</p>', unsafe_allow_html=True)

    c13, c14 = st.columns(2)
    with c13:
        competing     = st.checkbox("Competing offer disclosed")
        remote_match  = st.checkbox("Remote policy match", value=True)
    with c14:
        relocation    = st.checkbox("Relocation required")
        visa          = st.checkbox("Visa / work permit required")

    has_tech = st.checkbox("Technical test completed")
    tech_score = 0.0
    if has_tech:
        tech_score = st.slider("Technical test score (0–100)", 0.0, 100.0, 68.0, 1.0)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Run prediction")


# ══════════════════════════════════════════════════════════════
#  UI — RESULTS
# ══════════════════════════════════════════════════════════════

with col_right:

    if predict_btn:

        vals = {
            "current_level":          current_level,
            "offered_level":          offered_level,
            "years_experience":       years_exp,
            "months_in_current_role": months_role,
            "notice_period_days":     notice_period,
            "current_salary":         current_salary,
            "offered_salary":         offered_salary,
            "remote_policy_match":    remote_match,
            "interview_score":        interview_score,
            "recruiter_sentiment":    recruiter_sent,
            "hiring_manager_sentiment": hm_sent,
            "process_length_days":    process_days,
            "response_time_hours":    response_hours,
            "competing_offers":       competing,
            "employer_brand_score":   employer_brand,
            "technical_test_score":   tech_score,
            "candidate_nps":          candidate_nps,
            "relocation_required":    relocation,
            "visa_required":          visa,
            "has_tech_test":          has_tech,
        }

        row_df, salary_pct, jump, stage_drop = build_input(vals)
        prob = model.predict_proba(row_df)[0][1]

        # ── Probability display ──────────────────────────────
        st.markdown('<p class="section-label">Prediction result</p>', unsafe_allow_html=True)

        prob_pct = int(round(prob * 100))
        bar_col  = "#1D9E75" if prob >= 0.70 else ("#EF9F27" if prob >= 0.50 else "#E24B4A")

        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Predicted acceptance probability</div>
          <div class="metric-value" style="color:{bar_col};">{prob_pct}%</div>
          <div style="margin:10px 0 6px;">
            <div style="background:#EEE;border-radius:4px;height:8px;overflow:hidden;">
              <div style="width:{prob_pct}%;height:100%;background:{bar_col};border-radius:4px;"></div>
            </div>
          </div>
          <div class="metric-sub">
            {'Strong yes — candidate highly likely to accept'
             if prob >= 0.75 else
             'Likely to accept — monitor for competing signals'
             if prob >= 0.65 else
             'Uncertain — intervention recommended before offer'
             if prob >= 0.50 else
             'High decline risk — reassess before extending'}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Three quick-stats
        qs1, qs2, qs3 = st.columns(3)
        with qs1:
            jump_label = (
                f"+{jump} level up" if jump > 0
                else "Lateral" if jump == 0
                else f"{jump} step down"
            )
            st.markdown(f"""
            <div class="metric-card" style="padding:0.85rem 1rem;">
              <div class="metric-label">Seniority jump</div>
              <div style="font-size:1rem;font-weight:500;color:#1A1A1A;">{jump_label}</div>
            </div>""", unsafe_allow_html=True)
        with qs2:
            st.markdown(f"""
            <div class="metric-card" style="padding:0.85rem 1rem;">
              <div class="metric-label">Salary lift</div>
              <div style="font-size:1rem;font-weight:500;color:{bar_col};">{salary_pct:+.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with qs3:
            st.markdown(f"""
            <div class="metric-card" style="padding:0.85rem 1rem;">
              <div class="metric-label">Stage drop risk</div>
              <div style="font-size:1rem;font-weight:500;color:{'#E24B4A' if stage_drop > 0.4 else '#EF9F27' if stage_drop > 0.25 else '#1D9E75'};">
                {stage_drop:.3f}
              </div>
            </div>""", unsafe_allow_html=True)

        # ── Feature contributions ────────────────────────────
        st.markdown(
            '<p class="section-label" style="margin-top:1.25rem;">What is driving this prediction</p>',
            unsafe_allow_html=True
        )

        contribs = get_feature_contributions(row_df, prob)
        max_mag  = max(abs(c["magnitude"]) for c in contribs) if contribs else 1

        for c in contribs:
            pct_bar = int(min(abs(c["magnitude"]) / max_mag * 100, 100))
            direction_label = "Positive" if c["direction"] > 0 else "Negative"
            bar_color = "#1D9E75" if c["direction"] > 0 else "#E24B4A"
            icon = "▲" if c["direction"] > 0 else "▼"

            st.markdown(f"""
            <div class="insight-box">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                <strong style="font-size:13px;">{icon} {c['feature']}</strong>
                <span style="font-size:11px;color:{'#1D9E75' if c['direction']>0 else '#E24B4A'};">
                  {direction_label}
                </span>
              </div>
              <div style="background:#F0EDE8;border-radius:4px;height:6px;margin-bottom:6px;overflow:hidden;">
                <div style="width:{pct_bar}%;height:100%;background:{bar_color};border-radius:4px;"></div>
              </div>
              <span style="font-size:12px;color:#666;">{c['detail']}</span>
            </div>""", unsafe_allow_html=True)

        # ── Recruiter recommendations ────────────────────────
        st.markdown(
            '<p class="section-label" style="margin-top:1.25rem;">Recruiter recommendations</p>',
            unsafe_allow_html=True
        )

        risk, title, actions = get_recommendations(prob, row_df, jump, salary_pct)
        css_class = f"risk-{risk}"

        actions_html = "".join(f"<li style='margin-bottom:6px;'>{a}</li>" for a in actions)
        st.markdown(f"""
        <div class="{css_class}">
          <h3>{title}</h3>
          <ul style="margin:0.75rem 0 0;padding-left:1.25rem;">
            {actions_html}
          </ul>
        </div>""", unsafe_allow_html=True)

        # Behavioural science note
        if prob < 0.50 and vals["competing_offers"]:
            bs_note = ("BATNA is active. The candidate is no longer evaluating your offer "
                       "— they are comparing two offers. At this point, salary alone rarely "
                       "closes the gap. Speed, HM involvement, and a compelling career "
                       "narrative are your most effective tools.")
        elif jump >= 2:
            bs_note = ("A two-level promotion activates identity-based motivation — the "
                       "candidate is not just weighing salary, they are weighing who they "
                       "will become. Lead with the career narrative in your closing conversation.")
        elif months_role < 12:
            bs_note = ("This candidate has been in their role under 12 months. Temporal "
                       "anchoring means they are psychologically closer to their previous "
                       "role than their next one. Manage expectations on close probability "
                       "regardless of offer quality.")
        else:
            bs_note = ("Prospect theory predicts candidates evaluate offers relative to "
                       "their current situation, not in absolute terms. Frame the offer "
                       "around what they are gaining — career growth, autonomy, impact — "
                       "not just what they are being paid.")

        st.markdown(f"""
        <div class="insight-box" style="margin-top:0.75rem;border-left:3px solid #1A1A1A;border-radius:0 8px 8px 0;">
          <strong style="font-size:11px;text-transform:uppercase;letter-spacing:0.08em;color:#999;">
            Behavioural science note
          </strong><br>
          <span style="font-size:13px;color:#444;">{bs_note}</span>
        </div>""", unsafe_allow_html=True)

    else:
        # Placeholder before prediction
        st.markdown("""
        <div style="
            background:white;
            border-radius:12px;
            border:0.5px solid #E0DDD6;
            padding:3rem 2rem;
            text-align:center;
            margin-top:2rem;
        ">
            <div style="font-size:2.5rem;margin-bottom:1rem;">🎯</div>
            <p style="font-size:15px;font-weight:500;color:#1A1A1A;margin-bottom:0.5rem;">
                Ready to predict
            </p>
            <p style="font-size:13px;color:#888;line-height:1.7;max-width:280px;margin:0 auto;">
                Fill in the candidate profile on the left and click
                <strong>Run prediction</strong> to get an instant
                acceptance probability with full breakdown.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
  <div>
    <span style="font-size:12px;color:#999;">
      Built by
      <a href="https://solomonobie.com" target="_blank"
         style="color:#1A1A1A;text-decoration:none;font-weight:500;">
        Solomon F Torèn
      </a>
      · Senior Talent Partner · Talent Scientist
    </span>
  </div>
  <div style="display:flex;gap:16px;">
    <a href="https://github.com/solomonbie/offer-acceptance-predictor"
       target="_blank"
       style="font-size:12px;color:#999;text-decoration:none;">
      GitHub →
    </a>
    <a href="https://solomonobie.com/projects/offer-acceptance-modelling"
       target="_blank"
       style="font-size:12px;color:#999;text-decoration:none;">
      Full case study →
    </a>
  </div>
</div>
<p style="font-size:11px;color:#bbb;margin-top:0.5rem;">
  This tool uses a Random Forest model trained on anonymized recruitment data.
  No candidate data is stored or transmitted. For portfolio, opensource and
  educational use.
</p>
""", unsafe_allow_html=True)
