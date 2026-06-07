import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Google Analytics ──────────────────────────────────────────
def inject_ga():
    st.markdown("""
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-TK23Y5CN23"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'G-TK23Y5CN23');
    </script>
    """, unsafe_allow_html=True)

inject_ga()

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
    model_path    = "offer_acceptance_model.pkl"
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
    cur_num   = level_map[vals["current_level"]]
    off_num   = level_map[vals["offered_level"]]
    jump      = off_num - cur_num

    stage_drop = compute_stage_drop_risk(
        vals["response_time_hours"],
        vals["recruiter_sentiment"],
        vals["process_length_days"],
        vals["competing_offers"]
    )

    salary_pct = (
        (vals["offered_salary"] - vals["current_salary"])
        / vals["current_salary"] * 100
    ) if vals["current_salary"] > 0 else 0

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
    importances = dict(zip(feature_names, model.feature_importances_))
    row         = row_df.iloc[0]
    contributions = []

    sal     = row.get("salary_increase_pct", 0)
    sal_dir = 1 if sal > 10 else (-1 if sal < 0 else 0)
    sal_mag = min(abs(sal) / 30, 1.0) * importances.get("salary_increase_pct", 0)
    if sal_dir != 0:
        contributions.append({
            "feature":   "Salary increase",
            "direction": sal_dir,
            "magnitude": sal_mag,
            "detail":    f"{sal:.1f}% lift — {'strong pull' if sal > 20 else 'modest lift' if sal > 10 else 'at-risk territory'}"
        })

    jump = row.get("seniority_jump", 0)
    if jump != 0:
        contributions.append({
            "feature":   "Career trajectory",
            "direction": 1 if jump > 0 else -1,
            "magnitude": min(abs(jump) / 2, 1.0) * importances.get("seniority_jump", 0),
            "detail":    f"{'Promotion offer' if jump > 0 else 'Step down'} ({'+' if jump > 0 else ''}{jump} level{'s' if abs(jump) > 1 else ''})"
        })

    sdr     = row.get("stage_drop_risk", 0)
    sdr_dir = -1 if sdr > 0.4 else (1 if sdr < 0.2 else 0)
    if sdr_dir != 0:
        contributions.append({
            "feature":   "Engagement signal",
            "direction": sdr_dir,
            "magnitude": min(sdr, 1.0) * importances.get("stage_drop_risk", 0),
            "detail":    f"Stage drop risk: {sdr:.3f} — {'high concern' if sdr > 0.5 else 'moderate concern' if sdr > 0.3 else 'low concern'}"
        })

    proc     = row.get("process_length_days", 30)
    proc_dir = 1 if proc < 21 else (-1 if proc > 55 else 0)
    if proc_dir != 0:
        contributions.append({
            "feature":   "Process speed",
            "direction": proc_dir,
            "magnitude": min(proc / 90, 1.0) * importances.get("process_length_days", 0),
            "detail":    f"{proc} days — {'fast process, positive signal' if proc < 21 else 'slow process, reducing pull'}"
        })

    if row.get("competing_offers", 0) == 1:
        contributions.append({
            "feature":   "Competing offer",
            "direction": -1,
            "magnitude": importances.get("competing_offers", 0) * 1.5,
            "detail":    "BATNA active — candidate is comparing, not deciding"
        })

    hm     = row.get("hiring_manager_sentiment", 3)
    hm_dir = 1 if hm >= 4 else (-1 if hm <= 2.5 else 0)
    if hm_dir != 0:
        contributions.append({
            "feature":   "HM impression",
            "direction": hm_dir,
            "magnitude": ((hm - 3) / 2) * importances.get("hiring_manager_sentiment", 0),
            "detail":    f"HM sentiment {hm}/5 — {'strong human connection' if hm >= 4 else 'weak connection, risk factor'}"
        })

    months     = row.get("months_in_current_role", 24)
    months_dir = 1 if months > 36 else (-1 if months < 12 else 0)
    if months_dir != 0:
        contributions.append({
            "feature":   "Tenure readiness",
            "direction": months_dir,
            "magnitude": min(months / 84, 1.0) * importances.get("months_in_current_role", 0),
            "detail":    f"{months} months in role — {'ready to move' if months > 36 else 'reluctant mover, under 12 months'}"
        })

    nps     = row.get("candidate_nps", 0)
    nps_dir = 1 if nps > 50 else (-1 if nps < 0 else 0)
    if nps_dir != 0:
        contributions.append({
            "feature":   "Process NPS",
            "direction": nps_dir,
            "magnitude": abs(nps / 100) * importances.get("candidate_nps", 0),
            "detail":    f"NPS {nps} — {'process promoter' if nps > 50 else 'process detractor, flag for follow-up'}"
        })

    contributions.sort(key=lambda x: abs(x["magnitude"]), reverse=True)
    return contributions[:6]


def get_recommendations(prob, row_df, jump, salary_pct):
    row = row_df.iloc[0]

    if prob >= 0.70:
        risk  = "low"
        title = "Low risk — proceed with confidence"
        actions = [
            "Confirm verbal acceptance in a personal call before issuing the written "
            "offer letter. Never let the written offer be the first time the candidate "
            "hears the number.",
            "Brief the hiring manager with three things before the offer call: "
            "the candidate's stated career motivators, their current timeline, and "
            "any concerns raised during the process. The HM should not go in blind.",
            "Set a clear offer deadline — 5 to 7 business days. An open-ended offer "
            "creates uncertainty and invites competing offers to fill the vacuum.",
            "Lead the offer narrative with career trajectory and role impact first. "
            "Introduce the compensation package second. Candidates who feel excited "
            "about the role negotiate less aggressively on salary.",
        ]
        if row.get("competing_offers", 0) == 1:
            actions.append(
                "A competing offer is in play even at low risk — do not wait for the "
                "candidate to compare. Accelerate the timeline by 2 to 3 days and "
                "arrange a personal call from the hiring manager before the written "
                "offer arrives. Personal contact at this stage outperforms salary."
            )

    elif prob >= 0.50:
        risk  = "medium"
        title = "Medium risk — intervene before extending"
        actions = [
            "Do not extend the written offer without a pre-offer conversation. "
            "Call the candidate and open with: 'Before I send anything formal, "
            "I want to make sure we have answered everything for you. "
            "Is there anything that would make this decision difficult?'",
            "This single question surfaces unstated objections that no offer letter "
            "can address. The most common answers are: competing offer, partner "
            "concerns, relocation anxiety, or uncertainty about the team. All "
            "recoverable — but only if you know before the offer.",
            "Bring the hiring manager into the closing conversation directly. "
            "At medium risk, HM rapport is your strongest lever. A genuine personal "
            "call from the HM can move a hesitant candidate more than any salary "
            "adjustment.",
        ]
        if salary_pct < 12:
            actions.append(
                f"Salary lift is {salary_pct:.1f}% — below the 12% threshold where "
                "acceptance risk increases significantly. Explore whether the "
                "compensation band has any flex before extending."
            )
        if row.get("competing_offers", 0) == 1:
            actions.append(
                "Competing offer is active. Ask the candidate directly: 'What is "
                "the other offer doing that this one is not?' Listen without defending. "
                "Pick the dimension where you are genuinely stronger and make that case."
            )
        if jump <= 0:
            actions.append(
                "This is a lateral or step-down offer. Reframe the conversation around "
                "non-compensation value: brand, stability, learning environment, "
                "team quality, and long-term trajectory beyond this role."
            )

    else:
        risk  = "high"
        title = "High risk — pause and reassess before extending"
        actions = [
            "Do not extend this offer without a full pre-offer debrief with the "
            "hiring manager. Align on what levers are available and what the cost "
            "of a decline would be versus strengthening the offer now.",
            "Identify the primary risk driver from the signal breakdown and address "
            "that specific driver in the pre-offer conversation. One well-addressed "
            "concern is more effective than five partially addressed ones.",
            "Explore whether title, scope, start date flexibility, signing bonus, "
            "or remote arrangement can compensate for a weaker base salary.",
            "If the process has taken more than 55 days, acknowledge it directly: "
            "'I know this took longer than we would have liked — I want to make "
            "sure we make it worth the wait.' Transparency rebuilds trust.",
        ]
        if row.get("stage_drop_risk", 0) > 0.5:
            actions.append(
                "Stage drop risk is critically high. The candidate's engagement "
                "pattern suggests a possible silent withdrawal is already in progress. "
                "Make direct personal contact before sending any written offer."
            )
        if row.get("months_in_current_role", 24) < 12:
            actions.append(
                "This candidate has been in their current role under 12 months — "
                "the single strongest structural indicator of a reluctant mover. "
                "Manage the hiring manager's expectations on close probability explicitly."
            )
        actions.append(
            "Document the predicted decline risk in your ATS. If the offer is declined, "
            "this becomes valuable calibration data for future hiring on similar roles."
        )

    return risk, title, actions


def get_science_note(prob, vals, jump, salary_pct, stage_drop):
    competing_active = vals["competing_offers"]
    hm_score         = vals["hiring_manager_sentiment"]
    nps_score        = vals["candidate_nps"]
    months_in_role   = vals["months_in_current_role"]

    if competing_active and prob < 0.65:
        return (
            "<strong>BATNA effect (Fisher &amp; Ury, 1981)</strong><br><br>"
            "This candidate has a competing offer — which means they are no longer deciding "
            "whether to join you. They are comparing two options. Behavioural economics calls "
            "this the BATNA effect: once a person has a Best Alternative To a Negotiated "
            "Agreement, their entire reference point shifts. Your offer is no longer evaluated "
            "on its own merits — it is evaluated against the alternative.<br><br>"
            "<strong>What this means for the recruiter:</strong> Stop selling the role. "
            "Start differentiating it. Ask the candidate directly what the competing offer "
            "is doing that yours is not — then decide whether to compete on compensation, "
            "career trajectory, flexibility, or speed. Pick the dimension where you are "
            "genuinely stronger and make that case clearly."
        )
    elif months_in_role < 12:
        return (
            "<strong>Temporal anchoring and readiness to move</strong><br><br>"
            "This candidate has been in their current role for under 12 months. Research on "
            "career transition psychology shows that people who have recently started a role "
            "are psychologically anchored to it — they have not yet reached the natural "
            "inflection point where dissatisfaction or ambition outweighs the comfort of "
            "familiarity. This is not a compensation problem. It is a timing and readiness "
            "problem.<br><br>"
            "<strong>What this means for the recruiter:</strong> This is a structurally harder "
            "close regardless of offer quality. Explore what motivated the candidate to engage "
            "with this process at all — that signal is your leverage. Manage the hiring "
            "manager's expectations on close probability explicitly before the offer goes out."
        )
    elif jump >= 2:
        return (
            "<strong>Identity-based motivation (Oyserman, 2009)</strong><br><br>"
            "A two-level promotion offer triggers something more powerful than financial "
            "calculation — it activates the candidate's future identity. Research by Oyserman "
            "shows that people are strongly motivated to act in ways consistent with who they "
            "see themselves becoming, not just who they are today. This is a non-compensatory "
            "motivator: it cannot be fully substituted by a salary increase because it operates "
            "in a different psychological domain entirely.<br><br>"
            "<strong>What this means for the recruiter:</strong> Lead with the title, the scope, "
            "and the narrative of what this role makes possible for the candidate's career. "
            "Open the offer call with career, not salary. Save the numbers for when they ask."
        )
    elif stage_drop > 0.45:
        return (
            "<strong>Engagement decay and temporal discounting</strong><br><br>"
            "This candidate's engagement signals suggest that initial enthusiasm may have "
            "decayed during the hiring process. Behavioural science research on temporal "
            "discounting shows that the perceived value of a future reward decreases the "
            "longer a person must wait for it. A candidate excited at week two may be "
            "significantly less invested at week eight — not because the role changed, "
            "but because the wait eroded the emotional pull.<br><br>"
            "<strong>What this means for the recruiter:</strong> Before the offer call, "
            "re-engage the candidate personally. A direct call from the hiring manager "
            "expressing genuine enthusiasm can partially reverse engagement decay. "
            "Do not let the offer letter be the first contact after a long silence."
        )
    elif hm_score < 3.5 and prob < 0.65:
        return (
            "<strong>Interpersonal fit and future work experience signalling</strong><br><br>"
            "The hiring manager sentiment score is lower than the threshold associated with "
            "confident acceptance. Research consistently shows that candidates use the quality "
            "of their relationship with the hiring manager during interviews as a direct proxy "
            "for what it will be like to work for that person every day. A lukewarm HM "
            "impression creates a negative forecast of the daily work experience.<br><br>"
            "<strong>What this means for the recruiter:</strong> Before extending the offer, "
            "arrange an informal touchpoint between the candidate and the hiring manager. "
            "A warm personal message from the HM referencing something specific from the "
            "interview can create a meaningfully warmer final impression before the decision."
        )
    elif nps_score < 0:
        return (
            "<strong>Process experience as a culture signal (Talent Board, 2024)</strong><br><br>"
            "This candidate has rated the hiring process negatively. Research from Talent Board "
            "and IBM's Smarter Workforce Institute shows that candidates use the recruitment "
            "experience as a direct proxy for organisational culture. A negative NPS score "
            "suggests the candidate has already updated their beliefs about the organisation "
            "in a negative direction — independently of offer quality.<br><br>"
            "<strong>What this means for the recruiter:</strong> Acknowledge the friction "
            "before making the offer. A brief honest conversation about what the onboarding "
            "and first 90 days will look like can partially reverse a negative process "
            "impression. Silence about the friction makes it significantly worse."
        )
    elif salary_pct < 8:
        return (
            "<strong>Prospect theory and reference point anchoring "
            "(Kahneman &amp; Tversky, 1979)</strong><br><br>"
            "Prospect theory demonstrates that people evaluate gains and losses relative to "
            "a reference point — in this case the candidate's current salary. A lift below 8% "
            "is unlikely to feel like a meaningful gain once the disruption of changing jobs "
            "and the uncertainty of a new environment are factored in. Below a certain "
            "threshold the gain simply does not feel worth the friction of moving.<br><br>"
            "<strong>What this means for the recruiter:</strong> If the salary cannot flex, "
            "the non-financial value must be made explicit. Equity, bonus, learning budget, "
            "title change, flexibility, and career acceleration are all part of total "
            "compensation — but only if the recruiter names them clearly."
        )
    else:
        return (
            "<strong>Prospect theory and gain framing "
            "(Kahneman &amp; Tversky, 1979)</strong><br><br>"
            "Candidates do not evaluate offers in absolute terms — they evaluate them relative "
            "to where they currently are. This is the core insight of prospect theory: people "
            "respond to perceived gains from a personal reference point, not to objective "
            "outcomes. The emotional perception of gain matters more than the gain itself — "
            "and that perception is shaped as much by how the offer is framed as by what "
            "the offer contains.<br><br>"
            "<strong>What this means for the recruiter:</strong> Frame the offer conversation "
            "around what the candidate is gaining — career progression, new challenges, team "
            "quality, autonomy, and impact. Lead with the narrative. Close with the package."
        )


# ══════════════════════════════════════════════════════════════
#  UI — HEADER
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div style="margin-bottom:0.25rem;">
  <span style="font-size:11px;text-transform:uppercase;letter-spacing:0.1em;color:#999;">
    Solomon Fredrick · Talent Partner &amp; Talent Scientist
  </span>
</div>
""", unsafe_allow_html=True)

st.markdown('<h1 class="hero-title">Offer Acceptance Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
<p class="hero-sub">
  A decision support tool for Talent Acquisition professionals and Hiring Managers.
  Enter a candidate profile to receive an instant acceptance probability,
  risk classification, signal breakdown, and evidence-based recommendations.<br>
  <span style="font-size:12px;color:#aaa;">
    Random Forest model · 2,000 anonymised candidate records · 75% accuracy ·
    Grounded in behavioural economics
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
        competing    = st.checkbox("Competing offer disclosed")
        remote_match = st.checkbox("Remote policy match", value=True)
    with c14:
        relocation = st.checkbox("Relocation required")
        visa       = st.checkbox("Visa / work permit required")

    has_tech   = st.checkbox("Domain assessment completed")
    tech_score = 0.0
    if has_tech:
        tech_score = st.slider("Domain assessment score (0–100)", 0.0, 100.0, 68.0, 1.0)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Run prediction")


# ══════════════════════════════════════════════════════════════
#  UI — RESULTS
# ══════════════════════════════════════════════════════════════

with col_right:

    if predict_btn:

        vals = {
            "current_level":            current_level,
            "offered_level":            offered_level,
            "years_experience":         years_exp,
            "months_in_current_role":   months_role,
            "notice_period_days":       notice_period,
            "current_salary":           current_salary,
            "offered_salary":           offered_salary,
            "remote_policy_match":      remote_match,
            "interview_score":          interview_score,
            "recruiter_sentiment":      recruiter_sent,
            "hiring_manager_sentiment": hm_sent,
            "process_length_days":      process_days,
            "response_time_hours":      response_hours,
            "competing_offers":         competing,
            "employer_brand_score":     employer_brand,
            "technical_test_score":     tech_score,
            "candidate_nps":            candidate_nps,
            "relocation_required":      relocation,
            "visa_required":            visa,
            "has_tech_test":            has_tech,
        }

        row_df, salary_pct, jump, stage_drop = build_input(vals)
        prob = model.predict_proba(row_df)[0][1]

        # ── GA prediction event ──────────────────────────────
        risk_label = "low" if prob >= 0.70 else "medium" if prob >= 0.50 else "high"
        st.markdown(f"""
        <script>
            window.dataLayer = window.dataLayer || [];
            function gtag(){{dataLayer.push(arguments);}}
            gtag('event', 'prediction_run', {{
                'event_category': 'engagement',
                'event_label': 'offer_prediction',
                'risk_level': '{risk_label}',
                'probability': {round(prob * 100)}
            }});
        </script>
        """, unsafe_allow_html=True)

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
            sdr_col = '#E24B4A' if stage_drop > 0.4 else '#EF9F27' if stage_drop > 0.25 else '#1D9E75'
            st.markdown(f"""
            <div class="metric-card" style="padding:0.85rem 1rem;">
              <div class="metric-label">Stage drop risk</div>
              <div style="font-size:1rem;font-weight:500;color:{sdr_col};">
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
            pct_bar         = int(min(abs(c["magnitude"]) / max_mag * 100, 100))
            direction_label = "Positive" if c["direction"] > 0 else "Negative"
            bar_color       = "#1D9E75" if c["direction"] > 0 else "#E24B4A"
            icon            = "▲" if c["direction"] > 0 else "▼"

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
        css_class    = f"risk-{risk}"
        actions_html = "".join(f"<li style='margin-bottom:8px;'>{a}</li>" for a in actions)

        st.markdown(f"""
        <div class="{css_class}">
          <h3>{title}</h3>
          <ul style="margin:0.75rem 0 0;padding-left:1.25rem;">
            {actions_html}
          </ul>
        </div>""", unsafe_allow_html=True)

        # ── Behavioural science note ─────────────────────────
        bs_note = get_science_note(prob, vals, jump, salary_pct, stage_drop)

        st.markdown(f"""
        <div class="insight-box" style="margin-top:1rem;border-left:3px solid #1A1A1A;border-radius:0 8px 8px 0;">
          <strong style="font-size:11px;text-transform:uppercase;letter-spacing:0.08em;color:#999;">
            Behavioural science note
          </strong>
          <div style="font-size:13px;color:#444;margin-top:10px;line-height:1.85;">
            {bs_note}
          </div>
        </div>""", unsafe_allow_html=True)

    else:
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
            <p style="font-size:13px;color:#888;line-height:1.7;max-width:300px;margin:0 auto;">
                Fill in the candidate profile on the left and click
                <strong>Run prediction</strong> to get an instant
                acceptance probability with full breakdown and
                recruiter recommendations.
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
        Solomon Fredrick
      </a>
      · Talent Partner &amp; Talent Scientist ·
      Created for TA experts and Hiring Managers
    </span>
  </div>
  <div style="display:flex;gap:16px;">
    <a href="https://github.com/solomonbie/Candidate-offer-acceptance-predictor"
       target="_blank"
       style="font-size:12px;color:#999;text-decoration:none;">
      GitHub →
    </a>
    <a href="https://solomonobie.com/case-study/offer-acceptance-modelling"
       target="_blank"
       style="font-size:12px;color:#999;text-decoration:none;">
      Full case study →
    </a>
  </div>
</div>
<p style="font-size:11px;color:#bbb;margin-top:0.5rem;">
  This tool uses a Random Forest model trained on 2,000 anonymised candidate
  records. No candidate data is stored or transmitted. Free to use — no login
  required.
</p>
""", unsafe_allow_html=True)
