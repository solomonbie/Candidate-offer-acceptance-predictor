# Offer Acceptance Predictor — A Talent Science Approach

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)
![Status](https://img.shields.io/badge/status-active-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![Streamlit](https://img.shields.io/badge/app-Streamlit-red)

> A machine learning model predicting offer acceptance probability using 22 behavioural and process signals — built by a Senior Talent Acquisition Partner grounded in behavioural economics, people analytics, and talent science.

**[Live Simulator →](https://solomon-candidate-offer-acceptance-predictor.streamlit.app)** &nbsp;|&nbsp; **[Portfolio Case Study →](https://solomonobie.com/projects/offer-acceptance-modelling)** &nbsp;|&nbsp; **[Author →](https://solomonobie.com)**

---

## The problem

Offer declines are one of the most expensive and avoidable failures in talent acquisition. The average cost of a declined offer — factoring in time-to-fill extension, hiring manager hours, and pipeline rebuild — ranges between £8,000 and £25,000 per role depending on seniority.

Most organisations discover a candidate will decline only after the offer letter is sent. By then, leverage is gone.

This project asks: **what signals, all available before the offer is extended, reliably predict whether a candidate will accept?**

---

## Project overview

| Item | Detail |
|---|---|
| Algorithm | Random Forest Classifier |
| Dataset | 2,000 anonymized candidates data |
| Features | 22 behavioural and process signals |
| Target | offer_accepted (binary: 0/1) |
| Accuracy | 75% on held-out test data |
| ROC-AUC | 0.638 |
| Baseline | 73% (naive always-accept) |
| Hypotheses tested | 7 |
| Hypotheses supported | 6/7 |

---

## Theoretical framework

This model is grounded in three behavioural economics frameworks:

**Prospect Theory** (Kahneman & Tversky, 1979) — candidates evaluate offers relative to a reference point, not in absolute terms. A 15% lift means different things to different people.

**BATNA** (Fisher & Ury, 1981) — a candidate with a competing offer is comparing, not deciding. Their reference point shifts entirely. The data confirms a 15.5 percentage point acceptance drop when a competing offer exists.

**Temporal Discounting** — the perceived value of a future state decays over time. Slow processes signal dysfunction and erode enthusiasm. Every week above 21 days costs approximately 1.5 percentage points of acceptance rate.

**Identity-Based Motivation** (Oyserman, 2009) — candidates accept roles aligned with who they are becoming. Seniority jump ranked 4th in feature importance, independent of salary lift.

---

## Dataset — 22 features

| Feature | Type | Description |
|---|---|---|
| `candidate_id` | ID | Unique reference — excluded from modelling |
| `current_level` | Category | Junior / Mid / Senior / Lead / Director |
| `offered_level` | Category | Level of the role being offered |
| `seniority_jump` | Integer | Delta between offered and current level (-1 to +2) |
| `years_experience` | Integer | Total career tenure in years |
| `months_in_current_role` | Integer | Tenure in current position — readiness signal |
| `notice_period_days` | Integer | Notice period (0 / 14 / 30 / 60 / 90 days) |
| `current_salary` | Integer | Current annual compensation (SEK) |
| `offered_salary` | Integer | Offered annual compensation (SEK) |
| `salary_increase_pct` | Float | Percentage lift over current salary |
| `remote_policy_match` | Binary | 1 = role matches candidate work preference |
| `interview_score` | Float | Average assessment score 1–5 |
| `recruiter_sentiment` | Float | Recruiter engagement assessment 1–5 |
| `hiring_manager_sentiment` | Float | HM culture/fit impression 1–5 |
| `process_length_days` | Integer | Total days from application to offer |
| `response_time_hours` | Float | Average candidate response time (hours) |
| `competing_offers` | Binary | 1 = candidate has disclosed a competing offer |
| `employer_brand_score` | Float | Perceived organisation attractiveness 1–5 |
| `technical_test_score` | Float | Technical assessment score 0–100 (NaN if no test) |
| `candidate_nps` | Integer | Post-interview process satisfaction (-100 to +100) |
| `relocation_required` | Binary | 1 = role requires relocation |
| `visa_required` | Binary | 1 = visa/work permit required |
| `stage_drop_risk` | Float | Engineered composite — see below |
| `offer_accepted` | Binary | **Target variable** (0 = declined, 1 = accepted) |

### Engineered feature: `stage_drop_risk`

```python
stage_drop_risk = (
    (response_time_hours / 120) * 0.30 +
    (1 - recruiter_sentiment / 5) * 0.30 +
    (process_length_days / 90)   * 0.20 +
    (competing_offers)           * 0.20
)
```

This composite signal became the model's **#1 most important predictor** at importance score 0.228 — outperforming salary lift, seniority jump, and all individual features. The machine independently discovered that the combination of these four signals outperforms any single variable alone.

---

## Hypothesis testing results

| # | Hypothesis | Test | Result | p-value | Key finding |
|---|---|---|---|---|---|
| H1 | Salary lift predicts acceptance | t-test | ✅ Supported | <0.001 | 84% vs 61% (>20% vs <5% lift) |
| H2 | Competing offers reduce acceptance | Chi-square | ✅ Supported | <0.001 | 80% vs 64.5% |
| H3 | Longer processes reduce acceptance | ANOVA | ✅ Supported | <0.001 | 79% vs 65.5% (fast vs slow) |
| H4 | Remote policy match moderates acceptance | Chi-square | ❌ Not supported | 0.622 | 74% vs 73% — self-selection upstream |
| H5 | Sentiment scores predict acceptance | Pearson r | ✅ Partial | <0.001 | HM (r=0.128) > Recruiter (r=0.105) |
| H6 | Candidate NPS predicts acceptance | Pearson r | ✅ Supported | 0.011 | 75% promoters vs 70% detractors |
| H7 | Seniority jump increases acceptance | ANOVA | ✅ Supported | <0.001 | 62% step-down vs 82% two-levels-up |

**6 of 7 hypotheses supported.** H4 failure reveals that remote policy operates as an upstream funnel filter — candidates self-screen before reaching offer stage.

---

## Model performance

```
Accuracy:               75.0%
ROC-AUC:                0.638
Cross-validation avg:   75.4% (±0.018 across 5 folds)
Naive baseline:         73.0% (always predict accept)
Uplift over baseline:   +2.0 percentage points
```

### Feature importance ranking

```
1. stage_drop_risk          0.228  ████████████████████░  Composite engagement signal
2. salary_increase_pct      0.145  █████████████░░░░░░░░  Compensation lift
3. months_in_current_role   0.133  ████████████░░░░░░░░░  Tenure readiness
4. seniority_jump           0.116  ██████████░░░░░░░░░░░  Career trajectory
5. candidate_nps            0.091  ████████░░░░░░░░░░░░░  Process experience
6. process_length_days      0.072  ██████░░░░░░░░░░░░░░░  Hiring speed
7. response_time_hours      0.053  █████░░░░░░░░░░░░░░░░  Candidate engagement
8. hiring_manager_sentiment 0.043  ████░░░░░░░░░░░░░░░░░  Human connection
```

---

## Key findings for TA practitioners

**1. Engagement composite beats salary**
`stage_drop_risk` — a combination of response speed, recruiter sentiment, process length, and competing offer status — is a stronger predictor than salary lift alone. Monitor the system, not the number.

**2. Tenure readiness is underused**
`months_in_current_role` ranked 3rd. Candidates under 12 months in role accept at significantly lower rates regardless of offer quality. Screen for readiness early.

**3. Process speed is a conversion metric**
Every week above 21 days costs ~1.5 percentage points of acceptance. A 6-week process loses roughly 10 points vs a 3-week process. Time-to-offer is a revenue metric.

**4. Hiring managers are your best closers**
HM sentiment (r=0.128) outpredicts recruiter sentiment (r=0.105). Candidates decide based on who they will work FOR. Prepare and brief your HMs accordingly.

**5. Career trajectory is non-compensatory**
Seniority jump adds 5–8 percentage points per level, independent of salary. No compensation increase fully substitutes for perceived career progression.

---

## Repository structure

```
offer-acceptance-predictor/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── synthetic_candidates.csv          # 2,000 candidate records
│
├── notebooks/
│   ├── 01_data_generation.ipynb          # Synthetic dataset creation
│   ├── 02_eda_hypothesis_testing.ipynb   # EDA and 7 hypothesis tests
│   └── 03_model_training.ipynb           # Random Forest + evaluation
│
├── app/
│   └── streamlit_app.py                  # Live offer acceptance simulator
│
├── models/
│   ├── offer_acceptance_model.pkl        # Trained Random Forest
│   └── model_features.pkl               # Feature list for inference
│
└── docs/
    └── methodology.md                    # Extended methodology notes
```

---

## How to run locally

```bash
# 1. Clone the repo
git clone https://github.com/solomonbie/offer-acceptance-predictor.git
cd offer-acceptance-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run notebooks in order
jupyter notebook notebooks/01_data_generation.ipynb

# 4. Run the Streamlit app
streamlit run app/streamlit_app.py
```

---

## Requirements

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.4
matplotlib>=3.7
seaborn>=0.12
scipy>=1.10
streamlit>=1.30
joblib>=1.3
```

---

## References

- Kahneman, D. & Tversky, A. (1979). Prospect theory: An analysis of decision under risk. *Econometrica*, 47(2), 263–291.
- Fisher, R. & Ury, W. (1981). *Getting to Yes: Negotiating Agreement Without Giving In*. Penguin Books.
- Oyserman, D. (2009). Identity-based motivation. *Journal of Consumer Psychology*, 19(3), 250–260.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.

---

## Author

**Solomon F. Torén**
Senior Talent Partner | Talent Scientist | People Analytics
[solomonobie.com](https://solomonobie.com) · [LinkedIn](https://linkedin.com/in/solomonobie)

---

*This project uses candidate anonymized data. No real candidate data was stored. Built as a portfolio demonstration of applied talent science and machine learning.*
