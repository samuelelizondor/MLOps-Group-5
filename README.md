# NYC Taxi ETA Prediction Platform

> The NYC taxi operator should invest in a tested ETA prediction service with live baseline comparison and rehearsed fallback, not an unmonitored, one-off model.

## 1. Business case

### Context
**Client:** Fictional NYC taxi fleet operator (about 1,000 vehicles) in the transportation industry, Operations and Fleet Management.

**Maturity (all assumed for the fictional client):**
- Data MEDIUM: historical trip records in a centralised SQL database; data quality checks are inconsistent
- Tools MEDIUM: analysts use Python and SQL, with some cloud infrastructure
- Processes LOW: manual training and deployment, no standardised CI/CD or model monitoring
- People MEDIUM: analysts and data engineers exist, but dedicated MLOps expertise is limited

**Target maturity:** automated training and evaluation pipeline with controlled deployment and human approval. This is an incremental step from current manual processes; no numbered maturity framework is claimed.

**Ownership:** Operations and Fleet Management owns the business outcome; Data Science and Engineering are responsible for technical delivery. An ML Engineer owns day-to-day operation: monitoring, alert investigation, pipeline execution and rollback coordination. Production releases require technical approval from the ML Engineer and business approval from an Operations representative.

### Problem and baseline
Passengers experience unreliable journey-time estimates because the operator uses a simplified distance-based method that does not account for variations in effective travel speed. Large differences between estimated and actual durations can cause customer dissatisfaction and complaints.

**Current process:** straight-line pickup-to-drop-off distance divided by historical median effective speed.

**Evaluation method:** the Kaggle `train.csv` file is cleaned with fixed, documented data-quality rules and split chronologically into 80% training and 20% holdout. The Kaggle test file is not used because it has no trip-duration labels. Baseline speeds are computed from the training partition only. Baselines and all candidate models use identical cleaning rules and the same holdout records.

### Goal and success metrics
| Type | Metric (exact name) | Baseline | Target or threshold |
| --- | --- | --- | --- |
| Business KPI | `unreliable_eta_rate` (% of trips with absolute ETA error > 5 min) | Pending calculation (distance method, holdout) | At least 15% relative reduction vs distance method |
| Business KPI (secondary) | Customer complaint volume | To be measured (support or CRM data) | To be defined |
| ML metric | `mae` (minutes) | Pending calculation (distance method, holdout) | At least 15% relative improvement vs distance method |
| Incremental value | `unreliable_eta_rate` vs enhanced rule (median speed by hour of day and pickup area) | Pending calculation (holdout) | At least 5% relative improvement, otherwise deploy the enhanced rule |
| Limit | `latency_p95_ms` | n/a | At most 200 ms, validated by API load testing |
| Limit | `cost_per_prediction_usd` | n/a | At most 0.001 USD (provisional), measured against forecast and actual prediction volume |

`mae` and `unreliable_eta_rate` must pass independently because they do not always move together.

**Offline release gates:** do not release if any threshold above fails, or if technical and business approvals are not both obtained.

**Production monitoring:** the distance-based estimator runs alongside the ML model on the same completed trips. `unreliable_eta_rate` is measured monthly against this live control, with a target of at least 15% relative improvement within three months of deployment. Passing offline evaluation does not guarantee production success.

**Underperformance response:** the ML Engineer investigates and coordinates corrective action. Retraining requires validation and approval before deployment. If performance remains unacceptable or operational risk is identified, the ML Engineer coordinates rollback to the distance-based estimator and informs Operations.

### Solution
Passengers receive an estimated journey duration when confirming an immediate booking and use it to decide whether the journey fits their schedule.

The model predicts `trip_duration` (seconds) from pickup and destination coordinates, booking timestamp, passenger count and derived features (straight-line distance, hour of day, day of week). The booking system calls a real-time prediction API when a booking is confirmed.

**Excluded inputs:** `dropoff_datetime` and `trip_duration` (post-trip); `store_and_fwd_flag` (not reliably available at booking); `vendor_id` (identifies the provider rather than the journey and may reduce generalisation).

**Timestamp assumption:** the dataset has actual pickup times but no booking times, so `pickup_datetime` is used as a proxy for booking time in training. This assumes immediate bookings, where confirmation and pickup happen at about the same time. Advance bookings are out of scope.

**Fallback:** if the service is unavailable, times out or returns an invalid or implausible duration, the system uses the distance-based estimator. If both fail, the booking system shows "ETA unavailable". Plausibility thresholds are set through data profiling, validated with Operations before the first production release, and implemented as automated validation tests.

**Scope:** passenger ETAs only. Driver allocation, pricing and fleet management remain with Operations.

### Data
**Project data:** NYC Taxi Trip Duration (Kaggle competition, 2016), labelled `train.csv`, about 1.46 million trip records, subject to Kaggle's competition data terms. It is used to prototype and demonstrate the end-to-end pipeline, not to authorise production deployment.

**Production data:** the operator's own trip records. Owner: Operations and Fleet Management (proposed). Access requires operator authorisation and compliance with its data governance policies. Before production release, the model must be retrained and evaluated on sufficiently recent operator data and must pass all gates and approvals.

**Label:** `trip_duration` is computed from actual pickup and drop-off times. In production, it becomes available once a trip is completed. New records arrive as trips complete; ingestion and retraining frequency will be confirmed with the operator.

**Personal data:** precise coordinates combined with timestamps may be personal data because repeated journeys can reveal an identifiable person's movements. Production data is access-controlled, encrypted in transit and at rest, and retained only as long as necessary. Training and analysis datasets use pseudonymous trip identifiers. Raw coordinates and timestamps are excluded from application logs, which contain only request identifiers, prediction outcomes and operational metrics.

### Value, cost and risks
**Benefit:** at least 15% fewer trips with ETA errors above five minutes, compared with the current distance method on the same trips. ML is selected only if it also beats the enhanced rule by at least 5% on `unreliable_eta_rate`; otherwise the operator uses the enhanced rule.

**Cost (illustrative estimates, not supplier quotes; ongoing staffing excluded):**
- Phase 1, Kaggle prototype: 2 people (Data Scientist, ML Engineer) for 4 weeks; infrastructure 50 to 200 USD per month
- Phase 2, production preparation and deployment: 3 people (Data Scientist, ML Engineer, Data Engineer) for 6 to 8 weeks, plus part-time Operations input; infrastructure 200 to 500 USD per month during development and testing

| Risk | Mitigation |
| --- | --- |
| Data quality: missing, invalid or inconsistent records | Documented cleaning rules, profiling of incoming data, automated data validation tests |
| Outdated training data: 2016 data may not reflect current NYC traffic | Kaggle for prototyping only; retrain and evaluate on recent operator data before release; monitor live against the distance-based control |
| Skills gap and single-person dependency on the ML Engineer | Designated backup engineer who completes a documented rollback rehearsal before go-live |
| Train/serve timestamp skew: pickup time as a proxy for booking time | Immediate bookings only; validate with Operations; test training-serving feature consistency before release |
| Location privacy: coordinates and timestamps reveal movement patterns | Encryption, restricted access, pseudonymous trip identifiers, no raw coordinates or timestamps in logs |
| Prediction service failure or implausible output | Automatic fallback to the distance-based estimator; "ETA unavailable" if both fail |

**Scales to:** dispatch and fleet-planning use cases, after separate business validation, data assessment and model evaluation. These are outside the initial scope.

### Open gaps
- Baseline values pending calculation: `mae` and `unreliable_eta_rate` for the distance method and the enhanced rule
- 15% target and 5% margin over the enhanced rule not yet validated with Operations
- Five-minute tolerance not yet validated with Operations
- No concrete rollback trigger (for example, missing the target for N consecutive months)
- Booking time ≈ pickup time assumption not yet validated with Operations
- Customer complaint volume to be measured
- Exact date range of `train.csv` to be confirmed by profiling
- Production data owner to be confirmed
- Ingestion and retraining frequency to be confirmed with the operator
- "Sufficiently recent" operator data not yet defined

---

# MLOps Project Skeleton (reference only)

This repo is a **map, not a codebase**. Every file is a placeholder: no code, only a short note on
what the file does, where it sits in the pipeline and which files it talks to.

Create the same folders and files in **your own** project repo, then write the code for **your** use case.

## Stages = branches

| Branch | Stage | Sessions | What it adds |
|---|---|---|---|
| `main` (you are here) | 1. Kickoff | 1 to 3 | Folder structure, business case, one placeholder per pipeline step |
| `stage-2-modular` | 2. Modular code | 4 to 9 | Tests, style and test settings, first CI (Continuous Integration) workflow |
| `stage-3-cicd` | 3. CI/CD | 10 to 14 | Config loader, logging, experiment tracking, API, Docker, deployment |

Tip: on GitHub, open **Compare** (`main...stage-2-modular`) to see exactly what each stage adds.

## The pipeline at a glance

```text
                           main.py  (orchestrator, one command)
                              |
 data/raw/ -> load_data -> clean_data -> validate -> split -> features -> train -> evaluate
                                                (train/val/test) (recipe)   (fit)   (val split)
                                                                              |
                     reports/predictions.csv <- infer <- models/model.joblib (save)
```

## Stage 1 structure

```text
.
├── README.md                 # business case, success metrics, how to run
├── config.yaml               # non-secret settings (paths, target, split sizes)
├── environment.yml           # dependencies (conda)
├── .env.example              # names of secrets, never real values
├── .gitignore                # keeps data, secrets and generated files out of Git
├── LICENSE
├── .github/
│   └── pull_request_template.md
├── notebooks/                # the "lab": exploration only
├── src/                      # the "factory": one job per module
│   ├── load_data.py          # 1. read raw data
│   ├── clean_data.py         # 2. stateless cleaning
│   ├── validate.py           # 3. data quality gate
│   ├── features.py           # 4. unfitted feature recipe
│   ├── train.py              # 5. fit the pipeline
│   ├── evaluate.py           # 6. metrics on unseen data
│   ├── infer.py              # 7. predictions on new data
│   ├── utils.py              # shared read and write helpers
│   └── main.py               # 0. orchestrator: python -m src.main
├── data/  raw/ processed/ inference/
├── models/                   # saved model (generated)
├── reports/                  # predictions, plots (generated)
└── tests/                    # automated tests (stage 2)
```

## How to use this skeleton

1. Open a file and read its note
2. Create the equivalent file in your repo
3. Write your own code: the note describes the job, not the solution
