# UrbanGlide — ML Capstone (23CSE301)

## Problem statement

UrbanGlide is a micro-mobility (e-bike / e-scooter) operator. This project builds an end-to-end ML pipeline across three tracks on their trip data:

- **Regression** — predict `energy_consumed_wh`, the energy a trip draws from the battery, from trip and environmental features.
- **Classification** — predict `incident_type` (Normal, Hard_Braking, Battery_Critical, Mechanical_Fault, Improper_Parking) from the same kind of trip telemetry.
- **Clustering** *(Review 2)* — group rides by behavioural pattern without using the incident labels, to see whether natural clusters line up with real incident types.

## Datasets

- `data/urbanglide_regression_energy.csv` — 12,035 trips, 15 columns (14 predictors + `energy_consumed_wh`).
- `data/urbanglide_classification_incident.csv` — 12,030 trips, 14 columns (13 predictors + `incident_type`).

Both are provided by the instructor for this capstone; no external download needed.

## Repository structure

```
/
├── README.md
├── requirements.txt
├── data/           raw datasets
├── notebooks/      regression.ipynb, classification.ipynb, clustering.ipynb (Review 2)
├── models/         saved model files (.pkl via joblib) — added when the bonus GUI is built
└── app/            Streamlit/Gradio GUI + deployment code (bonus, Review 2)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Open `notebooks/regression.ipynb` or `notebooks/classification.ipynb` and run all cells top to bottom. Both read their data via a relative path (`../data/...`), so they run unmodified on any machine once the repo is cloned — no hardcoded paths.

## Results summary (Review 1)

**Regression — 10 required algorithms, ranked by single-split test R²:**

| Model | R² | RMSE (Wh) | MAE (Wh) |
|---|---|---|---|
| Polynomial Regression (Degree 2) | *see notebook output* | | |
| Gradient Boosting Regressor | | | |
| Random Forest Regressor | | | |
| ... | | | |

5-fold cross-validation on the top 2 models by single-split R² overturns this ranking — see `notebooks/regression.ipynb`, Section 5, for the full result and why **Gradient Boosting**, not Polynomial Regression, is the model actually selected.

**Classification (Part A) — 5 required algorithms, ranked by macro F1:**

| Algorithm | Accuracy | Macro F1 | Weighted F1 | ROC-AUC (OvR) |
|---|---|---|---|---|
| *(see notebook output — table regenerates on every run)* | | | | |

Class imbalance (Normal ≈ 52% of rows) means accuracy alone is misleading; macro F1 is the metric actually used to rank models. Part B (5 more algorithms) and the consolidated 10-algorithm table are added in Review 2.

Full commentary, real printed numbers, and the reasoning behind every modelling decision are in the notebooks themselves — every code cell is followed by a markdown cell explaining what its output actually shows.

## AI assistance disclosure (section 7.5)

Generative AI (Claude) was used for code scaffolding, pipeline structure, and drafting the explanatory markdown cells. All analysis, interpretation, feature-engineering rationale and the cross-validation-driven model-selection decision were reviewed and are owned by the team. Per the capstone guidelines, AI was not used to generate the interpretive judgement calls themselves (e.g. which model is the "champion" and why) without verification against the actual executed output.

## Team

Three-member team — see commit history for individual ownership: Person A (regression EDA + linear family + final comparison), Person B (classification, both reviews), Person C (regression ensembles + tuning + CV + repository).
