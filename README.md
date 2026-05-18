# CardioCheck — Cardiovascular Disease Detection 

## Project Description

This project is a machine learning proof of concept aimed at predicting the presence or absence of cardiovascular disease (CVD) from clinical and nutritional data. The goal is to build a non-invasive pre-diagnostic tool that can flag at-risk patients before serious complications occur.

Three classification models are trained and compared — Logistic Regression (baseline), Random Forest, and XGBoost — with **Recall** as the primary metric, since missing a sick patient is more costly than a false alarm. The final XGBoost model achieves a recall of ~92% on the test set.

The project also includes a **Streamlit web application** (*CardioCheck ML PoC*) with three pages: project context and preprocessing pipeline, model comparison and performance visualizations, and an interactive patient form for live predictions.

The full pipeline covers data cleaning (physiological outlier filtering), feature engineering (BMI, age conversion, NHANES enrichment via profile-based statistical join), model training, evaluation, and deployment.

---

## Data

The project uses two datasets that must be downloaded manually from Kaggle and placed in the `data/raw/` folder.

### 1. Primary dataset — Cardiovascular Disease Dataset

70,000 patient records with 12 clinical variables (age, sex, height, weight, blood pressure, cholesterol, glucose, smoking, alcohol, physical activity).

[Download on Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

> Save as `data/raw/cardio_train.csv`

### 2. Enrichment dataset — NHANES CVD Raw Data (2017–2023)

Survey data from the CDC including advanced biomarkers such as C-reactive protein, sodium intake, and saturated fat consumption, used to enrich the primary dataset via a demographic profile join.

[Download on Kaggle](https://www.kaggle.com/datasets/ahiduzzaman28/nhanes-cvd-raw-data-2017-23)

> Save as `data/raw/Nhanes_cvd_raw.csv`
