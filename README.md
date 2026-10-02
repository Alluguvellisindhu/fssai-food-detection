# Food Safety Risk Assessment using Machine Learning

## Project Overview

This project is an educational AI/ML prototype for **food-safety risk assessment and sample prioritization**.

The system takes food-sample parameters such as food category, pH, moisture, bacterial count, contaminant level, storage temperature, and storage duration, then predicts whether a sample belongs to a **Lower Risk** or **At Risk** class.

The application is designed as a portfolio project related to food-safety analytics. It is **not an official FSSAI system** and does not determine regulatory compliance.

## Why this project is relevant to food safety

Food-safety work includes laboratory analysis, surveillance, sampling, risk assessment, and monitoring. FSSAI describes risk assessment as supporting risk management and risk communication, including identification of hazards and use of data on contaminants/adulterants.

This project demonstrates how an ML workflow could be used as a **screening/prioritization prototype** before appropriate laboratory or regulatory action.

## Important data note

The included file:

`data/food_safety_demo.csv`

contains **synthetic demonstration data** created for this project. It is not FSSAI laboratory data, and the labels are not FSSAI regulatory classifications.

For a real research/internship deployment, use an authorized dataset with documented provenance and domain-approved target definitions.

## Features

- Data preprocessing with scikit-learn pipelines
- Missing-value handling
- StandardScaler for numeric features
- OneHotEncoder for food category
- Logistic Regression
- Random Forest
- Accuracy, precision, recall and F1-score
- Automatic model comparison
- Model selection using F1-score for the `At Risk` class
- Saved model using Joblib
- Flask web interface
- Responsive HTML/CSS UI

## Project Structure

```text
food-safety-risk-prediction/
│
├── data/
│   └── food_safety_demo.csv
│
├── model/
│   ├── food_safety_model.pkl
│   └── metrics.json
│
├── notebooks/
│   ├── food_safety_risk_assessment.ipynb
│   └── README.md
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

## Dataset columns

| Column | Meaning |
|---|---|
| sample_id | Demonstration sample identifier |
| food_category | Food category |
| moisture_percent | Moisture percentage |
| ph | pH value |
| protein_g_100g | Protein per 100g |
| fat_g_100g | Fat per 100g |
| bacterial_count_cfu_g | Illustrative bacterial count |
| contaminant_level_mg_kg | Illustrative contaminant measurement |
| storage_temperature_c | Storage temperature |
| storage_days | Storage duration |
| risk_label | Synthetic target label |

## Google Colab / Jupyter

Open `notebooks/food_safety_risk_assessment.ipynb` to see the complete AIML workflow including EDA, preprocessing, model training, evaluation, confusion matrix, and a sample prediction.

## How to run

### 1. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train_model.py
```

This creates:

```text
model/food_safety_model.pkl
model/metrics.json
```

### 4. Start Flask

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## ML workflow

```text
Input data
    ↓
Data validation
    ↓
Missing-value handling
    ↓
Numerical scaling + categorical encoding
    ↓
Train/Test Split
    ↓
Logistic Regression + Random Forest
    ↓
Evaluation
    ↓
Best model selected by F1-score
    ↓
Saved model
    ↓
Flask prediction interface
```

## Why F1-score?

In a screening/prioritization setting, accuracy alone can hide poor performance on the class we care about. Precision, recall and F1-score provide additional information about `At Risk` predictions.

## Limitations

- Synthetic dataset
- Synthetic target labels
- No claim of regulatory compliance
- No replacement for laboratory testing
- No replacement for FSSAI standards or official decisions
- Real-world deployment requires validated data, domain expertise, regulatory review, and appropriate model validation

## Future improvements

1. Replace synthetic data with authorized laboratory/surveillance data.
2. Add proper regulatory threshold features from the relevant product-specific standards.
3. Add explainable AI such as SHAP.
4. Add model monitoring and data-drift checks.
5. Add role-based access and audit logging.
6. Add sample-priority ranking for inspection planning.
7. Validate the model prospectively on an independent dataset.

## Suggested resume description

**Food Safety Risk Assessment using Machine Learning | Python, Pandas, Scikit-learn, Flask**

Developed an ML-based prototype to prioritize food samples for further safety assessment using laboratory-style food parameters. Implemented preprocessing, feature encoding, Logistic Regression and Random Forest models, evaluated predictions using precision, recall and F1-score, and deployed the trained model through a Flask web interface.

## Disclaimer

This is an educational project. It is not developed, endorsed, certified, or deployed by FSSAI. The demo dataset and labels are synthetic.
