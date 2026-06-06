# FraudGuard — Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![LightGBM](https://img.shields.io/badge/LightGBM-F1%3A0.837-brightgreen?style=flat-square)
![Tests](https://img.shields.io/badge/Tests-11%2F11%20Passed-brightgreen?style=flat-square)
![Deploy](https://img.shields.io/badge/Deploy-Render-purple?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=flat-square)

Real-time fraud detection system trained on 284,807 credit card transactions. Built end-to-end — data pipeline, feature engineering, model training, REST API, and a live dashboard.

**Live:** https://fraudguard.onrender.com &nbsp;|&nbsp; **API Docs:** https://fraudguard.onrender.com/api/docs &nbsp;|&nbsp; **GitHub:** https://github.com/atifbashir-ju/FraudGuard

---

## Screenshots

### Overview Dashboard
![Overview]![alt text](<Screenshot 2026-06-05 111553.png>)

### Real-Time Fraud Detection
![Detect](c:\Users\shame\OneDrive\Pictures\Screenshots\Screenshot 2026-06-05 111634.png)


## Results

| Model | F1 Score | AUC-ROC | AUC-PR | Threshold |
|-------|----------|---------|--------|-----------|
| Logistic Regression | 0.7362 | 0.9338 | 0.6991 | 1.000 |
| Random Forest | 0.8214 | 0.9610 | 0.8160 | 0.570 |
| XGBoost (Optuna) | 0.8000 | 0.9705 | 0.8093 | 0.991 |
| **LightGBM (Optuna)** ✅ | **0.8372** | **0.9627** | **0.8124** | **0.787** |

Dataset: 284,807 transactions &nbsp;|&nbsp; Fraud rate: 0.17% &nbsp;|&nbsp; After SMOTE: 385,224 balanced samples

---

## What This Project Does

Takes a credit card transaction and tells you in under a second whether it looks fraudulent — with a probability score and SHAP explanation of which features drove the decision.

The main challenge was the severe class imbalance (0.17% fraud). A naive model predicting "legit" every time gets 99.83% accuracy while missing all fraud. I solved this with SMOTETomek, then compared four algorithms with proper metrics (F1, AUC-PR — not accuracy).

---

## Project Structure

```
FraudGuard/
├── server.py                        ← Single entry point (FastAPI + Django)
├── manage.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── render.yaml
│
├── src/
│   ├── pipeline/
│   │   ├── data_pipeline.py         ← Cleaning + 30+ features
│   │   └── train.py                 ← 4 models + Optuna + SHAP + MLflow
│   ├── api/
│   │   └── main.py                  ← FastAPI routes
│   └── dashboard/
│       ├── settings.py
│       ├── models.py                ← Transaction DB model
│       └── views.py                 ← Django views
│
├── templates/dashboard/
│   └── index.html                   ← Full dashboard UI (HTML+CSS+JS)
│
├── models/                          ← Trained .pkl files
├── notebooks/eda.ipynb              ← EDA with findings
├── screenshots/                     ← Dashboard screenshots
└── tests/test_pipeline.py           ← 11 unit tests
```

---

## How to Run Locally

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Get the dataset**

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) → place in `data/raw/`

**3. Train models** (one time only — ~45 min)
```bash
python src/pipeline/train.py
```

**4. Start server**
```bash
python manage.py migrate
python server.py
```

Open `http://127.0.0.1:10000`

**5. MLflow UI** (optional)
```bash
mlflow ui --backend-store-uri mlflow_runs
```

---

## API Usage

```bash
# Single prediction
curl -X POST https://fraudguard.onrender.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 0.01, "hour": 3, "V14": -9.3, "V17": -8.7}'
```

```json
{
  "is_fraud": true,
  "fraud_probability": 0.9231,
  "risk_level": "CRITICAL",
  "model_used": "lightgbm",
  "explanation": "FRAUD: late-night transaction, suspiciously small amount. Probability: 92.3%"
}
```

---

## Dashboard Features

| Feature | Description |
|---------|-------------|
| **Overview** | Stats, 7-day trend chart, risk distribution, recent transactions |
| **Detect** | Single transaction analysis with SHAP feature impact |
| **Batch Upload** | CSV upload — analyze 1000 transactions at once |
| **History** | Paginated transaction log with filters |
| **Analytics** | Daily volume, fraud rate over time |
| **Model Comparison** | 4 models side by side with metrics chart |
| **Explainability** | Global SHAP feature importance |

---

## ML Decisions

**Why SMOTETomek?**
Combines oversampling with Tomek links removal — cleans borderline samples near the decision boundary. Better precision than plain SMOTE.

**Why optimize threshold?**
Default 0.5 isn't optimal on imbalanced data. Used Precision-Recall curve to find optimal threshold (0.787 for LightGBM). Missed fraud costs more than false alarms in banking.

**Why Optuna over GridSearch?**
Bayesian optimization finds good hyperparameters in 5 trials vs exhaustive grid that would take days on this dataset size.

**Why SHAP?**
Banking regulations require explainable AI. SHAP shows exactly which features drove each prediction — critical for audits and compliance.

---

## Key Findings from EDA

- Fraud peaks late night (10pm–6am) — 3x higher rate than daytime
- 23% of fraud transactions involve amounts under $10 (test transactions)
- V14 is the single most predictive feature (SHAP: 3.07)
- Engineered `risk_score` feature ranked #3 in SHAP — confirms feature engineering added value

---

## Tests

```bash
pytest tests/ -v --cov=src
```
11/11 tests pass &nbsp;|&nbsp; Pipeline coverage: 62%

---

## Docker

```bash
docker-compose up --build
```

---


