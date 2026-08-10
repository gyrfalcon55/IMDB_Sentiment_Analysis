
<img width="1376" height="768" alt="Thumbnail" src="https://github.com/user-attachments/assets/3feb350f-b37f-4d96-8c1e-5b75a0b15b3a" />

---

# 🎬 IMDB Sentiment Analysis

**An end-to-end, production-style Machine Learning pipeline** that classifies IMDB movie reviews as **Positive** or **Negative**, complete with NLP preprocessing, multi-model training, experiment tracking on DagsHub/MLflow, structured logging, custom exception handling, and a FastAPI web interface for training and live predictions.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Web%20App-teal.svg)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-orange.svg)
![DagsHub](https://img.shields.io/badge/DagsHub-Integrated-blueviolet.svg)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-purple.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Project Overview](#-project-overview)
- [Tech Stack](#️-tech-stack)
- [Project Architecture](#️-project-architecture)
- [Repository Structure](#-repository-structure)
- [Module Breakdown](#-module-breakdown)
- [Models Trained](#-models-trained)
- [Experiment Tracking (MLflow + DagsHub)](#-experiment-tracking-mlflow--dagshub)
- [Model Performance](#-model-performance)
- [Logging & Exception Handling](#-logging--exception-handling)
- [Installation & Setup](#-installation--setup)
- [Running the App](#-running-the-app)
- [Configuration Files](#-configuration-files)
- [Artifacts Generated](#-artifacts-generated)
- [Author](#️-author)

---

## 🧩 Problem Statement

Movie reviews carry rich, unstructured textual signals that reflect audience sentiment. This project builds a complete NLP + ML system to automatically classify a given movie review as **Positive** or **Negative**, benchmarking multiple classical ML algorithms and promoting the best-performing model to production for real-time inference.

---

## 🧠 Project Overview

The pipeline performs the following stages end-to-end:

1. **Data Ingestion** — Load raw review data from CSV, versioned with DVC.
2. **Text Preprocessing** — Clean, tokenize, remove stopwords, and stem review text.
3. **Train/Test Split** — Persist processed splits as reusable artifacts.
4. **Feature Engineering** — Convert text to numerical vectors via `CountVectorizer`.
5. **Multi-Model Training** — Train and tune **4 candidate models**: Logistic Regression, SVM, Random Forest, and Naive Bayes using `RandomizedSearchCV`.
6. **Experiment Tracking** — Log parameters, metrics, and artifacts for every model run to **MLflow via DagsHub**.
7. **Model Evaluation & Selection** — Compare all 4 models on CV score, F1 score, and ROC-AUC, then automatically select and persist the **best-performing model**.
8. **Serving** — Expose training and prediction through a **FastAPI** web application.
9. **Observability** — Every stage is wrapped in custom logging and centralized exception handling for traceability and debugging.

---

## ⚙️ Tech Stack

| Category | Tools / Libraries |
|---|---|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **NLP Preprocessing** | NLTK (tokenization, stopwords, stemming) |
| **Vectorization** | Scikit-learn `CountVectorizer` |
| **Modeling** | Logistic Regression, SVM, Random Forest, Naive Bayes |
| **Hyperparameter Tuning** | `RandomizedSearchCV` |
| **Evaluation** | Accuracy, F1-score, CV Score, ROC-AUC |
| **Experiment Tracking** | MLflow (via DagsHub) |
| **Data & Model Versioning** | DVC |
| **Serialization** | Joblib |
| **Logging** | Custom rotating file logger |
| **Error Handling** | Custom exception module with traceback context |
| **Deployment** | FastAPI + Jinja2 templates |

---

## 🏗️ Project Architecture

```
        ┌───────────────────────────────┐
        │        config.yaml            │
        └──────────────┬────────────────┘
                        │
             ┌──────────▼──────────┐
             │   Data Ingestion    │
             │  (DVC-tracked CSV)  │
             └──────────┬──────────┘
                        │
             ┌──────────▼──────────┐
             │ Data Preprocessing  │
             │  • Clean text       │
             │  • Tokenize         │
             │  • Remove stopwords │
             │  • Apply stemming   │
             │  • Train/test split │
             └──────────┬──────────┘
                        │
             ┌──────────▼──────────┐
             │   Model Training    │
             │  • CountVectorizer  │
             │  • 4 candidate ML   │
             │    models           │
             │  • RandomizedSearchCV│
             │  • MLflow/DagsHub   │
             │    logging          │
             └──────────┬──────────┘
                        │
             ┌──────────▼──────────┐
             │  Model Evaluation   │
             │  • CV / F1 / ROC-AUC│
             │  • Best model select│
             │  • Save artifacts   │
             └──────────┬──────────┘
                        │
             ┌──────────▼──────────┐
             │  Model Prediction   │
             │  • Clean input      │
             │  • Vectorize        │
             │  • Predict via      │
             │    best model       │
             └─────────────────────┘
```

Every stage above is instrumented with **custom logging** and **custom exception handling**, so failures are captured with full context instead of raw stack traces.

---

## 📂 Repository Structure

```
IMDB_Sentiment_Analysis/
│
├── app.py                                  # FastAPI app — training & prediction endpoints
├── requirements.txt
├── setup.py
├── README.md
│
├── config/
│   └── config.yaml                         # File & directory paths used across the project
│
├── data/
│   ├── raw_data/
│   │   └── movie.csv (DVC-tracked)
│   └── processed/
│       ├── train_data/train.csv
│       └── test_data/test.csv
│
├── artifacts/
│   ├── models/best_model.pkl               # Best model selected after evaluation
│   └── vectorizers/count_vectorizer.pkl    # Fitted CountVectorizer
│
├── logs/                                   # Timestamped run logs
│
├── notebooks/                              # EDA & experimentation notebooks
│
├── src/IMDB_Project/
│   ├── components/
│   │   ├── data_ingestion.py               # Load raw dataset
│   │   ├── data_processing.py              # Clean, tokenize, stem, split
│   │   ├── model_trainer.py                # Train 4 models + MLflow logging
│   │   ├── model_evaluation.py             # Evaluate & select best model
│   │   └── model_prediction.py             # Real-time inference logic
│   │
│   ├── pipeline/
│   │   ├── training_pipeline.py            # Orchestrates the full training flow
│   │   └── prediction_pipeline.py          # Orchestrates inference flow
│   │
│   ├── config/
│   │   └── model_configuration.yml         # Model classes & hyperparameter grids
│   │
│   ├── entity/
│   │   └── config_entity.py                # Typed config/data classes
│   │
│   ├── constants/                          # Project-wide constants
│   ├── utils/common.py                     # YAML/IO helper functions
│   ├── logger.py                           # Custom logging setup
│   └── exception.py                        # Custom exception handling
│
├── templates/                              # HTML templates (index, model training UI)
└── static/                                 # CSS & static assets
```

---

## 🧾 Module Breakdown

### 1️⃣ Data Ingestion
Reads paths from `config.yaml` and loads the DVC-tracked raw IMDB dataset into a DataFrame for downstream processing.

### 2️⃣ Data Processing
Performs full text-cleaning pipeline:
- Lowercasing
- Removing special characters (`[^a-zA-Z0-9\s]`)
- Tokenization (`word_tokenize`)
- Stopword removal
- Stemming (`PorterStemmer`)
- Rejoining tokens into cleaned text
- 80–20 train/test split, saved as CSV artifacts

### 3️⃣ Model Trainer
The core training engine:
- Loads processed train/test data
- Vectorizes text using `CountVectorizer` (`max_features=10000`)
- Reads model classes & hyperparameter search space from `model_configuration.yml`
- Trains and tunes **4 models** — Logistic Regression, SVM, Random Forest, Naive Bayes — via `RandomizedSearchCV`
- Logs every run (params, CV score, F1, ROC-AUC, model artifact) to **MLflow on DagsHub**
- Saves the fitted vectorizer as an artifact

### 4️⃣ Model Evaluation
- Reloads all trained models and their logged metrics
- Compares CV score, F1-score, and ROC-AUC across the 4 candidates
- Selects and persists the **best-performing model** as `best_model.pkl`
- Generates classification reports & confusion matrices

### 5️⃣ Model Prediction
Powers real-time inference from the FastAPI UI:
- Loads `best_model.pkl` and `count_vectorizer.pkl`
- Applies the identical preprocessing pipeline to user input
- Vectorizes and predicts — returns `"Positive"` or `"Negative"`

---

## 🤖 Models Trained

The pipeline benchmarks **four classical ML algorithms**, each tuned via `RandomizedSearchCV` over its respective hyperparameter grid defined in `model_configuration.yml`:

| Model | Description |
|---|---|
| **Logistic Regression** | Linear baseline classifier, tuned for regularization strength |
| **Support Vector Machine (SVM)** | Margin-based classifier, tuned for kernel & C parameter |
| **Random Forest** | Ensemble of decision trees, tuned for depth & estimators |
| **Naive Bayes** | Probabilistic classifier, well-suited for text classification |

After tuning, all four are evaluated head-to-head and the top performer is automatically promoted to `artifacts/models/best_model.pkl` for serving.

---

## 📊 Experiment Tracking (MLflow + DagsHub)

Every training run — across all four models — is logged to **MLflow**, hosted via **DagsHub**, so experiments are fully reproducible and comparable.

```python
dagshub.init(repo_owner='gyrfalcon55', repo_name='IMDB_Sentiment_Analysis', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/gyrfalcon55/IMDB_Sentiment_Analysis.mlflow")
mlflow.set_experiment("IMDB_Sentiment_Analysis_v1")
```

**Tracked per model run:**
- Hyperparameters (best params from `RandomizedSearchCV`)
- CV Score
- F1 Score
- ROC-AUC Score
- Serialized model artifact (`.pkl`)
- Confusion matrix & classification report

Visit the DagsHub MLflow dashboard to visually compare runs across all four algorithms.

---

## 📈 Model Performance

Final benchmark results across all four candidate models:

| Model | CV Score | F1 Score | ROC-AUC |
|---|---|---|---|
| **Logistic Regression** ⭐ | 0.8764 | 0.8805 | 0.8788 |
| **SVM** | 0.8756 | 0.8805 | 0.8788 |
| **Naive Bayes** | 0.8459 | 0.8470 | 0.8466 |
| **Random Forest** | 0.8412 | 0.8397 | 0.8371 |

⭐ **Logistic Regression** narrowly edges out SVM on CV score and is selected as the best model for deployment, with both models performing near-identically on F1 and ROC-AUC.

---

## 🪵 Logging & Exception Handling

- **Custom Logger** (`logger.py`): Every pipeline run generates a timestamped log file under `logs/`, capturing stage-level progress, warnings, and errors for full traceability.
- **Custom Exception Handling** (`exception.py`): All components raise a unified custom exception that wraps the original error with file name, line number, and contextual message — making debugging failures across the pipeline fast and consistent.

---

## 🚀 Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/gyrfalcon55/IMDB_Sentiment_Analysis.git
cd IMDB_Sentiment_Analysis
```

### 2️⃣ Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App

### Launch the FastAPI server
```bash
uvicorn app:app --reload
```

Visit **http://127.0.0.1:8000** in your browser to:
- Enter a review and get a real-time **Positive/Negative** prediction
- Trigger a full **training run** (all 4 models + evaluation) directly from the **"Train Model"** button in the UI

---

## 🛠️ Configuration Files

| File | Purpose |
|---|---|
| `config/config.yaml` | Central registry of file & directory paths used across the project |
| `src/IMDB_Project/config/model_configuration.yml` | Defines the 4 model classes and their `RandomizedSearchCV` hyperparameter grids — edit this to add/remove models or tune search space |

---

## 📁 Artifacts Generated

| Folder | Contents |
|---|---|
| `artifacts/models/` | Best trained model (`best_model.pkl`) |
| `artifacts/vectorizers/` | Fitted `CountVectorizer` (`count_vectorizer.pkl`) |
| `data/processed/train_data/`, `data/processed/test_data/` | Cleaned, split datasets |
| `logs/` | Timestamped execution logs |
| MLflow (DagsHub) | Params, metrics, confusion matrices, and model artifacts per run |

---

## ✍️ Author

**Shaik Junaid**
🔗 [GitHub: gyrfalcon55](https://github.com/gyrfalcon55)
