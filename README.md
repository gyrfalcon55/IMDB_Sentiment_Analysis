# 🎬 IMDB Sentiment Analysis  

<img width="1897" height="901" alt="image" src="https://github.com/user-attachments/assets/30c8ad90-32a5-4de0-881c-b37ef697a18a" />


An **end-to-end machine learning pipeline** that classifies IMDB movie reviews as **Positive** or **Negative** using NLP preprocessing, vectorization, ML model training, MLflow experiment tracking, and deployment via a FastAPI web app.  

---

## 🧩 Problem Statement  

Movie reviews contain rich textual data that reflect audience sentiment. The objective of this project is to automatically predict whether a given movie review expresses **positive** or **negative** sentiment using Natural Language Processing (NLP) and Machine Learning.

---

## 🧠 Project Overview  

The project performs:
1. Data ingestion from raw CSV files  
2. Text preprocessing (cleaning, tokenizing, stemming, removing stopwords)  
3. Splitting data into train/test sets  
4. Converting text into numerical features using **CountVectorizer**  
5. Hyperparameter tuning across multiple models using **RandomizedSearchCV**  
6. Logging experiments with **MLflow** (via **DagsHub**)  
7. Saving the best model & vectorizer as artifacts  
8. Evaluating model performance on test data  
9. Serving predictions via FastAPI web interface  

---

## ⚙️ Tech Stack  

| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **NLP Preprocessing** | NLTK (tokenization, stopwords, stemming) |
| **Vectorization** | Scikit-learn `CountVectorizer` |
| **Modeling** | Logistic Regression, Naive Bayes, SVM (from sklearn) |
| **Evaluation** | Accuracy, F1-score, ROC AUC, Classification Report |
| **Experiment Tracking** | MLflow (DagsHub integration) |
| **Serialization** | Joblib |
| **Deployment** | FastAPI |
| **Versioning** | DVC (optional), Git |

---

## 🏗️ Project Structure  

```
IMDB_Sentiment_Analysis/
│
├── app.py                               # FastAPI web app for training & predictions
├── config/config.yaml                   # File paths & directories configuration
├── config/model_config.yaml             # Model class paths & hyperparameters
├── requirements.txt                     # Python dependencies
├── setup.py                             # Project setup
│
├── src/
│   └── IMDB_Project/
│       ├── components/
│       │   ├── data_ingestion.py        # Load raw dataset
│       │   ├── data_processing.py       # Clean text, tokenize, stem, split data
│       │   ├── model_trainer.py         # Train models, hyperparameter tuning, MLflow logging
│       │   ├── model_evaluation.py      # Evaluate model on test data
│       │   └── model_prediction.py      # Handle live user input prediction
│       │
│       ├── utils/common.py              # Helper functions (YAML reading, CSV operations)
│       ├── logger.py                    # Project-wide logging setup
│       ├── exception.py                 # Custom exception handling
│       └── pipeline/
│           ├── data_ingestion.py        # Load raw dataset
│           ├── data_processing.py       # Orchestrates entire training pipeline
│
├── templates/                           # HTML templates for FastAPI UI
├── static/                              # CSS / JS for FastAPI app
└── artifacts/                           # Saved models & vectorizers
```

---

## 🧾 Each File Explained  

### 1️⃣ **`data_ingestion.py`**
- Reads file paths from `config.yaml`
- Loads the raw IMDB dataset (CSV) into a pandas DataFrame
- Returns the loaded data for downstream use  

---

### 2️⃣ **`data_processing.py`**
Handles **text cleaning, tokenization, stemming**, and **train-test split**.  

**Steps performed:**
- Lowercase conversion  
- Remove special characters (`[^a-zA-Z0-9\s]`)  
- Tokenization (`word_tokenize`)  
- Stopword removal  
- Apply PorterStemmer for stemming  
- Join words back into cleaned text  
- Split dataset (80–20) and save as CSVs  

Artifacts generated:
- `data/processed/train_data.csv`
- `data/processed/test_data.csv`

---

### 3️⃣ **`model_trainer.py`**
Core training module performing:
1. Loads train/test datasets  
2. Converts text → vector using **CountVectorizer** (max_features=10000)
3. Reads model configurations from `config/model_config.yaml`
4. Runs **RandomizedSearchCV** for each model
5. Logs metrics & params to **MLflow (via DagsHub)**
6. Saves best model & vectorizer as artifacts  

**Artifacts saved:**
```
artifacts/models/best_model.pkl
artifacts/vectorizers/count_vectorizer.pkl
```

**MLflow Integration:**
```python
dagshub.init(repo_owner='gyrfalcon55', repo_name='IMDB_Sentiment_Analysis', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/gyrfalcon55/IMDB_Sentiment_Analysis.mlflow")
mlflow.set_experiment("IMDB_Sentiment_Analysis_v1")
```

---

### 4️⃣ **`model_evaluation.py`**
- Loads the trained model & vectorizer  
- Evaluates using the test dataset  
- Computes:
  * Accuracy
  * F1-score
  * ROC-AUC
  * Confusion matrix
  * Classification report  

---

### 5️⃣ **`model_prediction.py`**
Used for **real-time sentiment prediction** through the FastAPI UI.  

**Flow:**
- Load saved model & vectorizer
- Clean user input using the same preprocessing logic
- Vectorize input
- Predict sentiment:  
  `"Positive"` if label = 1 else `"Negative"`

---

## 🔄 Complete Pipeline Architecture  

```
        ┌───────────────────────────────┐
        │         config.yaml           │
        └──────────────┬────────────────┘
                       │
            ┌──────────▼──────────┐
            │   Data Ingestion    │
            │ (read CSV file)     │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │ Data Preprocessing  │
            │  - Clean text       │
            │  - Tokenize         │
            │  - Remove stopwords │
            │  - Apply stemming   │
            │  - Split train/test │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │   Model Training    │
            │  - CountVectorizer  │
            │  - RandomSearchCV   │
            │  - MLflow logging   │
            │  - Save artifacts   │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  Model Evaluation   │
            │  - Accuracy / F1    │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  Model Prediction   │
            │  - Clean Input      │
            │  - Predict          │
            │  - Return Sentiment │
            └─────────────────────┘
```

---

## 🧮 Model Evaluation Metrics  

| Metric | Description |
|---------|-------------|
| **Accuracy** | Correct predictions / Total predictions |
| **F1 Score** | Harmonic mean of precision and recall |
| **ROC AUC** | Area under the ROC curve |
| **Classification Report** | Precision, recall, f1 per class |

---

## 🚀 Installation & Local Setup  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/gyrfalcon55/IMDB_Sentiment_Analysis.git
cd IMDB_Sentiment_Analysis
```

### 2️⃣ Create virtual environment  
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# OR
source venv/bin/activate       # macOS/Linux
```

### 3️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```

### 4️⃣ Launch the FastAPI app  
```bash
uvicorn app:app --reload
```

Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.  
Enter a review and get real-time sentiment predictions.

---

### 5️⃣ Training   
```bash
Traing can be done from the webapp also. Just click the 'Train Model' and the training will start
```

### 6️⃣ config/config.yaml  
```bash
This file contains all the paths of the files which are used inside the project

```

### 7️⃣ src/IMDB_Project/config/config.yaml
```bash
This file contains all the details of Machine Learning Models and parameters of each model. 
Can just modify this file to train on models as your choice
```


## 📊 MLflow Tracking (DagsHub Integration)  

All experiments are logged to MLflow via DagsHub.  
The configuration inside `model_trainer.py` ensures:
```python
dagshub.init(repo_owner='gyrfalcon55', repo_name='IMDB_Sentiment_Analysis', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/gyrfalcon55/IMDB_Sentiment_Analysis.mlflow")
mlflow.set_experiment("IMDB_Sentiment_Analysis_v1")
```

**Tracked Artifacts:**
- Model Parameters
- RandomizedSearchCV best params
- Accuracy, F1, ROC AUC
- Serialized model (`.pkl`)  
- Confusion Matrix & Reports (if logged manually)

Access MLflow dashboard on DagsHub to visualize experiment performance and compare models.

---

## 📁 Artifacts Generated  

| Folder | Contents |
|---------|-----------|
| `artifacts/models/` | Best trained model (`best_model.pkl`) |
| `artifacts/vectorizers/` | Trained CountVectorizer (`count_vectorizer.pkl`) |
| `data/processed/` | `train_data.csv`, `test_data.csv` |
| `mlruns/` | MLflow local tracking directory (if configured locally) |

---


## ✍️ Author  
**Shaik Junaid**  
🔗 [GitHub: gyrfalcon55](https://github.com/gyrfalcon55)
