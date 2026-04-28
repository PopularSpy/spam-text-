# Real-Time SMS Spam Detection Application

**Group Members**  
- Abdul Rafay (B24S0348AI074)  
- Farooq Awan (B24S0960AI065)  

**Instructor:** Mr Waqas Yousaf — Machine Learning Lab  

---

## Project Overview

A machine learning web application that classifies SMS messages as **Spam** or **Ham** in real time. It uses NLP preprocessing, TF-IDF feature extraction, and two classifiers (Multinomial Naive Bayes and Linear SVM). The best model is served through a Streamlit web interface.

---

## Recommended File Structure

```
spam-text-/
├── data/
│   └── spam.csv                  # UCI SMS Spam Collection dataset
├── models/
│   ├── vectorizer.pkl            # Saved TF-IDF vectorizer
│   └── best_model.pkl            # Saved best classifier
├── notebooks/
│   └── spam_detection.ipynb      # Full EDA, preprocessing, training, evaluation
├── app/
│   └── app.py                    # Streamlit web application
├── requirements.txt
└── README.md
```

---

## Step-by-Step Blueprint

### Phase 1 — Dataset

**Source:** [UCI SMS Spam Collection on Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

Download `spam.csv` and place it in the `data/` folder.  
The file has two relevant columns:

| Column | Description |
|--------|-------------|
| `v1`   | Label — `ham` or `spam` |
| `v2`   | Raw SMS text |

---

### Phase 2 — Data Loading & Preprocessing

**Goal:** Transform raw SMS text into clean tokens suitable for vectorization.

**Steps (in order):**

1. **Load** the CSV with `pandas`, keeping only columns `v1` (rename to `label`) and `v2` (rename to `text`).
2. **Encode labels** — map `ham` → 0, `spam` → 1.
3. **Lowercase** — `.str.lower()` on the text column.
4. **Remove punctuation and numbers** — use `re.sub(r'[^a-z\s]', '', text)`.
5. **Tokenize** — `nltk.word_tokenize(text)`.
6. **Remove stopwords** — use `nltk.corpus.stopwords.words('english')`.
7. **Rejoin** tokens back into a single cleaned string per message.

**Tip:** Download NLTK resources once at the top of your notebook:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

### Phase 3 — Exploratory Data Analysis (EDA)

Before modelling, understand your data:

- Print class distribution (`value_counts`) — expect ~87 % ham / 13 % spam.
- Plot a bar chart of class counts with `seaborn`.
- Compare average message length between spam and ham.
- Optional: generate a `WordCloud` for each class to visualise dominant terms.

---

### Phase 4 — TF-IDF Vectorization

**Recommended Parameters:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),   # unigrams + bigrams (captures "free offer", "call now")
    min_df=2,             # ignore terms appearing in fewer than 2 documents (reduces noise)
    max_features=5000,    # cap vocabulary size for speed
    sublinear_tf=True,    # apply log normalization to term frequencies
)
```

**Why these settings?**
- `ngram_range=(1,2)` — bigrams catch multi-word spam phrases that single words miss.
- `min_df=2` — removes typos/rare tokens that add noise without signal.
- `max_features=5000` — keeps memory manageable without sacrificing accuracy.
- `sublinear_tf=True` — dampens the effect of very high raw term frequencies.

**Train/Test Split (before fitting the vectorizer):**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['label'],
    test_size=0.20,
    random_state=42,
    stratify=df['label'],   # preserves class imbalance ratio in both splits
)
```

Fit the vectorizer **only on the training set**, then transform both sets:

```python
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)
```

---

### Phase 5 — Model Training & Evaluation

#### 5.1 Multinomial Naive Bayes

```python
from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB(alpha=0.1)  # alpha < 1 sharpens predictions for sparse text
nb_model.fit(X_train_tfidf, y_train)
```

**Why `alpha=0.1`?** The default `alpha=1.0` (Laplace smoothing) is conservative. For SMS spam with TF-IDF, `0.1` typically improves F1 for the spam class.

#### 5.2 Linear SVM

```python
from sklearn.svm import LinearSVC

svm_model = LinearSVC(C=1.0, max_iter=2000)
svm_model.fit(X_train_tfidf, y_train)
```

**Why `LinearSVC`?** It is dramatically faster than kernel SVM on sparse, high-dimensional TF-IDF matrices and consistently achieves top F1 scores on text classification benchmarks.

#### 5.3 Evaluation (both models)

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    print(f"\n=== {name} ===")
    print(classification_report(y_test, preds, target_names=['Ham', 'Spam']))
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'])
    plt.title(f'Confusion Matrix — {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
```

**Key metrics to focus on** (because of class imbalance):

| Metric | What it tells you |
|--------|------------------|
| **Precision (spam)** | Of all messages flagged spam, how many truly were? High precision = fewer false alarms on legitimate messages. |
| **Recall (spam)** | Of all actual spam, how many were caught? High recall = fewer spam messages slipping through. |
| **F1-score (spam)** | Harmonic mean — use this as the single deciding number when comparing models. |

---

### Phase 6 — Selecting & Saving the Best Model

Compare the `f1-score` for the **spam class** from the classification reports of both models. Whichever is higher becomes `best_model`.

```python
import joblib

joblib.dump(vectorizer, 'models/vectorizer.pkl')
joblib.dump(best_model, 'models/best_model.pkl')
print("Model and vectorizer saved.")
```

Use `joblib` (not `pickle`) — it handles large NumPy arrays more efficiently.

---

### Phase 7 — Streamlit App (`app/app.py`)

See `app/app.py` for the full skeleton. The app:

1. Loads `models/vectorizer.pkl` and `models/best_model.pkl` on startup (cached with `@st.cache_resource`).
2. Accepts free-text SMS input from the user.
3. Applies the **same preprocessing pipeline** as the notebook.
4. Vectorizes with the saved vectorizer.
5. Predicts and displays a colour-coded result (🟢 Ham / 🔴 Spam).
6. Shows a confidence bar (when the model supports `predict_proba`; for `LinearSVC` it uses `decision_function` as a proxy).

---

## Run Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NLTK Data (one-time)

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### 3. Run the Notebook

Open Jupyter and run all cells in `notebooks/spam_detection.ipynb`. This will:
- Preprocess the data
- Train and evaluate both models
- Save `models/vectorizer.pkl` and `models/best_model.pkl`

### 4. Launch the Web App

```bash
streamlit run app/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Expected Results (Typical Benchmarks on This Dataset)

| Model | Precision (Spam) | Recall (Spam) | F1 (Spam) |
|-------|-----------------|--------------|-----------|
| Multinomial NB (alpha=0.1) | ~0.97 | ~0.91 | ~0.94 |
| Linear SVM (C=1.0) | ~0.98 | ~0.95 | ~0.97 |

> SVM usually wins on F1; Naive Bayes is simpler to explain. Save whichever scores higher.

---

## Technology Stack

| Category | Library |
|----------|---------|
| Data manipulation | `pandas`, `numpy` |
| ML & NLP | `scikit-learn`, `nltk` |
| Visualization | `matplotlib`, `seaborn` |
| Model persistence | `joblib` |
| Web UI | `streamlit` |
