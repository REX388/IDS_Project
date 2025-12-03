# Code Explanation — IDS Project

This document explains the main code snippets and behavior in `all.py` and `realtime_ids.py`. It aims to help you understand the data flow, preprocessing, model training, saving/loading artifacts, and the real-time simulation.

---

## 1) Top-level path handling (robust approach)

Snippet (conceptual):

```python
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
train_csv = os.path.join(base_dir, 'kddtrain.csv')
test_csv = os.path.join(base_dir, 'kddtest.csv')
```

Explanation:
- `base_dir` uses the script file location as the anchor so the script will work even if the current working directory (`cwd`) differs from the project root.
- Always use `os.path.join(base_dir, ...)` when referencing data or artifact files to avoid `FileNotFoundError` when launching scripts from other folders.

---

## 2) Reading datasets and sampling (in `all.py`)

Snippet (conceptual):

```python
import pandas as pd
traindata = pd.read_csv(os.path.join(base_dir, 'kddtrain.csv'), header=None)
testdata = pd.read_csv(os.path.join(base_dir, 'kddtest.csv'), header=None)
# Use 5% for speed on limited systems
train_frac = int(0.05 * len(traindata))
traindata = traindata.iloc[:train_frac]
```

Explanation:
- The code reads raw CSV files without headers; labels are expected in column 0.
- A fraction of rows is used to limit run-time on low-resource systems. Adjust the 0.05 factor to match your system.

---

## 3) Feature selection and normalization

Snippet:

```python
X = traindata.iloc[:, 1:42]
Y = traindata.iloc[:, 0]
from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(X)
trainX = scaler.transform(X)
```

Explanation:
- The training features are columns 1..41 (pandas `1:42` is end-exclusive), and column 0 is the label.
- `Normalizer` rescales each sample vector to unit norm (useful for distance-based models and some linear models). If you prefer standardization (zero mean, unit variance), consider `StandardScaler` instead.
- The fitted `scaler` should be saved to disk (e.g., with `joblib.dump`) and re-used at inference time to ensure consistent preprocessing.

---

## 4) Training and saving models (Logistic Regression example)

Snippet:

```python
from sklearn.linear_model import LogisticRegression
from joblib import dump
model = LogisticRegression()
model.fit(trainX, Y)
# Ensure output directory exists then save
dump(model, os.path.join(base_dir, 'classical', 'logistic_model.joblib'))
dump(scaler, os.path.join(base_dir, 'classical', 'normalizer.joblib'))
```

Explanation:
- After training, save both the model and the scaler. This allows the real-time script to load and apply identical preprocessing.
- Confirm the `classical` directory exists before saving (scripts in this repo create it if missing).

---

## 5) Predictions and metric calculation

Snippet:

```python
predicted = model.predict(testdata_transformed)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(expected, predicted)
precision = precision_score(expected, predicted, average='binary')
```

Explanation:
- Predictions on the test subset are compared to expected labels to compute standard metrics.
- Use appropriate `average` arguments (e.g., `binary`, `macro`, `weighted`) depending on label distribution.

---

## 6) Multiple classifiers loop (pattern)

Typical pattern used in `all.py` for each classifier:

1. Instantiate classifier (e.g., `KNeighborsClassifier()`).
2. Call `fit(traindata, trainlabel)`.
3. Use `predict(testdata)` and `predict_proba(testdata)` where available.
4. Save predicted labels/probabilities to `classical/` for further analysis.
5. Compute and print metrics.

This pattern is repeated for Naive Bayes, KNN, Decision Tree, AdaBoost, RandomForest, SVM (rbf and linear), etc.

---

## 7) Real-time simulation (`realtime_ids.py`)

Key responsibilities:
- Load `classical/logistic_model.joblib` and `classical/normalizer.joblib` with `joblib.load`.
- Read `kddtest.csv` (or a stream source) and preprocess each row with the loaded scaler.
- For each sample, call `model.predict` and print `"Normal"` or `"Attack"` with a timestamp.
- Optionally include a `time.sleep()` between prints to simulate streaming.

Example (conceptual):

```python
from joblib import load
model = load(os.path.join(base_dir, 'classical', 'logistic_model.joblib'))
scaler = load(os.path.join(base_dir, 'classical', 'normalizer.joblib'))
for sample in test_samples:
    x = scaler.transform(sample.reshape(1, -1))
    pred = model.predict(x)[0]
    print('Normal' if pred==0 else 'Attack')
```

Notes:
- Use `reshape(1, -1)` when predicting a single row.
- If `predict_proba` is available and you want confidence thresholds, use `model.predict_proba(x)` to inspect class probabilities.

---

## 8) File and folder expectations

- `classical/` must exist and contain: `logistic_model.joblib`, `normalizer.joblib` after a successful run of `all.py`.
- Prediction outputs like `predictedlabelLR.txt` and `predictedprobaLR.txt` will be saved into `classical/` by `all.py`.

---


- Add CLI arguments to `all.py` to configure datasets and output folders.
- Add a `--save-artifacts` flag that controls whether models are saved.
- Replace `Normalizer` with `StandardScaler` if modeling benefits from standardization.
- Add error handling around file reads/writes and model load calls.
- Consider using logging instead of print statements for production use.

---

## 9) Quick run commands (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python all.py
python realtime_ids.py
```

---

