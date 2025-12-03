# CODE EXPLANATION — Real-Time IDS Project (Ultra-Detailed Guide)

This document provides an **extremely detailed, line-by-line explanation** of how `all.py` and `realtime_ids.py` work together to create a Real-Time Intrusion Detection System using Logistic Regression.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Understanding the Assignment](#understanding-the-assignment)
3. [Complete Breakdown of all.py](#complete-breakdown-of-allpy)
4. [Complete Breakdown of realtime_ids.py](#complete-breakdown-of-realtime_idspy)
5. [How Both Scripts Work Together](#how-both-scripts-work-together)
6. [Machine Learning Concepts](#machine-learning-concepts)
7. [File Outputs and Usage](#file-outputs-and-usage)
8. [Troubleshooting Guide](#troubleshooting-guide)

---

## Project Overview

### What is this project?

This is a **Real-Time Intrusion Detection System (IDS)** that uses machine learning to classify network traffic as either "Normal" or "Attack". The project consists of two complementary scripts:

1. **Training Script (`all.py`)**: Trains a Logistic Regression model on historical network traffic data and saves it
2. **Real-Time Script (`realtime_ids.py`)**: Loads the trained model and uses it to detect attacks in simulated real-time

### Why Intrusion Detection Matters

- Networks face constant cyber threats
- Traditional signature-based detection misses new/unknown attacks
- Machine learning can learn patterns and detect anomalies
- Real-time detection enables immediate response

---

## Understanding the Assignment

### Assignment Requirements (CYB 213 - Project 12)

- **Objective**: Simulate live intrusion detection on streaming data
- **Algorithm**: **Logistic Regression** (specified in assignment)
- **Output**: Real-time "Normal" or "Attack" alerts in console
- **Dataset**: KDD-style network traffic (kddtrain.csv, kddtest.csv)

### Why Logistic Regression?

1. **Binary Classification**: Perfect for Normal vs. Attack (2 classes)
2. **Probabilistic**: Provides confidence scores
3. **Efficient**: Fast training and prediction (critical for real-time)
4. **Interpretable**: Shows which features matter
5. **Industry Standard**: Widely used in cybersecurity

---

## Complete Breakdown of all.py

### Script Structure

`all.py` follows 8 major steps:

1. **Path setup and data loading**
2. **Data sampling (5% for efficiency)**
3. **Feature and label extraction**
4. **Feature normalization**
5. **Model training (Logistic Regression)**
6. **Save model artifacts (.joblib files)**
7. **Model evaluation on test data**
8. **Display results (metrics and reports)**

---

### IMPORTS EXPLAINED

#### Import: `import numpy as np`

**What it does**: NumPy = Numerical Python, fundamental library for arrays and math

**Why we need it**:
- Efficient multi-dimensional arrays
- Mathematical operations on arrays
- Save predictions to text files (`np.savetxt`)
- Convert pandas DataFrames to arrays

**In this project**: Array operations and saving output files

---

#### Import: `import pandas as pd`

**What it does**: Pandas = data manipulation and analysis library

**Why we need it**:
- Read CSV files easily (`pd.read_csv`)
- DataFrame structure (like Excel tables)
- Column selection and slicing
- Handle missing data

**In this project**: Load and manipulate KDD CSV datasets

---

#### Import: `from joblib import dump`

**What it does**: Joblib saves Python objects to disk

**Why we need it**:
- Save trained models for later use
- More efficient than pickle for numpy arrays
- Preserves exact model state
- Essential for deployment

**In this project**: Save trained model and scaler so `realtime_ids.py` can load them

---

#### Import: `import os`

**What it does**: Operating system interface

**Why we need it**:
- Cross-platform file paths
- Get script directory location
- Create directories
- Join paths correctly (Windows/Linux/Mac)

**In this project**: Build reliable paths and create `classical/` directory

---

#### Import: `from sklearn.linear_model import LogisticRegression`

**What it does**: Imports the Logistic Regression classifier

**What is Logistic Regression?**

- **Type**: Linear model for binary classification (2 classes)
- **Despite the name**: It's CLASSIFICATION, not regression
- **How it works**: Uses sigmoid function to map inputs to probabilities (0 to 1)
- **Decision**: Predicts class 1 if probability ≥ 0.5, else class 0

**Mathematical formula**:
```
P(Attack|X) = 1 / (1 + e^(-z))
where z = w₀ + w₁x₁ + w₂x₂ + ... + w₄₁x₄₁
```

- Learns weights (w) that best separate Normal from Attack
- Uses optimization (LBFGS or gradient descent)

**In this project**: The core algorithm that learns to detect attacks

---

#### Import: `from sklearn.preprocessing import Normalizer`

**What it does**: Normalizes data for consistent scale

**What is Normalization?**

- Scales each **sample (row)** to have unit norm
- Unit norm = vector length = 1
- Formula: ||x|| = √(x₁² + x₂² + ... + xₙ²)
- Divides each feature by the sample's norm

**Why normalize?**

- Features have different scales (packets vs. bytes)
- Large values can dominate the model
- Algorithms perform better with normalized data
- Fair contribution from all features

**Normalizer vs. StandardScaler**:
- **Normalizer**: Scales each row (used here)
- **StandardScaler**: Scales each column (different method)

**In this project**: Ensures all samples are on the same scale

---

#### Import: Evaluation Metrics

```python
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                            accuracy_score, classification_report)
```

**What each metric measures**:

1. **Accuracy**: (TP + TN) / Total
   - Overall correctness
   - Misleading with imbalanced data

2. **Precision**: TP / (TP + FP)
   - "Of predicted attacks, how many were real?"
   - Minimizes false alarms

3. **Recall**: TP / (TP + FN)
   - "Of real attacks, how many were detected?"
   - Critical for security

4. **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
   - Balance of precision and recall
   - Harmonic mean

5. **Classification Report**: Detailed breakdown per class

**Terms**:
- TP = True Positive (correctly detected attack)
- TN = True Negative (correctly identified normal)
- FP = False Positive (false alarm)
- FN = False Negative (missed attack)

---

### STEP 1: PATH SETUP AND DATA LOADING

```python
base_dir = os.path.dirname(os.path.abspath(__file__))
```

**Breaking it down**:

- `__file__`: Special variable = script's file path
- `os.path.abspath(__file__)`: Convert to absolute path
  - Example: `C:\Users\Web developer\Documents\PROJECTS\IDS_Project\all.py`
- `os.path.dirname(...)`: Extract directory (remove filename)
  - Result: `C:\Users\Web developer\Documents\PROJECTS\IDS_Project`
- `base_dir`: Store project root directory

**Why this is critical**:

- Scripts might run from different directories
- Relative paths like `'kddtrain.csv'` fail if not in right folder
- This ensures paths ALWAYS work

**Example**:
- Script at: `C:\Projects\IDS_Project\all.py`
- base_dir = `C:\Projects\IDS_Project`
- Data file = `C:\Projects\IDS_Project\kddtrain.csv`

---

```python
print("Loading datasets...")
traindata = pd.read_csv(os.path.join(base_dir, 'kddtrain.csv'), header=None)
testdata = pd.read_csv(os.path.join(base_dir, 'kddtest.csv'), header=None)
```

**What happens**:

1. `os.path.join(base_dir, 'kddtrain.csv')`: Creates full path
2. `pd.read_csv(...)`: Reads CSV into DataFrame
3. `header=None`: No column names (CSV has no header row)
4. Result: Columns numbered 0, 1, 2, ..., 41

**About KDD Dataset**:

- **Column 0**: Label (0 = Normal, 1 = Attack)
- **Columns 1-41**: 41 network traffic features
- **Features**: duration, protocol, service, byte counts, flags, etc.
- **KDD'99**: Standard benchmark for IDS research

---

```python
print(f"Training data shape: {traindata.shape}")
print(f"Test data shape: {testdata.shape}")
```

**What `.shape` shows**:

- Returns: `(rows, columns)`
- Example: `(125973, 42)` = 125,973 samples × 42 columns
- Verifies data loaded correctly

---

### STEP 2: DATA SAMPLING

```python
train_frac = int(0.05 * len(traindata))
test_frac = int(0.05 * len(testdata))
traindata = traindata.iloc[:train_frac]
testdata = testdata.iloc[:test_frac]
```

**Why only 5%?**

1. **Memory**: Full dataset can be 100k+ samples
2. **Speed**: Training on full data takes much longer
3. **Assignment**: 5% sufficient for demonstration
4. **Resources**: Works on lower-spec machines

**How it works**:

- `len(traindata)`: Total rows (e.g., 125,973)
- `0.05 * 125973`: 5% = 6,298.65
- `int(...)`: Round to integer = 6,298
- `.iloc[:6298]`: Select first 6,298 rows

**Adjustable**: Change `0.05` to `0.10` for 10%, etc.

---

```python
print(f"Using 5% of data - Training samples: {train_frac}, Test samples: {test_frac}")
```

**User feedback**: Shows exact sample count

---

### STEP 3: FEATURE AND LABEL EXTRACTION

```python
X = traindata.iloc[:, 1:42]  # Training features
Y = traindata.iloc[:, 0]      # Training labels
T = testdata.iloc[:, 1:42]    # Test features
C = testdata.iloc[:, 0]       # Test labels
```

**Understanding pandas `.iloc`**:

- `.iloc[rows, columns]`: Select by position
- `[:, 1:42]`: All rows, columns 1-41 (42 exclusive)
- `[:, 0]`: All rows, column 0 only

**Variable names**:

- `X`: Feature matrix (standard ML notation)
- `Y`: Labels/targets (standard ML notation)
- `T`: Test features
- `C`: Test labels (Correct/Class)

**Why separate?**

- Training: Model learns from X → Y
- Testing: Model predicts Y from X, we compare to C

---

```python
print(f"Feature matrix shape: {X.shape}")
print(f"Label distribution in training: {Y.value_counts().to_dict()}")
```

**Example output**:

- Shape: `(6298, 41)` = 6,298 samples × 41 features
- Distribution: `{0: 3145, 1: 3153}` = balanced classes

**Why check distribution?**

- Imbalanced data (e.g., 95% Normal) biases model
- Balanced classes = better learning
- Helps understand data

---

### STEP 4: FEATURE NORMALIZATION

```python
scaler = Normalizer().fit(X)
```

**What `.fit()` does**:

- Analyzes training data `X`
- Learns transformation parameters
- **Does NOT transform yet**
- **Critical**: Only fit on training data!

**Data Leakage Warning**:

- NEVER fit on test data!
- Test = "future" unknown data
- Fitting on test = unfair advantage
- Rule: **fit on train, transform on both**

---

```python
trainX = scaler.transform(X)
testT = scaler.transform(T)
```

**What `.transform()` does**:

For each sample (row):
1. Calculate L2 norm: √(sum of squares)
2. Divide each feature by this norm
3. Result: normalized vector (length = 1)

**Example**:

- Original: `[10, 20, 30]`
- Norm: √(10² + 20² + 30²) = √1400 ≈ 37.42
- Normalized: `[0.267, 0.534, 0.802]`

---

```python
traindata = np.array(trainX)
trainlabel = np.array(Y)
testdata = np.array(testT)
testlabel = np.array(C)
```

**Why NumPy arrays?**

- More memory-efficient
- Faster numerical operations
- scikit-learn compatibility

---

### STEP 5: MODEL TRAINING

```python
model = LogisticRegression(max_iter=1000, random_state=42)
```

**Parameters explained**:

1. **`max_iter=1000`**:
   - Max optimization iterations
   - Default = 100 (often too low)
   - Increase if convergence fails
   - Higher = more computation

2. **`random_state=42`**:
   - Random seed for reproducibility
   - Same results every run
   - 42 = arbitrary (ML tradition)
   - Remove for different results

**Other defaults** (not specified):

- `solver='lbfgs'`: Optimization algorithm
- `C=1.0`: Regularization strength
- `penalty='l2'`: Regularization type

---

```python
model.fit(traindata, trainlabel)
```

**What training does** (simplified):

1. **Initialize**: Start with random weights
2. **Forward pass**: For each sample:
   - Calculate: z = w₀ + w₁x₁ + ... + w₄₁x₄₁
   - Apply sigmoid: p = 1 / (1 + e^(-z))
   - Get probability of Attack
3. **Loss**: Measure error (log loss)
4. **Backprop**: Adjust weights to reduce error
5. **Repeat**: Until convergence or max_iter
6. **Result**: Optimal weights

**What model learns**:

- 41 weights (one per feature) + 1 bias
- Positive weight = feature increases Attack probability
- Negative weight = feature decreases Attack probability

---

### STEP 6: SAVE MODEL ARTIFACTS

```python
os.makedirs('classical', exist_ok=True)
```

**What this does**:

- Creates `classical/` directory
- `exist_ok=True`: No error if exists
- Like `mkdir classical` in terminal

**Why?**

- Organize output files
- Separate models from code
- ML project convention

---

```python
dump(model, 'classical/logistic_model.joblib')
dump(scaler, 'classical/normalizer.joblib')
```

**What gets saved**:

1. **logistic_model.joblib**:
   - Complete trained model
   - All learned weights
   - Can predict without retraining

2. **normalizer.joblib**:
   - Fitted scaler
   - Knows how to transform new data
   - **CRITICAL** for correct predictions

**Why save both?**

- New data needs same normalization
- Without scaler = wrong predictions
- Scaler is part of the pipeline

**File format**:

- `.joblib` = compressed binary
- Smaller/faster than pickle
- Only readable by joblib

---

### STEP 7: MODEL EVALUATION

```python
predicted = model.predict(testdata)
proba = model.predict_proba(testdata)
```

**`.predict()`**:

- Input: Normalized features
- Output: Predicted classes (0 or 1)
- Threshold: 0.5 (probability ≥ 0.5 → Attack)
- Example: `[0, 1, 0, 0, 1, ...]`

**`.predict_proba()`**:

- Input: Normalized features
- Output: Probabilities (shape: n_samples × 2)
- Column 0: P(Normal)
- Column 1: P(Attack)
- Example row: `[0.73, 0.27]` = 73% Normal

**Why both?**

- `predict()`: Final decision
- `predict_proba()`: Confidence/analysis
- Can adjust threshold later

---

```python
np.savetxt('classical/expected.txt', testlabel, fmt='%01d')
np.savetxt('classical/predictedlabelLR.txt', predicted, fmt='%01d')
np.savetxt('classical/predictedprobaLR.txt', proba)
```

**Saved files**:

- **expected.txt**: Ground truth
- **predictedlabelLR.txt**: Model predictions
- **predictedprobaLR.txt**: Probability scores

**Why save?**

- Manual inspection
- External analysis
- Visualization
- Debugging

---

### STEP 8: METRICS

```python
accuracy = accuracy_score(testlabel, predicted)
precision = precision_score(testlabel, predicted, average="binary")
recall = recall_score(testlabel, predicted, average="binary")
f1 = f1_score(testlabel, predicted, average="binary")
```

**`average="binary"`**:

- For 2-class problems
- Calculates for positive class (Attack = 1)

**Interpreting**:

- Accuracy = 0.95 → 95% correct
- Precision = 0.92 → 92% of predicted attacks are real
- Recall = 0.88 → 88% of real attacks detected
- F1 = 0.90 → Balanced score

**Good scores for IDS**:

- Recall > 0.85 (catch most attacks)
- Precision > 0.80 (minimize false alarms)
- F1 > 0.85 (production ready)

---

```python
print(classification_report(testlabel, predicted, target_names=['Normal', 'Attack']))
```

**Sample output**:

```
              precision    recall  f1-score   support

      Normal       0.96      0.97      0.97      3145
      Attack       0.92      0.88      0.90      3153

    accuracy                           0.93      6298
   macro avg       0.94      0.93      0.93      6298
weighted avg       0.94      0.93      0.93      6298
```

**Understanding**:

- **support**: Sample count per class
- **macro avg**: Simple average
- **weighted avg**: Weighted by support

---

## Complete Breakdown of realtime_ids.py

### Purpose

Simulates real-time IDS:
1. Load trained model + scaler
2. Stream traffic samples one-by-one
3. Classify in real-time
4. Print alerts

---

### IMPORTS

```python
import time
```

**Used for**:
- `time.sleep()`: Pause between samples
- `time.strftime()`: Format timestamps

---

```python
import numpy as np
```

**Used for**: `.reshape()` for predictions

---

```python
from joblib import load
```

**Critical**: Loads saved model (counterpart to `dump`)

---

```python
import os
```

**Same as all.py**: File path handling

---

### LOAD ARTIFACTS

```python
base_dir = os.path.dirname(os.path.abspath(__file__))
```

**Same logic**: Get script directory

---

```python
model = load(os.path.join(base_dir, 'classical', 'logistic_model.joblib'))
scaler = load(os.path.join(base_dir, 'classical', 'normalizer.joblib'))
```

**What happens**:

1. Find .joblib files in classical/
2. Deserialize model object
3. Deserialize scaler object
4. Ready to use!

**Error if missing**: Run `all.py` first!

---

### LOAD TEST DATA

```python
import pandas as pd
testdata = pd.read_csv(os.path.join(base_dir, 'kddtest.csv'), header=None)
```

**Why here?**

- Simulates traffic stream
- Production = live packets
- Demo = saved data

---

```python
test_frac = int(0.05 * len(testdata))
testdata = testdata.iloc[:test_frac]
T = testdata.iloc[:,1:42]
C = testdata.iloc[:,0]
```

**Same as all.py**:

- 5% sampling
- Extract features (columns 1-41)
- Extract labels (column 0)

---

### NORMALIZE

```python
T_scaled = scaler.transform(T)
```

**CRITICAL**:

- Uses SAME scaler from training
- Same transformation
- Without this = garbage predictions

**Why not fit?**

- Fitting = new parameters from test data
- Must use training parameters
- Ensures consistency

---

### REAL-TIME LOOP

```python
print('--- Real-Time IDS Simulation (5% of test data) ---')
```

**Feedback**: Simulation starting

---

```python
for i, (sample, label) in enumerate(zip(T_scaled, C)):
```

**What this does**:

1. `zip(T_scaled, C)`: Pair sample with label
2. `enumerate()`: Add index (0, 1, 2, ...)
3. Loop one-by-one

**Why?**

- Simulates streaming
- Production = one at a time from network
- Demonstrates online prediction

---

```python
pred = model.predict(sample.reshape(1, -1))[0]
```

**Breakdown**:

1. `sample`: 41 features (shape: `(41,)`)
2. `reshape(1, -1)`: Convert to `(1, 41)` (2D)
   - Why? scikit-learn expects 2D
   - `-1` = auto-calculate
3. `model.predict()`: Returns array `[0]` or `[1]`
4. `[0]`: Extract scalar value

---

```python
out = 'Normal' if pred == 0 else 'Attack'
```

**Ternary operator**:

- pred == 0 → "Normal"
- pred == 1 → "Attack"

---

```python
print(f'[{time.strftime('%Y-%m-%d %H:%M:%S')}] Sample {i+1}: {out}')
```

**Output example**:

```
[2025-12-03 14:23:45] Sample 1: Normal
[2025-12-03 14:23:46] Sample 2: Attack
```

**Format**:

- Timestamp: Year-Month-Day Hour:Minute:Second
- Sample number: i+1 (starts from 1)
- Result: Normal or Attack

---

```python
time.sleep(0.5)
```

**Why pause?**

- Simulates real-time (500ms between samples)
- Without = instant processing
- Adjust for speed (0.1 = faster, 1.0 = slower)

**Production**: No sleep, process as fast as possible

---

## How Both Scripts Work Together

### Workflow

```
PHASE 1: TRAINING (all.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Load kddtrain.csv & kddtest.csv
2. Sample 5% of data
3. Extract features (columns 1-41)
4. Fit Normalizer on training features
5. Transform both train & test features
6. Train Logistic Regression model
7. Evaluate on test set
8. Save:
   ├─ classical/logistic_model.joblib
   └─ classical/normalizer.joblib
   
        ↓ SAVED FILES ↓
        
PHASE 2: INFERENCE (realtime_ids.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Load saved model
2. Load saved scaler  
3. Load test data (simulated stream)
4. Transform each sample with scaler
5. Predict with model
6. Display: "Normal" or "Attack"
```

### Critical Dependencies

1. **Execution Order**:
   - MUST run all.py FIRST
   - THEN run realtime_ids.py

2. **Same Normalization**:
   - Training: fit + transform
   - Inference: transform only (with saved scaler)

3. **File Structure**:
   - Both expect 41 features (columns 1-41)
   - Column 0 = label

---

## Machine Learning Concepts

### Supervised Learning

- Learn from labeled examples (X → Y)
- Training data = features + correct answers
- Goal = predict labels for new data

### Binary Classification

- Two classes: 0 or 1 (Normal or Attack)
- Decision boundary separates classes
- Discrete output (not continuous)

### Training vs. Inference

- **Training**: Learn patterns (all.py)
- **Inference**: Apply patterns (realtime_ids.py)

### Overfitting vs. Underfitting

- **Overfitting**: Memorizes training, fails on new data
- **Underfitting**: Too simple, misses patterns
- **Solution**: Regularization (parameter C)

### Train/Test Split

- Training: Learn patterns
- Testing: Evaluate (unseen data)
- Simulates real-world performance

---

## File Outputs and Usage

### Generated Files

| File | Contents | Format |
|------|----------|--------|
| `classical/logistic_model.joblib` | Trained model | Binary |
| `classical/normalizer.joblib` | Fitted scaler | Binary |
| `classical/expected.txt` | True labels | Text (0/1) |
| `classical/predictedlabelLR.txt` | Predictions | Text (0/1) |
| `classical/predictedprobaLR.txt` | Probabilities | Text (float) |

### Analysis Examples

**Compare predictions**:

```python
import numpy as np
expected = np.loadtxt('classical/expected.txt')
predicted = np.loadtxt('classical/predictedlabelLR.txt')
errors = expected != predicted
print(f"Errors: {errors.sum()}")
```

**Analyze confidence**:

```python
proba = np.loadtxt('classical/predictedprobaLR.txt')
attack_prob = proba[:, 1]
uncertain = (attack_prob > 0.4) & (attack_prob < 0.6)
print(f"Uncertain: {uncertain.sum()}")
```

---

## Troubleshooting Guide

### Common Errors

**FileNotFoundError: kddtrain.csv**

- **Cause**: CSV not in right location
- **Fix**: Put CSV files with scripts

**FileNotFoundError: logistic_model.joblib**

- **Cause**: Running realtime_ids.py before all.py
- **Fix**: Run all.py first!

**Import "numpy" could not be resolved**

- **Cause**: Wrong Python interpreter in VS Code
- **Fix**: Ctrl+Shift+P → Select Interpreter → Choose Python 3.13

**ConvergenceWarning**

- **Cause**: max_iter too low
- **Fix**: Increase to 2000

**Poor Performance (accuracy < 0.8)**

- **Causes**:
  - Wrong normalization
  - Imbalanced classes
  - Need more data
- **Fixes**:
  - Verify scaler usage
  - Use class_weight='balanced'
  - Increase from 5% to 10%

**Simulation too slow/fast**

- **Fix**: Adjust `time.sleep()` value

---

## Summary

### What You Learned

1. Train Logistic Regression for IDS
2. Save/load models with joblib
3. Normalize data consistently
4. Evaluate with proper metrics
5. Build real-time prediction system

### Key Takeaways

- ✓ Always use training scaler for test data
- ✓ Run all.py before realtime_ids.py
- ✓ Logistic Regression perfect for binary classification
- ✓ Real-time IDS needs fast models
- ✓ Multiple metrics > just accuracy

### Next Steps

1. Run all.py and review metrics
2. Run realtime_ids.py for real-time demo
3. Experiment with sampling (try 10%)
4. Tune hyperparameters (try C=0.1, C=10)
5. Analyze saved prediction files

---

**End of Ultra-Detailed Explanation**

For questions or issues, refer to the [README.md](./README.md) file.

