# ============================================================================
# Real-Time IDS Training Script - Logistic Regression Implementation
# ============================================================================
# This script trains a Logistic Regression model for intrusion detection
# and saves the trained model and normalizer for real-time use.
#
# Assignment: CYB 213 - Project 12: Real-Time IDS Simulation
# Algorithm: Logistic Regression (as per assignment requirements)
# ============================================================================

import numpy as np
import pandas as pd
from joblib import dump
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                            accuracy_score, classification_report)

# ============================================================================
# STEP 1: Path Setup and Data Loading
# ============================================================================
# Get the absolute path to the script's directory to ensure paths work
# regardless of where the script is run from
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load the KDD dataset files
# header=None means the CSV has no column names in the first row
print("Loading datasets...")
traindata = pd.read_csv(os.path.join(base_dir, 'kddtrain.csv'), header=None)
testdata = pd.read_csv(os.path.join(base_dir, 'kddtest.csv'), header=None)
print(f"Training data shape: {traindata.shape}")
print(f"Test data shape: {testdata.shape}")

# ============================================================================
# STEP 2: Data Sampling (5% for system resource constraints)
# ============================================================================
# Use only 5% of the data for training/testing due to system specs
# This reduces memory usage and computation time while still allowing
# meaningful model training
train_frac = int(0.05 * len(traindata))
test_frac = int(0.05 * len(testdata))
traindata = traindata.iloc[:train_frac]
testdata = testdata.iloc[:test_frac]
print(f"Using 5% of data - Training samples: {train_frac}, Test samples: {test_frac}")

# ============================================================================
# STEP 3: Feature and Label Extraction
# ============================================================================
# Column 0 contains the label (0 = Normal, 1 = Attack)
# Columns 1-41 contain the 41 network traffic features
X = traindata.iloc[:, 1:42]  # Training features
Y = traindata.iloc[:, 0]      # Training labels
T = testdata.iloc[:, 1:42]    # Test features
C = testdata.iloc[:, 0]       # Test labels (ground truth)

print(f"Feature matrix shape: {X.shape}")
print(f"Label distribution in training: {Y.value_counts().to_dict()}")

# ============================================================================
# STEP 4: Feature Normalization
# ============================================================================
# Normalizer() scales each sample (row) to have unit norm (L2 norm = 1)
# This ensures that no single feature dominates due to scale differences
# Important: We fit on training data only to avoid data leakage
print("Normalizing features...")
scaler = Normalizer().fit(X)
trainX = scaler.transform(X)
testT = scaler.transform(T)

# Convert to numpy arrays for compatibility with scikit-learn
traindata = np.array(trainX)
trainlabel = np.array(Y)
testdata = np.array(testT)
testlabel = np.array(C)

print("Normalization complete.")

# ============================================================================
# STEP 5: Model Training - Logistic Regression
# ============================================================================
print("\n" + "="*80)
print("TRAINING LOGISTIC REGRESSION MODEL")
print("="*80)

# Initialize Logistic Regression with default parameters
# Default solver='lbfgs' is efficient for small-to-medium datasets
# max_iter=100 by default, but can be increased if convergence fails
model = LogisticRegression(max_iter=1000, random_state=42)

# Train the model on normalized training data
print("Fitting model...")
model.fit(traindata, trainlabel)
print("Model training complete!")

# ============================================================================
# STEP 6: Save Model and Scaler for Real-Time Use
# ============================================================================
# Create the classical directory if it doesn't exist
os.makedirs('classical', exist_ok=True)

# Save the trained model and scaler using joblib
# These will be loaded by realtime_ids.py for live predictions
print("\nSaving model and scaler...")
dump(model, 'classical/logistic_model.joblib')
dump(scaler, 'classical/normalizer.joblib')
print("✓ Saved: classical/logistic_model.joblib")
print("✓ Saved: classical/normalizer.joblib")

# ============================================================================
# STEP 7: Model Evaluation on Test Data
# ============================================================================
print("\n" + "="*80)
print("EVALUATING MODEL PERFORMANCE")
print("="*80)

# Make predictions on the test set
predicted = model.predict(testdata)
proba = model.predict_proba(testdata)

# Save predictions for later analysis
np.savetxt('classical/expected.txt', testlabel, fmt='%01d')
np.savetxt('classical/predictedlabelLR.txt', predicted, fmt='%01d')
np.savetxt('classical/predictedprobaLR.txt', proba)
print("✓ Saved: classical/expected.txt")
print("✓ Saved: classical/predictedlabelLR.txt")
print("✓ Saved: classical/predictedprobaLR.txt")

# ============================================================================
# STEP 8: Calculate and Display Metrics
# ============================================================================
# Calculate performance metrics
accuracy = accuracy_score(testlabel, predicted)
precision = precision_score(testlabel, predicted, average="binary")
recall = recall_score(testlabel, predicted, average="binary")
f1 = f1_score(testlabel, predicted, average="binary")

# Display results
print("\n" + "-"*80)
print("PERFORMANCE METRICS")
print("-"*80)
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-Score:  {f1:.3f}")
print("-"*80)

# Display detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(testlabel, predicted, target_names=['Normal', 'Attack']))

# ============================================================================
# Training Complete
# ============================================================================
print("\n" + "="*80)
print("TRAINING COMPLETED SUCCESSFULLY")
print("="*80)
print("\nNext steps:")
print("1. Review the metrics above to assess model performance")
print("2. Run 'python realtime_ids.py' to start the real-time IDS simulation")
print("="*80)



