"""
DIAGNOSTIC: Tests if the model can correctly predict its own training data.
If this fails, the model file is wrong.
If this passes, the live inference pipeline has a mismatch.
"""
import os, sys, json, zipfile
import numpy as np

# Compatibility patch
try:
    import google.protobuf.runtime_version as rv
except (ImportError, AttributeError):
    from unittest.mock import MagicMock
    proto_mod = sys.modules.get("google.protobuf")
    if proto_mod:
        proto_mod.runtime_version = MagicMock()

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input

# Load label map
with open('models/label_map.json', 'r') as f:
    label_map = json.load(f)
actions = sorted(label_map.keys(), key=lambda x: label_map[x])
num_classes = len(actions)

# Reconstruct model
model = Sequential([
    Input(shape=(30, 126)),
    Bidirectional(LSTM(64, return_sequences=True, activation='tanh')),
    Dropout(0.2),
    Bidirectional(LSTM(128, return_sequences=True, activation='tanh')),
    Dropout(0.2),
    Bidirectional(LSTM(64, return_sequences=False, activation='tanh')),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])
model.load_weights('models/isl_model.h5')
print(f"[OK] Model loaded with {num_classes} classes.\n")

# Extract a few samples from the normalized dataset
ZIP_PATH = 'isl_dataset_normalized.zip'
EXTRACT_DIR = 'temp_diag'

if os.path.exists(ZIP_PATH):
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(EXTRACT_DIR)
    data_dir = os.path.join(EXTRACT_DIR, 'custom_isl_data')
    if not os.path.exists(data_dir):
        data_dir = EXTRACT_DIR
else:
    print("ERROR: Cannot find isl_dataset_normalized.zip")
    sys.exit(1)

# Test a sample from each class
print(f"{'SIGN':<15} {'PREDICTED':<15} {'CONF':>8} {'RESULT':>8}")
print("-" * 50)

correct = 0
total = 0
wrong_list = []

folders = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

for word in folders:
    word_dir = os.path.join(data_dir, word)
    npy_files = [f for f in os.listdir(word_dir) if f.endswith('.npy')]
    
    if not npy_files:
        continue
    
    # Load the first sequence
    seq = np.load(os.path.join(word_dir, npy_files[0]))
    
    # Predict
    res = model.predict(np.expand_dims(seq, axis=0), verbose=0)[0]
    pred_idx = np.argmax(res)
    pred_word = actions[pred_idx]
    conf = res[pred_idx] * 100
    
    match = "PASS" if pred_word == word else "FAIL"
    if pred_word == word:
        correct += 1
    else:
        wrong_list.append((word, pred_word, conf))
    total += 1
    
    print(f"{word:<15} {pred_word:<15} {conf:>7.1f}% {match:>8}")

print("-" * 50)
print(f"\nRESULT: {correct}/{total} correct ({correct/total*100:.1f}%)")

if wrong_list:
    print(f"\nFAILED SIGNS:")
    for w, p, c in wrong_list:
        print(f"  {w} -> predicted as {p} ({c:.1f}%)")

# Cleanup
import shutil
try:
    shutil.rmtree(EXTRACT_DIR, ignore_errors=True)
except:
    pass

if correct == total:
    print("\n>>> MODEL IS PERFECT ON TRAINING DATA.")
    print(">>> The problem is in the LIVE CAMERA pipeline (landmark extraction mismatch).")
else:
    print("\n>>> MODEL ITSELF HAS ISSUES.")
    print(">>> The training data or model weights are corrupted.")
