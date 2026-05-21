# ==================================================================================
#           INDIAN SIGN LANGUAGE (ISL) RECOGNITION - TRAINING NOTEBOOK
# ==================================================================================
# This notebook is designed to run on Google Colab to train a BiLSTM model
# on hand landmark sequences.
# ==================================================================================

# ─── 1. MOUNT GOOGLE DRIVE ────────────────────────────────────────────────────────
# This step gives Colab access to your 'isl_dataset.zip' file.
from google.colab import drive
drive.mount('/content/drive')

# ─── 2. EXTRACT DATASET ───────────────────────────────────────────────────────────
# We unzip the data into the local Colab environment for super-fast reading.
import os
import zipfile

ZIP_PATH = '/content/drive/MyDrive/isl_dataset.zip' # Make sure it is in MyDrive!
EXTRACT_PATH = '/content/dataset'

if not os.path.exists(EXTRACT_PATH):
    print(f"[INFO] Unzipping {ZIP_PATH}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)
    print("[SUCCESS] Dataset extracted.")
else:
    print("[INFO] Dataset already extracted.")

# ─── 3. IMPORTS ───────────────────────────────────────────────────────────────────
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json

# ─── 4. LOAD & PREPROCESS DATA ────────────────────────────────────────────────────
RAW_DATA_PATH = os.path.join(EXTRACT_PATH, 'custom_isl_data')
DATA_DIR = RAW_DATA_PATH if os.path.exists(RAW_DATA_PATH) else EXTRACT_PATH

sequences, labels = [], []
label_map = {}

# We create an alphabetical list of folders to ensure labels are consistent
words = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

print(f"\n[INTERMEDIATE STEP] Found {len(words)} classes.")
print("Creating Label Mapping...")

for idx, word in enumerate(words):
    label_map[word] = idx
    word_path = os.path.join(DATA_DIR, word)
    
    # Load all .npy files in this folder
    for seq_file in os.listdir(word_path):
        if seq_file.endswith('.npy'):
            res = np.load(os.path.join(word_path, seq_file))
            sequences.append(res)
            labels.append(idx)

X = np.array(sequences)
y = tf.keras.utils.to_categorical(labels).astype(int)

print(f"Total Sequences: {X.shape[0]}")
print(f"Sequence Shape : {X.shape[1:]} (Frames, Landmarks)")
print(f"Total Classes  : {len(words)}")

# Save label map back to Drive so we don't lose it
with open('/content/drive/MyDrive/label_map.json', 'w') as f:
    json.dump(label_map, f)
print("[SUCCESS] Saved label_map.json to Google Drive.")

# ─── 5. TRAIN/TEST SPLIT ──────────────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
print(f"Training Set  : {X_train.shape[0]} samples")
print(f"Validation Set: {X_val.shape[0]} samples")

# ─── 6. BUILD BiLSTM ARCHITECTURE ─────────────────────────────────────────────────
model = Sequential([
    Input(shape=(30, 126)),
    Bidirectional(LSTM(64, return_sequences=True, activation='relu')),
    Dropout(0.2),
    Bidirectional(LSTM(128, return_sequences=True, activation='relu')),
    Dropout(0.2),
    Bidirectional(LSTM(64, return_sequences=False, activation='relu')),
    Dense(64, activation='relu'),
    Dense(len(words), activation='softmax')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("\n[INTERMEDIATE STEP] Model Architecture Summary:")
model.summary()

# ─── 7. TRAINING PHASE ────────────────────────────────────────────────────────────
checkpoint = ModelCheckpoint('/content/drive/MyDrive/isl_model.h5', monitor='val_categorical_accuracy', save_best_only=True, mode='max')
stop_early = EarlyStopping(monitor='val_loss', patience=15)

print("\nStarting Training... (Intermediate steps will show below)")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, stop_early]
)

# ─── 8. VISUALIZATION (Show your Sir!) ───────────────────────────────────────────
plt.figure(figsize=(12, 4))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['categorical_accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_categorical_accuracy'], label='Val Accuracy', color='orange')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Val Loss', color='orange')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/training_metrics.png')
plt.show()

print("\n[FINAL SUCCESS] Model and Graphs saved to your Google Drive!")
