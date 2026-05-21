"""
AI DATA MULTIPLIER & NORMALIZER
================================
Extracts your intact 106-word dataset directly from the zip file you trained on!
 mathematically multiplies your 5 recorded sequences into 40 highly robust 
sequences per word (4,240 sequences total!).
"""
import os
import numpy as np
import shutil
import zipfile

ZIP_FILE = 'isl_dataset_normalized.zip'
if not os.path.exists(ZIP_FILE):
    print("ERROR: Could not find isl_dataset_normalized.zip")
    exit(1)

print(f"[INFO] Extracting 106 words from {ZIP_FILE}...")
EXTRACT_DIR = "temp_multiplier"
shutil.rmtree(EXTRACT_DIR, ignore_errors=True)
with zipfile.ZipFile(ZIP_FILE, 'r') as z:
    z.extractall(EXTRACT_DIR)

# The dataset structure is sometimes nested inside 'custom_isl_data'
DATA_DIR = os.path.join(EXTRACT_DIR, 'custom_isl_data')
if not os.path.exists(DATA_DIR):
    DATA_DIR = EXTRACT_DIR

OUTPUT_DIR = "mega_dataset"
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── NORMALIZATION LOGIC (Idempotent - safe to run on normalized data) ──
def normalize_landmarks(raw_landmarks):
    normalized = np.zeros(126)
    
    # Left Hand (Indices 0-62)
    left_h = raw_landmarks[:63]
    if np.any(left_h):
        wrist_x, wrist_y, wrist_z = left_h[0], left_h[1], left_h[2]
        for i in range(0, 63, 3):
            normalized[i]   = left_h[i] - wrist_x
            normalized[i+1] = left_h[i+1] - wrist_y
            normalized[i+2] = left_h[i+2] - wrist_z
        max_val = np.max(np.abs(normalized[:63]))
        if max_val > 0:
            normalized[:63] = normalized[:63] / max_val
            
    # Right Hand (Indices 63-125)
    right_h = raw_landmarks[63:]
    if np.any(right_h):
        wrist_x, wrist_y, wrist_z = right_h[0], right_h[1], right_h[2]
        for i in range(0, 63, 3):
            normalized[63+i]   = right_h[i] - wrist_x
            normalized[63+i+1] = right_h[i+1] - wrist_y
            normalized[63+i+2] = right_h[i+2] - wrist_z
        max_val = np.max(np.abs(normalized[63:]))
        if max_val > 0:
            normalized[63:] = normalized[63:] / max_val
            
    return normalized

# ── DATA AUGMENTATION LOGIC ──
def augment_sequence(seq, scale, shift_x, shift_y, noise_amount=0.015):
    """Applies scaling, shifting, and noise to raw coordinates before normalization"""
    aug_seq = np.zeros_like(seq)
    for frame_idx in range(len(seq)):
        frame = seq[frame_idx].copy()
        
        # We only shift the X and Y coordinates (not Z)
        for i in range(0, 126, 3):
            if frame[i] != 0 or frame[i+1] != 0: 
                frame[i]   = frame[i] + shift_x + np.random.normal(0, noise_amount)
                frame[i+1] = frame[i+1] + shift_y + np.random.normal(0, noise_amount)
                frame[i+2] = frame[i+2] + np.random.normal(0, noise_amount)
        aug_seq[frame_idx] = frame
    return aug_seq

# ── MULTIPLIER PROCESSOR ──
words = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
print(f"[INFO] Preparing to hyper-augment {len(words)} classes...\n")

total_sequences = 0

for word_idx, word in enumerate(words):
    word_dir = os.path.join(DATA_DIR, word)
    out_word_dir = os.path.join(OUTPUT_DIR, "custom_isl_data", word)
    os.makedirs(out_word_dir, exist_ok=True)
    
    npy_files = [f for f in os.listdir(word_dir) if f.endswith('.npy')]
    file_counter = 0
    
    if not npy_files:
        continue
        
    for npy_idx, npy_file in enumerate(npy_files):
        # 1. Load sequence
        filepath = os.path.join(word_dir, npy_file)
        original_seq = np.load(filepath)
        
        # 2. Add original sequence
        norm_orig = np.zeros_like(original_seq)
        for f in range(len(original_seq)): norm_orig[f] = normalize_landmarks(original_seq[f])
        np.save(os.path.join(out_word_dir, f"{file_counter}.npy"), norm_orig)
        file_counter += 1
        total_sequences += 1
        
        # 3. Create 7 Advanced Augmentations for EVERY original sequence
        augmentations = [
            (1.0,  0.05,  0.00, 0.02), # Shift right
            (1.0, -0.05,  0.00, 0.02), # Shift left
            (1.0,  0.00,  0.05, 0.02), # Shift down
            (1.0,  0.00, -0.05, 0.02), # Shift up
            (1.0,  0.03,  0.03, 0.03), # Shift diagonal
            (1.0, -0.03, -0.03, 0.03), # Shift diagonal 2
            (1.0,  0.00,  0.00, 0.04), # Heavy noise
        ]
        
        for aug in augmentations:
            scale, sx, sy, noise = aug
            aug_raw = augment_sequence(original_seq, scale, sx, sy, noise)
            
            aug_norm = np.zeros_like(aug_raw)
            for f in range(len(aug_raw)): aug_norm[f] = normalize_landmarks(aug_raw[f])
            
            np.save(os.path.join(out_word_dir, f"{file_counter}.npy"), aug_norm)
            file_counter += 1
            total_sequences += 1

    progress = ((word_idx + 1) / len(words)) * 100
    print(f"[{progress:.1f}%] Augmented '{word}' -> {file_counter} sequences")

print(f"\n[SUCCESS] Multiplied {len(words)} classes into {total_sequences} completely unique sequences!\n")

print("[INFO] Packaging mega-dataset for Colab...")
ZIP_NAME = 'isl_mega_dataset.zip'
shutil.make_archive(ZIP_NAME.replace('.zip', ''), 'zip', OUTPUT_DIR)

# Cleanup
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
shutil.rmtree(EXTRACT_DIR, ignore_errors=True)

print(f"\n[DONE] Saved precisely {total_sequences} sequences to '{ZIP_NAME}'")
print("\n>>> UPLOAD 'isl_mega_dataset.zip' to Google Colab to train the final robust model!")
