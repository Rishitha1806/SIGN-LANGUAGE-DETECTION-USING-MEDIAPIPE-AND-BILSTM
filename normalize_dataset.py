import os
import zipfile
import numpy as np
import shutil

ZIP_PATH = 'isl_dataset.zip'
EXTRACT_DIR = 'temp_dataset'
OUTPUT_ZIP = 'isl_dataset_normalized.zip'

def normalize_landmarks(landmarks_126):
    """
    Transforms RAW coordinates into NORMALIZED coordinates.
    1. Makes coordinates relative to the wrist (position invariant).
    2. Scales the hand to a standard size (distance invariant).
    """
    normalized = np.zeros(126)
    
    # Left hand (indices 0 to 62)
    left_hand = landmarks_126[:63]
    if np.any(left_hand):
        wrist_x, wrist_y, wrist_z = left_hand[0], left_hand[1], left_hand[2]
        for i in range(0, 63, 3):
            normalized[i]   = left_hand[i] - wrist_x
            normalized[i+1] = left_hand[i+1] - wrist_y
            normalized[i+2] = left_hand[i+2] - wrist_z
        # Scale to max absolute value
        max_val = np.max(np.abs(normalized[:63]))
        if max_val > 0:
            normalized[:63] = normalized[:63] / max_val

    # Right hand (indices 63 to 125)
    right_hand = landmarks_126[63:]
    if np.any(right_hand):
        wrist_x, wrist_y, wrist_z = right_hand[0], right_hand[1], right_hand[2]
        for i in range(63, 126, 3):
            normalized[i]   = right_hand[i-63] - wrist_x
            normalized[i+1] = right_hand[i-62] - wrist_y
            normalized[i+2] = right_hand[i-61] - wrist_z
        # Scale to max absolute value
        max_val = np.max(np.abs(normalized[63:]))
        if max_val > 0:
            normalized[63:] = normalized[63:] / max_val
            
    return normalized

def main():
    print("--- ISL Normalization Tool ---")
    
    if not os.path.exists(ZIP_PATH):
        print(f"ERROR: Could not find {ZIP_PATH}.")
        return

    # 1. Unzip the current dataset
    print("Unzipping dataset...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
        
    data_dir = os.path.join(EXTRACT_DIR, 'custom_isl_data')
    if not os.path.exists(data_dir):
        # Fallback if structure is different
        data_dir = EXTRACT_DIR

    # 2. Process all NPY files
    folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"Found {len(folders)} classes. Normalizing...")
    
    count = 0
    for word in folders:
        word_dir = os.path.join(data_dir, word)
        for file in os.listdir(word_dir):
            if file.endswith('.npy'):
                filepath = os.path.join(word_dir, file)
                
                # Load sequence (Shape: 30, 126)
                sequence = np.load(filepath)
                normalized_sequence = np.zeros_like(sequence)
                
                # Normalize each frame
                for frame_idx in range(len(sequence)):
                    normalized_sequence[frame_idx] = normalize_landmarks(sequence[frame_idx])
                
                # Overwrite file
                np.save(filepath, normalized_sequence)
                count += 1

    print(f"Normalized {count} sequences successfully.")

    # 3. Zip it back up
    print(f"Repackaging into {OUTPUT_ZIP}...")
    shutil.make_archive('isl_dataset_normalized', 'zip', EXTRACT_DIR)

    # 4. Cleanup
    shutil.rmtree(EXTRACT_DIR)
    
    print("✅ DONE! You can now upload 'isl_dataset_normalized.zip' to Google Drive.")

if __name__ == "__main__":
    main()
