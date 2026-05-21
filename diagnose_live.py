"""
DIAGNOSTIC #2: Compare LIVE camera landmarks vs TRAINING data landmarks.
Shows you exactly what the camera is sending vs what the model expects.
"""
import os, sys, json, zipfile
import numpy as np

try:
    import google.protobuf.runtime_version as rv
except (ImportError, AttributeError):
    from unittest.mock import MagicMock
    proto_mod = sys.modules.get("google.protobuf")
    if proto_mod:
        proto_mod.runtime_version = MagicMock()

import cv2
import mediapipe as mp

# ── MediaPipe Setup (matching collect_custom_data.py) ──
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_and_normalize_live(results):
    """Exact copy of what the live script does."""
    if not results.multi_hand_landmarks:
        return None

    left_hand  = np.zeros(63)
    right_hand = np.zeros(63)

    for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                          results.multi_handedness):
        coords = []
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        coords = np.array(coords)

        label = handedness.classification[0].label
        if label == "Left":
            left_hand = coords
        else:
            right_hand = coords

    raw_landmarks = np.concatenate([left_hand, right_hand])
    
    # Spatial Normalization
    normalized = np.zeros(126)
    
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

# ── Load a training sample for comparison ──
ZIP_PATH = 'isl_dataset_normalized.zip'
EXTRACT_DIR = 'temp_diag2'

with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    z.extractall(EXTRACT_DIR)

data_dir = os.path.join(EXTRACT_DIR, 'custom_isl_data')
if not os.path.exists(data_dir):
    data_dir = EXTRACT_DIR

# Pick a simple sign to test
TEST_SIGN = '5'  # Number 5 = open palm, easy to do
sign_dir = os.path.join(data_dir, TEST_SIGN)
training_seq = np.load(os.path.join(sign_dir, os.listdir(sign_dir)[0]))

print(f"=== DIAGNOSTIC: Comparing LIVE vs TRAINING data for sign '{TEST_SIGN}' ===\n")

# Show training data stats
train_frame = training_seq[15]  # Middle frame
print("TRAINING DATA (frame 15 of 30):")
print(f"  Left hand (first 6 values):  {train_frame[:6]}")
print(f"  Right hand (first 6 values): {train_frame[63:69]}")
print(f"  Left hand non-zero:  {np.any(train_frame[:63])}")
print(f"  Right hand non-zero: {np.any(train_frame[63:])}")
print(f"  Left wrist (should be 0,0,0): {train_frame[0:3]}")
print(f"  Right wrist (should be 0,0,0): {train_frame[63:66]}")
left_max = np.max(np.abs(train_frame[:63])) if np.any(train_frame[:63]) else 0
right_max = np.max(np.abs(train_frame[63:])) if np.any(train_frame[63:]) else 0
print(f"  Left hand max value:  {left_max:.4f}")
print(f"  Right hand max value: {right_max:.4f}")

print(f"\n--- Now show the '{TEST_SIGN}' sign to the camera. Press 'C' to capture. ---\n")

# ── Capture from camera ──
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    sys.exit(1)

captured = False
while not captured:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    # Draw landmarks
    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
    
    cv2.putText(frame, f"Show '{TEST_SIGN}' sign, then press 'C' to capture", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Diagnostic Capture", frame)
    
    key = cv2.waitKey(10) & 0xFF
    if key == ord('c'):
        live_data = extract_and_normalize_live(results)
        if live_data is not None:
            print("\nLIVE CAMERA DATA:")
            print(f"  Left hand (first 6 values):  {live_data[:6]}")
            print(f"  Right hand (first 6 values): {live_data[63:69]}")
            print(f"  Left hand non-zero:  {np.any(live_data[:63])}")
            print(f"  Right hand non-zero: {np.any(live_data[63:])}")
            print(f"  Left wrist (should be 0,0,0): {live_data[0:3]}")
            print(f"  Right wrist (should be 0,0,0): {live_data[63:66]}")
            left_max = np.max(np.abs(live_data[:63])) if np.any(live_data[:63]) else 0
            right_max = np.max(np.abs(live_data[63:])) if np.any(live_data[63:]) else 0
            print(f"  Left hand max value:  {left_max:.4f}")
            print(f"  Right hand max value: {right_max:.4f}")
            
            # KEY COMPARISON
            print("\n=== KEY COMPARISON ===")
            
            # Check which hand has data in training vs live
            train_left = np.any(train_frame[:63])
            train_right = np.any(train_frame[63:])
            live_left = np.any(live_data[:63])
            live_right = np.any(live_data[63:])
            
            print(f"  Training: Left={train_left}, Right={train_right}")
            print(f"  Live:     Left={live_left}, Right={live_right}")
            
            if train_left != live_left or train_right != live_right:
                print("\n  !!! HAND MISMATCH DETECTED !!!")
                print("  The training data has the hand in a DIFFERENT slot than live!")
                print("  This is the ROOT CAUSE of the misrecognition!")
                if train_left and not live_left and live_right:
                    print("  SOLUTION: Training used LEFT slot, but live puts it in RIGHT slot.")
                    print("  --> We need to SWAP the live data: put Right into Left slot.")
                elif train_right and not live_right and live_left:
                    print("  SOLUTION: Training used RIGHT slot, but live puts it in LEFT slot.")
                    print("  --> We need to SWAP the live data: put Left into Right slot.")
            else:
                print("\n  Hands match! Checking value similarity...")
                # Compare actual values
                diff = np.mean(np.abs(train_frame - live_data))
                print(f"  Average absolute difference: {diff:.4f}")
                if diff < 0.3:
                    print("  Values are reasonably similar - model should work!")
                else:
                    print("  Values differ significantly - the sign looks different from training.")
            
            captured = True
        else:
            print("No hand detected! Show your hand and try again.")
    
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Cleanup
import shutil
try:
    shutil.rmtree(EXTRACT_DIR, ignore_errors=True)
except:
    pass
