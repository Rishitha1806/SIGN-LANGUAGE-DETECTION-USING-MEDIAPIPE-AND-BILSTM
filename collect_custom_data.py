# ============================================================
#  collect_custom_data.py  —  MediaPipe Hand Landmark Dataset
#
#  Collects hand landmark data (21 points × 3 coords = 63 values)
#  per frame using MediaPipe Hands. Saves sequences as .npy files.
#
#  This is LIGHTING & BACKGROUND INDEPENDENT — no skin masking!
#
#  RUN:
#    pip install mediapipe opencv-python numpy
#    python collect_custom_data.py
# ============================================================

import cv2
import numpy as np
import os
import time

try:
    # Block TensorFlow from loading — data collection doesn't need it.
    # This prevents the protobuf version conflict between TF and MediaPipe.
    import sys
    import types
    
    # Create a fake tensorflow module so MediaPipe skips TF-dependent code
    fake_tf = types.ModuleType('tensorflow')
    fake_tf.__version__ = '0.0.0'
    fake_tf_tools = types.ModuleType('tensorflow.tools')
    fake_tf_docs = types.ModuleType('tensorflow.tools.docs')
    fake_tf_docs.doc_controls = types.ModuleType('tensorflow.tools.docs.doc_controls')
    fake_tf_docs.doc_controls.do_not_generate_docs = lambda f: f
    
    sys.modules['tensorflow'] = fake_tf
    sys.modules['tensorflow.tools'] = fake_tf_tools
    sys.modules['tensorflow.tools.docs'] = fake_tf_docs
    sys.modules['tensorflow.tools.docs.doc_controls'] = fake_tf_docs.doc_controls
    
    import mediapipe as mp
except ImportError as e:
    print(f"[ERROR] MediaPipe not installed or broken: {e}")
    print("Run: pip install mediapipe")
    exit(1)

# ─── CONFIGURATION ───────────────────────────────────────────
DATA_DIR = r"C:\Users\jvris\Desktop\custom_isl_data"

WORDS = [
    "HELLO", "THANK_YOU", "YES", "NO", "NAME"
]

SEQUENCES_PER_WORD  = 30   # Record each word 30 times for deep learning
FRAMES_PER_SEQUENCE = 30   # ~1 second of data per recording
COUNTDOWN_SECONDS   = 3    # Seconds to get ready before recording
# ─────────────────────────────────────────────────────────────

# ── MediaPipe Setup ──────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,           # Detect both hands (some ISL signs use 2)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_landmarks(results):
    """
    Extract hand landmarks from MediaPipe results.
    Returns a flat array of 126 values (2 hands × 21 landmarks × 3 coords).
    If only 1 hand detected, the other 63 values are zeros.
    If no hand detected, returns None.
    """
    if not results.multi_hand_landmarks:
        return None

    # Initialize both hands as zeros
    left_hand = np.zeros(63)
    right_hand = np.zeros(63)

    for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                          results.multi_handedness):
        # Get 21 landmarks × (x, y, z)
        coords = []
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        coords = np.array(coords)

        # Assign to correct hand
        label = handedness.classification[0].label
        if label == "Left":
            left_hand = coords
        else:
            right_hand = coords

    # Combine both hands: 126 total values
    return np.concatenate([left_hand, right_hand])


def draw_ui(frame, word, status, progress_text, hand_detected, recording=False):
    """Draw a clean UI overlay on the frame."""
    h, w = frame.shape[:2]

    # Dark overlay bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 130), (15, 15, 30), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # Title
    cv2.putText(frame, "ISL Dataset Collector (MediaPipe)",
                (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 150), 2)

    # Current word
    color = (0, 255, 0) if recording else (0, 200, 255)
    cv2.putText(frame, f"Sign: {word}",
                (15, 65), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)

    # Status text
    cv2.putText(frame, status,
                (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Progress
    cv2.putText(frame, progress_text,
                (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Hand detection indicator
    if hand_detected:
        cv2.putText(frame, "HAND DETECTED",
                    (w - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.circle(frame, (w - 235, 25), 8, (0, 255, 0), -1)
    else:
        cv2.putText(frame, "NO HAND",
                    (w - 160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.circle(frame, (w - 175, 25), 8, (0, 0, 255), -1)

    # Recording border flash
    if recording:
        cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 4)
        cv2.putText(frame, "REC", (w - 70, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Footer
    cv2.putText(frame, "SPACE=Skip Word | R=Re-record | Q=Quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

    return frame


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)

    print("[INFO] Attempting to connect to webcam...")
    # Try different camera ports and backends
    cap = None
    for port in [0, 1, 2]:
        for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
            try:
                temp_cap = cv2.VideoCapture(port, backend)
                if temp_cap.isOpened():
                    # Test if it returns frames
                    ret, frame = temp_cap.read()
                    if ret and frame is not None:
                        print(f"[INFO] Successfully connected to webcam on port {port}")
                        cap = temp_cap
                        break
                temp_cap.release()
            except:
                pass
        if cap is not None:
            break
            
    if cap is None:
        print("\n" + "!" * 60)
        print("[FATAL ERROR] Cannot connect to any webcam on any port!")
        print("Why does this happen on Windows?")
        print("  1. Windows Privacy Settings are blocking Python from the camera.")
        print("  2. Another hidden app (Zoom/Teams/Discord) is holding the camera.")
        print("  3. Your camera driver crashed (try unplugging and plugging it back in).")
        print("!" * 60 + "\n")
        return

    print("[INFO] Webcam connected and streaming successfully!\n")

    total_words = len(WORDS)
    total_sequences = total_words * SEQUENCES_PER_WORD
    completed = 0

    print("\n" + "=" * 55)
    print("  ISL CUSTOM DATASET COLLECTOR (MediaPipe Landmarks)")
    print("=" * 55)
    print(f"  Words to record : {total_words}")
    print(f"  Sequences/word  : {SEQUENCES_PER_WORD}")
    print(f"  Frames/sequence : {FRAMES_PER_SEQUENCE}")
    print(f"  Total sequences : {total_sequences}")
    print(f"  Save location   : {DATA_DIR}")
    print("=" * 55)
    print("\nControls:")
    print("  SPACE = Skip current word")
    print("  R     = Re-record last sequence")
    print("  Q     = Quit and save progress\n")

    word_idx = 0
    while word_idx < total_words:
        word = WORDS[word_idx]
        word_dir = os.path.join(DATA_DIR, word)
        os.makedirs(word_dir, exist_ok=True)

        # Check how many sequences already exist (resume support)
        existing = len([f for f in os.listdir(word_dir) if f.endswith('.npy')])

        seq_idx = existing  # Start from where we left off
        
        # ── PREPARATION PHASE (40s to get ready for a new word) ──
        if seq_idx == 0: # Only wait 40s if we are starting a completely new word
            quit_app = False
            skip_word = False
            for p_time in range(40, 0, -1):
                start = time.time()
                while time.time() - start < 1.0:
                    ret, frame = cap.read()
                    if not ret: break
                    frame = cv2.flip(frame, 1)
                    
                    progress_text = f"Word {word_idx+1}/{total_words} | Total: {completed}/{total_sequences}"
                    status = f"Preparation: '{word}' in {p_time}s. Press 'C' to start now!"
                    
                    # Draw UI
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 140), (45, 45, 45), -1)
                    cv2.putText(frame, f"SIGN: {word}", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                    cv2.putText(frame, status, (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, progress_text, (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                    cv2.putText(frame, "C=Start Now | SPACE=Skip Word | Q=Quit", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
                    
                    cv2.imshow("ISL Data Collection", frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    # Let the user press 'C' to bypass the 40 seconds
                    if key == ord('c'): break
                    if key == ord('q'): quit_app = True; break
                    if key == ord(' '): skip_word = True; break
                
                # Check breaks from inner loop
                if key == ord('c') or quit_app or skip_word: break
            
            if quit_app: break
            if skip_word:
                word_idx += 1
                continue

        while seq_idx < SEQUENCES_PER_WORD:
            progress = f"Word {word_idx+1}/{total_words} | Seq {seq_idx+1}/{SEQUENCES_PER_WORD} | Total: {completed}/{total_sequences}"

            # ── COUNTDOWN PHASE ──────────────────────────────
            skip_word = False
            quit_app = False

            for countdown in range(COUNTDOWN_SECONDS, 0, -1):
                start = time.time()
                while time.time() - start < 1.0:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)

                    # Run MediaPipe to show hand tracking during countdown
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)
                    hand_ok = results.multi_hand_landmarks is not None

                    # Draw hand landmarks on frame
                    if results.multi_hand_landmarks:
                        for hand_lms in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                                mp_styles.get_default_hand_landmarks_style(),
                                mp_styles.get_default_hand_connections_style()
                            )

                    status = f"Get ready in {countdown}... Show the sign for '{word}'"
                    frame = draw_ui(frame, word, status, progress, hand_ok)
                    cv2.imshow("ISL Data Collection", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        quit_app = True
                        break
                    elif key == ord(' '):
                        skip_word = True
                        break

                if quit_app or skip_word:
                    break

            if quit_app:
                break
            if skip_word:
                print(f"  [SKIP] Skipped: {word}")
                word_idx += 1
                break

            # ── RECORDING PHASE ──────────────────────────────
            sequence_data = []
            frames_captured = 0
            frames_missed = 0

            while frames_captured < FRAMES_PER_SEQUENCE:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

                # Extract landmarks
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                landmarks = extract_landmarks(results)

                hand_ok = landmarks is not None
                if hand_ok:
                    sequence_data.append(landmarks)
                    frames_captured += 1
                else:
                    frames_missed += 1

                # Draw hand landmarks
                if results.multi_hand_landmarks:
                    for hand_lms in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style()
                        )

                status = f"RECORDING! Frame {frames_captured}/{FRAMES_PER_SEQUENCE} (missed: {frames_missed})"
                frame = draw_ui(frame, word, status, progress, hand_ok, recording=True)
                cv2.imshow("ISL Data Collection", frame)

                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    quit_app = True
                    break

                # Safety: if too many misses, warn user
                if frames_missed > FRAMES_PER_SEQUENCE * 3:
                    print(f"  [WARN] Too many missed frames for '{word}'. Make sure hand is visible!")
                    break

            if quit_app:
                break

            # ── SAVE SEQUENCE ────────────────────────────────
            if len(sequence_data) == FRAMES_PER_SEQUENCE:
                seq_array = np.array(sequence_data)  # Shape: (30, 126)
                save_path = os.path.join(word_dir, f"seq_{seq_idx:03d}.npy")
                np.save(save_path, seq_array)
                completed += 1
                print(f"  [SAVED] {word}/seq_{seq_idx:03d}.npy  shape={seq_array.shape}  ({completed}/{total_sequences})")
                seq_idx += 1
            else:
                print(f"  [RETRY] Only got {len(sequence_data)}/{FRAMES_PER_SEQUENCE} frames. Try again...")
                # Don't increment seq_idx, so it re-records this one

        else:
            # All sequences for this word done, move to next
            word_idx += 1
            continue

        if quit_app:
            break
        # If we broke out of inner loop via skip, word_idx already incremented

    # ── DONE ─────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    print("\n" + "=" * 55)
    print(f"  DONE! Collected {completed} sequences.")
    print(f"  Dataset saved at: {DATA_DIR}")
    print("=" * 55)

    # Print summary
    print("\n  Summary per word:")
    for word in WORDS:
        word_dir = os.path.join(DATA_DIR, word)
        if os.path.exists(word_dir):
            count = len([f for f in os.listdir(word_dir) if f.endswith('.npy')])
            if count > 0:
                print(f"    {word:20s} : {count} sequences")


if __name__ == "__main__":
    main()
