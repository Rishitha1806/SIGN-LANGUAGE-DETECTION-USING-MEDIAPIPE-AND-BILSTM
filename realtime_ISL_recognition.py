import os
import sys
import time
import threading
import tempfile

# --- COMPATIBILITY PATCH ---
try:
    import google.protobuf.runtime_version as rv
except (ImportError, AttributeError):
    from unittest.mock import MagicMock
    proto_mod = sys.modules.get("google.protobuf")
    if proto_mod:
        proto_mod.runtime_version = MagicMock()

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
import mediapipe as mp
import json

# ==================================================================================
#          ISL REAL-TIME RECOGNITION + NLP GRAMMAR + TRANSLATION ENGINE
# ==================================================================================

# ── 1. MediaPipe Setup (MUST match collect_custom_data.py) ───────────────────────
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ── 2. Paths & Model Loading ────────────────────────────────────────────────────
MODEL_PATH     = 'models/isl_model.h5'
LABEL_MAP_PATH = 'models/label_map.json'

print("--- ISL Real-Time Recognition + NLP Translation System ---")

if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_MAP_PATH):
    print(f"ERROR: Model files not found. Place isl_model.h5 and label_map.json in 'models/' folder.")
    sys.exit(1)

with open(LABEL_MAP_PATH, 'r') as f:
    label_map = json.load(f)
actions = sorted(label_map.keys(), key=lambda x: label_map[x])
num_classes = len(actions)
print(f"[INFO] Loaded {num_classes} sign classes.")

# Reconstruct the exact trained architecture
print("[INFO] Reconstructing AI Architecture...")
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

print("[INFO] Loading AI weights...")
model.load_weights(MODEL_PATH)
print("[SUCCESS] AI Brain loaded!")

# ── 3. Landmark Extraction (EXACT COPY from collect_custom_data.py) ──────────────
def extract_landmarks(results):
    """
    Extract hand landmarks from MediaPipe results.
    Returns a flat array of 126 values (2 hands x 21 landmarks x 3 coords).
    If only 1 hand detected, the other 63 values are zeros.
    If no hand detected, returns None.
    """
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
    
    # ── SPATIAL NORMALIZATION ──
    normalized = np.zeros(126)
    
    # Normalize Left Hand
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

    # Normalize Right Hand
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


# ── 4. NLP GRAMMAR ENGINE (ISL SOV -> English SVO) ──────────────────────────────

SUBJECTS    = {'I', 'I_ME', 'ME', 'HE', 'SHE', 'YOU', 'WE', 'THEY'}
POSSESSIVES = {'MY', 'YOUR'}
VERBS       = {'EAT', 'DRINK', 'HELP', 'KNOW', 'LEARN', 'LIKE', 'LOVE',
               'NEED', 'PLAY', 'SLEEP', 'STOP', 'THINK', 'WANT', 'WORK'}
NOUNS       = {'BOOK', 'CAR', 'FAMILY', 'FATHER', 'FOOD', 'FRIEND', 'HOUSE',
               'MOTHER', 'NAME', 'PHONE', 'SCHOOL', 'SISTER', 'BROTHER', 'WATER', 'AGE'}
ADJECTIVES  = {'ANGRY', 'BAD', 'BEAUTIFUL', 'GOOD', 'HAPPY', 'HUNGRY', 'SAD',
               'SICK', 'TIRED'}
TIME_WORDS  = {'AFTERNOON', 'DAY', 'LATER', 'MORNING', 'NIGHT', 'NOW',
               'TODAY', 'TOMORROW', 'YESTERDAY'}
GREETINGS   = {'BYE', 'HELLO', 'PLEASE', 'SORRY', 'THANK_YOU', 'YES', 'NO'}
QUESTION_WDS = {'HOW', 'WHAT', 'WHERE', 'WHO', 'WHY', 'WHEN'}

# Auxiliary verb mapping based on subject
AUX_MAP = {
    'I': 'am', 'I_ME': 'am', 'ME': 'am',
    'HE': 'is', 'SHE': 'is',
    'YOU': 'are', 'WE': 'are', 'THEY': 'are'
}

# Verb conjugation (present continuous for natural flow)
VERB_ING = {
    'EAT': 'eating', 'DRINK': 'drinking', 'HELP': 'helping',
    'KNOW': 'knowing', 'LEARN': 'learning', 'LIKE': 'liking',
    'LOVE': 'loving', 'NEED': 'needing', 'PLAY': 'playing',
    'SLEEP': 'sleeping', 'STOP': 'stopping', 'THINK': 'thinking',
    'WANT': 'wanting', 'WORK': 'working'
}

# Simple present for some verbs that sound better
VERB_SIMPLE = {
    'EAT': 'eat', 'DRINK': 'drink', 'HELP': 'help',
    'KNOW': 'know', 'LEARN': 'learn', 'LIKE': 'like',
    'LOVE': 'love', 'NEED': 'need', 'PLAY': 'play',
    'SLEEP': 'sleep', 'STOP': 'stop', 'THINK': 'think',
    'WANT': 'want', 'WORK': 'work'
}

# Verbs that sound better in simple present
SIMPLE_VERBS = {'KNOW', 'LIKE', 'LOVE', 'NEED', 'WANT', 'THINK'}

# Subject display mapping
SUBJECT_MAP = {
    'I': 'I', 'I_ME': 'I', 'ME': 'I',
    'HE': 'He', 'SHE': 'She',
    'YOU': 'You', 'WE': 'We', 'THEY': 'They'
}


def isl_to_english(glosses):
    """
    Convert a list of ISL glosses (SOV order) to a grammatically
    correct English sentence (SVO order).
    ISL: I WATER DRINK  ->  English: I am drinking water
    ISL: HE FOOD EAT    ->  English: He is eating food
    """
    if not glosses:
        return ""

    # If single word, return it directly (handles greetings, etc.)
    if len(glosses) == 1:
        w = glosses[0]
        if w == 'THANK_YOU': return "Thank you"
        if w == 'I_ME':      return "I"
        return w.capitalize()

    # Classify each word
    subj_list  = []
    poss_list  = []
    verb_list  = []
    noun_list  = []
    adj_list   = []
    time_list  = []
    greet_list = []
    q_list     = []
    other_list = []

    for g in glosses:
        if g in SUBJECTS:      subj_list.append(g)
        elif g in POSSESSIVES: poss_list.append(g)
        elif g in VERBS:       verb_list.append(g)
        elif g in NOUNS:       noun_list.append(g)
        elif g in ADJECTIVES:  adj_list.append(g)
        elif g in TIME_WORDS:  time_list.append(g)
        elif g in GREETINGS:   greet_list.append(g)
        elif g in QUESTION_WDS: q_list.append(g)
        else:                  other_list.append(g)

    parts = []

    # 1. Greetings first
    for g in greet_list:
        parts.append("Thank you" if g == 'THANK_YOU' else g.capitalize())

    # 2. Time words
    for t in time_list:
        parts.append(t.capitalize())

    # 3. Subject
    subject = subj_list[0] if subj_list else None
    if subject:
        parts.append(SUBJECT_MAP.get(subject, subject.capitalize()))

    # 4. Possessives + Adjectives + Nouns (before verb if no verb, after if verb exists)
    obj_parts = []
    for p in poss_list:
        obj_parts.append("my" if p == "MY" else "your")
    for a in adj_list:
        obj_parts.append(a.lower())
    for n in noun_list:
        obj_parts.append(n.lower())

    # 5. Verb with auxiliary
    if verb_list:
        v = verb_list[0]
        if subject:
            aux = AUX_MAP.get(subject, 'is')
            if v in SIMPLE_VERBS:
                # "I want water" not "I am wanting water"
                if subject in ('HE', 'SHE'):
                    parts.append(VERB_SIMPLE.get(v, v.lower()) + 's')
                else:
                    parts.append(VERB_SIMPLE.get(v, v.lower()))
            else:
                parts.append(aux)
                parts.append(VERB_ING.get(v, v.lower() + 'ing'))
        else:
            parts.append(VERB_ING.get(v, v.lower() + 'ing'))

    # Add adjective-only sentences ("I am happy")
    if adj_list and not verb_list and subject:
        aux = AUX_MAP.get(subject, 'is')
        parts.append(aux)
        # obj_parts already has the adjectives
    
    # 6. Object parts come after the verb
    parts.extend(obj_parts)

    # 7. Question words at end (ISL style) -> move to front (English style)
    if q_list:
        q_word = q_list[0].capitalize()
        # Insert question word at the beginning
        parts.insert(0, q_word)
        # Add question mark handling
    
    # 8. Other/unknown words
    for o in other_list:
        if o == 'THANK_YOU': parts.append("Thank you")
        elif o == 'I_ME':    parts.append("I")
        elif o == 'I_ALPHA': pass  # Skip the alphabet 'I'
        else: parts.append(o.capitalize())

    sentence = " ".join(parts)
    
    # Post-processing
    sentence = sentence.strip()
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]  # Capitalize first letter
    if q_list and not sentence.endswith('?'):
        sentence += '?'

    return sentence


# ── 5. TRANSLATION MODULE ────────────────────────────────────────────────────────

LANG_CODES = {'Hindi': 'hi'}
current_lang = 'Hindi'  # Only Hindi

def translate_text(text, target_lang_code):
    """Translate English text to target language using deep-translator."""
    try:
        from deep_translator import GoogleTranslator
        result = GoogleTranslator(source='auto', target=target_lang_code).translate(text)
        return result
    except Exception as e:
        return f"[Translation error: {e}]"


# ── 6. TEXT-TO-SPEECH MODULE ─────────────────────────────────────────────────────

def speak_text(text, lang_code='en'):
    """Speak text using gTTS + pygame in a background thread."""
    def _speak():
        try:
            import pygame
            import os, tempfile, time, random
            
            # Use gTTS to fetch English audio
            from gtts import gTTS
            tts = gTTS(text=text, lang=lang_code, slow=False)
            
            # Generate a RANDOM file name so it never collides or gets locked!
            tmp_file = os.path.join(tempfile.gettempdir(), f'isl_speech_{random.randint(1000,9999)}.mp3')
            tts.save(tmp_file)
            
            # Initialize Pygame mixer natively
            if not pygame.mixer.get_init():
                pygame.mixer.init()
                
            pygame.mixer.music.load(tmp_file)
            pygame.mixer.music.play()
            
            # Wait for audio to finish playing safely
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
            # Safely unload the file from memory
            pygame.mixer.music.unload()
            
            # Clean up the file from the hard drive
            try:
                os.remove(tmp_file)
            except:
                pass
        except Exception as e:
            print(f"[SPEECH ERROR] {e}")
            
    # Run heavily isolated so it never halts the video frame loop
    t = threading.Thread(target=_speak, daemon=True)
    t.start()


# ── 7. REAL-TIME RECOGNITION LOOP ───────────────────────────────────────────────

sequence     = []
sentence_buf = []   # Raw ISL glosses for the current sentence
display_words = []  # Last few recognized words for display
predictions  = []
threshold    = 0.70

# NLP output state
english_sentence   = ""
translated_sentence = ""
last_word_time      = time.time()
SENTENCE_TIMEOUT    = 3.0  # seconds of pause to auto-translate

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Give the webcam driver time to fully initialize
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Warmup: read and discard frames until the webcam is actually ready
for _ in range(10):
    cap.read()

if not cap.isOpened():
    print("[ERROR] Cannot open webcam!")
    sys.exit(1)

print("\n[SUCCESS] Webcam connected!")
print("\n" + "=" * 55)
print("  ISL REAL-TIME RECOGNITION + NLP TRANSLATION")
print("=" * 55)
print("  Controls:")
print("    SPACE = Translate current sentence")
print("    C     = Clear sentence buffer")
print("    S     = Speak the translation aloud")
print("    Q     = Quit")
print("=" * 55 + "\n")


failed_frames = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        failed_frames += 1
        print(f"[WARN] Failed to read frame ({failed_frames}/10)...")
        if failed_frames >= 10:
            print("[ERROR] Webcam stopped sending frames. Exiting.")
            break
        time.sleep(0.1)
        continue
    failed_frames = 0  # Reset counter on successful frame

    # Mirror the frame (MUST match data collection)
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Process with MediaPipe Hands
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_lms, handedness_info in zip(results.multi_hand_landmarks,
                                              results.multi_handedness):
            mp_drawing.draw_landmarks(
                frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )
            # Label each hand
            label = handedness_info.classification[0].label
            wrist = hand_lms.landmark[0]
            cx, cy = int(wrist.x * w), int(wrist.y * h)
            color = (0, 255, 0) if label == "Left" else (0, 0, 255)
            cv2.putText(frame, label, (cx - 20, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Extract landmarks
    keypoints = extract_landmarks(results)
    
    if keypoints is not None:
        sequence.append(keypoints)
        hand_is_visible = True
    else:
        # Hand dropped -> pad with zeros, but mark as invisible to prevent predicting empty space!
        sequence.append(np.zeros(126))
        hand_is_visible = False
    
    sequence = sequence[-30:]

    # Prediction
    confidence = 0.0
    detected_word = ""

    # ONLY PREDICT IF A HAND IS ACTUALLY IN THE FRAME!
    if len(sequence) == 30 and hand_is_visible:
        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        
        # Standard 106-word prediction (unfiltered)
        action_idx = np.argmax(res)
        confidence = res[action_idx]

        predictions.append(action_idx)
        
        if len(predictions) > 15:
            recent = predictions[-15:]
            most_common_idx = max(set(recent), key=recent.count)
            count = recent.count(most_common_idx)
            
            # Majority voting: Must be the dominant prediction (10+ out of 15 frames)
            if count >= 10 and action_idx == most_common_idx and confidence > threshold:
                detected_word = actions[action_idx]
                
                # Only add if different from last word
                if len(display_words) == 0 or detected_word != display_words[-1]:
                    display_words.append(detected_word)
                    sentence_buf.append(detected_word)

                    if len(display_words) > 7:
                        display_words = display_words[-7:]
                        
        # Continually reset the sentence timeout as long as hands are in the frame!
        last_word_time = time.time()

    # Auto-translate on 3-second pause (if buffer has words)
    if sentence_buf and (time.time() - last_word_time) > SENTENCE_TIMEOUT:
        english_sentence = isl_to_english(sentence_buf)
        lang_code = LANG_CODES[current_lang]
        translated_sentence = translate_text(english_sentence, lang_code)
        
        # Speak the original English sentence out loud (forced English Audio)
        speak_text(english_sentence, 'en')
        
        # Clear UI boards for the next sentence
        sentence_buf = []
        display_words = []

    # ── UI RENDERING ─────────────────────────────────────────────────────────

    # Bar 1: Raw ISL Glosses (Top - Blue)
    cv2.rectangle(frame, (0, 0), (w, 45), (180, 90, 20), -1)
    gloss_text = "ISL: " + (" ".join(display_words) if display_words else "Show your sign...")
    cv2.putText(frame, gloss_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Bar 2: English Sentence (Middle - Dark Green)
    cv2.rectangle(frame, (0, 45), (w, 85), (40, 100, 40), -1)
    eng_text = "EN: " + (english_sentence if english_sentence else "Waiting for sentence...")
    cv2.putText(frame, eng_text, (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2, cv2.LINE_AA)

    # Bar 3: Translation (Bottom - Dark Purple)
    cv2.rectangle(frame, (0, 85), (w, 125), (100, 40, 80), -1)
    
    if current_lang != 'English' and translated_sentence:
        trans_text = f"{current_lang}: {translated_sentence}"
        try:
            from PIL import Image, ImageDraw, ImageFont
            # Convert to PIL for Unicode Rendering
            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)
            # Nirmala.ttc is standard on Windows for Hindi, Telugu, and Tamil
            font = ImageFont.truetype("C:/Windows/Fonts/Nirmala.ttc", 26)
            draw.text((10, 92), trans_text, font=font, fill=(255, 200, 255))
            frame = np.array(img_pil)
        except Exception as e:
            # Fallback if font is somehow missing on this Windows build
            cv2.putText(frame, f"[Nirmala.ttc Font Missing] {english_sentence}", (10, 112),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 255), 2, cv2.LINE_AA)
    else:
        trans_text = f"{current_lang}: " + (translated_sentence if translated_sentence else "...")
        cv2.putText(frame, trans_text, (10, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 255), 2, cv2.LINE_AA)

    # Confidence meter (bottom-right)
    conf_text = f"Conf: {confidence*100:.0f}%"
    cv2.putText(frame, conf_text, (w - 150, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Language indicator (bottom-left)
    cv2.putText(frame, f"Lang: {current_lang} | SPACE=Translate | C=Clear | Q=Quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    cv2.imshow('ISL Live Recognition + NLP Translation', frame)

    # ── KEY HANDLING ─────────────────────────────────────────────────────────
    key = cv2.waitKey(10) & 0xFF

    if key == ord('q'):
        break

    elif key == ord(' '):
        # Manual translate trigger
        if sentence_buf:
            english_sentence = isl_to_english(sentence_buf)
            lang_code = LANG_CODES[current_lang]
            translated_sentence = translate_text(english_sentence, lang_code)
            sentence_buf = []

    elif key == ord('c'):
        # Clear everything
        sentence_buf = []
        display_words = []
        english_sentence = ""
        translated_sentence = ""
        predictions = []
        sequence = []
        print("[CLEARED] Sentence buffer reset.")


    elif key == ord('s'):
        # Speak the forced English translation
        if english_sentence:
            speak_text(english_sentence, 'en')
            print(f"[SPEECH] Speaking: {english_sentence}")

cap.release()
cv2.destroyAllWindows()
print("\n[DONE] ISL Recognition System closed.")
