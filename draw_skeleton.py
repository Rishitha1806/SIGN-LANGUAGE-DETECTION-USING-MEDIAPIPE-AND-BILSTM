import os
import cv2
import numpy as np
import zipfile

# Extract the dataset again
ZIP_PATH = 'isl_dataset_normalized.zip'
if os.path.exists(ZIP_PATH):
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall('temp_draw')

DATA_DIR = 'temp_draw/custom_isl_data'
if not os.path.exists(DATA_DIR):
    DATA_DIR = 'temp_draw'

def draw_landmarks_from_npy(word, seq_idx=0):
    word_dir = os.path.join(DATA_DIR, word)
    if not os.path.exists(word_dir):
        print(f"Directory {word_dir} not found.")
        return
        
    files = [f for f in os.listdir(word_dir) if f.endswith('.npy')]
    if not files:
        print(f"No npy files in {word_dir}")
        return
        
    seq = np.load(os.path.join(word_dir, files[seq_idx]))
    
    # We will pick the middle frame (frame 15) to draw
    frame_idx = 15
    landmarks = seq[frame_idx]
    
    # Create a blank black image
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    
    # Left hand is indices 0-62
    left_hand = landmarks[:63]
    right_hand = landmarks[63:]
    
    connections = [
        (0,1), (1,2), (2,3), (3,4), # Thumb
        (0,5), (5,6), (6,7), (7,8), # Index
        (5,9), (9,10), (10,11), (11,12), # Middle
        (9,13), (13,14), (14,15), (15,16), # Ring
        (13,17), (17,18), (18,19), (19,20), # Pinky
        (0,17) # Wrist to Pinky base
    ]
    
    def draw_hand(hand_data, color):
        if not np.any(hand_data): return
        
        points = []
        for i in range(0, 63, 3):
            # Scale up coordinates for visualization (original is roughly -1 to 1)
            x = int((hand_data[i] * 150) + 250)
            y = int((hand_data[i+1] * 150) + 250)
            points.append((x, y))
            cv2.circle(img, (x, y), 3, color, -1)
            
        for a, b in connections:
            if a < len(points) and b < len(points):
                cv2.line(img, points[a], points[b], color, 1)

    draw_hand(left_hand, (0, 255, 0))   # Left=Green
    draw_hand(right_hand, (0, 0, 255))  # Right=Red
    
    cv2.putText(img, f"Data recorded for: '{word}'", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.imwrite(f"recorded_{word}.png", img)
    print(f"Created visual representing the training data for {word}")

draw_landmarks_from_npy('1')
draw_landmarks_from_npy('HELLO')
draw_landmarks_from_npy('4')
draw_landmarks_from_npy('BYE')

# Cleanup
import shutil
shutil.rmtree('temp_draw', ignore_errors=True)
