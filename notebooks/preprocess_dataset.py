import os
import re
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

DATASET_DIR = "dataset"
OUTPUT_FILE = os.path.join(DATASET_DIR, "preprocessed_data.npz")
WINDOW_SIZE = 16
IMG_SIZE = (128, 128)
MAX_GAP_MS = 2000  # Threshold to split continuous driving runs

def parse_filename(filename):
    # Format: frame_{timestamp}_{direction}_{speed}_{alpha}.jpg
    # Or: frame_{timestamp}_aug_{num}_{direction}_{speed}_{alpha}.jpg
    pattern = r"frame_(\d+)(?:_aug_\d+)?_(\d+)_(\d+)_(\d+)\.jpg"
    match = re.match(pattern, filename)
    if match:
        timestamp = int(match.group(1))
        direction = int(match.group(2))
        speed = int(match.group(3))
        alpha = int(match.group(4))
        return {
            "filename": filename,
            "timestamp": timestamp,
            "direction": direction,
            "speed": speed,
            "alpha": alpha
        }
    return None

def load_metadata():
    files = os.listdir(DATASET_DIR)
    metadata_list = []
    for f in files:
        if f.endswith(".jpg"):
            meta = parse_filename(f)
            if meta:
                meta["path"] = os.path.join(DATASET_DIR, f)
                metadata_list.append(meta)
    
    # Sort chronologically
    metadata_list.sort(key=lambda x: x["timestamp"])
    return metadata_list

def group_into_sessions(metadata_list):
    if not metadata_list:
        return []
    
    sessions = []
    current_session = [metadata_list[0]]
    
    for i in range(1, len(metadata_list)):
        gap = metadata_list[i]["timestamp"] - metadata_list[i-1]["timestamp"]
        if gap > MAX_GAP_MS:
            sessions.append(current_session)
            current_session = [metadata_list[i]]
        else:
            current_session.append(metadata_list[i])
            
    sessions.append(current_session)
    return sessions

def build_stacked_windows(sessions):
    X = []
    Y_steer = []
    Y_throt = []
    Y_obst = []
    
    total_sessions_processed = 0
    total_windows_created = 0
    
    for s_idx, session in enumerate(sessions):
        if len(session) < WINDOW_SIZE:
            continue
            
        total_sessions_processed += 1
        
        # Preload and resize all images in this session to save disk reads during sliding window
        session_images = []
        for frame in session:
            img = cv2.imread(frame["path"], cv2.IMREAD_GRAYSCALE)
            if img is None:
                # Fallback to zeros if file is corrupted
                img = np.zeros(IMG_SIZE, dtype=np.uint8)
            else:
                img = cv2.resize(img, IMG_SIZE)
            session_images.append(img)
            
        # Create sliding windows
        for i in range(WINDOW_SIZE - 1, len(session)):
            # Stack WINDOW_SIZE consecutive frames
            stack = session_images[i - WINDOW_SIZE + 1 : i + 1] # List of 16 images
            stacked_tensor = np.stack(stack, axis=-1)           # Shape: (128, 128, 16)
            
            # Use label of the last frame in the window
            target_frame = session[i]
            
            # Map steering angle (-1.0 to 1.0)
            # LEFT (3) -> -1.0, RIGHT (4) -> 1.0, STRAIGHT (1) -> 0.0
            direction = target_frame["direction"]
            if direction == 3:
                steer = -1.0
            elif direction == 4:
                steer = 1.0
            else:
                steer = 0.0 # Straight or stop
                
            # Map throttle (0.0 to 1.0)
            speed = target_frame["speed"]
            throt = speed / 255.0
            
            # Map obstacle probability (0.0 to 1.0)
            # If speed is 0 or direction is STOP (0), set obstacle probability to 1.0 (Emergency stop / manual stop)
            if speed == 0 or direction == 0:
                obst = 1.0
            else:
                obst = 0.0
                
            X.append(stacked_tensor)
            Y_steer.append(steer)
            Y_throt.append(throt)
            Y_obst.append(obst)
            total_windows_created += 1
            
    print(f"Processed {total_sessions_processed}/{len(sessions)} sessions.")
    print(f"Created {total_windows_created} sliding window stacks.")
    return np.array(X, dtype=np.uint8), np.array(Y_steer, dtype=np.float32), np.array(Y_throt, dtype=np.float32), np.array(Y_obst, dtype=np.float32)

def main():
    print("Loading dataset metadata...")
    metadata = load_metadata()
    print(f"Found {len(metadata)} total images.")
    
    if len(metadata) == 0:
        print("Error: No images found in dataset folder!")
        return
        
    print("Grouping frames into continuous driving sessions...")
    sessions = group_into_sessions(metadata)
    print(f"Detected {len(sessions)} separate driving sessions.")
    
    print(f"Building stacked windows of size {WINDOW_SIZE}...")
    X, Y_steer, Y_throt, Y_obst = build_stacked_windows(sessions)
    
    if len(X) == 0:
        print("Error: Not enough frames to create even one window! Please collect more data.")
        return
        
    # Split: 70% Train, 20% Val, 10% Test
    print("Splitting dataset (70% Train, 20% Val, 10% Test)...")
    # First split off the 10% Test set
    x_train_val, x_test, y_s_train_val, y_s_test, y_t_train_val, y_t_test, y_o_train_val, y_o_test = train_test_split(
        X, Y_steer, Y_throt, Y_obst, test_size=0.10, random_state=42
    )
    
    # Then split the remaining 90% into Train (70% total) and Val (20% total)
    # Val ratio out of Train+Val is 20/90 = 2/9
    val_ratio_of_subset = 2.0 / 9.0
    x_train, x_val, y_s_train, y_s_val, y_t_train, y_t_val, y_o_train, y_o_val = train_test_split(
        x_train_val, y_s_train_val, y_t_train_val, y_o_train_val, test_size=val_ratio_of_subset, random_state=42
    )
    
    print(f"Train set: {len(x_train)} samples")
    print(f"Val set:   {len(x_val)} samples")
    print(f"Test set:  {len(x_test)} samples")
    
    print(f"Saving preprocessed data to {OUTPUT_FILE}...")
    np.savez_compressed(
        OUTPUT_FILE,
        x_train=x_train,
        y_train_steer=y_s_train,
        y_train_throt=y_t_train,
        y_train_obst=y_o_train,
        x_val=x_val,
        y_val_steer=y_s_val,
        y_val_throt=y_t_val,
        y_val_obst=y_o_val,
        x_test=x_test,
        y_test_steer=y_s_test,
        y_test_throt=y_t_test,
        y_test_obst=y_o_test
    )
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
