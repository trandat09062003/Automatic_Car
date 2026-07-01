import os
import socket
import cv2
import numpy as np
import time

# --- CONFIGURATION ---
LISTEN_PORT = 3000          # Port receiving JPEG frames from ESP32-S3
CONTROL_PORT = 3001         # Port sending control strings to ESP32-S3
MODEL_PATH = "models/best_model.keras"
TFLITE_PATH = "models/tiny_nav_quantized.tflite"
WINDOW_SIZE = 16
IMG_SIZE = (128, 128)

# Motor speed control settings
BASE_SPEED = 80             # Base speed of motors (optimized between 60 and 95 to prevent stalling)
TURN_ALPHA = 35             # Default fallback steering offset

# Movement state codes
STOP = 0
STRAIGHT = 1
LEFT = 3
RIGHT = 4

# --- MODEL LOADING (DUAL-MODE: TFLITE / KERAS FALLBACK) ---
use_tflite = False
interpreter = None

if os.path.exists(TFLITE_PATH):
    print("Attempting to load lightweight TFLite model...", flush=True)
    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=TFLITE_PATH)
        use_tflite = True
        print("TFLite model loaded successfully using tflite_runtime!", flush=True)
    except ImportError:
        try:
            import tensorflow.lite as tflite
            interpreter = tflite.Interpreter(model_path=TFLITE_PATH)
            use_tflite = True
            print("TFLite model loaded successfully using tensorflow.lite!", flush=True)
        except ImportError:
            print("tflite-runtime or tensorflow.lite not found. Falling back to full Keras model...", flush=True)

if use_tflite:
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']
    
    # Avoid division by zero if unquantized
    if input_scale == 0.0: input_scale = 1.0
    if output_scale == 0.0: output_scale = 1.0
    
    print("TFLite model initialized successfully!", flush=True)
else:
    print("Loading full Keras AI model (this may take 1-2 minutes to initialize TensorFlow)...", flush=True)
    
    # Suppress TensorFlow verbose logging on import
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    import tensorflow as tf
    
    # --- CUSTOM LOSS & METRICS (REQUIRED TO LOAD KERAS MODEL) ---
    def custom_loss(y_true, y_pred):
        y_true_steer = y_true[:, 0:1]
        y_true_throt = y_true[:, 1:2]
        y_true_obst = y_true[:, 2:3]
        
        y_pred_steer = y_pred[:, 0:1]
        y_pred_throt = y_pred[:, 1:2]
        y_pred_obst = y_pred[:, 2:3]
        
        steer_pred_act = tf.tanh(y_pred_steer)
        throt_pred_act = tf.sigmoid(y_pred_throt)
        obst_pred_act = tf.sigmoid(y_pred_obst)
        
        steer_loss = tf.reduce_mean(tf.square(y_true_steer - steer_pred_act))
        throt_loss = tf.reduce_mean(tf.square(y_true_throt - throt_pred_act))
        obst_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true_obst, obst_pred_act))
        
        return steer_loss * 1.0 + throt_loss * 1.0 + obst_loss * 0.5

    def steer_mae(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true[:, 0:1] - tf.tanh(y_pred[:, 0:1])))

    def throt_mae(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true[:, 1:2] - tf.sigmoid(y_pred[:, 1:2])))

    def obst_accuracy(y_true, y_pred):
        pred_rounded = tf.round(tf.sigmoid(y_pred[:, 2:3]))
        return tf.reduce_mean(tf.cast(tf.equal(y_true[:, 2:3], pred_rounded), tf.float32))

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained Keras model not found at {MODEL_PATH}. Make sure to train it first.")
        
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "custom_loss": custom_loss,
            "steer_mae": steer_mae,
            "throt_mae": throt_mae,
            "obst_accuracy": obst_accuracy
        }
    )
    print("Keras model loaded successfully!", flush=True)

# Initialize UDP Socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # Enable broadcast for auto-discovery
sock.bind(("0.0.0.0", LISTEN_PORT))
sock.settimeout(0.01)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)

# Sliding window buffer (holds last 16 processed grayscale frames)
frame_queue = []

buffer = bytearray()
end_marker = b"\xff\xd9"
esp32_ip = None

# Manual override & auto-pilot toggles
auto_pilot = False

print("\n--- AUTOPILOT CONTROL INTERFACE ---", flush=True)
print("  Space (Phím cách) : Bật/Tắt chế độ Tự Lái (Autopilot)", flush=True)
print("  S                 : Phanh khẩn cấp / Dừng xe (STOP)", flush=True)
print("  Q                 : Thoát chương trình", flush=True)
print("------------------------------------\n", flush=True)

last_frame_time = time.time()
last_command_time = 0
command_interval = 0.05  # Send control command at max ~20Hz (every 50ms) to avoid flooding UDP

# Initialize variables before the loop to prevent NameError
steering = 0.0
throttle = 0.0
obstacle = 0.0
status_msg = "Waiting for camera stream..."
last_sent_speed = 0
current_frame = np.zeros((240, 320, 3), dtype=np.uint8)

last_heartbeat_time = 0

while True:
    # Send discovery heartbeat to ESP32-S3 via UDP Broadcast if stream is disconnected
    now = time.time()
    if now - last_frame_time > 1.0 and now - last_heartbeat_time > 1.0:
        try:
            sock.sendto(b"HB", ("255.255.255.255", CONTROL_PORT))
            last_heartbeat_time = now
        except Exception:
            pass

    try:
        data, addr = sock.recvfrom(65535)
        if data == b"HB" or data == b"HB\n":
            continue
            
        if esp32_ip is None or esp32_ip != addr[0]:
            esp32_ip = addr[0]
            print(f"-> ESP32-S3 connected at IP: {esp32_ip}", flush=True)
            
        # Fast JPEG check
        if data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9"):
            buffer.clear()
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        else:
            buffer.extend(data)
            
            # JPEG start marker synchronization
            start_pos = buffer.find(b"\xff\xd8")
            if start_pos != -1:
                buffer = buffer[start_pos:]
            else:
                if len(buffer) > 65535:
                    buffer.clear()
                continue
                
            # JPEG end marker check
            pos = buffer.find(end_marker)
            if pos != -1:
                pos += 2
                img_bytes = buffer[:pos]
                buffer = buffer[pos:]
                frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            else:
                continue
                
        # Fallback for corrupted JPEG frames to prevent AI queue starvation
        if frame is None and current_frame is not None:
            frame = current_frame

        if frame is not None:
            current_frame = frame
            last_frame_time = time.time()
            
            # --- PREPROCESSING FOR AI (ROI CROP & RESIZE) ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            crop_top = int(h * 0.4)  # Cut top 40% (room background) to prevent overfitting
            roi = gray[crop_top:, :]
            resized = cv2.resize(roi, IMG_SIZE)
            
            # Maintain sliding window queue
            frame_queue.append(resized)
            if len(frame_queue) > WINDOW_SIZE:
                frame_queue.pop(0)
            
            # Run inference if stack is full
            if len(frame_queue) == WINDOW_SIZE:
                # Stack frames: (128, 128, 16)
                stacked_tensor = np.stack(frame_queue, axis=-1)
                
                # Normalize: scale to [0.0, 1.0] and add batch dimension (1, 128, 128, 16)
                input_data = np.expand_dims(stacked_tensor.astype(np.float32) / 255.0, axis=0)
                
                # AI Prediction
                if use_tflite:
                    # Quantize input if necessary
                    if input_details[0]['dtype'] == np.int8:
                        input_data_q = (input_data / input_scale + input_zero_point).astype(np.int8)
                    else:
                        input_data_q = input_data
                        
                    interpreter.set_tensor(input_details[0]['index'], input_data_q)
                    interpreter.invoke()
                    output_raw = interpreter.get_tensor(output_details[0]['index'])[0]
                    
                    # Dequantize output if necessary
                    if output_details[0]['dtype'] == np.int8:
                        output_raw = (output_raw.astype(np.float32) - output_zero_point) * output_scale
                        
                    steer_raw = output_raw[0]
                    throt_raw = output_raw[1]
                    obst_raw = output_raw[2]
                else:
                    output = model.predict(input_data, verbose=0)[0]
                    steer_raw = output[0]
                    throt_raw = output[1]
                    obst_raw = output[2]
                
                # Apply activations manually to match model metrics
                steering = np.tanh(steer_raw)
                throttle = 1.0 / (1.0 + np.exp(-throt_raw)) # Sigmoid
                obstacle = 1.0 / (1.0 + np.exp(-obst_raw))  # Sigmoid
                
                # Translate to vehicle controls using Proportional Control to prevent tail-wagging / oscillation
                STEER_DEADZONE = 0.05
                steer_val = abs(steering)
                
                if steer_val > STEER_DEADZONE:
                    target_dir = LEFT if steering < 0 else RIGHT
                    
                    # Proportional speed boost for outer wheel: ranges from BASE_SPEED up to BASE_SPEED + 45
                    target_speed = int(BASE_SPEED + (steer_val * 45))
                    target_speed = min(130, target_speed)  # Cap maximum speed for safety
                    
                    # Proportional alpha (wheel speed difference) based on steering magnitude
                    # As steering approaches 1.0, alpha approaches target_speed (inner wheel stops)
                    target_alpha = int(steer_val * target_speed)
                    
                    dir_name = "LEFT" if target_dir == LEFT else "RIGHT"
                    status_msg = f"Steering {dir_name} (S:{steering:+.2f} | Spd:{target_speed} | Alpha:{target_alpha})"
                else:
                    target_dir = STRAIGHT
                    target_speed = BASE_SPEED
                    target_alpha = 0
                    status_msg = f"Steering STRAIGHT (S:{steering:+.2f})"
                
                # --- UDP AUTOMATIC TRANSMISSION ---
                now = time.time()
                if auto_pilot and esp32_ip and (now - last_command_time > command_interval):
                    # Kick-start (xung kích khởi hành) to overcome static friction when starting from a stop
                    if last_sent_speed == 0 and target_dir == STRAIGHT and target_speed < 110:
                        kick_cmd = f"{target_dir} 130 {target_alpha}"
                        sock.sendto(kick_cmd.encode(), (esp32_ip, CONTROL_PORT))
                        time.sleep(0.12)
                        
                    cmd = f"{target_dir} {target_speed} {target_alpha}"
                    sock.sendto(cmd.encode(), (esp32_ip, CONTROL_PORT))
                    last_command_time = now
                    last_sent_speed = target_speed
            else:
                status_msg = f"Buffering stack... ({len(frame_queue)}/16)"
                steering, throttle, obstacle = 0.0, 0.0, 0.0

    except socket.timeout:
        # Buffer clear on timeout
        if time.time() - last_frame_time > 2.0:
            buffer.clear()
            frame_queue.clear()
            last_frame_time = time.time()
            
    # --- VISUAL UI OVERLAY ---
    display_frame = current_frame.copy()
    h, w, _ = display_frame.shape
    
    # Check connection timeout (WiFi or ESP32 signal loss)
    connected = (time.time() - last_frame_time < 1.5)
    if not connected:
        # Draw a semi-transparent red overlay over the image
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 150), -1)
        cv2.addWeighted(overlay, 0.4, display_frame, 0.6, 0, display_frame)
        cv2.putText(display_frame, "!!! CONNECTION LOST !!!", (w // 2 - 110, h // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(display_frame, "Check WiFi/ESP32 Power", (w // 2 - 100, h // 2 + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    
    # Draw ROI Crop Line
    crop_top = int(h * 0.4)
    cv2.line(display_frame, (0, crop_top), (w, crop_top), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(display_frame, "AI VIEW ZONE (CROP)", (w - 130, crop_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)

    # 1. Gauge/HUD indicators
    cv2.rectangle(display_frame, (0, h - 50), (w, h), (0, 0, 0), -1) # Footer background
    
    # Draw steer pointer
    center_x = w // 2
    pointer_x = int(center_x + (steering * (w // 2 - 20)))
    cv2.line(display_frame, (center_x, h - 35), (pointer_x, h - 15), (0, 255, 0), 2)
    cv2.circle(display_frame, (center_x, h - 35), 3, (255, 255, 255), -1)
    
    # Text displays
    pilot_text = "AUTO-PILOT ACTIVE" if auto_pilot else "MANUAL STANDBY"
    pilot_color = (0, 255, 0) if auto_pilot else (0, 255, 255)
    
    cv2.putText(display_frame, pilot_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pilot_color, 1, cv2.LINE_AA)
    cv2.putText(display_frame, f"Steer: {steering:+.2f} | Speed: {BASE_SPEED} | Obstacle: {obstacle:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(display_frame, f"State: {status_msg}", (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    
    # Obstacle indicator
    if obstacle > 0.8:
        cv2.putText(display_frame, "!!! DANGER !!!", (w - 110, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(display_frame, (w - 20, 15), 6, (0, 0, 255), -1)
    else:
        cv2.circle(display_frame, (w - 20, 15), 6, (0, 255, 0), -1)
        
    cv2.imshow("TinyNav PC Autonomous Driver HUD", display_frame)
    
    # --- KEYBOARD CONTROLS ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        # Emergency stop on exit
        if esp32_ip:
            sock.sendto(b"0 0 0", (esp32_ip, CONTROL_PORT))
        break
    elif key == ord(' '):
        # Toggle autopilot mode
        auto_pilot = not auto_pilot
        print(f"==> Autopilot mode: {'ENABLED' if auto_pilot else 'DISABLED'}", flush=True)
        if not auto_pilot and esp32_ip:
            # STOP car when turning off autopilot
            sock.sendto(b"0 0 0", (esp32_ip, CONTROL_PORT))
            last_sent_speed = 0
    elif key == ord('s') or key == ord('S'):
        # Force stop
        auto_pilot = False
        print("==> EMERGENCY STOP triggered!", flush=True)
        if esp32_ip:
            sock.sendto(b"0 0 0", (esp32_ip, CONTROL_PORT))
            last_sent_speed = 0
    elif key == ord('+') or key == ord('='):
        BASE_SPEED = min(220, BASE_SPEED + 10)
        print(f"==> BASE_SPEED increased to: {BASE_SPEED}", flush=True)
    elif key == ord('-') or key == ord('_'):
        BASE_SPEED = max(80, BASE_SPEED - 10)
        print(f"==> BASE_SPEED decreased to: {BASE_SPEED}", flush=True)

sock.close()
cv2.destroyAllWindows()
print("Autopilot script terminated.", flush=True)
