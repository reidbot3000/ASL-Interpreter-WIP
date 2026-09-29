print("APP STARTING...")
# TO RUN US {PYTHON# APP.PY IN TERMINAL!!! MAKE SURE YOU ARE IN THE RIGTH DIRECTORY!!!}
import os
import urllib.request
import base64
import json
import time
import cv2
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("APP CONTINUING TO BE STARTING...")

MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading hand_landmarker.task model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("✅ Model downloaded successfully!")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('process_frame')
def handle_frame(data):
    try:
        img_data = base64.b64decode(data.split(',')[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        detection_result = detector.detect(mp_image)

        hand_data = []
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                points = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks]
                hand_data.append(points)

        emit('frame_results', json.dumps(hand_data))
    except Exception as e:
        print(f"Error processing frame: {e}")
        emit('frame_results', json.dumps([]))

@socketio.on('save_data')
def handle_save(data):
    hand_data_str = data.get('coords', '[]')
    hand_data = json.loads(hand_data_str)
    
    if hand_data:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        txt_filename = f"hand_coords_{timestamp}.txt"
        
        # Write formatted coordinates to a plain text file
        with open(txt_filename, 'w') as f:
            f.write(f"Hand Landmarks Capture - {timestamp}\n")
            f.write("=" * 45 + "\n\n")
            
            for hand_idx, hand in enumerate(hand_data):
                f.write(f"Hand #{hand_idx + 1}:\n")
                f.write("-" * 20 + "\n")
                for pt_idx, point in enumerate(hand):
                    f.write(f"Point {pt_idx:02d}: x={point['x']:.6f}, y={point['y']:.6f}, z={point['z']:.6f}\n")
                f.write("\n")
                
        print(f"\n📸 SUCCESS: Saved coordinates to {txt_filename}\n")
    else:
        print("\n⚠️ Spacebar pressed, but no hands were detected to save.\n")

if __name__ == '__main__':
    print("🚀 Starting local server! Open your browser to http://127.0.0.1:5000")
    socketio.run(app, host='127.0.0.1', port=5000)